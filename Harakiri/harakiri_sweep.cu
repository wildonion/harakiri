// Harakiri Phase 2 host driver + self-test.
// Sweeps a scalar range [range_start, range_start+range_len) for a target hash160.
// Self-test mode (no --target): picks a random scalar in range, computes its hash160,
// then verifies the sweep finds it.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

#include "field.h"
#include "hash.h"

#define FOUND_NONE   0
#define FOUND_LOCK   1
#define FOUND_READY  2

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif

#include "sweep_kernel.cu"   // kernel + __constant__ defs + FoundResult

// Every CUDA call is checked. Without this, a launch that fails (out of
// resources, bad device pointer, unmapped memory) leaves d_any_left == 0 and
// d_hashes == 0, so the host loop reads it as "nothing left to do" and exits
// silently with iterations=0, total keys=0.
#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) { \
  fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n           -> %s\n", \
          __FILE__, __LINE__, #expr, cudaGetErrorString(_e)); \
  fflush(stderr); exit(2); } } while (0)

// ---- device helper for self-test: hash160(prefix || x) ----
__global__ void hash160_kernel(uint8_t prefix, const uint64_t* x_le, uint32_t* out160) {
  hash160_33_from_x(prefix, x_le, out160);
}

// ---- host-side SHA-256 (single block) for Base58Check checksum ----
static void host_sha256(const uint8_t* msg, size_t len, uint8_t out[32]) {
  static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
  };
  uint32_t h0=0x6a09e667,h1=0xbb67ae85,h2=0x3c6ef372,h3=0xa54ff53a,
         h4=0x510e527f,h5=0x9b05688c,h6=0x1f83d9ab,h7=0x5be0cd19;
  size_t padded = ((len + 8) / 64 + 1) * 64;
  std::vector<uint8_t> buf(padded, 0);
  memcpy(buf.data(), msg, len);
  buf[len] = 0x80;
  uint64_t bits = (uint64_t)len * 8;
  for (int i = 0; i < 8; ++i) buf[padded - 1 - i] = (uint8_t)(bits >> (8*i));
  for (size_t off = 0; off < padded; off += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t)buf[off+i*4]<<24 | (uint32_t)buf[off+i*4+1]<<16 |
             (uint32_t)buf[off+i*4+2]<<8  | (uint32_t)buf[off+i*4+3];
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = ((w[i-15]>>7)|(w[i-15]<<25)) ^ ((w[i-15]>>18)|(w[i-15]<<14)) ^ (w[i-15]>>3);
      uint32_t s1 = ((w[i-2]>>17)|(w[i-2]<<15)) ^ ((w[i-2]>>19)|(w[i-2]<<13)) ^ (w[i-2]>>10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,hh=h7;
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = ((e>>6)|(e<<26)) ^ ((e>>11)|(e<<21)) ^ ((e>>25)|(e<<7));
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      uint32_t S0 = ((a>>2)|(a<<30)) ^ ((a>>13)|(a<<19)) ^ ((a>>22)|(a<<10));
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + maj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h0+=a; h1+=b; h2+=c; h3+=d; h4+=e; h5+=f; h6+=g; h7+=hh;
  }
  uint32_t H[8]={h0,h1,h2,h3,h4,h5,h6,h7};
  for (int i=0;i<8;++i){ out[i*4+0]=(uint8_t)(H[i]>>24); out[i*4+1]=(uint8_t)(H[i]>>16); out[i*4+2]=(uint8_t)(H[i]>>8); out[i*4+3]=(uint8_t)H[i]; }
}

// ---- Base58Check decode: address -> 20-byte hash160 (pubkey-hash, version 0x00) ----
// Returns true and fills out[20] on success (version byte + checksum verified).
static bool base58check_decode(const char* s, uint8_t out[20]) {
  static const char* A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  int map[128]; for (int i=0;i<128;++i) map[i]=-1;
  for (int i=0;i<58;++i) map[(int)A[i]]=i;
  // decode into big number (max 25 bytes)
  std::vector<uint8_t> num;
  for (const char* p=s; *p; ++p) {
    int d = (*p<128) ? map[(int)*p] : -1;
    if (d < 0) return false;
    int carry = d;
    for (size_t i=0;i<num.size();++i) { int x = num[i]*58 + carry; num[i]=(uint8_t)(x&0xFF); carry=x>>8; }
    while (carry) { num.push_back((uint8_t)(carry&0xFF)); carry>>=8; }
  }
  // count leading '1' (zero bytes)
  size_t nz = 0;
  while (s[nz]=='1') ++nz;
  // pad to 25 bytes
  std::vector<uint8_t> full(25, 0);
  for (size_t i=0;i<num.size() && i<25;++i) full[25-1-i] = num[i];
  if (nz > 25) return false;
  uint8_t ck[32];
  host_sha256(full.data(), 21, ck);   // Base58Check: checksum = dbl-sha256(version+hash160)
  host_sha256(ck, 32, ck);
  if (memcmp(ck, full.data()+21, 4) != 0) return false;   // bad checksum
  if (full[0] != 0x00) return false;                      // expect pubkey-hash v0
  memcpy(out, full.data()+1, 20);
  return true;
}

// ---- host helpers ----------------------------------------------------------
static void hex_to_le256(const char* hex, uint64_t* out) {
  // skip optional 0x prefix
  if (hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) hex += 2;
  // big-endian 64-hex-digit value -> 4 LE limbs
  char buf[128];
  int len = (int)strlen(hex);
  for (int i = 0; i < len; i++) buf[len-1-i] = hex[i];
  for (int i = 0; i < len; i++)
    buf[i] = (buf[i] >= 'a') ? (char)(buf[i]-'a'+10) : (char)(buf[i]-'0');
  for (int i = 0; i < 4; i++) {
    uint64_t v = 0;
    for (int j = 0; j < 16 && i*16+j < len; j++) v |= (uint64_t)buf[i*16+j] << (4*j);
    out[i] = v;
  }
}

static void le256_to_hex(const uint64_t* in, char* out) {
  static const char* hexd = "0123456789abcdef";
  for (int i = 3; i >= 0; i--) {
    for (int j = 56; j >= 0; j -= 8) {
      uint8_t b = (uint8_t)(in[i] >> j);
      out[0] = hexd[b >> 4]; out[1] = hexd[b & 0xF]; out += 2;
    }
  }
  *out = 0;
}

static void inc256(uint64_t a[4], uint64_t inc) {
  uint64_t cur = a[0] + inc;
  uint64_t carry = (cur < a[0]) ? 1ull : 0ull;
  a[0] = cur;
  for (int i = 1; i < 4 && carry; ++i) {
    uint64_t nv = a[i] + carry;
    carry = (nv < a[i]) ? 1ull : 0ull;
    a[i] = nv;
  }
}

static void sub256_u64(uint64_t a[4], uint64_t dec) {
  uint64_t borrow = (a[0] < dec) ? 1ull : 0ull;
  a[0] = a[0] - dec;
  for (int i = 1; i < 4; ++i) {
    uint64_t ai = a[i];
    uint64_t bi = borrow;
    a[i] = ai - bi;
    borrow = (ai < bi) ? 1ull : 0ull;
    if (!borrow) break;
  }
}

// a >= b ? (256-bit compare)
static bool ge256(const uint64_t a[4], const uint64_t b[4]) {
  for (int i = 3; i >= 0; --i) { if (a[i] != b[i]) return a[i] > b[i]; }
  return true;
}

// out = a - b   (requires a >= b)
static void sub256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
  uint64_t borrow = 0;
  for (int i = 0; i < 4; ++i) {
    uint64_t bi = b[i] + borrow;
    uint64_t nv = a[i] - bi;
    borrow = (a[i] < bi) ? 1ull : 0ull;
    out[i] = nv;
  }
}

// a += v  (in place, v small)
static void add256_u64(uint64_t a[4], uint64_t v) {
  uint64_t s = a[0] + v;
  uint64_t c = (s < a[0]) ? 1ull : 0ull;
  a[0] = s;
  for (int i = 1; i < 4 && c; ++i) {
    uint64_t ns = a[i] + c;
    c = (ns < a[i]) ? 1ull : 0ull;
    a[i] = ns;
  }
}

// a += b  (in place, 256-bit)
static void add256(uint64_t a[4], const uint64_t b[4]) {
  uint64_t carry = 0;
  for (int i = 0; i < 4; ++i) {
    uint64_t s = a[i] + b[i];
    uint64_t c1 = (s < a[i]) ? 1ull : 0ull;
    uint64_t s2 = s + carry;
    uint64_t c2 = (s2 < s) ? 1ull : 0ull;
    a[i] = s2;
    carry = c1 | c2;
  }
}

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

// q = a / d, rem = a % d  (256-bit a, 64-bit d; uses 128/64 HW division on x64)
static void div256_u64(const uint64_t a[4], uint64_t d, uint64_t q[4], uint64_t* rem) {
  uint64_t r = 0;
  for (int i = 3; i >= 0; --i) {
#if defined(_MSC_VER) && defined(_M_X64)
    q[i] = _udiv128(r, a[i], d, &r);
#else
    // portable fallback: binary long division is unnecessary for our call sites
    // (d < 2^32, so use the simple chained 64-bit divide)
    __uint128_t cur = ((__uint128_t)r << 64) | a[i];
    q[i] = (uint64_t)(cur / d);
    r = (uint64_t)(cur % d);
#endif
  }
  *rem = r;
}

// scalar_le * G -> (X, Y) on host using Python-verified path (device kernel scalarMulBaseAffine)
__global__ void scalarMulBaseKernel(const uint64_t* scalars, uint64_t* ox, uint64_t* oy, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  scalarMulBaseAffine(scalars + (size_t)idx*4, ox + (size_t)idx*4, oy + (size_t)idx*4);
}

static void scalar_mul_g(uint64_t k_le[4], uint64_t x[4], uint64_t y[4]) {
  uint64_t *d_s, *d_x, *d_y;
  cudaMalloc(&d_s, 32); cudaMalloc(&d_x, 32); cudaMalloc(&d_y, 32);
  cudaMemcpy(d_s, k_le, 32, cudaMemcpyHostToDevice);
  scalarMulBaseKernel<<<1,1>>>(d_s, d_x, d_y, 1);
  cudaDeviceSynchronize();
  cudaMemcpy(x, d_x, 32, cudaMemcpyDeviceToHost);
  cudaMemcpy(y, d_y, 32, cudaMemcpyDeviceToHost);
  cudaFree(d_s); cudaFree(d_x); cudaFree(d_y);
}

// --- warmup: burn-in kernel to ramp GPU clocks before a short sweep ----------
// Short ranges (2^30..2^40) finish in seconds; a cold GPU spends that entire
// time at idle clocks (we measured 159 Mkeys/s cold vs ~500 warm on the 3050).
// Running a few seconds of dense ALU work first lets boost clocks kick in so
// the actual sweep runs at full speed.
__global__ void warmup_kernel(unsigned long long* dummy, unsigned long long rounds) {
  unsigned long long x = ((unsigned long long)blockIdx.x << 32) ^ (unsigned long long)threadIdx.x;
  unsigned long long y = 0x9E3779B97F4A7C15ULL;
  for (unsigned long long i = 0; i < rounds; ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    y = (y ^ x) * 0xBF58476D1CE4E5B9ULL;
  }
  if (y == 0xDEADBEEFDEADBEEFULL) *dummy = y;   // never true; defeats DCE
}

// ---- section ranges: split [range_start, range_start+range_len) into n slices ----
struct SectionRange {
  uint64_t start[4];   // section start scalar
  uint64_t len[4];     // section length in keys
};

static void partition_range(const uint64_t range_start[4], const uint64_t range_len[4],
                            int n, std::vector<SectionRange>& out) {
  out.clear();
  if (n < 1) n = 1;
  // sec_len = ceil(range_len / n)
  uint64_t sec_len[4];
  {
    uint64_t q[4], r;
    div256_u64(range_len, (uint64_t)n, q, &r);
    if (r) add256_u64(q, 1);
    for (int k=0;k<4;++k) sec_len[k] = q[k];
  }
  uint64_t cur[4] = {range_start[0], range_start[1], range_start[2], range_start[3]};
  uint64_t end[4]  = {range_start[0], range_start[1], range_start[2], range_start[3]};
  add256(end, range_len);                    // end = start + len
  for (int s = 0; s < n; ++s) {
    SectionRange sr;
    for (int k=0;k<4;++k) sr.start[k] = cur[k];
    uint64_t rem[4];
    sub256(end, cur, rem);                   // remaining keys
    bool rem_lt = false;                     // remaining < sec_len ?
    for (int k=3;k>=0;--k) { if (rem[k] < sec_len[k]) { rem_lt = true; break; } if (rem[k] > sec_len[k]) break; }
    if (rem_lt) { for (int k=0;k<4;++k) sr.len[k] = rem[k]; }
    else        { for (int k=0;k<4;++k) sr.len[k] = sec_len[k]; }
    out.push_back(sr);
    add256(cur, sr.len);
  }
}

// ---- per-device sweep worker (one std::thread per GPU) ----
struct WorkerCtx {
  int dev;
  std::vector<SectionRange> sections;   // sections this device owns (as groups in one grid)
  uint32_t B;
  uint32_t threads_per_block;
  uint64_t threadsTotal;                // 0 = auto-detect per device
  bool jobs_explicit;
  uint32_t maxbatch_cap;
  int iters;
  bool verbose;
  bool self_test;
  uint8_t target_hash160[20];
  uint32_t prefix_le;
  int warmup_sec;
  bool backward;                          // two-front race: forward half + backward half per section
  bool range_is_infinite;
  uint64_t full_range_len[4];           // whole range (for ETA)
  // shared across devices:
  int* h_global_stop;                   // host-mapped; kernel reads it via device pointer
  FoundResult* shared_found;            // winner's result
  std::mutex* mtx;
  // output:
  unsigned long long total_hashes;
  double secs;
  int iter;                 // loop index at exit (not launch count)
  int launches_done;        // launches actually executed, find launch included
  int found_on_launch;      // 0-based launch that found the key, -1 if none
  bool found;
};

static void device_worker(WorkerCtx& W) {
  cudaSetDevice(W.dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, W.dev);
  printf("GPU %d       : %s (cc %d.%d, %d SMs)\n", W.dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount);

  uint64_t threadsTotal = W.threadsTotal;
  if (!W.jobs_explicit) {
    uint64_t sm = (uint64_t)prop.multiProcessorCount;
    threadsTotal = (sm * 256) & ~0xFFull;
    if (threadsTotal < 256) threadsTotal = 256;
  }
  if (W.self_test && !W.jobs_explicit && threadsTotal > 256) threadsTotal = 256;

  int S = (int)W.sections.size();
  if (S < 1) S = 1;
  // --backward: each section becomes TWO racing fronts (a forward half from the
  // section start + a backward half from the section end), so the effective
  // group count doubles. Without backward: one forward group per section.
  const bool back = W.backward;
  int G = back ? 2*S : S;                     // total groups in the grid
  // groupSize = threadsTotal / G, rounded down to a multiple of 256
  uint64_t groupSize = (threadsTotal / (uint64_t)G) & ~0xFFull;
  if (groupSize < 256) groupSize = 256;
  threadsTotal = groupSize * (uint64_t)G;

  // ---- constants + table generation (constant memory is per-device) ----
  cudaMemcpyToSymbol(c_target_prefix, &W.prefix_le, 4);
  cudaMemcpyToSymbol(c_target_hash160, W.target_hash160, 20);

  const uint32_t half = W.B/2;
  if (half == 0 || (half & 1u) || half > MAX_BATCH_SIZE/2) {
    fprintf(stderr, "error: invalid batch size %u (must be even, 2..%d)\n", W.B, MAX_BATCH_SIZE);
    exit(1);
  }
  {
    std::vector<uint64_t> h_sc(half*4), h_gx(half*4), h_gy(half*4);
    for (uint32_t k = 0; k < half; ++k) {
      h_sc[k*4+0] = (uint64_t)(k+1); h_sc[k*4+1]=0; h_sc[k*4+2]=0; h_sc[k*4+3]=0;
    }
    uint64_t *d_s, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_s, half*32)); CUDA_CHECK(cudaMalloc(&d_x, half*32)); CUDA_CHECK(cudaMalloc(&d_y, half*32));
    CUDA_CHECK(cudaMemcpy(d_s, h_sc.data(), half*32, cudaMemcpyHostToDevice));
    int blk = (int)((half + 255)/256);
    scalarMulBaseKernel<<<blk,256>>>(d_s, d_x, d_y, (int)half);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_gx.data(), d_x, half*32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gy.data(), d_y, half*32, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Gx, h_gx.data(), half*32));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Gy, h_gy.data(), half*32));
    cudaFree(d_s); cudaFree(d_x); cudaFree(d_y);
  }

  // ---- per-group layout (backward: two racing fronts per section) ----
  // group 2s   = FORWARD half of section s: [A, A+len/2),   dir +1, marches up
  // group 2s+1 = BACKWARD half of section s: [A+len/2, A+len), dir -1, marches down
  // Without backward: group s = whole section [A, A+len), dir +1.
  std::vector<int>         grp_sec(G);
  std::vector<int>         grp_dir(G);
  std::vector<uint64_t>    grp_start(G*4);   // group's base A
  std::vector<uint64_t>    grp_len(G*4);     // keys this group owns
  for (int g = 0; g < G; ++g) {
    int s, d;
    if (!back) { s = g; d = +1; }
    else { s = g / 2; d = (g & 1) ? -1 : +1; }
    grp_sec[g] = s; grp_dir[g] = d;
    for (int k=0;k<4;++k) grp_start[g*4+k] = W.sections[s].start[k];
    if (!back) {
      for (int k=0;k<4;++k) grp_len[g*4+k] = W.sections[s].len[k];
    } else {
      // halfk = ceil(len/2); forward gets the lower half, backward the upper half.
      // Named halfk to avoid shadowing the batch `half` (W.B/2) from above.
      uint64_t halfk[4], rem;
      div256_u64(W.sections[s].len, 2, halfk, &rem);    // halfk = len/2 (floor)
      if (rem) add256_u64(halfk, 1);                    // halfk = ceil(len/2)
      if (d > 0) { for (int k=0;k<4;++k) grp_len[g*4+k] = halfk[k]; }
      else {
        // len - ceil(len/2) wraps to ~2^256 when len == 0; clamp to 0.
        bool zero = (W.sections[s].len[0]|W.sections[s].len[1]|
                     W.sections[s].len[2]|W.sections[s].len[3]) == 0;
        for (int k=0;k<4;++k) grp_len[g*4+k] = zero ? 0 : (W.sections[s].len[k] - halfk[k]);
      }
    }
  }

  // ---- per-group chunk lengths (keys per thread) ----
  std::vector<uint64_t> grp_chunk(G*4);
  for (int g=0;g<G;++g) {
    uint64_t q[4], r;
    div256_u64(&grp_len[g*4], groupSize, q, &r);
    if (r) add256_u64(q, 1);
    uint64_t tmp[4], rm;
    div256_u64(q, (uint64_t)W.B, tmp, &rm);
    if (rm) add256_u64(q, (uint64_t)W.B - rm);
    for (int k=0;k<4;++k) grp_chunk[g*4+k] = q[k];
    if ((grp_chunk[g*4+0]|grp_chunk[g*4+1]|grp_chunk[g*4+2]|grp_chunk[g*4+3]) == 0)
      add256_u64(&grp_chunk[g*4], (uint64_t)W.B);
  }

  if ((threadsTotal & (W.threads_per_block - 1)) != 0)
    fprintf(stderr, "warning: jobs=%llu is not a multiple of threads/block=%u; "
                    "%llu threads will launch\n",
            (unsigned long long)threadsTotal, W.threads_per_block,
            (unsigned long long)(((threadsTotal + W.threads_per_block - 1) / W.threads_per_block) * W.threads_per_block));

  // ---- per-thread start/count: thread gid -> group = gid/groupSize, local = gid%groupSize ----
  // forward thread : S0 = A + local*B + half   (march up)
  // backward thread: S0 = A+len - local*B - half (march down; A+len = section end)
  std::vector<uint64_t> h_start(threadsTotal*4), h_cnt(threadsTotal*4);
  for (uint64_t gid = 0; gid < threadsTotal; ++gid) {
    uint64_t group = gid / groupSize;
    uint64_t local = gid % groupSize;
    int s = grp_sec[(int)group];
    int d = grp_dir[(int)group];
    uint64_t S0[4];
    if (d > 0) {
      for (int k=0;k<4;++k) S0[k] = W.sections[s].start[k];
      uint64_t addv = (uint64_t)local * (uint64_t)W.B;
      for (int k=0;k<4 && addv;++k){ uint64_t old=S0[k]; S0[k]=old+addv; addv=(S0[k]<old)?1ull:0ull; }
      inc256(S0, (uint64_t)(W.B / 2));
    } else {
      // section end A+len
      uint64_t end4[4] = {W.sections[s].start[0], W.sections[s].start[1],
                          W.sections[s].start[2], W.sections[s].start[3]};
      add256(end4, W.sections[s].len);
      uint64_t subv = (uint64_t)local * (uint64_t)W.B + (uint64_t)(W.B / 2);
      for (int k=0;k<4 && subv;++k){ uint64_t old=end4[k]; end4[k]=old-subv; subv=(old<subv)?1ull:0ull; }
      for (int k=0;k<4;++k) S0[k] = end4[k];
    }
    for (int k=0;k<4;++k) { h_start[gid*4+k] = S0[k]; h_cnt[gid*4+k] = grp_chunk[(int)group*4+k]; }
  }

  // ---- per-group stride point c_Jx[g] = (groupSize*B)*G ----
  // The J magnitude is the same for every group: (groupSize*B)*G. Forward groups
  // upload +J; backward groups upload -J = (Jx, p-Jy) so the identical point-add
  // in the kernel advances the base point DOWNWARD.
  {
    uint64_t jsc[4] = {groupSize * (uint64_t)W.B, 0, 0, 0}, Jx[4], Jy[4];
    scalar_mul_g(jsc, Jx, Jy);
    const uint64_t P_LE[4] = {0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
                              0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    uint64_t Jneg[4];
    sub256(P_LE, Jy, Jneg);                     // Jneg = p - Jy  (negated y)
    for (int g=0;g<G;++g) {
      int dir = grp_dir[g];
      cudaMemcpyToSymbol(c_Jx, Jx, 32, (size_t)g*32);
      cudaMemcpyToSymbol(c_Jy, (dir > 0) ? Jy : Jneg, 32, (size_t)g*32);
      cudaMemcpyToSymbol(c_dir, &dir, sizeof(int), (size_t)g*sizeof(int));
    }
  }

  uint64_t *d_start, *d_cnt, *d_Px, *d_Py, *d_Rx, *d_Ry;
  int *d_found_flag;
  FoundResult *d_found_result;
  unsigned long long *d_hashes;
  unsigned int *d_any_left;

  CUDA_CHECK(cudaMalloc(&d_start, threadsTotal*32));
  CUDA_CHECK(cudaMalloc(&d_cnt,   threadsTotal*32));
  CUDA_CHECK(cudaMalloc(&d_Px,    threadsTotal*32));
  CUDA_CHECK(cudaMalloc(&d_Py,    threadsTotal*32));
  CUDA_CHECK(cudaMalloc(&d_Rx,    threadsTotal*32));
  CUDA_CHECK(cudaMalloc(&d_Ry,    threadsTotal*32));
  CUDA_CHECK(cudaMalloc(&d_found_flag, 4));
  CUDA_CHECK(cudaMalloc(&d_found_result, sizeof(FoundResult)));
  CUDA_CHECK(cudaMalloc(&d_hashes, 8));
  CUDA_CHECK(cudaMalloc(&d_any_left, 4));

  CUDA_CHECK(cudaMemcpy(d_start, h_start.data(), threadsTotal*32, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cnt,   h_cnt.data(),   threadsTotal*32, cudaMemcpyHostToDevice));

  // compute P = start*G
  int blk = (int)((threadsTotal + 255)/256);
  scalarMulBaseKernel<<<blk,256>>>(d_start, d_Px, d_Py, (int)threadsTotal);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- VERIFY: hash thread 30's base point (should equal target for selftest) ----
  if (W.self_test) {
    uint64_t h_x[4], h_y[4];
    cudaMemcpy(h_x, d_Px + 30*4, 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_Py + 30*4, 32, cudaMemcpyDeviceToHost);
    uint8_t prefix = (uint8_t)(h_y[0] & 1) ? 0x03 : 0x02;
    uint64_t *d_x2; uint32_t *d_h2;
    cudaMalloc(&d_x2, 32); cudaMalloc(&d_h2, 20);
    cudaMemcpy(d_x2, h_x, 32, cudaMemcpyHostToDevice);
    hash160_kernel<<<1,1>>>(prefix, d_x2, d_h2);
    cudaDeviceSynchronize();
    uint32_t h2[5];
    cudaMemcpy(h2, d_h2, 20, cudaMemcpyDeviceToHost);
    cudaFree(d_x2); cudaFree(d_h2);
    printf("  [verify] thread30 base hash: ");
    for (int k=0;k<5;++k) printf("%08x", h2[k]);
    printf("\n");
    printf("  [verify] target           : ");
    for (int k=0;k<20;++k) printf("%02x", W.target_hash160[k]);
    printf("\n");
    bool same = true;
    for (int k=0;k<5;++k) {
      uint32_t t = (uint32_t)W.target_hash160[k*4+0]
                 | ((uint32_t)W.target_hash160[k*4+1] << 8)
                 | ((uint32_t)W.target_hash160[k*4+2] << 16)
                 | ((uint32_t)W.target_hash160[k*4+3] << 24);
      if (t != h2[k]) same = false;
    }
    printf("  [verify] match=%s\n", same ? "YES" : "NO");
  }

  // ---- warmup: ramp GPU clocks before short sweeps (skipped in selftest) ----
  if (!W.self_test && W.warmup_sec > 0) {
    unsigned long long *d_dummy;
    cudaMalloc(&d_dummy, 8);
    printf("[GPU %d warmup] running %d s burn-in to ramp clocks ...\n", W.dev, W.warmup_sec);
    auto w0 = std::chrono::high_resolution_clock::now();
    double el = 0.0;
    while (el < (double)W.warmup_sec) {
      warmup_kernel<<<80, 256>>>(d_dummy, 20000000ULL);
      cudaDeviceSynchronize();
      el = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - w0).count();
    }
    cudaFree(d_dummy);
    printf("[GPU %d warmup] done (%.2f s)\n", W.dev, el);
  }

  // ---- launch loop ----
  unsigned long long total_hashes = 0;
  auto t0 = std::chrono::high_resolution_clock::now();
  int iter = 0;
  // Launches actually executed, including the one that found the key. 'iter' is
  // the 0-based loop index and is only bumped on the non-found path, so a run
  // that stops on a find would otherwise report 0 regardless of the work done.
  int launches_done = 0;
  int found_on_launch = -1;     // -1 = not found; else 0-based launch index
  bool stop = false;
  FoundResult h_found;
  memset(&h_found, 0, sizeof(h_found));

  // max batches per launch: use the LARGEST chunk (threads with smaller chunks
  // exit early when their rem < B; any_left only set by threads with work left)
  uint64_t max_chunk[4] = {0,0,0,0};
  for (int g=0;g<G;++g)
    if (ge256(&grp_chunk[g*4], max_chunk))
      for (int k=0;k<4;++k) max_chunk[k] = grp_chunk[g*4+k];
  uint64_t mb_rem;
  uint64_t mbq[4];
  div256_u64(max_chunk, (uint64_t)W.B, mbq, &mb_rem);
  uint64_t mb = mbq[0];
  if (mb > 0xFFFFFFFFull) mb = 0xFFFFFFFFull;
  uint32_t max_batches = (uint32_t)mb;
  if (W.maxbatch_cap > 0 && max_batches > W.maxbatch_cap) max_batches = W.maxbatch_cap;
  bool auto_timebox = (W.maxbatch_cap == 0);
  if (auto_timebox) {
    const uint64_t target_keys_per_launch = 5000000000ull;
    uint64_t keys_per_batch = (uint64_t)threadsTotal * (uint64_t)W.B;
    uint64_t cap_batches = target_keys_per_launch / keys_per_batch;
    if (cap_batches < 1) cap_batches = 1;
    if ((uint64_t)max_batches > cap_batches) max_batches = (uint32_t)cap_batches;
  }
  {
    char buf[65]; le256_to_hex(max_chunk, buf);
    int grid = (int)((threadsTotal + W.threads_per_block - 1) / W.threads_per_block);
    printf("config      : batch=%u keys per inversion | threads/block=%u | "
           "jobs=%llu total threads | grid=%u blocks | sections=%d (groupSize=%llu) | backward=%s | max_batches=%u per launch%s | max_chunk=%s\n",
           W.B, W.threads_per_block, (unsigned long long)threadsTotal,
           grid, S, (unsigned long long)groupSize, back ? "yes" : "no", max_batches,
           auto_timebox ? " (auto-timeboxed)" : "", buf);
    if (W.dev == 0) {
      printf("\n=== Harakiri Config Explained ===\n");
      printf("- batch            keys hashed per one field-inversion batch (B)\n"
             "- threads/block    block size (fixed 256, kernel launch bounds)\n"
             "- jobs             total worker threads = threads/block x grid blocks\n"
             "- grid             number of blocks = jobs / threads/block\n"
             "- sections         range split into N parallel slices; groupSize = jobs/N\n"
             "                   threads per slice (one slice per GPU on multi-GPU boxes)\n"
             "- backward         two-front race per section: a forward half from the\n"
             "                   section start + a backward half from the section end\n"
             "- max_batches      batches per thread per kernel launch (auto-timeboxed to ~10s; add --maxbatch to override)\n"
             "- max_chunk        keys each thread owns = range / jobs\n");
    }
  }

  // device pointer to shared host-mapped stop flag
  int* d_global_stop = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&d_global_stop, W.h_global_stop, 0));

  while (!stop) {
    if (*W.h_global_stop != 0) { stop = true; break; }   // another device found
    if (W.iters > 0 && iter >= W.iters) break;
    cudaMemset(d_found_flag, 0, 4);
    cudaMemset(d_hashes, 0, 8);
    cudaMemset(d_any_left, 0, 4);

    int grid = (int)((threadsTotal + W.threads_per_block - 1) / W.threads_per_block);
    kernel_point_add_and_check_oneinv<<<grid, W.threads_per_block>>>(
        d_Px, d_Py, d_Rx, d_Ry, d_start, d_cnt, threadsTotal, groupSize, W.B, max_batches,
        d_found_flag, d_global_stop, d_found_result, d_hashes, d_any_left);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    ++launches_done;

    unsigned long long h_hashes = 0;
    cudaMemcpy(&h_hashes, d_hashes, 8, cudaMemcpyDeviceToHost);
    total_hashes += h_hashes;

    int h_flag = 0;
    cudaMemcpy(&h_flag, d_found_flag, 4, cudaMemcpyDeviceToHost);
    if (h_flag == FOUND_READY) {
      cudaMemcpy(&h_found, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
      found_on_launch = iter;
      printf("[GPU %d][dbg] FOUND at launch %d: threadId=%d finder_batch=%d\n",
             W.dev, iter, h_found.threadId, h_found.iter);
      // publish winner under lock
      {
        std::lock_guard<std::mutex> lk(*W.mtx);
        if (W.shared_found->scalar[0] == 0)
          memcpy(W.shared_found, &h_found, sizeof(FoundResult));
      }
      *W.h_global_stop = 1;
      stop = true;
      break;
    }

    unsigned int h_any = 0;
    cudaMemcpy(&h_any, d_any_left, 4, cudaMemcpyDeviceToHost);
    if (W.verbose && (iter < 3 || (iter % 1000 == 0))) {
      uint64_t dbg_cnt[4];
      cudaMemcpy(dbg_cnt, d_cnt, 32, cudaMemcpyDeviceToHost);
      printf("  [GPU %d][dbg] launch %d: hashes=%llu flag=%d any_left=%u rem0=%llu\n",
             W.dev, iter, h_hashes, h_flag, h_any, dbg_cnt[0]);
    }
    // If the first launch did zero work while threads still have a positive
    // remaining count, the kernel bailed at its entry gate. Say so instead of
    // printing "0 keys, done" and looking like a finished (empty) sweep.
    if (iter == 0 && h_hashes == 0 && h_any == 0) {
      uint64_t dbg_cnt[4];
      cudaMemcpy(dbg_cnt, d_cnt, 32, cudaMemcpyDeviceToHost);
      fprintf(stderr,
        "[GPU %d][FATAL] first launch hashed nothing and no thread reported work left.\n"
        "         thread0 remaining keys = %llu (expected >= %u).\n"
        "         The kernel exited at its entry gate before hashing.\n", W.dev,
        (unsigned long long)dbg_cnt[0], W.B);
      fflush(stderr);
      exit(3);
    }
    if (h_any == 0) { stop = true; break; }

    cudaMemcpy(d_Px, d_Rx, threadsTotal*32, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_Py, d_Ry, threadsTotal*32, cudaMemcpyDeviceToDevice);
    ++iter;

    if (!W.self_test && !W.range_is_infinite && (auto_timebox || W.verbose)) {
      auto now = std::chrono::high_resolution_clock::now();
      double el = std::chrono::duration<double>(now - t0).count();
      if (el > 0 && total_hashes > 0) {
        double rate = total_hashes / el;
        double total_keys = (double)W.full_range_len[0]
                          + (double)W.full_range_len[1] * 18446744073709551616.0
                          + (double)W.full_range_len[2] * 340282366920938463463374607431768211456.0
                          + (double)W.full_range_len[3] * 6277101735386680763835789423207666416102355444464034512896.0;
        double done = (total_hashes < total_keys) ? total_hashes : total_keys;
        double rem = total_keys - done;
        char ebuf[64];
        if (rem / rate >= 86400.0*365.0)       sprintf(ebuf, "%.1f years", rem/rate/(86400.0*365.0));
        else if (rem/rate >= 86400.0)          sprintf(ebuf, "%.1f days",  rem/rate/86400.0);
        else if (rem/rate >= 3600.0)           sprintf(ebuf, "%.1f hours", rem/rate/3600.0);
        else if (rem/rate >= 60.0)             sprintf(ebuf, "%.1f min",   rem/rate/60.0);
        else                                   sprintf(ebuf, "%.1f s",     rem/rate);
        printf("  [GPU %d][%7.1f s] %13.3f Mkeys/s | done %.3e / %.3e keys | ETA %s\n",
               W.dev, el, rate/1e6, done, total_keys, ebuf);
      }
    } else if (!W.self_test && W.range_is_infinite && W.verbose) {
      auto now = std::chrono::high_resolution_clock::now();
      double el = std::chrono::duration<double>(now - t0).count();
      if (el > 0)
        printf("  [GPU %d][%7.1f s] %13.3f Mkeys/s | scanned %.3e keys\n",
               W.dev, el, total_hashes/el/1e6, (double)total_hashes);
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();

  cudaFree(d_start); cudaFree(d_cnt); cudaFree(d_Px); cudaFree(d_Py);
  cudaFree(d_Rx); cudaFree(d_Ry); cudaFree(d_found_flag);
  cudaFree(d_found_result); cudaFree(d_hashes); cudaFree(d_any_left);

  W.total_hashes = total_hashes;
  W.secs = secs;
  W.iter = iter;
  W.launches_done = launches_done;
  W.found_on_launch = found_on_launch;
  W.found = (h_found.scalar[0] != 0);
}

// ---- main ------------------------------------------------------------------
int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);   // unbuffered for live output
  uint32_t B = 128;                       // points per inversion (even, <= 1024)
  uint32_t threads_per_block = 256;
  uint64_t threadsTotal = 16 * 256;       // default: 16 SM * 256 threads
  int iters = 0;                          // 0 = run until range exhausted or found
  uint32_t maxbatch_cap = 0;              // 0 = no cap (whole chunk per launch)
  std::string range_hex = "";             // "start_hex:length_dec" or just "start_hex"
  uint64_t range_start[4];
  uint64_t range_len[4];
  bool self_test = true;
  uint8_t target_hash160[20];
  bool have_target = false;
  bool target_was_address = false;
  bool verbose = false;
  std::string out_file;   // --out FILE: also write result (and summary) to file
  int warmup_sec = 0;     // --warmup N: run N s of GPU burn-in before the sweep
  int sections = 1;       // --sections N: split range into N parallel sections
  bool backward = false;  // --backward: two-front race (forward + backward halves per section)

  // parse args
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--batch" && i+1<argc) B = (uint32_t)atoi(argv[++i]);
    else if (a == "--threads" && i+1<argc) threads_per_block = (uint32_t)atoi(argv[++i]);
    else if (a == "--jobs" && i+1<argc) threadsTotal = (uint64_t)atoll(argv[++i]);
    else if (a == "--range" && i+1<argc) range_hex = argv[++i];
    else if (a == "--target" && i+1<argc) {
      std::string h = argv[++i];
      // accept either 40-hex hash160 or a Base58Check address
      bool is_hex = (h.size() == 40);
      for (char c : h) if (!isxdigit((unsigned char)c)) { is_hex = false; break; }
      if (is_hex) {
        for (int k = 0; k < 20; ++k) {
          unsigned v; sscanf(h.substr(k*2,2).c_str(), "%x", &v);
          target_hash160[k] = (uint8_t)v;
        }
      } else {
        if (!base58check_decode(h.c_str(), target_hash160)) {
          fprintf(stderr, "invalid target (need 40-hex hash160 or valid Bitcoin address)\n");
          return 1;
        }
        target_was_address = true;
      }
      have_target = true; self_test = false;
    }
    else if (a == "--iters" && i+1<argc) iters = atoi(argv[++i]);
    else if (a == "--maxbatch" && i+1<argc) maxbatch_cap = (uint32_t)atoi(argv[++i]);
    else if (a == "--out" && i+1<argc) out_file = argv[++i];
    else if (a == "--warmup" && i+1<argc) warmup_sec = atoi(argv[++i]);
    else if (a == "--sections" && i+1<argc) sections = atoi(argv[++i]);
    else if (a == "--backward") backward = true;
    else if (a == "--verbose") verbose = true;
    else if (a == "--selftest") self_test = true;
  }

  if (range_hex.empty()) {
    fprintf(stderr, "usage: harakiri_sweep [--batch B] [--threads T] [--range start_hex[:len_dec]] "
            "[--target hex160] [--iters N] [--sections N] [--backward] [--selftest]\n");
    return 1;
  }

  size_t colon = range_hex.find(':');
  std::string start_hex = colon==std::string::npos ? range_hex : range_hex.substr(0, colon);
  hex_to_le256(start_hex.c_str(), range_start);
  bool range_is_infinite = (colon == std::string::npos);
  range_len[0]=range_len[1]=range_len[2]=range_len[3]=0;
  if (colon != std::string::npos) {
    std::string tail = range_hex.substr(colon+1);
    // "start:len" (decimal length) vs "start:end" (hex, inclusive)
    bool hex_end = (tail.size() >= 2 && tail[0]=='0' && (tail[1]=='x'||tail[1]=='X'));
    if (!hex_end) { for (char c : tail) if (isxdigit((unsigned char)c) && !isdigit((unsigned char)c)) { hex_end=true; break; } }
    if (hex_end) {
      // end must be >= start; length = end - start + 1
      uint64_t end[4]; hex_to_le256(tail.c_str(), end);
      if (!ge256(end, range_start)) { fprintf(stderr, "error: end < start\n"); return 1; }
      sub256(end, range_start, range_len);
      add256_u64(range_len, 1);          // inclusive end
    } else {
      uint64_t len = strtoull(tail.c_str(), nullptr, 10);
      range_len[0] = len;
    }
  } else {
    // infinite range -> use 2^96 as "huge" cap
    range_len[1] = 0x1ULL;
  }

  printf("\n=== Harakiri Phase 2 sweep ===\n");
  {
    char buf[65]; le256_to_hex(range_start, buf);
    printf("range start : %s\n", buf);
  }
  printf("batch size  : %u\n", B);
  printf("selftest    : %s\n", self_test ? "yes" : "no");
  printf("backward    : %s\n", backward ? "yes (two-front race)" : "no");
  if (have_target) {
    printf("target      : ");
    for (int k=0;k<20;++k) printf("%02x", target_hash160[k]);
    printf("  (from %s)\n", target_was_address ? "address" : "hex");
  }

  // ---- self test: pick random scalar in range, compute target hash160 ----
  uint64_t found_target[4];
  if (self_test) {
    // seed from range_start, use deterministic pseudo-random offset within first 2^20
    uint64_t pick = 0x12345ULL;
    if (range_len[0] > 1) pick = (uint64_t)(((uint64_t)range_start[0] * 2654435761ULL) % range_len[0]);
    uint64_t sc[4] = {range_start[0], range_start[1], range_start[2], range_start[3]};
    inc256(sc, pick);
    uint64_t X[4], Y[4];
    scalar_mul_g(sc, X, Y);
    uint8_t prefix = (uint8_t)(Y[0] & 1) ? 0x03 : 0x02;
    uint32_t h[5];
    // device hash via kernel
    uint64_t *d_x; uint32_t *d_h;
    cudaMalloc(&d_x, 32); cudaMalloc(&d_h, 20);
    cudaMemcpy(d_x, X, 32, cudaMemcpyHostToDevice);
    hash160_kernel<<<1,1>>>(prefix, d_x, d_h);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d_h, 20, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_h);
    for (int k=0;k<5;++k) {
      target_hash160[k*4+0]=(uint8_t)(h[k]      );
      target_hash160[k*4+1]=(uint8_t)(h[k] >> 8 );
      target_hash160[k*4+2]=(uint8_t)(h[k] >> 16);
      target_hash160[k*4+3]=(uint8_t)(h[k] >> 24);
    }
    for (int k=0;k<4;++k) found_target[k] = sc[k];
    printf("selftest scalar : ");
    { char buf[65]; le256_to_hex(found_target, buf); printf("%s\n", buf); }
    printf("selftest target : ");
    for (int k=0;k<20;++k) printf("%02x", target_hash160[k]);
    printf("\n");
  }


  // ---- constants: c_target_prefix (LE u32 of first 4 hash bytes) ----
  uint32_t prefix_le = (uint32_t)target_hash160[0]
                     | ((uint32_t)target_hash160[1] << 8)
                     | ((uint32_t)target_hash160[2] << 16)
                     | ((uint32_t)target_hash160[3] << 24);

  // ---- device plan: sections across GPUs ----
  bool jobs_explicit = false;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--jobs") jobs_explicit = true;

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 1) { fprintf(stderr, "no CUDA device found\n"); return 1; }
  {
    int drv = 0;
    cudaDriverGetVersion(&drv);
    printf("CUDA driver/runtime: driver %d, %d device(s) visible\n", drv, ndev);
    for (int i = 0; i < ndev; ++i) {
      cudaDeviceProp p;
      if (cudaGetDeviceProperties(&p, i) == cudaSuccess)
        printf("  device %d: %s (cc %d.%d, %d SMs, canMapHostMemory=%d, "
               "unifiedAddressing=%d)\n", i, p.name, p.major, p.minor,
               p.multiProcessorCount, (int)p.canMapHostMemory,
               (int)p.unifiedAddressing);
    }
  }
if (sections < 1) { fprintf(stderr, "--sections must be >= 1\n"); return 1; }
  if (sections > MAX_SECTIONS) { fprintf(stderr, "--sections too large (max %d)\n", MAX_SECTIONS); return 1; }
  if (backward && 2*sections > MAX_SECTIONS) { fprintf(stderr, "--backward doubles the grid: max --sections with --backward is %d\n", MAX_SECTIONS/2); return 1; }
  if (self_test) sections = 1;   // selftest uses one section on device 0

  int used_gpus = (ndev > 1 && sections > 1) ? ((sections < ndev) ? sections : ndev) : 1;
  if (used_gpus > 1)
    printf("GPU plan    : using %d of %d devices, %d section(s) spread across them\n",
           used_gpus, ndev, sections);

  // ---- partition the range into N sections, assign round-robin to GPUs ----
  std::vector<SectionRange> all_sections;
  partition_range(range_start, range_len, sections, all_sections);

  std::vector<WorkerCtx> workers(used_gpus);
  for (int d = 0; d < used_gpus; ++d) {
    workers[d].dev = d;
    workers[d].B = B;
    workers[d].threads_per_block = threads_per_block;
    workers[d].threadsTotal = jobs_explicit ? threadsTotal : 0;
    workers[d].jobs_explicit = jobs_explicit;
    workers[d].maxbatch_cap = maxbatch_cap;
    workers[d].iters = iters;
    workers[d].verbose = verbose;
    workers[d].self_test = self_test;
    for (int k=0;k<20;++k) workers[d].target_hash160[k] = target_hash160[k];
    workers[d].prefix_le = prefix_le;
workers[d].warmup_sec = warmup_sec;
    workers[d].backward = backward;
    workers[d].range_is_infinite = range_is_infinite;
    for (int k=0;k<4;++k) workers[d].full_range_len[k] = range_len[k];
  }
  for (int s = 0; s < sections; ++s) {
    int d = (used_gpus > 1) ? (s % used_gpus) : 0;
    workers[d].sections.push_back(all_sections[s]);
  }

  // shared host-mapped stop flag + winner slot
  int* h_global_stop = nullptr;
  if (cudaHostAlloc(&h_global_stop, 4, cudaHostAllocMapped) != cudaSuccess) {
    fprintf(stderr, "error: cannot allocate host-mapped stop flag\n");
    return 1;
  }
  *h_global_stop = 0;
  FoundResult shared_found;
  memset(&shared_found, 0, sizeof(shared_found));
  std::mutex mtx;
  for (int d = 0; d < used_gpus; ++d) {
    workers[d].h_global_stop = h_global_stop;
    workers[d].shared_found = &shared_found;
    workers[d].mtx = &mtx;
  }

  // ---- run: one thread per GPU (or inline for a single GPU) ----
  auto w0 = std::chrono::high_resolution_clock::now();
  if (used_gpus == 1) {
    device_worker(workers[0]);
  } else {
    std::vector<std::thread> ths;
    for (int d = 0; d < used_gpus; ++d)
      ths.emplace_back(device_worker, std::ref(workers[d]));
    for (auto& t : ths) t.join();
  }
  auto w1 = std::chrono::high_resolution_clock::now();
  double wall_secs = std::chrono::duration<double>(w1 - w0).count();

  // ---- aggregate results ----
  unsigned long long total_hashes = 0;
  int total_iter = 0;
  int win_launch = -1;
  double total_secs = 0.0;
  for (int d = 0; d < used_gpus; ++d) {
    total_hashes += workers[d].total_hashes;
    total_iter  += workers[d].launches_done;
    if (workers[d].found_on_launch >= 0) win_launch = workers[d].found_on_launch;
    if (workers[d].secs > total_secs) total_secs = workers[d].secs;
  }
  bool found = (shared_found.scalar[0] != 0);

  printf("\n=== results ===\n");
  if (used_gpus > 1) printf("devices     : %d GPUs\n", used_gpus);
  printf("iterations : %d\n", total_iter);
  if (win_launch >= 0) printf("found on    : launch %d\n", win_launch);
  printf("total keys : %llu\n", total_hashes);
  printf("sweep time : %.3f s\n", total_secs);
  printf("wall time  : %.3f s (incl. warmup/table-gen)\n", wall_secs);
  if (total_secs > 0) printf("throughput : %.3f Mkeys/s\n", total_hashes / 1e6 / total_secs);

  // ETA for the requested range at measured throughput. Only meaningful when the
  // range was exhausted WITHOUT a find: on a find the remaining keys are irrelevant.
  if (total_secs > 0 && !self_test && !range_is_infinite && !found) {
    double total_keys = (double)range_len[0]
                      + (double)range_len[1] * 18446744073709551616.0
                      + (double)range_len[2] * 340282366920938463463374607431768211456.0
                      + (double)range_len[3] * 6277101735386680763835789423207666416102355444464034512896.0;
    double done = (double)total_hashes;
    double rate = total_hashes / total_secs;
    double effective_done = (done < total_keys) ? done : total_keys;
    if (rate > 0 && total_keys > effective_done) {
      double eta = (total_keys - effective_done) / rate;
      char buf[64];
      if (eta >= 86400.0*365.0)       sprintf(buf, "%.1f years", eta/(86400.0*365.0));
      else if (eta >= 86400.0)        sprintf(buf, "%.1f days", eta/86400.0);
      else if (eta >= 3600.0)         sprintf(buf, "%.1f hours", eta/3600.0);
      else if (eta >= 60.0)           sprintf(buf, "%.1f min", eta/60.0);
      else                            sprintf(buf, "%.1f s", eta);
      printf("ETA        : %s (range %.3e keys, done %.3e)\n", buf, total_keys, done);
    }
  }

  if (found) {
    printf("FOUND scalar: ");
    { char buf[65]; le256_to_hex(shared_found.scalar, buf); printf("%s\n", buf); }
    if (!self_test) {
      char pk_buf[65]; le256_to_hex(shared_found.scalar, pk_buf);
      printf("PRIVATE KEY : %064s\n", pk_buf);
      printf("target      : ");
      for (int k=0;k<20;++k) printf("%02x", target_hash160[k]);
      printf("\n");
      printf("MATCH       : YES\n");
      // How much of the requested range was covered before the hit. The other
      // GPUs stop draining once the global stop flag is set, so this is an
      // upper bound on the fraction actually swept.
      if (!range_is_infinite) {
        double total_keys = (double)range_len[0]
                          + (double)range_len[1] * 18446744073709551616.0
                          + (double)range_len[2] * 340282366920938463463374607431768211456.0
                          + (double)range_len[3] * 6277101735386680763835789423207666416102355444464034512896.0;
        double done = (double)total_hashes;
        if (total_keys > 0) {
          double pct = (done < total_keys ? done : total_keys) / total_keys * 100.0;
          printf("searched    : %.3e / %.3e keys (%.2f%% of range)\n",
                 done, total_keys, pct);
        }
      }
    }
    if (!out_file.empty()) {
      char pk_buf[65]; le256_to_hex(shared_found.scalar, pk_buf);
      FILE* f = fopen(out_file.c_str(), "w");
      if (f) {
        fprintf(f, "Harakiri sweep result\n");
        fprintf(f, "=====================\n");
        fprintf(f, "range start : ");
        { char buf[65]; le256_to_hex(range_start, buf); fprintf(f, "%s\n", buf); }
        fprintf(f, "range end   : ");
        { uint64_t end[4]={range_start[0],range_start[1],range_start[2],range_start[3]};
          uint64_t len4[4]={range_len[0],range_len[1],range_len[2],range_len[3]};
          add256(end, len4); char buf[65]; le256_to_hex(end, buf); fprintf(f, "%s\n", buf); }
        fprintf(f, "target      : ");
        for (int k=0;k<20;++k) fprintf(f, "%02x", target_hash160[k]);
        fprintf(f, "\n");
        fprintf(f, "FOUND scalar: %s\n", pk_buf);
        fprintf(f, "PRIVATE KEY : %064s\n", pk_buf);
        fprintf(f, "total keys  : %llu\n", total_hashes);
        fprintf(f, "launches    : %d\n", total_iter);
        fprintf(f, "sweep time  : %.3f s\n", total_secs);
        if (total_secs > 0)
          fprintf(f, "throughput  : %.3f Mkeys/s\n", total_hashes / 1e6 / total_secs);
        fclose(f);
        printf("result written to %s\n", out_file.c_str());
      } else {
        fprintf(stderr, "error: cannot write %s\n", out_file.c_str());
      }
    }
    if (self_test) {
      printf("expected     : ");
      { char buf[65]; le256_to_hex(found_target, buf); printf("%s\n", buf); }
      bool ok = true;
      for (int k=0;k<4;++k) if (shared_found.scalar[k] != found_target[k]) ok = false;
      printf("%s\n", ok ? "SELFTEST PASS" : "SELFTEST FAIL");
      return ok ? 0 : 1;
    }
  } else if (self_test) {
    printf("SELFTEST FAIL (not found)\n");
    return 1;
  }

  printf("done\n");
  return 0;
}
