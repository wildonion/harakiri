// Harakiri Phase 1 ? register-resident fused SHA-256 + RIPEMD-160.
// Use case: hash160(prefix || x) where x is a 32-byte secp256k1 x-coordinate.
//   SHA-256 input   = 33 bytes  -> exactly ONE 64-byte block
//   RIPEMD-160 input = 32 bytes -> exactly ONE 64-byte block
// No dynamic memory, no __uint128_t. CUDA 11.8 compatible.
#pragma once

#include <cstdint>

__device__ __forceinline__ uint32_t ror32(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32 - n));
}
__device__ __forceinline__ uint32_t rol32(uint32_t x, uint32_t n) {
  return (x << n) | (x >> (32 - n));
}

// Load 32-bit big-endian from 4 bytes
__device__ __forceinline__ uint32_t be32(const uint8_t *b) {
  return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | (uint32_t)b[3];
}

// Load 32-bit little-endian from 4 bytes (RIPEMD-160 word order)
__device__ __forceinline__ uint32_t le32(const uint8_t *b) {
  return ((uint32_t)b[3] << 24) | ((uint32_t)b[2] << 16) | ((uint32_t)b[1] << 8) | (uint32_t)b[0];
}

// ---------------------------------------------------------------------------
// SHA-256: msg MUST be 33..55 bytes (fits one block after padding).
// out: 8 x uint32 (big-endian digest words)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void sha256_33bytes(const uint8_t *msg, uint32_t out[8]) {
  uint32_t W[64];
  uint32_t h[8];

  #pragma unroll
  for (int i = 0; i < 8; i++) W[i] = be32(msg + 4 * i);
  // 33-byte message: byte32 = last x byte, byte33 = 0x80 pad, bytes34..63 = 0
  W[8] = ((uint32_t)msg[32] << 24) | 0x00800000U;
  W[9] = 0; W[10] = 0; W[11] = 0; W[12] = 0; W[13] = 0;
  // length = 33*8 = 264 = 0x108, 64-bit BE at bytes 56..63 -> W[14]=0, W[15]=0x00000108
  W[14] = 0;
  W[15] = 0x00000108U;

  #pragma unroll
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = ror32(W[i-15], 7) ^ ror32(W[i-15], 18) ^ (W[i-15] >> 3);
    uint32_t s1 = ror32(W[i-2], 17) ^ ror32(W[i-2], 19) ^ (W[i-2] >> 10);
    W[i] = W[i-16] + s0 + W[i-7] + s1;
  }

  h[0] = 0x6a09e667U; h[1] = 0xbb67ae85U; h[2] = 0x3c6ef372U; h[3] = 0xa54ff53aU;
  h[4] = 0x510e527fU; h[5] = 0x9b05688cU; h[6] = 0x1f83d9abU; h[7] = 0x5be0cd19U;

  uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
  uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

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

  #pragma unroll
  for (int i = 0; i < 64; i++) {
    uint32_t S1 = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t t1 = hh + S1 + ch + K[i] + W[i];
    uint32_t S0 = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = S0 + maj;
    hh = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
  }

  h[0] += a; h[1] += b; h[2] += c; h[3] += d;
  h[4] += e; h[5] += f; h[6] += g; h[7] += hh;

  out[0] = h[0]; out[1] = h[1]; out[2] = h[2]; out[3] = h[3];
  out[4] = h[4]; out[5] = h[5]; out[6] = h[6]; out[7] = h[7];
}

// Generic single-block SHA-256 for 1..55 byte messages (test vector support)
__device__ __forceinline__ void sha256_oneblock(const uint8_t *msg, uint32_t msglen, uint32_t out[8]) {
  uint8_t block[64];
  #pragma unroll
  for (int i = 0; i < 64; i++) block[i] = 0;
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    if (i * 4 < (int)msglen) {
      block[i*4+0] = msg[i*4+0];
      block[i*4+1] = msg[i*4+1];
      block[i*4+2] = msg[i*4+2];
      block[i*4+3] = msg[i*4+3];
    }
  }
  block[msglen] = 0x80;
  uint64_t bitlen = (uint64_t)msglen * 8;
  block[56] = (uint8_t)(bitlen >> 56);
  block[57] = (uint8_t)(bitlen >> 48);
  block[58] = (uint8_t)(bitlen >> 40);
  block[59] = (uint8_t)(bitlen >> 32);
  block[60] = (uint8_t)(bitlen >> 24);
  block[61] = (uint8_t)(bitlen >> 16);
  block[62] = (uint8_t)(bitlen >> 8);
  block[63] = (uint8_t)(bitlen);

  uint32_t W[64];
  #pragma unroll
  for (int i = 0; i < 16; i++) W[i] = be32(block + 4 * i);

  #pragma unroll
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = ror32(W[i-15], 7) ^ ror32(W[i-15], 18) ^ (W[i-15] >> 3);
    uint32_t s1 = ror32(W[i-2], 17) ^ ror32(W[i-2], 19) ^ (W[i-2] >> 10);
    W[i] = W[i-16] + s0 + W[i-7] + s1;
  }

  uint32_t h[8];
  h[0] = 0x6a09e667U; h[1] = 0xbb67ae85U; h[2] = 0x3c6ef372U; h[3] = 0xa54ff53aU;
  h[4] = 0x510e527fU; h[5] = 0x9b05688cU; h[6] = 0x1f83d9abU; h[7] = 0x5be0cd19U;

  uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
  uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

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

  #pragma unroll
  for (int i = 0; i < 64; i++) {
    uint32_t S1 = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t t1 = hh + S1 + ch + K[i] + W[i];
    uint32_t S0 = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = S0 + maj;
    hh = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
  }

  h[0] += a; h[1] += b; h[2] += c; h[3] += d;
  h[4] += e; h[5] += f; h[6] += g; h[7] += hh;

  out[0] = h[0]; out[1] = h[1]; out[2] = h[2]; out[3] = h[3];
  out[4] = h[4]; out[5] = h[5]; out[6] = h[6]; out[7] = h[7];
}

// ---------------------------------------------------------------------------
// RIPEMD-160: msg MUST be 32..55 bytes (fits one block after padding).
// out: 5 x uint32 digest words (output directly as LE words, per spec)
// ---------------------------------------------------------------------------
__device__ uint32_t ripemd_f(int j, uint32_t x, uint32_t y, uint32_t z) {
  if (j < 16)      return x ^ y ^ z;
  else if (j < 32) return (x & y) | (~x & z);
  else if (j < 48) return (x | ~y) ^ z;
  else if (j < 64) return (x & z) | (y & ~z);
  else             return x ^ (y | ~z);
}

__device__ __forceinline__ void ripemd160_32bytes(const uint8_t *msg, uint32_t out[5]) {
  uint32_t X[16];
  #pragma unroll
  for (int i = 0; i < 16; i++) X[i] = le32(msg + 4 * i);
  X[8] = 0x00000080U;  // pad 0x80 (LE word at bytes 32..35: 80 00 00 00)
  X[9] = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0;
  X[14] = 0x00000100U; // bit length = 32*8 = 256 (LE 64-bit at bytes 56..63)
  X[15] = 0;

  uint32_t h0 = 0x67452301U, h1 = 0xefcdab89U, h2 = 0x98badcfeU, h3 = 0x10325476U, h4 = 0xc3d2e1f0U;

  uint32_t A = h0, B = h1, C = h2, D = h3, E = h4;
  uint32_t A2 = h0, B2 = h1, C2 = h2, D2 = h3, E2 = h4;

  // left line tables
  static const uint8_t rL[80] = {
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8,
     3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12,
     1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2,
     4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13
  };
  static const uint8_t sL[80] = {
    11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
     7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
    11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
    11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
     9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6
  };
  // right line tables
  static const uint8_t rR[80] = {
     5,14, 7, 0, 9, 2,11, 4,13, 6,15, 8, 1,10, 3,12,
     6,11, 3, 7, 0,13, 5,10,14,15, 8,12, 4, 9, 1, 2,
    15, 5, 1, 3, 7,14, 6, 9,11, 8,12, 2,10, 0, 4,13,
     8, 6, 4, 1, 3,11,15, 0, 5,12, 2,13, 9, 7,10,14,
    12,15,10, 4, 1, 5, 8, 7, 6, 2,13,14, 0, 3, 9,11
  };
  static const uint8_t sR[80] = {
     8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6,
     9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11,
     9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5,
    15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,12, 5,15, 8,
     8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11
  };
  static const uint32_t K[5] = {0x00000000U, 0x5a827999U, 0x6ed9eba1U, 0x8f1bbcdcU, 0xa953fd4eU};
  static const uint32_t K2[5] = {0x50a28be6U, 0x5c4dd124U, 0x6d703ef3U, 0x7a6d76e9U, 0x00000000U};

  #pragma unroll
  for (int j = 0; j < 80; j++) {
    int g = j >> 4;
    uint32_t f  = ripemd_f(j, B, C, D);
    uint32_t fr = ripemd_f(79 - j, B2, C2, D2);

    uint32_t T  = rol32(A + f + X[rL[j]] + K[g], sL[j]) + E;
    A = E; E = D; D = rol32(C, 10); C = B; B = T;

    uint32_t T2 = rol32(A2 + fr + X[rR[j]] + K2[g], sR[j]) + E2;
    A2 = E2; E2 = D2; D2 = rol32(C2, 10); C2 = B2; B2 = T2;
  }

  uint32_t T = h1 + C + D2;
  h1 = h2 + D + E2;
  h2 = h3 + E + A2;
  h3 = h4 + A + B2;
  h4 = h0 + B + C2;
  h0 = T;

  out[0] = h0; out[1] = h1; out[2] = h2; out[3] = h3; out[4] = h4;
}

// Generic single-block RIPEMD-160 for 1..55 byte messages (test vector support)
__device__ __forceinline__ void ripemd160_oneblock(const uint8_t *msg, uint32_t msglen, uint32_t out[5]) {
  uint8_t block[64];
  #pragma unroll
  for (int i = 0; i < 64; i++) block[i] = 0;
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    if (i * 4 < (int)msglen) {
      block[i*4+0] = msg[i*4+0];
      block[i*4+1] = msg[i*4+1];
      block[i*4+2] = msg[i*4+2];
      block[i*4+3] = msg[i*4+3];
    }
  }
  block[msglen] = 0x80;
  uint64_t bitlen = (uint64_t)msglen * 8;
  // RIPEMD-160 length field: 64-bit little-endian at bytes 56..63
  block[56] = (uint8_t)(bitlen);
  block[57] = (uint8_t)(bitlen >> 8);
  block[58] = (uint8_t)(bitlen >> 16);
  block[59] = (uint8_t)(bitlen >> 24);
  block[60] = (uint8_t)(bitlen >> 32);
  block[61] = (uint8_t)(bitlen >> 40);
  block[62] = (uint8_t)(bitlen >> 48);
  block[63] = (uint8_t)(bitlen >> 56);

  uint32_t X[16];
  #pragma unroll
  for (int i = 0; i < 16; i++) X[i] = le32(block + 4 * i);

  uint32_t h0 = 0x67452301U, h1 = 0xefcdab89U, h2 = 0x98badcfeU, h3 = 0x10325476U, h4 = 0xc3d2e1f0U;
  uint32_t A = h0, B = h1, C = h2, D = h3, E = h4;
  uint32_t A2 = h0, B2 = h1, C2 = h2, D2 = h3, E2 = h4;

  static const uint8_t rL[80] = {
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8,
     3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12,
     1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2,
     4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13
  };
  static const uint8_t sL[80] = {
    11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
     7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
    11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
    11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
     9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6
  };
  static const uint8_t rR[80] = {
     5,14, 7, 0, 9, 2,11, 4,13, 6,15, 8, 1,10, 3,12,
     6,11, 3, 7, 0,13, 5,10,14,15, 8,12, 4, 9, 1, 2,
    15, 5, 1, 3, 7,14, 6, 9,11, 8,12, 2,10, 0, 4,13,
     8, 6, 4, 1, 3,11,15, 0, 5,12, 2,13, 9, 7,10,14,
    12,15,10, 4, 1, 5, 8, 7, 6, 2,13,14, 0, 3, 9,11
  };
  static const uint8_t sR[80] = {
     8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6,
     9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11,
     9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5,
    15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,12, 5,15, 8,
     8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11
  };
  static const uint32_t K[5] = {0x00000000U, 0x5a827999U, 0x6ed9eba1U, 0x8f1bbcdcU, 0xa953fd4eU};
  static const uint32_t K2[5] = {0x50a28be6U, 0x5c4dd124U, 0x6d703ef3U, 0x7a6d76e9U, 0x00000000U};

  #pragma unroll
  for (int j = 0; j < 80; j++) {
    int g = j >> 4;
    uint32_t f  = ripemd_f(j, B, C, D);
    uint32_t fr = ripemd_f(79 - j, B2, C2, D2);

    uint32_t T  = rol32(A + f + X[rL[j]] + K[g], sL[j]) + E;
    A = E; E = D; D = rol32(C, 10); C = B; B = T;

    uint32_t T2 = rol32(A2 + fr + X[rR[j]] + K2[g], sR[j]) + E2;
    A2 = E2; E2 = D2; D2 = rol32(C2, 10); C2 = B2; B2 = T2;
  }

  uint32_t T = h1 + C + D2;
  h1 = h2 + D + E2;
  h2 = h3 + E + A2;
  h3 = h4 + A + B2;
  h4 = h0 + B + C2;
  h0 = T;

  out[0] = h0; out[1] = h1; out[2] = h2; out[3] = h3; out[4] = h4;
}

// ---------------------------------------------------------------------------
// Fused: hash160(prefix || x) where x is 32-byte x-coord, LE limbs in x_le[4].
// out160: 5 x uint32 (LE words per RIPEMD-160 spec)
// prefix: 0x02 or 0x03 (compressed pubkey parity byte)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void hash160_33_from_x(uint8_t prefix, const uint64_t x_le[4], uint32_t out160[5]) {
  // fully reduce x into [0,p) so the pubkey encoding is canonical
  uint64_t xr[4] = {x_le[0], x_le[1], x_le[2], x_le[3]};
  fieldReduce(xr);

  // Build 33-byte message BE: prefix || x[31]..x[0] (x big-endian, MSB limb first)
  uint8_t msg[33];
  msg[0] = prefix;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    uint64_t v = xr[3 - i];  // most significant limb first
    msg[1 + i*8 + 0] = (uint8_t)(v >> 56);
    msg[1 + i*8 + 1] = (uint8_t)(v >> 48);
    msg[1 + i*8 + 2] = (uint8_t)(v >> 40);
    msg[1 + i*8 + 3] = (uint8_t)(v >> 32);
    msg[1 + i*8 + 4] = (uint8_t)(v >> 24);
    msg[1 + i*8 + 5] = (uint8_t)(v >> 16);
    msg[1 + i*8 + 6] = (uint8_t)(v >> 8);
    msg[1 + i*8 + 7] = (uint8_t)(v);
  }

  uint32_t sha[8];
  sha256_33bytes(msg, sha);

  uint8_t sha_bytes[32];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    sha_bytes[i*4+0] = (uint8_t)(sha[i] >> 24);
    sha_bytes[i*4+1] = (uint8_t)(sha[i] >> 16);
    sha_bytes[i*4+2] = (uint8_t)(sha[i] >> 8);
    sha_bytes[i*4+3] = (uint8_t)(sha[i]);
  }

  ripemd160_32bytes(sha_bytes, out160);
}

// ---------------------------------------------------------------------------
// Register-reduced variant: builds the SHA-256 message schedule W[0..8]
// DIRECTLY from the x limbs and the RIPEMD-160 message words X[0..7] as
// byte-swaps of the SHA words, so no msg[]/sha_bytes[] byte arrays exist ->
// far fewer live registers and no byte-shuffling instructions in the per-key
// hot loop.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void hash160_33_from_x_red(uint8_t prefix, const uint64_t x_le[4], uint32_t out160[5]) {
  const uint64_t xr0 = x_le[0];
  const uint64_t xr1 = x_le[1];
  const uint64_t xr2 = x_le[2];
  const uint64_t xr3 = x_le[3];

  // ---- SHA-256 message words (33-byte msg, big-endian) ----
  // msg = prefix(1) || xr3(8 BE) || xr2(8 BE) || xr1(8 BE) || xr0(8 BE)
  uint32_t W[16];
  W[0] = ((uint32_t)prefix << 24) | (uint32_t)((xr3 >> 40) & 0xFFFFFFULL);
  W[1] = (uint32_t)(xr3 >> 8);
  W[2] = ((uint32_t)(xr3 & 0xFFULL) << 24) | (uint32_t)((xr2 >> 40) & 0xFFFFFFULL);
  W[3] = (uint32_t)(xr2 >> 8);
  W[4] = ((uint32_t)(xr2 & 0xFFULL) << 24) | (uint32_t)((xr1 >> 40) & 0xFFFFFFULL);
  W[5] = (uint32_t)(xr1 >> 8);
  W[6] = ((uint32_t)(xr1 & 0xFFULL) << 24) | (uint32_t)((xr0 >> 40) & 0xFFFFFFULL);
  W[7] = (uint32_t)(xr0 >> 8);
  W[8] = ((uint32_t)(xr0 & 0xFFULL) << 24) | 0x00800000U;  // last x byte + 0x80 pad
  W[9]  = 0; W[10] = 0; W[11] = 0; W[12] = 0; W[13] = 0; W[14] = 0;
  W[15] = 0x00000108U;  // bit length 33*8 = 264

  // ---- SHA-256 schedule + rounds ----
  uint32_t wsch[64];
  #pragma unroll
  for (int i = 0; i < 16; ++i) wsch[i] = W[i];
  #pragma unroll
  for (int i = 16; i < 64; ++i) {
    uint32_t s0 = ror32(wsch[i-15], 7) ^ ror32(wsch[i-15], 18) ^ (wsch[i-15] >> 3);
    uint32_t s1 = ror32(wsch[i-2], 17) ^ ror32(wsch[i-2], 19) ^ (wsch[i-2] >> 10);
    wsch[i] = wsch[i-16] + s0 + wsch[i-7] + s1;
  }

  uint32_t h0 = 0x6a09e667U, h1 = 0xbb67ae85U, h2 = 0x3c6ef372U, h3 = 0xa54ff53aU;
  uint32_t h4 = 0x510e527fU, h5 = 0x9b05688cU, h6 = 0x1f83d9abU, h7 = 0x5be0cd19U;
  uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, hh = h7;

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

  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    uint32_t S1 = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t t1 = hh + S1 + ch + K[i] + wsch[i];
    uint32_t S0 = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = S0 + maj;
    hh = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
  }
  h0 += a; h1 += b; h2 += c; h3 += d;
  h4 += e; h5 += f; h6 += g; h7 += hh;

  // ---- RIPEMD-160 message words: X[i<8] = byte-swap of SHA word (LE) ----
  uint32_t X[16];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t v;
    switch (i) {
      case 0: v = h0; break;
      case 1: v = h1; break;
      case 2: v = h2; break;
      case 3: v = h3; break;
      case 4: v = h4; break;
      case 5: v = h5; break;
      case 6: v = h6; break;
      default: v = h7; break;
    }
    X[i] = ((v & 0x000000FFU) << 24) | ((v & 0x0000FF00U) << 8) |
           ((v & 0x00FF0000U) >> 8)  | ((v & 0xFF000000U) >> 24);
  }
  X[8]  = 0x00000080U;   // pad byte
  X[9]  = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0;
  X[14] = 0x00000100U;   // bit length 32*8 = 256 (LE)
  X[15] = 0;

  uint32_t rh0 = 0x67452301U, rh1 = 0xefcdab89U, rh2 = 0x98badcfeU, rh3 = 0x10325476U, rh4 = 0xc3d2e1f0U;
  uint32_t A = rh0, B = rh1, C = rh2, D = rh3, E = rh4;
  uint32_t A2 = rh0, B2 = rh1, C2 = rh2, D2 = rh3, E2 = rh4;

  static const uint8_t rL[80] = {
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8,
     3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12,
     1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2,
     4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13
  };
  static const uint8_t sL[80] = {
    11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
     7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
    11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
    11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
     9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6
  };
  static const uint8_t rR[80] = {
     5,14, 7, 0, 9, 2,11, 4,13, 6,15, 8, 1,10, 3,12,
     6,11, 3, 7, 0,13, 5,10,14,15, 8,12, 4, 9, 1, 2,
    15, 5, 1, 3, 7,14, 6, 9,11, 8,12, 2,10, 0, 4,13,
     8, 6, 4, 1, 3,11,15, 0, 5,12, 2,13, 9, 7,10,14,
    12,15,10, 4, 1, 5, 8, 7, 6, 2,13,14, 0, 3, 9,11
  };
  static const uint8_t sR[80] = {
     8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6,
     9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11,
     9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5,
    15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,12, 5,15, 8,
     8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11
  };
  static const uint32_t KL[5] = {0x00000000U, 0x5a827999U, 0x6ed9eba1U, 0x8f1bbcdcU, 0xa953fd4eU};
  static const uint32_t KR[5] = {0x50a28be6U, 0x5c4dd124U, 0x6d703ef3U, 0x7a6d76e9U, 0x00000000U};

  #pragma unroll
  for (int j = 0; j < 80; ++j) {
    int g = j >> 4;
    uint32_t fl, fr;
    // left line f
    if      (j < 16)      fl = B ^ C ^ D;
    else if (j < 32)      fl = (B & C) | (~B & D);
    else if (j < 48)      fl = (B | ~C) ^ D;
    else if (j < 64)      fl = (B & D) | (C & ~D);
    else                  fl = B ^ (C | ~D);
    // right line f (79-j in [0..79])
    int jr = 79 - j;
    if      (jr < 16)     fr = B2 ^ C2 ^ D2;
    else if (jr < 32)     fr = (B2 & C2) | (~B2 & D2);
    else if (jr < 48)     fr = (B2 | ~C2) ^ D2;
    else if (jr < 64)     fr = (B2 & D2) | (C2 & ~D2);
    else                  fr = B2 ^ (C2 | ~D2);

    uint32_t T  = rol32(A + fl + X[rL[j]] + KL[g], sL[j]) + E;
    A = E; E = D; D = rol32(C, 10); C = B; B = T;

    uint32_t T2 = rol32(A2 + fr + X[rR[j]] + KR[g], sR[j]) + E2;
    A2 = E2; E2 = D2; D2 = rol32(C2, 10); C2 = B2; B2 = T2;
  }

  uint32_t T = rh1 + C + D2;
  rh1 = rh2 + D + E2;
  rh2 = rh3 + E + A2;
  rh3 = rh4 + A + B2;
  rh4 = rh0 + B + C2;
  rh0 = T;

  out160[0] = rh0; out160[1] = rh1; out160[2] = rh2; out160[3] = rh3; out160[4] = rh4;
}

// Fast filter variant: returns first 4 bytes (uint32 BE) of hash160
__device__ __forceinline__ uint32_t hash160_prefix4(uint8_t prefix, const uint64_t x_le[4]) {
  uint32_t h[5];
  hash160_33_from_x(prefix, x_le, h);
  // digest bytes = h[0] little-endian: b0=(h0)&0xFF, b1=(h0>>8)&0xFF, b2=(h0>>16)&0xFF, b3=(h0>>24)
  // pack as BE u32: b0<<24 | b1<<16 | b2<<8 | b3
  return ((h[0] & 0x000000FFU) << 24) | ((h[0] & 0x0000FF00U) << 8) |
         ((h[0] & 0x00FF0000U) >> 8)  | ((h[0] & 0xFF000000U) >> 24);
}


// ---------------------------------------------------------------------------
// Folded variant: hash160(prefix || x). All SHA-256 schedule constants that
// arise from the fixed message structure (33-byte pubkey => W[9..14]=0,
// W[15]=0x108) are eliminated at the source. The 64 rounds are written
// explicitly with their per-round W value, and each round's K+W add is folded
// by the compiler (K is a compile-time immediate). RIPEMD-160 is unchanged
// from _red (its message words X[8..15] are already constants that nvcc
// folds into the round immediates). Verified bit-exact vs hash160_33_from_x_red
// and vs Python hashlib/ripemd160 references.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void hash160_33_from_x_fold(uint8_t prefix, const uint64_t x_le[4], uint32_t out160[5]) {
  const uint64_t xr0 = x_le[0];
  const uint64_t xr1 = x_le[1];
  const uint64_t xr2 = x_le[2];
  const uint64_t xr3 = x_le[3];

  // ---- SHA-256 message words (33-byte msg, big-endian) ----
  uint32_t W0 = ((uint32_t)prefix << 24) | (uint32_t)((xr3 >> 40) & 0xFFFFFFULL);
  uint32_t W1 = (uint32_t)(xr3 >> 8);
  uint32_t W2 = ((uint32_t)(xr3 & 0xFFULL) << 24) | (uint32_t)((xr2 >> 40) & 0xFFFFFFULL);
  uint32_t W3 = (uint32_t)(xr2 >> 8);
  uint32_t W4 = ((uint32_t)(xr2 & 0xFFULL) << 24) | (uint32_t)((xr1 >> 40) & 0xFFFFFFULL);
  uint32_t W5 = (uint32_t)(xr1 >> 8);
  uint32_t W6 = ((uint32_t)(xr1 & 0xFFULL) << 24) | (uint32_t)((xr0 >> 40) & 0xFFFFFFULL);
  uint32_t W7 = (uint32_t)(xr0 >> 8);
  uint32_t W8 = ((uint32_t)(xr0 & 0xFFULL) << 24) | 0x00800000U;
  const uint32_t W15 = 0x00000108U;   // bit length 33*8 = 264 (W[9..14] = 0)

  // ---- folded schedule (all zero/const contributions eliminated) ----
#define FS0(x) (ror32((x),7) ^ ror32((x),18) ^ ((uint32_t)(x) >> 3))
#define FS1(x) (ror32((x),17) ^ ror32((x),19) ^ ((uint32_t)(x) >> 10))

  uint32_t w16 = (FS0(W1) + W0);
  uint32_t w17 = (FS1(W15) + FS0(W2) + W1);
  uint32_t w18 = (FS1(w16) + FS0(W3) + W2);
  uint32_t w19 = (FS1(w17) + FS0(W4) + W3);
  uint32_t w20 = (FS1(w18) + FS0(W5) + W4);
  uint32_t w21 = (FS1(w19) + FS0(W6) + W5);
  uint32_t w22 = (FS1(w20) + W15 + FS0(W7) + W6);
  uint32_t w23 = (FS1(w21) + w16 + FS0(W8) + W7);
  uint32_t w24 = (FS1(w22) + w17 + W8);
  uint32_t w25 = (FS1(w23) + w18);
  uint32_t w26 = (FS1(w24) + w19);
  uint32_t w27 = (FS1(w25) + w20);
  uint32_t w28 = (FS1(w26) + w21);
  uint32_t w29 = (FS1(w27) + w22);
  uint32_t w30 = (FS1(w28) + w23 + FS0(W15));
  uint32_t w31 = (FS1(w29) + w24 + FS0(w16) + W15);
  uint32_t w32 = (FS1(w30) + w25 + FS0(w17) + w16);
  uint32_t w33 = (FS1(w31) + w26 + FS0(w18) + w17);
  uint32_t w34 = (FS1(w32) + w27 + FS0(w19) + w18);
  uint32_t w35 = (FS1(w33) + w28 + FS0(w20) + w19);
  uint32_t w36 = (FS1(w34) + w29 + FS0(w21) + w20);
  uint32_t w37 = (FS1(w35) + w30 + FS0(w22) + w21);
  uint32_t w38 = (FS1(w36) + w31 + FS0(w23) + w22);
  uint32_t w39 = (FS1(w37) + w32 + FS0(w24) + w23);
  uint32_t w40 = (FS1(w38) + w33 + FS0(w25) + w24);
  uint32_t w41 = (FS1(w39) + w34 + FS0(w26) + w25);
  uint32_t w42 = (FS1(w40) + w35 + FS0(w27) + w26);
  uint32_t w43 = (FS1(w41) + w36 + FS0(w28) + w27);
  uint32_t w44 = (FS1(w42) + w37 + FS0(w29) + w28);
  uint32_t w45 = (FS1(w43) + w38 + FS0(w30) + w29);
  uint32_t w46 = (FS1(w44) + w39 + FS0(w31) + w30);
  uint32_t w47 = (FS1(w45) + w40 + FS0(w32) + w31);
  uint32_t w48 = (FS1(w46) + w41 + FS0(w33) + w32);
  uint32_t w49 = (FS1(w47) + w42 + FS0(w34) + w33);
  uint32_t w50 = (FS1(w48) + w43 + FS0(w35) + w34);
  uint32_t w51 = (FS1(w49) + w44 + FS0(w36) + w35);
  uint32_t w52 = (FS1(w50) + w45 + FS0(w37) + w36);
  uint32_t w53 = (FS1(w51) + w46 + FS0(w38) + w37);
  uint32_t w54 = (FS1(w52) + w47 + FS0(w39) + w38);
  uint32_t w55 = (FS1(w53) + w48 + FS0(w40) + w39);
  uint32_t w56 = (FS1(w54) + w49 + FS0(w41) + w40);
  uint32_t w57 = (FS1(w55) + w50 + FS0(w42) + w41);
  uint32_t w58 = (FS1(w56) + w51 + FS0(w43) + w42);
  uint32_t w59 = (FS1(w57) + w52 + FS0(w44) + w43);
  uint32_t w60 = (FS1(w58) + w53 + FS0(w45) + w44);
  uint32_t w61 = (FS1(w59) + w54 + FS0(w46) + w45);
  uint32_t w62 = (FS1(w60) + w55 + FS0(w47) + w46);
  uint32_t w63 = (FS1(w61) + w56 + FS0(w48) + w47);

#undef FS0
#undef FS1

  // ---- SHA-256 rounds (explicit, W folded per round) ----
  uint32_t h0 = 0x6a09e667U, h1 = 0xbb67ae85U, h2 = 0x3c6ef372U, h3 = 0xa54ff53aU;
  uint32_t h4 = 0x510e527fU, h5 = 0x9b05688cU, h6 = 0x1f83d9abU, h7 = 0x5be0cd19U;
  uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, hh = h7;
#define SHA_ROUND(I, WV) do { \
    uint32_t S1 = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25); \
    uint32_t ch = (e & f) ^ (~e & g); \
    uint32_t t1 = hh + S1 + ch + K[I] + (WV); \
    uint32_t S0 = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22); \
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c); \
    uint32_t t2 = S0 + maj; \
    hh = g; g = f; f = e; e = d + t1; \
    d = c; c = b; b = a; a = t1 + t2; \
} while (0)
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

    SHA_ROUND(0, W0);   SHA_ROUND(1, W1);   SHA_ROUND(2, W2);   SHA_ROUND(3, W3);   SHA_ROUND(4, W4);   SHA_ROUND(5, W5);   SHA_ROUND(6, W6);   SHA_ROUND(7, W7);
    SHA_ROUND(8, W8);   SHA_ROUND(9, 0u);   SHA_ROUND(10, 0u);   SHA_ROUND(11, 0u);   SHA_ROUND(12, 0u);   SHA_ROUND(13, 0u);   SHA_ROUND(14, 0u);   SHA_ROUND(15, W15);
    SHA_ROUND(16, w16);   SHA_ROUND(17, w17);   SHA_ROUND(18, w18);   SHA_ROUND(19, w19);   SHA_ROUND(20, w20);   SHA_ROUND(21, w21);   SHA_ROUND(22, w22);   SHA_ROUND(23, w23);
    SHA_ROUND(24, w24);   SHA_ROUND(25, w25);   SHA_ROUND(26, w26);   SHA_ROUND(27, w27);   SHA_ROUND(28, w28);   SHA_ROUND(29, w29);   SHA_ROUND(30, w30);   SHA_ROUND(31, w31);
    SHA_ROUND(32, w32);   SHA_ROUND(33, w33);   SHA_ROUND(34, w34);   SHA_ROUND(35, w35);   SHA_ROUND(36, w36);   SHA_ROUND(37, w37);   SHA_ROUND(38, w38);   SHA_ROUND(39, w39);
    SHA_ROUND(40, w40);   SHA_ROUND(41, w41);   SHA_ROUND(42, w42);   SHA_ROUND(43, w43);   SHA_ROUND(44, w44);   SHA_ROUND(45, w45);   SHA_ROUND(46, w46);   SHA_ROUND(47, w47);
    SHA_ROUND(48, w48);   SHA_ROUND(49, w49);   SHA_ROUND(50, w50);   SHA_ROUND(51, w51);   SHA_ROUND(52, w52);   SHA_ROUND(53, w53);   SHA_ROUND(54, w54);   SHA_ROUND(55, w55);
    SHA_ROUND(56, w56);   SHA_ROUND(57, w57);   SHA_ROUND(58, w58);   SHA_ROUND(59, w59);   SHA_ROUND(60, w60);   SHA_ROUND(61, w61);   SHA_ROUND(62, w62);   SHA_ROUND(63, w63);

#undef SHA_ROUND
  h0 += a; h1 += b; h2 += c; h3 += d;
  h4 += e; h5 += f; h6 += g; h7 += hh;

  // ---- RIPEMD-160 (identical to _red) ----
  uint32_t X[16];

  X[0] = ((h0 & 0x000000FFU) << 24) | ((h0 & 0x0000FF00U) << 8) |
           ((h0 & 0x00FF0000U) >> 8) | ((h0 & 0xFF000000U) >> 24);
  X[1] = ((h1 & 0x000000FFU) << 24) | ((h1 & 0x0000FF00U) << 8) |
           ((h1 & 0x00FF0000U) >> 8) | ((h1 & 0xFF000000U) >> 24);
  X[2] = ((h2 & 0x000000FFU) << 24) | ((h2 & 0x0000FF00U) << 8) |
           ((h2 & 0x00FF0000U) >> 8) | ((h2 & 0xFF000000U) >> 24);
  X[3] = ((h3 & 0x000000FFU) << 24) | ((h3 & 0x0000FF00U) << 8) |
           ((h3 & 0x00FF0000U) >> 8) | ((h3 & 0xFF000000U) >> 24);
  X[4] = ((h4 & 0x000000FFU) << 24) | ((h4 & 0x0000FF00U) << 8) |
           ((h4 & 0x00FF0000U) >> 8) | ((h4 & 0xFF000000U) >> 24);
  X[5] = ((h5 & 0x000000FFU) << 24) | ((h5 & 0x0000FF00U) << 8) |
           ((h5 & 0x00FF0000U) >> 8) | ((h5 & 0xFF000000U) >> 24);
  X[6] = ((h6 & 0x000000FFU) << 24) | ((h6 & 0x0000FF00U) << 8) |
           ((h6 & 0x00FF0000U) >> 8) | ((h6 & 0xFF000000U) >> 24);
  X[7] = ((h7 & 0x000000FFU) << 24) | ((h7 & 0x0000FF00U) << 8) |
           ((h7 & 0x00FF0000U) >> 8) | ((h7 & 0xFF000000U) >> 24);
  X[8] = 0x00000080U; X[9] = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0; X[14] = 0x00000100U; X[15] = 0;

  uint32_t rh0 = 0x67452301U, rh1 = 0xefcdab89U, rh2 = 0x98badcfeU, rh3 = 0x10325476U, rh4 = 0xc3d2e1f0U;
  uint32_t A = rh0, B = rh1, C = rh2, D = rh3, E = rh4;
  uint32_t A2 = rh0, B2 = rh1, C2 = rh2, D2 = rh3, E2 = rh4;

  static const uint8_t rL[80] = {
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8,
     3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12,
     1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2,
     4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13
  };
  static const uint8_t sL[80] = {
    11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
     7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
    11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
    11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
     9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6
  };
  static const uint8_t rR[80] = {
     5,14, 7, 0, 9, 2,11, 4,13, 6,15, 8, 1,10, 3,12,
     6,11, 3, 7, 0,13, 5,10,14,15, 8,12, 4, 9, 1, 2,
    15, 5, 1, 3, 7,14, 6, 9,11, 8,12, 2,10, 0, 4,13,
     8, 6, 4, 1, 3,11,15, 0, 5,12, 2,13, 9, 7,10,14,
    12,15,10, 4, 1, 5, 8, 7, 6, 2,13,14, 0, 3, 9,11
  };
  static const uint8_t sR[80] = {
     8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6,
     9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11,
     9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5,
    15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,12, 5,15, 8,
     8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11
  };
  static const uint32_t KL[5] = {0x00000000U, 0x5a827999U, 0x6ed9eba1U, 0x8f1bbcdcU, 0xa953fd4eU};
  static const uint32_t KR[5] = {0x50a28be6U, 0x5c4dd124U, 0x6d703ef3U, 0x7a6d76e9U, 0x00000000U};

  #pragma unroll
  for (int j = 0; j < 80; ++j) {
    int g = j >> 4;
    uint32_t fl, fr;
    if      (j < 16)      fl = B ^ C ^ D;
    else if (j < 32)      fl = (B & C) | (~B & D);
    else if (j < 48)      fl = (B | ~C) ^ D;
    else if (j < 64)      fl = (B & D) | (C & ~D);
    else                  fl = B ^ (C | ~D);
    int jr = 79 - j;
    if      (jr < 16)     fr = B2 ^ C2 ^ D2;
    else if (jr < 32)     fr = (B2 & C2) | (~B2 & D2);
    else if (jr < 48)     fr = (B2 | ~C2) ^ D2;
    else if (jr < 64)     fr = (B2 & D2) | (C2 & ~D2);
    else                  fr = B2 ^ (C2 | ~D2);

    uint32_t T  = rol32(A + fl + X[rL[j]] + KL[g], sL[j]) + E;
    A = E; E = D; D = rol32(C, 10); C = B; B = T;

    uint32_t T2 = rol32(A2 + fr + X[rR[j]] + KR[g], sR[j]) + E2;
    A2 = E2; E2 = D2; D2 = rol32(C2, 10); C2 = B2; B2 = T2;
  }

  uint32_t T = rh1 + C + D2;
  rh1 = rh2 + D + E2;
  rh2 = rh3 + E + A2;
  rh3 = rh4 + A + B2;
  rh4 = rh0 + B + C2;
  rh0 = T;

  out160[0] = rh0; out160[1] = rh1; out160[2] = rh2; out160[3] = rh3; out160[4] = rh4;
}

// Rolled-round variant: same folded schedule, but the SHA-256 and RIPEMD-160
// round engines run as loops with #pragma unroll 8. Static code footprint is
// ~10x smaller than the fully unrolled variants -> tests the L1I-fit hypothesis
// (the fully unrolled i-loop body is ~100KB+, borderline vs the 128KB I-cache).
__device__ __forceinline__ void hash160_33_from_x_roll8(uint8_t prefix, const uint64_t x_le[4], uint32_t out160[5]) {
  const uint64_t xr0 = x_le[0];
  const uint64_t xr1 = x_le[1];
  const uint64_t xr2 = x_le[2];
  const uint64_t xr3 = x_le[3];

  uint32_t W0 = ((uint32_t)prefix << 24) | (uint32_t)((xr3 >> 40) & 0xFFFFFFULL);
  uint32_t W1 = (uint32_t)(xr3 >> 8);
  uint32_t W2 = ((uint32_t)(xr3 & 0xFFULL) << 24) | (uint32_t)((xr2 >> 40) & 0xFFFFFFULL);
  uint32_t W3 = (uint32_t)(xr2 >> 8);
  uint32_t W4 = ((uint32_t)(xr2 & 0xFFULL) << 24) | (uint32_t)((xr1 >> 40) & 0xFFFFFFULL);
  uint32_t W5 = (uint32_t)(xr1 >> 8);
  uint32_t W6 = ((uint32_t)(xr1 & 0xFFULL) << 24) | (uint32_t)((xr0 >> 40) & 0xFFFFFFULL);
  uint32_t W7 = (uint32_t)(xr0 >> 8);
  uint32_t W8 = ((uint32_t)(xr0 & 0xFFULL) << 24) | 0x00800000U;
  const uint32_t W15 = 0x00000108U;

  uint32_t wsch[64];
  wsch[0]=W0; wsch[1]=W1; wsch[2]=W2; wsch[3]=W3; wsch[4]=W4; wsch[5]=W5; wsch[6]=W6; wsch[7]=W7; wsch[8]=W8;
  wsch[9]=0; wsch[10]=0; wsch[11]=0; wsch[12]=0; wsch[13]=0; wsch[14]=0; wsch[15]=W15;

#define RS0(x) (ror32((x),7) ^ ror32((x),18) ^ ((uint32_t)(x) >> 3))
#define RS1(x) (ror32((x),17) ^ ror32((x),19) ^ ((uint32_t)(x) >> 10))
  wsch[16] = (RS0(W1) + W0);
  wsch[17] = (RS1(W15) + RS0(W2) + W1);
  wsch[18] = (RS1(wsch[16]) + RS0(W3) + W2);
  wsch[19] = (RS1(wsch[17]) + RS0(W4) + W3);
  wsch[20] = (RS1(wsch[18]) + RS0(W5) + W4);
  wsch[21] = (RS1(wsch[19]) + RS0(W6) + W5);
  wsch[22] = (RS1(wsch[20]) + W15 + RS0(W7) + W6);
  wsch[23] = (RS1(wsch[21]) + wsch[16] + RS0(W8) + W7);
  wsch[24] = (RS1(wsch[22]) + wsch[17] + W8);
  wsch[25] = (RS1(wsch[23]) + wsch[18]);
  wsch[26] = (RS1(wsch[24]) + wsch[19]);
  wsch[27] = (RS1(wsch[25]) + wsch[20]);
  wsch[28] = (RS1(wsch[26]) + wsch[21]);
  wsch[29] = (RS1(wsch[27]) + wsch[22]);
  wsch[30] = (RS1(wsch[28]) + wsch[23] + RS0(W15));
  wsch[31] = (RS1(wsch[29]) + wsch[24] + RS0(wsch[16]) + W15);
  wsch[32] = (RS1(wsch[30]) + wsch[25] + RS0(wsch[17]) + wsch[16]);
  wsch[33] = (RS1(wsch[31]) + wsch[26] + RS0(wsch[18]) + wsch[17]);
  wsch[34] = (RS1(wsch[32]) + wsch[27] + RS0(wsch[19]) + wsch[18]);
  wsch[35] = (RS1(wsch[33]) + wsch[28] + RS0(wsch[20]) + wsch[19]);
  wsch[36] = (RS1(wsch[34]) + wsch[29] + RS0(wsch[21]) + wsch[20]);
  wsch[37] = (RS1(wsch[35]) + wsch[30] + RS0(wsch[22]) + wsch[21]);
  wsch[38] = (RS1(wsch[36]) + wsch[31] + RS0(wsch[23]) + wsch[22]);
  wsch[39] = (RS1(wsch[37]) + wsch[32] + RS0(wsch[24]) + wsch[23]);
  wsch[40] = (RS1(wsch[38]) + wsch[33] + RS0(wsch[25]) + wsch[24]);
  wsch[41] = (RS1(wsch[39]) + wsch[34] + RS0(wsch[26]) + wsch[25]);
  wsch[42] = (RS1(wsch[40]) + wsch[35] + RS0(wsch[27]) + wsch[26]);
  wsch[43] = (RS1(wsch[41]) + wsch[36] + RS0(wsch[28]) + wsch[27]);
  wsch[44] = (RS1(wsch[42]) + wsch[37] + RS0(wsch[29]) + wsch[28]);
  wsch[45] = (RS1(wsch[43]) + wsch[38] + RS0(wsch[30]) + wsch[29]);
  wsch[46] = (RS1(wsch[44]) + wsch[39] + RS0(wsch[31]) + wsch[30]);
  wsch[47] = (RS1(wsch[45]) + wsch[40] + RS0(wsch[32]) + wsch[31]);
  wsch[48] = (RS1(wsch[46]) + wsch[41] + RS0(wsch[33]) + wsch[32]);
  wsch[49] = (RS1(wsch[47]) + wsch[42] + RS0(wsch[34]) + wsch[33]);
  wsch[50] = (RS1(wsch[48]) + wsch[43] + RS0(wsch[35]) + wsch[34]);
  wsch[51] = (RS1(wsch[49]) + wsch[44] + RS0(wsch[36]) + wsch[35]);
  wsch[52] = (RS1(wsch[50]) + wsch[45] + RS0(wsch[37]) + wsch[36]);
  wsch[53] = (RS1(wsch[51]) + wsch[46] + RS0(wsch[38]) + wsch[37]);
  wsch[54] = (RS1(wsch[52]) + wsch[47] + RS0(wsch[39]) + wsch[38]);
  wsch[55] = (RS1(wsch[53]) + wsch[48] + RS0(wsch[40]) + wsch[39]);
  wsch[56] = (RS1(wsch[54]) + wsch[49] + RS0(wsch[41]) + wsch[40]);
  wsch[57] = (RS1(wsch[55]) + wsch[50] + RS0(wsch[42]) + wsch[41]);
  wsch[58] = (RS1(wsch[56]) + wsch[51] + RS0(wsch[43]) + wsch[42]);
  wsch[59] = (RS1(wsch[57]) + wsch[52] + RS0(wsch[44]) + wsch[43]);
  wsch[60] = (RS1(wsch[58]) + wsch[53] + RS0(wsch[45]) + wsch[44]);
  wsch[61] = (RS1(wsch[59]) + wsch[54] + RS0(wsch[46]) + wsch[45]);
  wsch[62] = (RS1(wsch[60]) + wsch[55] + RS0(wsch[47]) + wsch[46]);
  wsch[63] = (RS1(wsch[61]) + wsch[56] + RS0(wsch[48]) + wsch[47]);

#undef RS0
#undef RS1

  uint32_t h0 = 0x6a09e667U, h1 = 0xbb67ae85U, h2 = 0x3c6ef372U, h3 = 0xa54ff53aU;
  uint32_t h4 = 0x510e527fU, h5 = 0x9b05688cU, h6 = 0x1f83d9abU, h7 = 0x5be0cd19U;
  uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, hh = h7;

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
  #pragma unroll 8
  for (int i = 0; i < 64; ++i) {
    uint32_t S1 = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t t1 = hh + S1 + ch + K[i] + wsch[i];
    uint32_t S0 = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = S0 + maj;
    hh = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
  }
  h0 += a; h1 += b; h2 += c; h3 += d;
  h4 += e; h5 += f; h6 += g; h7 += hh;

  // ---- RIPEMD-160 ----
  uint32_t X[16];

  X[0] = ((h0 & 0x000000FFU) << 24) | ((h0 & 0x0000FF00U) << 8) |
           ((h0 & 0x00FF0000U) >> 8) | ((h0 & 0xFF000000U) >> 24);
  X[1] = ((h1 & 0x000000FFU) << 24) | ((h1 & 0x0000FF00U) << 8) |
           ((h1 & 0x00FF0000U) >> 8) | ((h1 & 0xFF000000U) >> 24);
  X[2] = ((h2 & 0x000000FFU) << 24) | ((h2 & 0x0000FF00U) << 8) |
           ((h2 & 0x00FF0000U) >> 8) | ((h2 & 0xFF000000U) >> 24);
  X[3] = ((h3 & 0x000000FFU) << 24) | ((h3 & 0x0000FF00U) << 8) |
           ((h3 & 0x00FF0000U) >> 8) | ((h3 & 0xFF000000U) >> 24);
  X[4] = ((h4 & 0x000000FFU) << 24) | ((h4 & 0x0000FF00U) << 8) |
           ((h4 & 0x00FF0000U) >> 8) | ((h4 & 0xFF000000U) >> 24);
  X[5] = ((h5 & 0x000000FFU) << 24) | ((h5 & 0x0000FF00U) << 8) |
           ((h5 & 0x00FF0000U) >> 8) | ((h5 & 0xFF000000U) >> 24);
  X[6] = ((h6 & 0x000000FFU) << 24) | ((h6 & 0x0000FF00U) << 8) |
           ((h6 & 0x00FF0000U) >> 8) | ((h6 & 0xFF000000U) >> 24);
  X[7] = ((h7 & 0x000000FFU) << 24) | ((h7 & 0x0000FF00U) << 8) |
           ((h7 & 0x00FF0000U) >> 8) | ((h7 & 0xFF000000U) >> 24);
  X[8] = 0x00000080U; X[9] = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0; X[14] = 0x00000100U; X[15] = 0;

  uint32_t rh0 = 0x67452301U, rh1 = 0xefcdab89U, rh2 = 0x98badcfeU, rh3 = 0x10325476U, rh4 = 0xc3d2e1f0U;
  uint32_t A = rh0, B = rh1, C = rh2, D = rh3, E = rh4;
  uint32_t A2 = rh0, B2 = rh1, C2 = rh2, D2 = rh3, E2 = rh4;

  static const uint8_t rL[80] = {
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8,
     3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12,
     1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2,
     4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13
  };
  static const uint8_t sL[80] = {
    11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
     7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
    11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
    11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
     9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6
  };
  static const uint8_t rR[80] = {
     5,14, 7, 0, 9, 2,11, 4,13, 6,15, 8, 1,10, 3,12,
     6,11, 3, 7, 0,13, 5,10,14,15, 8,12, 4, 9, 1, 2,
    15, 5, 1, 3, 7,14, 6, 9,11, 8,12, 2,10, 0, 4,13,
     8, 6, 4, 1, 3,11,15, 0, 5,12, 2,13, 9, 7,10,14,
    12,15,10, 4, 1, 5, 8, 7, 6, 2,13,14, 0, 3, 9,11
  };
  static const uint8_t sR[80] = {
     8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6,
     9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11,
     9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5,
    15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,12, 5,15, 8,
     8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11
  };
  static const uint32_t KL[5] = {0x00000000U, 0x5a827999U, 0x6ed9eba1U, 0x8f1bbcdcU, 0xa953fd4eU};
  static const uint32_t KR[5] = {0x50a28be6U, 0x5c4dd124U, 0x6d703ef3U, 0x7a6d76e9U, 0x00000000U};

  #pragma unroll 8
  for (int j = 0; j < 80; ++j) {
    int g = j >> 4;
    uint32_t fl, fr;
    if      (j < 16)      fl = B ^ C ^ D;
    else if (j < 32)      fl = (B & C) | (~B & D);
    else if (j < 48)      fl = (B | ~C) ^ D;
    else if (j < 64)      fl = (B & D) | (C & ~D);
    else                  fl = B ^ (C | ~D);
    int jr = 79 - j;
    if      (jr < 16)     fr = B2 ^ C2 ^ D2;
    else if (jr < 32)     fr = (B2 & C2) | (~B2 & D2);
    else if (jr < 48)     fr = (B2 | ~C2) ^ D2;
    else if (jr < 64)     fr = (B2 & D2) | (C2 & ~D2);
    else                  fr = B2 ^ (C2 | ~D2);

    uint32_t T  = rol32(A + fl + X[rL[j]] + KL[g], sL[j]) + E;
    A = E; E = D; D = rol32(C, 10); C = B; B = T;

    uint32_t T2 = rol32(A2 + fr + X[rR[j]] + KR[g], sR[j]) + E2;
    A2 = E2; E2 = D2; D2 = rol32(C2, 10); C2 = B2; B2 = T2;
  }

  uint32_t T = rh1 + C + D2;
  rh1 = rh2 + D + E2;
  rh2 = rh3 + E + A2;
  rh3 = rh4 + A + B2;
  rh4 = rh0 + B + C2;
  rh0 = T;

  out160[0] = rh0; out160[1] = rh1; out160[2] = rh2; out160[3] = rh3; out160[4] = rh4;
}
