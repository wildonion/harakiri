// Harakiri Phase 1 — register-resident fused SHA-256 + RIPEMD-160.
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
// Register-reduced variant: builds the SHA-256 message words W[0..15] directly
// from the x limbs, runs a rolling 16-word schedule fused into the 64 rounds
// (live state 16 words, not 64), and builds the RIPEMD-160 words as
// single-instruction byte-swaps (__byte_perm) of the SHA digest. No msg[] or
// sha_bytes[] arrays exist anywhere in the per-key path.
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

  // ---- SHA-256 schedule + rounds, rolling 16-word window ----
  // Slot i&15 holds W[i-16] (16 steps old), exactly the slot the recurrence
  // overwrites; W[i-15]/W[i-2]/W[i-7] live at (i+1)&15, (i+14)&15, (i+9)&15.
  // Live state is 16 words, not 64 (same pattern as the gated 2x variant).
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
    if (i >= 16) {
      uint32_t p15 = W[(i+1)&15], p2 = W[(i+14)&15], p7 = W[(i+9)&15];
      uint32_t s0 = ror32(p15, 7) ^ ror32(p15, 18) ^ (p15 >> 3);
      uint32_t s1 = ror32(p2, 17) ^ ror32(p2, 19) ^ (p2 >> 10);
      W[i&15] += s0 + p7 + s1;
    }
    uint32_t wi = W[i&15];
    uint32_t S1 = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25);
    uint32_t ch = (e & f) ^ (~e & g);
    uint32_t t1 = hh + S1 + ch + K[i] + wi;
    uint32_t S0 = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22);
    uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = S0 + maj;
    hh = g; g = f; f = e; e = d + t1;
    d = c; c = b; b = a; a = t1 + t2;
  }
  h0 += a; h1 += b; h2 += c; h3 += d;
  h4 += e; h5 += f; h6 += g; h7 += hh;

  // ---- RIPEMD-160 message words: X[i<8] = byte-swap of SHA word (LE) ----
  // __byte_perm(v,0,0x0123) is a single-instruction bswap32 (KeyHunt idiom).
  uint32_t X[16];
  {
    uint32_t hv[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    #pragma unroll
    for (int i = 0; i < 8; ++i) X[i] = __byte_perm(hv[i], 0, 0x0123);
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

// ---------------------------------------------------------------------------
// Two independent hash160(prefix || x) evaluations interleaved in one pass.
//
// Why: SHA-256 and RIPEMD-160 are strictly serial chains (round i+1 depends on
// round i), so a single chain leaves most of a GPU warp's issue width idle.
// Running two chains in the same thread gives the scheduler independent work and
// is the standard multi-buffer technique for GPU SHA.
//
// Register cost is kept flat by using a ROLLING 16-word SHA schedule instead of
// the 64-word table the 1x variant materialises: 2x(16+8) for SHA plus
// 2x(16+10) for RIPEMD ~= what 1x spends on (64+8) + (16+10).
//
// Correctness is gated by hash_selftest_kernel, which compares this against the
// reference hash160_33_from_x over a large randomised input set. A broken hash
// here would not crash - it would silently stop finding keys.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void hash160_33_from_x_red_2x(
    uint8_t prefixA, const uint64_t xa[4], uint32_t ha[5],
    uint8_t prefixB, const uint64_t xb[4], uint32_t hb[5]) {
  // ---- SHA-256 message words, built straight from the x limbs ----
  uint32_t wa[16], wb[16];
  {
    const uint64_t a0 = xa[0], a1 = xa[1], a2 = xa[2], a3 = xa[3];
    wa[0] = ((uint32_t)prefixA << 24) | (uint32_t)((a3 >> 40) & 0xFFFFFFULL);
    wa[1] = (uint32_t)(a3 >> 8);
    wa[2] = ((uint32_t)(a3 & 0xFFULL) << 24) | (uint32_t)((a2 >> 40) & 0xFFFFFFULL);
    wa[3] = (uint32_t)(a2 >> 8);
    wa[4] = ((uint32_t)(a2 & 0xFFULL) << 24) | (uint32_t)((a1 >> 40) & 0xFFFFFFULL);
    wa[5] = (uint32_t)(a1 >> 8);
    wa[6] = ((uint32_t)(a1 & 0xFFULL) << 24) | (uint32_t)((a0 >> 40) & 0xFFFFFFULL);
    wa[7] = (uint32_t)(a0 >> 8);
    wa[8] = ((uint32_t)(a0 & 0xFFULL) << 24) | 0x00800000U;
    wa[9] = 0; wa[10] = 0; wa[11] = 0; wa[12] = 0; wa[13] = 0; wa[14] = 0;
    wa[15] = 0x00000108U;
  }
  {
    const uint64_t b0 = xb[0], b1 = xb[1], b2 = xb[2], b3 = xb[3];
    wb[0] = ((uint32_t)prefixB << 24) | (uint32_t)((b3 >> 40) & 0xFFFFFFULL);
    wb[1] = (uint32_t)(b3 >> 8);
    wb[2] = ((uint32_t)(b3 & 0xFFULL) << 24) | (uint32_t)((b2 >> 40) & 0xFFFFFFULL);
    wb[3] = (uint32_t)(b2 >> 8);
    wb[4] = ((uint32_t)(b2 & 0xFFULL) << 24) | (uint32_t)((b1 >> 40) & 0xFFFFFFULL);
    wb[5] = (uint32_t)(b1 >> 8);
    wb[6] = ((uint32_t)(b1 & 0xFFULL) << 24) | (uint32_t)((b0 >> 40) & 0xFFFFFFULL);
    wb[7] = (uint32_t)(b0 >> 8);
    wb[8] = ((uint32_t)(b0 & 0xFFULL) << 24) | 0x00800000U;
    wb[9] = 0; wb[10] = 0; wb[11] = 0; wb[12] = 0; wb[13] = 0; wb[14] = 0;
    wb[15] = 0x00000108U;
  }

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

  uint32_t a0=0x6a09e667U,a1=0xbb67ae85U,a2=0x3c6ef372U,a3=0xa54ff53a,
           a4=0x510e527fU,a5=0x9b05688cU,a6=0x1f83d9abU,a7=0x5be0cd19U;
  uint32_t b0=a0,b1=a1,b2=a2,b3=a3,b4=a4,b5=a5,b6=a6,b7=a7;

  // ---- interleaved SHA-256, rolling schedule ----
  #pragma unroll
  for (int i = 0; i < 64; ++i) {
    if (i >= 16) {
      uint32_t p15 = wa[(i+1)&15], p2 = wa[(i+14)&15], p7 = wa[(i+9)&15];
      uint32_t s0 = ror32(p15,7) ^ ror32(p15,18) ^ (p15 >> 3);
      uint32_t s1 = ror32(p2,17) ^ ror32(p2,19) ^ (p2 >> 10);
      wa[i&15] += s0 + p7 + s1;
      uint32_t q15 = wb[(i+1)&15], q2 = wb[(i+14)&15], q7 = wb[(i+9)&15];
      uint32_t t0 = ror32(q15,7) ^ ror32(q15,18) ^ (q15 >> 3);
      uint32_t t1 = ror32(q2,17) ^ ror32(q2,19) ^ (q2 >> 10);
      wb[i&15] += t0 + q7 + t1;
    }
    uint32_t wa_i = wa[i&15];
    uint32_t S1a = ror32(a4,6) ^ ror32(a4,11) ^ ror32(a4,25);
    uint32_t cha = (a4 & a5) ^ (~a4 & a6);
    uint32_t t1a = a7 + S1a + cha + K[i] + wa_i;
    uint32_t S0a = ror32(a0,2) ^ ror32(a0,13) ^ ror32(a0,22);
    uint32_t mja = (a0 & a1) ^ (a0 & a2) ^ (a1 & a2);
    uint32_t t2a = S0a + mja;
    a7 = a6; a6 = a5; a5 = a4; a4 = a3 + t1a; a3 = a2; a2 = a1; a1 = a0; a0 = t1a + t2a;

    uint32_t wb_i = wb[i&15];
    uint32_t S1b = ror32(b4,6) ^ ror32(b4,11) ^ ror32(b4,25);
    uint32_t chb = (b4 & b5) ^ (~b4 & b6);
    uint32_t t1b = b7 + S1b + chb + K[i] + wb_i;
    uint32_t S0b = ror32(b0,2) ^ ror32(b0,13) ^ ror32(b0,22);
    uint32_t mjb = (b0 & b1) ^ (b0 & b2) ^ (b1 & b2);
    uint32_t t2b = S0b + mjb;
    b7 = b6; b6 = b5; b5 = b4; b4 = b3 + t1b; b3 = b2; b2 = b1; b1 = b0; b0 = t1b + t2b;
  }
  a0 += 0x6a09e667U; a1 += 0xbb67ae85U; a2 += 0x3c6ef372U; a3 += 0xa54ff53a;
  a4 += 0x510e527fU; a5 += 0x9b05688cU; a6 += 0x1f83d9abU; a7 += 0x5be0cd19U;
  b0 += 0x6a09e667U; b1 += 0xbb67ae85U; b2 += 0x3c6ef372U; b3 += 0xa54ff53a;
  b4 += 0x510e527fU; b5 += 0x9b05688cU; b6 += 0x1f83d9abU; b7 += 0x5be0cd19U;

  // ---- RIPEMD-160 message words: byte-swap of the SHA words ----
  uint32_t XA[16], XB[16];
  {
    uint32_t v[8] = {a0,a1,a2,a3,a4,a5,a6,a7};
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      XA[i] = ((v[i] & 0x000000FFU) << 24) | ((v[i] & 0x0000FF00U) << 8) |
              ((v[i] & 0x00FF0000U) >> 8)  | ((v[i] & 0xFF000000U) >> 24);
    XA[8] = 0x00000080U; XA[9] = 0; XA[10] = 0; XA[11] = 0; XA[12] = 0;
    XA[13] = 0; XA[14] = 0x00000100U; XA[15] = 0;
  }
  {
    uint32_t v[8] = {b0,b1,b2,b3,b4,b5,b6,b7};
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      XB[i] = ((v[i] & 0x000000FFU) << 24) | ((v[i] & 0x0000FF00U) << 8) |
              ((v[i] & 0x00FF0000U) >> 8)  | ((v[i] & 0xFF000000U) >> 24);
    XB[8] = 0x00000080U; XB[9] = 0; XB[10] = 0; XB[11] = 0; XB[12] = 0;
    XB[13] = 0; XB[14] = 0x00000100U; XB[15] = 0;
  }

  static const uint8_t rL[80] = {
     0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8,
     3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12,
     1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2,
     4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13};
  static const uint8_t sL[80] = {
    11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
     7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
    11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
    11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
     9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6};
  static const uint8_t rR[80] = {
     5,14, 7, 0, 9, 2,11, 4,13, 6,15, 8, 1,10, 3,12,
     6,11, 3, 7, 0,13, 5,10,14,15, 8,12, 4, 9, 1, 2,
    15, 5, 1, 3, 7,14, 6, 9,11, 8,12, 2,10, 0, 4,13,
     8, 6, 4, 1, 3,11,15, 0, 5,12, 2,13, 9, 7,10,14,
    12,15,10, 4, 1, 5, 8, 7, 6, 2,13,14, 0, 3, 9,11};
  static const uint8_t sR[80] = {
     8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6,
     9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11,
     9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5,
    15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,12, 5,15, 8,
     8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11};
  static const uint32_t KL[5] = {0x00000000U,0x5a827999U,0x6ed9eba1U,0x8f1bbcdcU,0xa953fd4eU};
  static const uint32_t KR[5] = {0x50a28be6U,0x5c4dd124U,0x6d703ef3U,0x7a6d76e9U,0x00000000U};

#define HASH2X_RIPEMD_F(j,x,y,z) ( \
    (j) < 16 ? ((x) ^ (y) ^ (z)) : \
    (j) < 32 ? (((x) & (y)) | (~(x) & (z))) : \
    (j) < 48 ? (((x) | ~(y)) ^ (z)) : \
    (j) < 64 ? (((x) & (z)) | ((y) & ~(z))) : \
               ((x) ^ ((y) | ~(z))))

  uint32_t rh0=0x67452301U,rh1=0xefcdab89U,rh2=0x98badcfeU,rh3=0x10325476U,rh4=0xc3d2e1f0U;
  uint32_t A=rh0,B=rh1,C=rh2,D=rh3,E=rh4;
  uint32_t A2=rh0,B2=rh1,C2=rh2,D2=rh3,E2=rh4;

  // Two chains, four interleaved RIPEMD lines. Only the finalization differs
  // from the 1x version and it is written per chain below.
  #pragma unroll
  for (int j = 0; j < 80; ++j) {
    int g = j >> 4, jr = 79 - j;
    uint32_t Ta  = rol32(A  + HASH2X_RIPEMD_F(j ,B ,C ,D ) + XA[rL[j]] + KL[g], sL[j]) + E;
    uint32_t Ta2 = rol32(A2 + HASH2X_RIPEMD_F(jr,B2,C2,D2) + XA[rR[j]] + KR[g], sR[j]) + E2;
    A = E; E = D; D = rol32(C,10); C = B; B = Ta;
    A2 = E2; E2 = D2; D2 = rol32(C2,10); C2 = B2; B2 = Ta2;
  }
  uint32_t oA = rh1 + C  + D2;
  rh1 = rh2 + D  + E2;  rh2 = rh3 + E  + A2;  rh3 = rh4 + A  + B2;  rh4 = rh0 + B  + C2;  rh0 = oA;
  ha[0]=rh0; ha[1]=rh1; ha[2]=rh2; ha[3]=rh3; ha[4]=rh4;

  rh0=0x67452301U;rh1=0xefcdab89U;rh2=0x98badcfeU;rh3=0x10325476U;rh4=0xc3d2e1f0U;
  A=rh0;B=rh1;C=rh2;D=rh3;E=rh4; A2=rh0;B2=rh1;C2=rh2;D2=rh3;E2=rh4;
  #pragma unroll
  for (int j = 0; j < 80; ++j) {
    int g = j >> 4, jr = 79 - j;
    uint32_t Tb  = rol32(A  + HASH2X_RIPEMD_F(j ,B ,C ,D ) + XB[rL[j]] + KL[g], sL[j]) + E;
    uint32_t Tb2 = rol32(A2 + HASH2X_RIPEMD_F(jr,B2,C2,D2) + XB[rR[j]] + KR[g], sR[j]) + E2;
    A = E; E = D; D = rol32(C,10); C = B; B = Tb;
    A2 = E2; E2 = D2; D2 = rol32(C2,10); C2 = B2; B2 = Tb2;
  }
  oA = rh1 + C  + D2;
  rh1 = rh2 + D  + E2;  rh2 = rh3 + E  + A2;  rh3 = rh4 + A  + B2;  rh4 = rh0 + B  + C2;  rh0 = oA;
  hb[0]=rh0; hb[1]=rh1; hb[2]=rh2; hb[3]=rh3; hb[4]=rh4;
#undef HASH2X_RIPEMD_F
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