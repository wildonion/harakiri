// Harakiri Phase 0 — secp256k1 field arithmetic core
// 5x64-bit lazy "signed 320-bit" limbs, inline PTX carry chains.
// CUDA 11.8 compatible: NO __uint128_t / __int128_t anywhere.
// Ported from the proven CUDACyclone / KeyHunt-CUDA (JeanLucPons lineage) math.
#pragma once

#include <cstdint>

#define NBBLOCK 5

// --- PTX carry / mul primitives ---------------------------------------------
#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADD(c, a, b)  asm volatile ("addc.u64 %0, %1, %2;"   : "=l"(c) : "l"(a), "l"(b));

#define UADDO1(c, a)   asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADDC1(c, a)   asm volatile ("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADD1(c, a)    asm volatile ("addc.u64 %0, %0, %1;"   : "+l"(c) : "l"(a));

#define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUB(c, a, b)  asm volatile ("subc.u64 %0, %1, %2;"   : "=l"(c) : "l"(a), "l"(b));

#define USUBO1(c, a)   asm volatile ("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUBC1(c, a)   asm volatile ("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUB1(c, a)    asm volatile ("subc.u64 %0, %0, %1;"   : "+l"(c) : "l"(a));

#define UMULLO(lo, a, b) asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
#define UMULHI(hi, a, b) asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
#define MADDO(r, a, b, c)  asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADDC(r, a, b, c)  asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADD(r, a, b, c)   asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
#define MADDS(r, a, b, c)  asm volatile ("madc.hi.s64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));

// --- Constants --------------------------------------------------------------
__device__ __constant__ uint64_t MM64  = 0xD838091DD2253531ULL;
__device__ __constant__ uint64_t MSK62 = 0x3FFFFFFFFFFFFFFFULL;

// p = 2^256 - 2^32 - 977, little-endian limbs
static __device__ __constant__ uint64_t SECP_P_LE[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

static __device__ __constant__ uint64_t SECP_GX_LE[4] = {
    0x59f2815b16f81798ULL,
    0x029bfcdb2dce28d9ULL,
    0x55a06295ce870b07ULL,
    0x79be667ef9dcbbacULL
};
static __device__ __constant__ uint64_t SECP_GY_LE[4] = {
    0x9c47d08ffb10d4b8ULL,
    0xfd17b448a6855419ULL,
    0x5da4fbfc0e1108a8ULL,
    0x483ada7726a3c465ULL
};

// --- 5-limb predicates / helpers ---------------------------------------------
#define _IsPositive(x) (((int64_t)(x)[4]) >= 0LL)
#define _IsNegative(x) (((int64_t)(x)[4]) < 0LL)
#define _IsEqual(a, b)  ((a[4] == b[4]) && (a[3] == b[3]) && (a[2] == b[2]) && (a[1] == b[1]) && (a[0] == b[0]))
#define _IsZero(a)      ((a[4] | a[3] | a[2] | a[1] | a[0]) == 0ULL)
#define _IsOne(a)       ((a[4] == 0ULL) && (a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL))

#define __sright128(a, b, n) ((a) >> (n)) | ((b) << (64 - (n)))
#define __sleft128(a, b, n)  ((b) << (n)) | ((a) >> (64 - (n)))

#define AddP(r) { \
  UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  UADD1(r[4], 0ULL); }

#define SubP(r) { \
  USUBO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  USUBC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  USUB1(r[4], 0ULL); }

#define Sub2(r, a, b) { \
  USUBO(r[0], a[0], b[0]); \
  USUBC(r[1], a[1], b[1]); \
  USUBC(r[2], a[2], b[2]); \
  USUBC(r[3], a[3], b[3]); \
  USUB(r[4], a[4], b[4]); }

#define Neg(r) { \
  USUBO(r[0], 0ULL, r[0]); \
  USUBC(r[1], 0ULL, r[1]); \
  USUBC(r[2], 0ULL, r[2]); \
  USUBC(r[3], 0ULL, r[3]); \
  USUB(r[4], 0ULL, r[4]); }

#define UMult(r, a, b) { \
  UMULLO(r[0], a[0], b); \
  UMULLO(r[1], a[1], b); \
  MADDO(r[1], a[0], b, r[1]); \
  UMULLO(r[2], a[2], b); \
  MADDC(r[2], a[1], b, r[2]); \
  UMULLO(r[3], a[3], b); \
  MADDC(r[3], a[2], b, r[3]); \
  MADD(r[4], a[3], b, 0ULL); }

#define Load(r, a) { \
  (r)[0] = (a)[0]; (r)[1] = (a)[1]; (r)[2] = (a)[2]; \
  (r)[3] = (a)[3]; (r)[4] = (a)[4]; }

#define Load256(r, a) { \
  (r)[0] = (a)[0]; (r)[1] = (a)[1]; (r)[2] = (a)[2]; (r)[3] = (a)[3]; }

// --- 62-bit shift (delayed right shift used by binary EEA) ------------------
__device__ void ShiftR62(uint64_t r[5]) {
  r[0] = (r[1] << 2) | (r[0] >> 62);
  r[1] = (r[2] << 2) | (r[1] >> 62);
  r[2] = (r[3] << 2) | (r[2] >> 62);
  r[3] = (r[4] << 2) | (r[3] >> 62);
  r[4] = (int64_t)(r[4]) >> 62;
}

__device__ void ShiftR62(uint64_t dest[5], uint64_t r[5], uint64_t carry) {
  dest[0] = (r[1] << 2) | (r[0] >> 62);
  dest[1] = (r[2] << 2) | (r[1] >> 62);
  dest[2] = (r[3] << 2) | (r[2] >> 62);
  dest[3] = (r[4] << 2) | (r[3] >> 62);
  dest[4] = (carry << 2) | (r[4] >> 62);
}

// --- multiply 5-limb by small int64 -----------------------------------------
__device__ void IMult(uint64_t *r, uint64_t *a, int64_t b) {
  uint64_t t[NBBLOCK];
  if (b < 0) {
    b = -b;
    USUBO(t[0], 0ULL, a[0]);
    USUBC(t[1], 0ULL, a[1]);
    USUBC(t[2], 0ULL, a[2]);
    USUBC(t[3], 0ULL, a[3]);
    USUB(t[4], 0ULL, a[4]);
  } else {
    Load(t, a);
  }
  UMULLO(r[0], t[0], b);
  UMULLO(r[1], t[1], b);
  MADDO(r[1], t[0], b, r[1]);
  UMULLO(r[2], t[2], b);
  MADDC(r[2], t[1], b, r[2]);
  UMULLO(r[3], t[3], b);
  MADDC(r[3], t[2], b, r[3]);
  UMULLO(r[4], t[4], b);
  MADD(r[4], t[3], b, r[4]);
}

__device__ uint64_t IMultC(uint64_t *r, uint64_t *a, int64_t b) {
  uint64_t t[NBBLOCK];
  uint64_t carry;
  if (b < 0) {
    b = -b;
    USUBO(t[0], 0ULL, a[0]);
    USUBC(t[1], 0ULL, a[1]);
    USUBC(t[2], 0ULL, a[2]);
    USUBC(t[3], 0ULL, a[3]);
    USUB(t[4], 0ULL, a[4]);
  } else {
    Load(t, a);
  }
  UMULLO(r[0], t[0], b);
  UMULLO(r[1], t[1], b);
  MADDO(r[1], t[0], b, r[1]);
  UMULLO(r[2], t[2], b);
  MADDC(r[2], t[1], b, r[2]);
  UMULLO(r[3], t[3], b);
  MADDC(r[3], t[2], b, r[3]);
  UMULLO(r[4], t[4], b);
  MADDC(r[4], t[3], b, r[4]);
  MADDS(carry, t[4], b, 0ULL);
  return carry;
}

// --- multiply by 0x1000003D1 (2^32 + 977) with reduction folding ------------
__device__ void MulP(uint64_t *r, uint64_t a) {
  uint64_t ah, al;
  UMULLO(al, a, 0x1000003D1ULL);
  UMULHI(ah, a, 0x1000003D1ULL);
  USUBO(r[0], 0ULL, al);
  USUBC(r[1], 0ULL, ah);
  USUBC(r[2], 0ULL, 0ULL);
  USUBC(r[3], 0ULL, 0ULL);
  USUB(r[4], a, 0ULL);
}

// --- Add 5-limb with carry ------------------------------------------------
__device__ uint64_t AddCh(uint64_t r[5], uint64_t a[5], uint64_t carry) {
  uint64_t carryOut;
  UADDO1(r[0], a[0]);
  UADDC1(r[1], a[1]);
  UADDC1(r[2], a[2]);
  UADDC1(r[3], a[3]);
  UADDC1(r[4], a[4]);
  UADD(carryOut, carry, 0ULL);
  return carryOut;
}

// --- 4-limb modular ops (all PTX-based, no 128-bit) -------------------------

// r = a - b mod p
__device__ __forceinline__ void ModSub256(uint64_t *r, const uint64_t *a, const uint64_t *b) {
  uint64_t borrow;
  USUBO(r[0], a[0], b[0]);
  USUBC(r[1], a[1], b[1]);
  USUBC(r[2], a[2], b[2]);
  USUBC(r[3], a[3], b[3]);
  USUB(borrow, 0ULL, 0ULL);
  if (borrow) {
    UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL);
    UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL);
    UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL);
    UADD1(r[3], 0xFFFFFFFFFFFFFFFFULL);
  }
}

// r -= b mod p  (in place)
__device__ __forceinline__ void ModSub256(uint64_t *r, const uint64_t *b) {
  uint64_t borrow;
  USUBO1(r[0], b[0]);
  USUBC1(r[1], b[1]);
  USUBC1(r[2], b[2]);
  USUBC1(r[3], b[3]);
  USUB(borrow, 0ULL, 0ULL);
  if (borrow) {
    UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL);
    UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL);
    UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL);
    UADD1(r[3], 0xFFFFFFFFFFFFFFFFULL);
  }
}

// r = a + b mod p  (a,b in [0, 2^256); single conditional subtract, like original fieldAdd)
__device__ __forceinline__ void ModAdd256(uint64_t *r, const uint64_t *a, const uint64_t *b) {
  uint64_t c, br;
  uint64_t t[4];
  UADDO(r[0], a[0], b[0]);
  UADDC(r[1], a[1], b[1]);
  UADDC(r[2], a[2], b[2]);
  UADDC(r[3], a[3], b[3]);
  UADD(c, 0ULL, 0ULL);   // carry out (a+b >= 2^256)
  // subtract p once; if borrow occurred AND no carry-out, a+b < p so keep original
  USUBO(t[0], r[0], 0xFFFFFFFEFFFFFC2FULL);
  USUBC(t[1], r[1], 0xFFFFFFFFFFFFFFFFULL);
  USUBC(t[2], r[2], 0xFFFFFFFFFFFFFFFFULL);
  USUBC(t[3], r[3], 0xFFFFFFFFFFFFFFFFULL);
  USUB(br, 0ULL, 0ULL);
  if (c || !br) {
    r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
  }
}

// r = -a mod p
__device__ __forceinline__ void ModNeg256(uint64_t *r, const uint64_t *a) {
  uint64_t t[4];
  USUBO(t[0], 0ULL, a[0]);
  USUBC(t[1], 0ULL, a[1]);
  USUBC(t[2], 0ULL, a[2]);
  USUBC(t[3], 0ULL, a[3]);
  UADDO(r[0], t[0], 0xFFFFFFFEFFFFFC2FULL);
  UADDC(r[1], t[1], 0xFFFFFFFFFFFFFFFFULL);
  UADDC(r[2], t[2], 0xFFFFFFFFFFFFFFFFULL);
  UADD(r[3], t[3], 0xFFFFFFFFFFFFFFFFULL);
}

// parity of (a-b) mod p, without computing the full result
__device__ __forceinline__ void ModSub256isOdd(uint64_t *a, uint64_t *b, uint8_t *parity) {
  uint64_t t;
  uint64_t T[4];
  USUBO(T[0], a[0], b[0]);
  USUBC(T[1], a[1], b[1]);
  USUBC(T[2], a[2], b[2]);
  USUBC(T[3], a[3], b[3]);
  USUB(t, 0ULL, 0ULL);  // borrow
  *parity = (uint8_t)((T[0] & 1) ^ (t & 1));
}

__device__ __forceinline__ bool ge256_u64(const uint64_t a[4], uint64_t b) {
  if (a[3] | a[2] | a[1]) return true;
  return a[0] >= b;
}

// a -= dec  (in place, 256-bit minus u64, borrow propagation)
__device__ __forceinline__ void sub256_u64_inplace(uint64_t a[4], uint64_t dec) {
  uint64_t borrow = (a[0] < dec) ? 1ull : 0ull;
  a[0] = a[0] - dec;
  #pragma unroll
  for (int i = 1; i < 4; ++i) {
    uint64_t ai = a[i];
    uint64_t bi = borrow;
    a[i] = ai - bi;
    borrow = (ai < bi) ? 1ull : 0ull;
    if (!borrow) break;
  }
}

// a += inc  (in place, 256-bit plus u64, carry propagation)
__device__ __forceinline__ void inc256_device(uint64_t a[4], uint64_t inc) {
  uint64_t cur = a[0] + inc;
  uint64_t carry = (cur < a[0]) ? 1ull : 0ull;
  a[0] = cur;
  #pragma unroll
  for (int i = 1; i < 4 && carry; ++i) {
    uint64_t nv = a[i] + carry;
    carry = (nv < a[i]) ? 1ull : 0ull;
    a[i] = nv;
  }
}

__device__ __forceinline__ void ModNeg256(uint64_t *r) {
  uint64_t t[4];
  USUBO(t[0], 0ULL, r[0]);
  USUBC(t[1], 0ULL, r[1]);
  USUBC(t[2], 0ULL, r[2]);
  USUBC(t[3], 0ULL, r[3]);
  UADDO(r[0], t[0], 0xFFFFFFFEFFFFFC2FULL);
  UADDC(r[1], t[1], 0xFFFFFFFFFFFFFFFFULL);
  UADDC(r[2], t[2], 0xFFFFFFFFFFFFFFFFULL);
  UADD(r[3], t[3], 0xFFFFFFFFFFFFFFFFULL);
}

// r >= p ?
__device__ __forceinline__ bool ge256_p(const uint64_t r[4]) {
  if (r[3] != 0xFFFFFFFFFFFFFFFFULL) return false;
  if (r[2] != 0xFFFFFFFFFFFFFFFFULL) return false;
  if (r[1] != 0xFFFFFFFFFFFFFFFFULL) return false;
  return r[0] >= 0xFFFFFFFEFFFFFC2FULL;
}

// Fully reduce r into [0, p): r may be in [0, 2^256). Subtracts p at most 2 times.
__device__ __forceinline__ void fieldReduce(uint64_t r[4]) {
  uint64_t tmp[4];
  if (ge256_p(r)) {
    USUBO(tmp[0], r[0], 0xFFFFFFFEFFFFFC2FULL);
    USUBC(tmp[1], r[1], 0xFFFFFFFFFFFFFFFFULL);
    USUBC(tmp[2], r[2], 0xFFFFFFFFFFFFFFFFULL);
    USUBC(tmp[3], r[3], 0xFFFFFFFFFFFFFFFFULL);
    r[0]=tmp[0]; r[1]=tmp[1]; r[2]=tmp[2]; r[3]=tmp[3];
    if (ge256_p(r)) {
      USUBO(tmp[0], r[0], 0xFFFFFFFEFFFFFC2FULL);
      USUBC(tmp[1], r[1], 0xFFFFFFFFFFFFFFFFULL);
      USUBC(tmp[2], r[2], 0xFFFFFFFFFFFFFFFFULL);
      USUBC(tmp[3], r[3], 0xFFFFFFFFFFFFFFFFULL);
      r[0]=tmp[0]; r[1]=tmp[1]; r[2]=tmp[2]; r[3]=tmp[3];
    }
  }
}

// --- UMultSpecial: mul 4-limb by 0x1000003D1 with fold ---------------------
#define UMultSpecial(r, a) { \
  uint64_t temp; \
  r[0] = (a[0] << 32) + (a[0] << 9) + (a[0] << 8) + (a[0] << 7) + (a[0] << 6) + (a[0] << 4) + a[0]; \
  r[1] = (a[1] << 32) + (a[1] << 9) + (a[1] << 8) + (a[1] << 7) + (a[1] << 6) + (a[1] << 4) + a[1]; \
  MADDO(r[1], a[0], 0x1000003D1ULL, r[1]); \
  r[2] = (a[2] << 32) + (a[2] << 9) + (a[2] << 8) + (a[2] << 7) + (a[2] << 6) + (a[2] << 4) + a[2]; \
  MADDC(r[2], a[1], 0x1000003D1ULL, r[2]); \
  r[3] = (a[3] << 32) + (a[3] << 9) + (a[3] << 8) + (a[3] << 7) + (a[3] << 6) + (a[3] << 4) + a[3]; \
  temp = r[3]; \
  MADDC(r[3], a[2], 0x1000003D1ULL, r[3]); \
  r[4] = temp + a[3]; \
  MADD(r[4], a[3], 0x1000003D1ULL, 0ULL); \
}

// --- _ModMult: r = a*b mod p (256x256 -> 512 -> reduce) ---------------------
__device__ void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b) {
  uint64_t r512[8];
  uint64_t t[NBBLOCK];
  uint64_t ah, al;

  r512[5] = 0; r512[6] = 0; r512[7] = 0;

  UMult(r512, a, b[0]);
  UMult(t, a, b[1]);
  UADDO1(r512[1], t[0]);
  UADDC1(r512[2], t[1]);
  UADDC1(r512[3], t[2]);
  UADDC1(r512[4], t[3]);
  UADD1(r512[5], t[4]);
  UMult(t, a, b[2]);
  UADDO1(r512[2], t[0]);
  UADDC1(r512[3], t[1]);
  UADDC1(r512[4], t[2]);
  UADDC1(r512[5], t[3]);
  UADD1(r512[6], t[4]);
  UMult(t, a, b[3]);
  UADDO1(r512[3], t[0]);
  UADDC1(r512[4], t[1]);
  UADDC1(r512[5], t[2]);
  UADDC1(r512[6], t[3]);
  UADD1(r512[7], t[4]);

  // reduce 512 -> 320 by folding high 256 bits via 2^256 = 2^32+977
  UMultSpecial(t, (r512 + 4));
  UADDO1(r512[0], t[0]);
  UADDC1(r512[1], t[1]);
  UADDC1(r512[2], t[2]);
  UADDC1(r512[3], t[3]);

  // reduce 320 -> 256
  UADD1(t[4], 0ULL);
  UMULLO(al, t[4], 0x1000003D1ULL);
  UMULHI(ah, t[4], 0x1000003D1ULL);
  UADDO(r[0], r512[0], al);
  UADDC(r[1], r512[1], ah);
  UADDC(r[2], r512[2], 0ULL);
  UADD(r[3], r512[3], 0ULL);
}

// _ModMult(r, a): r = r * a mod p  (r in/out)
__device__ void _ModMult(uint64_t *r, uint64_t *a) {
  uint64_t r512[8];
  uint64_t t[NBBLOCK];
  uint64_t ah, al;
  r512[5] = 0; r512[6] = 0; r512[7] = 0;

  UMult(r512, a, r[0]);
  UMult(t, a, r[1]);
  UADDO1(r512[1], t[0]);
  UADDC1(r512[2], t[1]);
  UADDC1(r512[3], t[2]);
  UADDC1(r512[4], t[3]);
  UADD1(r512[5], t[4]);
  UMult(t, a, r[2]);
  UADDO1(r512[2], t[0]);
  UADDC1(r512[3], t[1]);
  UADDC1(r512[4], t[2]);
  UADDC1(r512[5], t[3]);
  UADD1(r512[6], t[4]);
  UMult(t, a, r[3]);
  UADDO1(r512[3], t[0]);
  UADDC1(r512[4], t[1]);
  UADDC1(r512[5], t[2]);
  UADDC1(r512[6], t[3]);
  UADD1(r512[7], t[4]);

  UMultSpecial(t, (r512 + 4));
  UADDO1(r512[0], t[0]);
  UADDC1(r512[1], t[1]);
  UADDC1(r512[2], t[2]);
  UADDC1(r512[3], t[3]);

  UADD1(t[4], 0ULL);
  UMULLO(al, t[4], 0x1000003D1ULL);
  UMULHI(ah, t[4], 0x1000003D1ULL);
  UADDO(r[0], r512[0], al);
  UADDC(r[1], r512[1], ah);
  UADDC(r[2], r512[2], 0ULL);
  UADD(r[3], r512[3], 0ULL);
}

// --- _ModSqr: r = a^2 mod p (fully unrolled cross-term squaring) ------------
__device__ void _ModSqr(uint64_t *rp, const uint64_t *up) {
  uint64_t r512[8];
  uint64_t SL, SH;

  {
    uint64_t r01L, r01H, r02L, r02H, r03L, r03H;
    UMULLO(SL, up[0], up[0]);
    UMULHI(SH, up[0], up[0]);
    UMULLO(r01L, up[0], up[1]);
    UMULHI(r01H, up[0], up[1]);
    UMULLO(r02L, up[0], up[2]);
    UMULHI(r02H, up[0], up[2]);
    UMULLO(r03L, up[0], up[3]);
    UMULHI(r03H, up[0], up[3]);

    r512[0] = SL;
    r512[1] = r01L;
    r512[2] = r02L;
    r512[3] = r03L;

    UADDO1(r512[1], SH);
    UADDC1(r512[2], r01H);
    UADDC1(r512[3], r02H);
    UADD(r512[4], r03H, 0ULL);

    uint64_t r12L, r12H, r13L, r13H;
    UMULLO(SL, up[1], up[1]);
    UMULHI(SH, up[1], up[1]);
    UMULLO(r12L, up[1], up[2]);
    UMULHI(r12H, up[1], up[2]);
    UMULLO(r13L, up[1], up[3]);
    UMULHI(r13H, up[1], up[3]);

    UADDO1(r512[1], r01L);
    UADDC1(r512[2], SL);
    UADDC1(r512[3], r12L);
    UADDC1(r512[4], r13L);
    UADD(r512[5], r13H, 0ULL);

    UADDO1(r512[2], r01H);
    UADDC1(r512[3], SH);
    UADDC1(r512[4], r12H);
    UADD1(r512[5], 0ULL);

    uint64_t r23L, r23H;
    UMULLO(SL, up[2], up[2]);
    UMULHI(SH, up[2], up[2]);
    UMULLO(r23L, up[2], up[3]);
    UMULHI(r23H, up[2], up[3]);

    UADDO1(r512[2], r02L);
    UADDC1(r512[3], r12L);
    UADDC1(r512[4], SL);
    UADDC1(r512[5], r23L);
    UADD(r512[6], r23H, 0ULL);

    UADDO1(r512[3], r02H);
    UADDC1(r512[4], r12H);
    UADDC1(r512[5], SH);
    UADD1(r512[6], 0ULL);

    UMULLO(SL, up[3], up[3]);
    UMULHI(SH, up[3], up[3]);

    UADDO1(r512[3], r03L);
    UADDC1(r512[4], r13L);
    UADDC1(r512[5], r23L);
    UADDC1(r512[6], SL);
    UADD(r512[7], SH, 0ULL);

    UADDO1(r512[4], r03H);
    UADDC1(r512[5], r13H);
    UADDC1(r512[6], r23H);
    UADD1(r512[7], 0ULL);
  }

  uint64_t t[NBBLOCK];
  UMult(t, (r512 + 4), 0x1000003D1ULL);
  UADDO1(r512[0], t[0]);
  UADDC1(r512[1], t[1]);
  UADDC1(r512[2], t[2]);
  UADDC1(r512[3], t[3]);

  UADD1(t[4], 0ULL);
  UMULLO(SL, t[4], 0x1000003D1ULL);
  UMULHI(SH, t[4], 0x1000003D1ULL);
  UADDO(rp[0], r512[0], SL);
  UADDC(rp[1], r512[1], SH);
  UADDC(rp[2], r512[2], 0ULL);
  UADD(rp[3], r512[3], 0ULL);
}

// --- count trailing zeros ----------------------------------------------------
__device__ __forceinline__ uint32_t ctz64(uint64_t x) {
  uint32_t n;
  asm("{\n\t"
      " .reg .u64 tmp;\n\t"
      " brev.b64 tmp, %1;\n\t"
      " clz.b64 %0, tmp;\n\t"
      "}"
      : "=r"(n) : "l"(x));
  return n;
}

#define SWAP(tmp, x, y) tmp = x; x = y; y = tmp;

// --- binary extended Euclidean step (62-bit delayed shifts) ------------------
__device__ void _DivStep62(uint64_t u[5], uint64_t v[5],
                           int32_t *pos,
                           int64_t *uu, int64_t *uv,
                           int64_t *vu, int64_t *vv) {
  *uu = 1; *uv = 0;
  *vu = 0; *vv = 1;

  uint32_t bitCount = 62;
  uint32_t zeros;
  uint64_t u0 = u[0];
  uint64_t v0 = v[0];

  uint64_t uh, vh;
  int64_t w, x, y, z;
  bitCount = 62;

  while (*pos > 0 && (u[*pos] | v[*pos]) == 0) (*pos)--;
  if (*pos == 0) {
    uh = u[0];
    vh = v[0];
  } else {
    uint32_t s = __clzll(u[*pos] | v[*pos]);
    if (s == 0) {
      uh = u[*pos];
      vh = v[*pos];
    } else {
      uh = __sleft128(u[*pos - 1], u[*pos], s);
      vh = __sleft128(v[*pos - 1], v[*pos], s);
    }
  }

  while (true) {
    zeros = ctz64(v0 | (1ULL << bitCount));
    v0 >>= zeros;
    vh >>= zeros;
    *uu <<= zeros;
    *uv <<= zeros;
    bitCount -= zeros;

    if (bitCount == 0)
      break;

    if (vh < uh) {
      SWAP(w, uh, vh);
      SWAP(x, u0, v0);
      SWAP(y, *uu, *vu);
      SWAP(z, *uv, *vv);
    }

    vh -= uh;
    v0 -= u0;
    *vv -= *uv;
    *vu -= *uu;
  }
}

__device__ void MatrixVecMulHalf(uint64_t dest[5], uint64_t u[5], uint64_t v[5],
                                 int64_t _11, int64_t _12, uint64_t *carry) {
  uint64_t t1[NBBLOCK];
  uint64_t t2[NBBLOCK];
  uint64_t c1, c2;

  c1 = IMultC(t1, u, _11);
  c2 = IMultC(t2, v, _12);

  UADDO(dest[0], t1[0], t2[0]);
  UADDC(dest[1], t1[1], t2[1]);
  UADDC(dest[2], t1[2], t2[2]);
  UADDC(dest[3], t1[3], t2[3]);
  UADDC(dest[4], t1[4], t2[4]);
  UADD(*carry, c1, c2);
}

__device__ void MatrixVecMul(uint64_t u[5], uint64_t v[5],
                             int64_t _11, int64_t _12, int64_t _21, int64_t _22) {
  uint64_t t1[NBBLOCK];
  uint64_t t2[NBBLOCK];
  uint64_t t3[NBBLOCK];
  uint64_t t4[NBBLOCK];

  IMult(t1, u, _11);
  IMult(t2, v, _12);
  IMult(t3, u, _21);
  IMult(t4, v, _22);

  UADDO(u[0], t1[0], t2[0]);
  UADDC(u[1], t1[1], t2[1]);
  UADDC(u[2], t1[2], t2[2]);
  UADDC(u[3], t1[3], t2[3]);
  UADD(u[4], t1[4], t2[4]);

  UADDO(v[0], t3[0], t4[0]);
  UADDC(v[1], t3[1], t4[1]);
  UADDC(v[2], t3[2], t4[2]);
  UADDC(v[3], t3[3], t4[3]);
  UADD(v[4], t3[4], t4[4]);
}

// --- _ModInv: modular inverse via binary EEA (5-limb, signed) ----------------
__device__ __noinline__ void _ModInv(uint64_t *R) {
  int64_t uu, uv, vu, vv;
  uint64_t mr0, ms0;
  int32_t pos = NBBLOCK - 1;

  uint64_t u[NBBLOCK];
  uint64_t v[NBBLOCK];
  uint64_t r[NBBLOCK];
  uint64_t s[NBBLOCK];
  uint64_t tr[NBBLOCK];
  uint64_t ts[NBBLOCK];
  uint64_t r0[NBBLOCK];
  uint64_t s0[NBBLOCK];
  uint64_t carryR;
  uint64_t carryS;

  u[0] = 0xFFFFFFFEFFFFFC2F;
  u[1] = 0xFFFFFFFFFFFFFFFF;
  u[2] = 0xFFFFFFFFFFFFFFFF;
  u[3] = 0xFFFFFFFFFFFFFFFF;
  u[4] = 0;
  Load(v, R);
  r[0] = 0; s[0] = 1;
  r[1] = 0; s[1] = 0;
  r[2] = 0; s[2] = 0;
  r[3] = 0; s[3] = 0;
  r[4] = 0; s[4] = 0;

  // Delayed right shift 62 bits
  while (true) {
    _DivStep62(u, v, &pos, &uu, &uv, &vu, &vv);
    MatrixVecMul(u, v, uu, uv, vu, vv);

    if (_IsNegative(u)) {
      Neg(u);
      uu = -uu;
      uv = -uv;
    }
    if (_IsNegative(v)) {
      Neg(v);
      vu = -vu;
      vv = -vv;
    }

    ShiftR62(u);
    ShiftR62(v);

    // Update r
    MatrixVecMulHalf(tr, r, s, uu, uv, &carryR);
    mr0 = (tr[0] * MM64) & MSK62;
    MulP(r0, mr0);
    carryR = AddCh(tr, r0, carryR);

    if (_IsZero(v)) {
      ShiftR62(r, tr, carryR);
      break;
    } else {
      // Update s
      MatrixVecMulHalf(ts, r, s, vu, vv, &carryS);
      ms0 = (ts[0] * MM64) & MSK62;
      MulP(s0, ms0);
      carryS = AddCh(ts, s0, carryS);
    }

    ShiftR62(r, tr, carryR);
    ShiftR62(s, ts, carryS);
  }

  if (!_IsOne(u)) {
    R[0] = 0ULL; R[1] = 0ULL; R[2] = 0ULL; R[3] = 0ULL; R[4] = 0ULL;
    return;
  }

  while (_IsNegative(r))
    AddP(r);
  while (!_IsNegative(r))
    SubP(r);
  AddP(r);

  Load(R, r);
}

// --- 4-limb wrappers ----------------------------------------------------------
__device__ void fieldMul(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
  _ModMult(out, (uint64_t *)a, (uint64_t *)b);
}
__device__ void fieldSqr(const uint64_t a[4], uint64_t out[4]) {
  _ModSqr(out, a);
}
__device__ void fieldInv(const uint64_t in[4], uint64_t out[4]) {
  uint64_t t[5];
  t[0] = in[0]; t[1] = in[1]; t[2] = in[2]; t[3] = in[3]; t[4] = 0;
  _ModInv(t);
  out[0] = t[0]; out[1] = t[1]; out[2] = t[2]; out[3] = t[3];
}

// --- Affine EC point ops (used only for seeding / scalar mult, not hot loop) --
struct ECPointA {
  uint64_t X[4];
  uint64_t Y[4];
  bool infinity;
};

__device__ __forceinline__ void pointSetInfinity(ECPointA &P) {
  P.infinity = true;
  P.X[0] = P.X[1] = P.X[2] = P.X[3] = 0ULL;
  P.Y[0] = P.Y[1] = P.Y[2] = P.Y[3] = 0ULL;
}
__device__ __forceinline__ void pointSetG(ECPointA &P) {
  pointSetInfinity(P);
  P.infinity = false;
  P.X[0] = SECP_GX_LE[0]; P.X[1] = SECP_GX_LE[1]; P.X[2] = SECP_GX_LE[2]; P.X[3] = SECP_GX_LE[3];
  P.Y[0] = SECP_GY_LE[0]; P.Y[1] = SECP_GY_LE[1]; P.Y[2] = SECP_GY_LE[2]; P.Y[3] = SECP_GY_LE[3];
}

__device__ void pointDoubleAffine(const ECPointA &P, ECPointA &R) {
  if (P.infinity) { pointSetInfinity(R); return; }
  uint64_t x2[4], two_x2[4], three_x2[4];
  uint64_t denom[4], invDen[4], lambda[4];

  fieldSqr(P.X, x2);
  ModAdd256(two_x2, x2, x2);
  ModAdd256(three_x2, two_x2, x2);

  ModAdd256(denom, P.Y, P.Y);
  fieldInv(denom, invDen);

  fieldMul(three_x2, invDen, lambda);

  uint64_t lambda2[4], twoX[4], newX[4];
  fieldSqr(lambda, lambda2);
  ModAdd256(twoX, P.X, P.X);
  ModSub256(newX, lambda2, twoX);

  uint64_t tmp[4], prod[4], newY[4];
  ModSub256(tmp, P.X, newX);
  fieldMul(lambda, tmp, prod);
  ModSub256(newY, prod, P.Y);

  R.X[0] = newX[0]; R.X[1] = newX[1]; R.X[2] = newX[2]; R.X[3] = newX[3];
  R.Y[0] = newY[0]; R.Y[1] = newY[1]; R.Y[2] = newY[2]; R.Y[3] = newY[3];
  R.infinity = false;
}

__device__ void pointAddAffine(const ECPointA &P, const ECPointA &Q, ECPointA &R) {
  if (P.infinity) { R = Q; return; }
  if (Q.infinity) { R = P; return; }

  bool sameX = (P.X[0] == Q.X[0] && P.X[1] == Q.X[1] && P.X[2] == Q.X[2] && P.X[3] == Q.X[3]);
  bool sameY = (P.Y[0] == Q.Y[0] && P.Y[1] == Q.Y[1] && P.Y[2] == Q.Y[2] && P.Y[3] == Q.Y[3]);

  if (sameX && sameY) { pointDoubleAffine(P, R); return; }
  if (sameX && !sameY) { pointSetInfinity(R); return; }

  uint64_t dx[4], dy[4], invdx[4], lambda[4], lambda2[4];
  uint64_t tmp1[4], prod[4], newX[4], newY[4];

  ModSub256(dx, Q.X, P.X);
  ModSub256(dy, Q.Y, P.Y);
  fieldInv(dx, invdx);
  fieldMul(dy, invdx, lambda);

  fieldSqr(lambda, lambda2);
  ModSub256(tmp1, lambda2, P.X);
  ModSub256(newX, tmp1, Q.X);

  ModSub256(tmp1, P.X, newX);
  fieldMul(lambda, tmp1, prod);
  ModSub256(newY, prod, P.Y);

  R.X[0] = newX[0]; R.X[1] = newX[1]; R.X[2] = newX[2]; R.X[3] = newX[3];
  R.Y[0] = newY[0]; R.Y[1] = newY[1]; R.Y[2] = newY[2]; R.Y[3] = newY[3];
  R.infinity = false;
}

// k*G via double-and-add (seeding only)
__device__ void scalarMulBaseAffine(const uint64_t scalar_le[4], uint64_t outX[4], uint64_t outY[4]) {
  ECPointA R;
  pointSetInfinity(R);

  int msb = -1;
  for (int limb = 3; limb >= 0; --limb) {
    uint64_t v = scalar_le[limb];
    if (v != 0) { msb = limb * 64 + 63 - __clzll(v); break; }
  }
  if (msb == -1) {
    outX[0] = outX[1] = outX[2] = outX[3] = 0ULL;
    outY[0] = outY[1] = outY[2] = outY[3] = 0ULL;
    return;
  }

  for (int bi = msb; bi >= 0; --bi) {
    if (!R.infinity) {
      ECPointA tmpD;
      pointDoubleAffine(R, tmpD);
      R = tmpD;
    }
    int limb = bi >> 6;
    int shift = bi & 63;
    uint64_t bit = (scalar_le[limb] >> shift) & 1ULL;
    if (bit) {
      ECPointA Gp;
      pointSetG(Gp);
      if (R.infinity) {
        R = Gp;
      } else {
        ECPointA tmpA;
        pointAddAffine(R, Gp, tmpA);
        R = tmpA;
      }
    }
  }

  if (R.infinity) {
    outX[0] = outX[1] = outX[2] = outX[3] = 0ULL;
    outY[0] = outY[1] = outY[2] = outY[3] = 0ULL;
  } else {
    outX[0] = R.X[0]; outX[1] = R.X[1]; outX[2] = R.X[2]; outX[3] = R.X[3];
    outY[0] = R.Y[0]; outY[1] = R.Y[1]; outY[2] = R.Y[2]; outY[3] = R.Y[3];
  }
}