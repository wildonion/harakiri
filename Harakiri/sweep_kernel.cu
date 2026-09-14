// Harakiri Phase 2 — affine sweep with one-inversion batch (Montgomery trick).
// Port of CUDACyclone's kernel_point_add_and_check_oneinv onto verified field.h/hash.h.
// Each thread: holds base point P (x1,y1) + start scalar S, sweeps a batch of B=2*half
// consecutive scalars per inversion, checking hash160(prefix||x) for each.
// Batch layout: checks S itself, then S+(i+1) and S-(i+1) for i in 0..half-2,
// then S-half (the last neg side), then advances base by J=B*G.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <string>

#include "field.h"
#include "hash.h"

// ---------------------------------------------------------------------------
// constants / protocol
// ---------------------------------------------------------------------------
#define FOUND_NONE   0
#define FOUND_LOCK   1
#define FOUND_READY  2

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif

#ifndef MAX_SECTIONS
#define MAX_SECTIONS 64
#endif

struct FoundResult {
  uint64_t scalar[4];
  uint64_t Rx[4];
  uint64_t Ry[4];
  int      threadId;
  int      iter;
};

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[MAX_SECTIONS][4];
__constant__ uint64_t c_Jy[MAX_SECTIONS][4];
__constant__ int      c_dir[MAX_SECTIONS];      // +1 forward group, -1 backward group
__constant__ uint32_t c_target_prefix;            // first 4 bytes of hash160 as LE u32
__constant__ uint8_t  c_target_hash160[20];

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
  return *((const volatile int*)p);
}
__device__ __forceinline__ bool warp_found_ready(const int* __restrict__ d_found_flag,
                                                 unsigned full_mask, unsigned lane) {
  int f = 0;
  if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
  f = __shfl_sync(full_mask, f, 0);
  // Stop on ANY nonzero flag: the finder claims with FOUND_LOCK before it has
  // written the result and flipped to FOUND_READY. Waiting for READY means every
  // other warp keeps hashing during that write window (microseconds of wasted
  // work across the whole grid, repeatedly). The host only reads the result
  // after cudaDeviceSynchronize, so stopping on LOCK is safe.
  return f != FOUND_NONE;
}
// any_warp_stop = local found flag OR global stop flag (host-mapped pinned
// memory, set when any OTHER device found the key; lets multi-GPU sweeps halt
// all devices within microseconds of the first find).
// NOTE: the host-mapped global flag lives in system RAM, so a volatile read of
// it is an UNCACHED PCIe access. Reading it every batch iteration (or worse,
// every inner-loop step) tanks throughput (we measured ~5.8x). We therefore
// only poll it when asked (callers gate it, e.g. every 8 batches), while the
// device-local found flag stays checked every iteration (fast, on-device).
__device__ __forceinline__ bool any_warp_stop(const int* __restrict__ d_found_flag,
                                              const int* __restrict__ d_global_stop,
                                              unsigned full_mask, unsigned lane) {
  int f = 0;
  if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
  f = __shfl_sync(full_mask, f, 0);
  if (f != FOUND_NONE) return true;
  if (d_global_stop != nullptr) {
    int g = 0;
    if (lane == 0) g = load_found_flag_relaxed(d_global_stop);
    g = __shfl_sync(full_mask, g, 0);
    if (g != 0) return true;
  }
  return false;
}
__device__ __forceinline__ unsigned long long warp_reduce_add_ull(unsigned long long v) {
  unsigned mask = 0xFFFFFFFFu;
  v += __shfl_down_sync(mask, v, 16);
  v += __shfl_down_sync(mask, v, 8);
  v += __shfl_down_sync(mask, v, 4);
  v += __shfl_down_sync(mask, v, 2);
  v += __shfl_down_sync(mask, v, 1);
  return v;
}

// ---------------------------------------------------------------------------
// sweep kernel
// ---------------------------------------------------------------------------
__launch_bounds__(256, 2)
__global__ void kernel_point_add_and_check_oneinv(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars,
    uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
    uint64_t groupSize,          // threads per section-group (stride = groupSize*B)
    uint32_t batch_size,
    uint32_t max_batches_per_launch,
    int* __restrict__ d_found_flag,
    const int* __restrict__ d_global_stop,   // host-mapped; nonzero => stop all (another device found)
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left)
{
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    // Round-robin stride: thread g processes batches g, g+T, g+2T, ...
    // so the whole grid sweeps the range in linear order (frontier advances
    // by T*B keys per round). The batch step (point advance) is T*B*G and
    // the scalar advance is T*B per batch. c_Jx/c_Jy hold (T*B)*G.
    // With sections (--sections N), the grid is split into N groups of
    // groupSize threads each; group gid/groupSize owns a contiguous slice of
    // the range and strides by groupSize*B within it. c_Jx[g]/c_Jy[g] holds
    // that group's (groupSize*B)*G advance point.
    const uint64_t stride = groupSize * (uint64_t)B;

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;
    const uint64_t group = (groupSize > 0) ? (gid / groupSize) : 0ull;

    const unsigned lane      = (unsigned)(threadIdx.x & 31u);
    const unsigned full_mask = 0xFFFFFFFFu;
    if (any_warp_stop(d_found_flag, d_global_stop, full_mask, lane)) return;

    // Hoist this group's stride point into registers ONCE (constant memory with
    // a runtime index [group] is slow; the old [4]-sized c_Jx was compile-time
    // indexed and got register-cached by the compiler. Read it a single time
    // here so the hot per-batch loop only touches registers).
    uint64_t Jx_group[4], Jy_group[4];
#pragma unroll
    for (int j=0;j<4;++j) { Jx_group[j] = c_Jx[group][j]; Jy_group[j] = c_Jy[group][j]; }

    // Sweep direction for this group: +1 = forward (march S upward, base point
    // advances by +J each batch), -1 = backward (march S downward, base point
    // advances by -J each batch). For backward groups the host uploads the
    // NEGATED J point (Jx, p-Jy) so the exact same point-add advances downward,
    // and only the scalar advance below branches on direction.
    const int dir = c_dir[group];

    const uint32_t target_prefix = c_target_prefix;


    unsigned int local_hashes = 0;
    // EXACT per-lane counter: each lane atomically adds its OWN count when it
    // hits a flush boundary (and its remainder on exit). The old version did a
    // warp reduction and reset ALL lanes whenever ANY lane hit the boundary,
    // silently discarding the partial counts of drifting lanes (measured ~3%
    // loss on normal-exit runs and up to ~30% on found runs). atomicAdd
    // contention is negligible: one add per 65536 hashes per thread.
    #define FLUSH_THRESHOLD 65536u
    #define WARP_FLUSH_HASHES() do { \
        if (local_hashes) { atomicAdd(hashes_accum, (unsigned long long)local_hashes); local_hashes = 0; } \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

#ifdef SWEEP_KERNEL_DEBUG
    // one sample thread per block (gid 0,256,512,...) + block0, reports its
    // batch at exit so we can see the block distribution / stop latency.
    #define DBG_EXIT(site) do { \
        if (gid < 256u || (gid & 255u) == 0u) \
            printf("EXIT-%s gid=%u batches=%u\\n", site, (unsigned)gid, batches_done); \
    } while (0)
#else
    #define DBG_EXIT(site) do { } while (0)
#endif

    uint64_t x1[4], y1[4], S[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        x1[i] = Px[gid*4 + i];
        y1[i] = Py[gid*4 + i];
        S[i]  = start_scalars[gid*4 + i];
    }
    uint64_t rem[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4 + i];

#ifdef SWEEP_KERNEL_DEBUG
    if (gid == 30 || gid == 0 || gid == 31) {
      printf("[gid=%llu] Px0=%016llx Py0=%016llx S0=%016llx rem=%016llx target_prefix=%08x\n",
        (unsigned long long)gid,
        (unsigned long long)x1[0], (unsigned long long)y1[0],
        (unsigned long long)S[0], (unsigned long long)rem[0], target_prefix);
    }
#endif

    if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { Rx[gid*4+i] = x1[i]; Ry[gid*4+i] = y1[i]; }
        WARP_FLUSH_HASHES(); return;
    }

    uint32_t batches_done = 0;

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        // local found flag: every batch (fast, on-device). Host-mapped global
        // stop flag: every 64 batches only (uncached PCIe read; gating keeps
        // it off the hot path - worst-case cross-device stop latency ~64
        // batches = microseconds of work, vs finishing the whole ~10s launch).
        if ((batches_done & 63u) == 0u) {
            if (any_warp_stop(d_found_flag, d_global_stop, full_mask, lane)) { DBG_EXIT("stop64"); WARP_FLUSH_HASHES(); return; }
        } else {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { DBG_EXIT("stop1"); WARP_FLUSH_HASHES(); return; }
        }

        // --- check base point S*G ---
        {
            uint32_t h[5];
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            hash160_33_from_x_red(prefix, x1, h);
            ++local_hashes; MAYBE_WARP_FLUSH();
#ifdef SWEEP_KERNEL_DEBUG
            if (gid == 30) {
              printf("[gid=30] base: prefix=%u x1[3..0]=%016llx %016llx %016llx %016llx h0=%08x h1=%08x pref=%d\n",
                (unsigned)prefix,
                (unsigned long long)x1[3], (unsigned long long)x1[2],
                (unsigned long long)x1[1], (unsigned long long)x1[0],
                h[0], h[1], (int)(h[0] == target_prefix));
            }
#endif
            bool pref = (h[0] == target_prefix);
            if (__any_sync(full_mask, pref)) {
                // full compare
                uint8_t h20[20];
                for (int k=0;k<5;++k) {
                    h20[k*4+0]=(uint8_t)(h[k]      );
                    h20[k*4+1]=(uint8_t)(h[k] >> 8 );
                    h20[k*4+2]=(uint8_t)(h[k] >> 16);
                    h20[k*4+3]=(uint8_t)(h[k] >> 24);
                }
                bool full = true;
                for (int k=0;k<20;++k) if (h20[k]!=c_target_hash160[k]) full=false;
                if (pref && full) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=S[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=x1[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y1[k];
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = (int)batches_done;
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                if (__any_sync(full_mask, pref && full)) {
                    __syncwarp(full_mask); DBG_EXIT("found-base"); WARP_FLUSH_HASHES(); return;
                }
            }
        }

        // --- build suffix product tree: subp[i] = prod_{j=i}^{half-1} (c_Gx[j+1]-x1) ... ---
        // Port of original: subp[half-1] = (Jx - x1); subp[i] = (Jx-x1)*prod(c_Gx[i+1..half-1]-x1)
        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

#pragma unroll
        for (int j=0;j<4;++j) acc[j] = Jx_group[j];
        ModSub256(acc, acc, x1);
#pragma unroll
        for (int j=0;j<4;++j) subp[half-1][j] = acc[j];

        for (int i = half - 2; i >= 0; --i) {
#pragma unroll
            for (int j=0;j<4;++j) tmp[j] = c_Gx[(size_t)(i+1)*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
#pragma unroll
            for (int j=0;j<4;++j) subp[i][j] = acc[j];
        }

        uint64_t d0[4], inverse[5];
#pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_Gx[0*4 + j];
        ModSub256(d0, d0, x1);
#pragma unroll
        for (int j=0;j<4;++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ull;
        _ModInv(inverse);

        uint64_t sy_neg[4], sx_neg[4];
        ModNeg256(sy_neg, y1);
        ModNeg256(sx_neg, x1);

        // --- main sweep: i in 0..half-2, both + and - sides ---
        for (int i = 0; i < half - 1; ++i) {
            // device-local flag only (on-device volatile read is fast; the
            // host-mapped global flag is polled once per batch at loop top)
            if (warp_found_ready(d_found_flag, full_mask, lane)) { DBG_EXIT("stop-inner"); WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            // + (i+1)*G
            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                fieldReduce(s);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint32_t h[5]; hash160_33_from_x_red(odd?0x03:0x02, px3, h);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool pref = (h[0] == target_prefix);
                if (__any_sync(full_mask, pref)) {
                    uint8_t h20[20];
                    for (int k=0;k<5;++k) {
                        h20[k*4+0]=(uint8_t)(h[k]      );
                        h20[k*4+1]=(uint8_t)(h[k] >> 8 );
                        h20[k*4+2]=(uint8_t)(h[k] >> 16);
                        h20[k*4+3]=(uint8_t)(h[k] >> 24);
                    }
                    bool full = true;
                    for (int k=0;k<20;++k) if (h20[k]!=c_target_hash160[k]) full=false;
                    if (pref && full) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t addv=(uint64_t)(i+1);
                            for (int k=0;k<4 && addv;++k){ uint64_t old=fs[k]; fs[k]=old+addv; addv=(fs[k]<old)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = (int)batches_done;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    if (__any_sync(full_mask, pref && full)) {
                        __syncwarp(full_mask); DBG_EXIT("found-x"); WARP_FLUSH_HASHES(); return;
                    }
                }
            }

            // - (i+1)*G
            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
                ModNeg256(py_i, py_i);

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                fieldReduce(s);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint32_t h[5]; hash160_33_from_x_red(odd?0x03:0x02, px3, h);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool pref = (h[0] == target_prefix);
                if (__any_sync(full_mask, pref)) {
                    uint8_t h20[20];
                    for (int k=0;k<5;++k) {
                        h20[k*4+0]=(uint8_t)(h[k]      );
                        h20[k*4+1]=(uint8_t)(h[k] >> 8 );
                        h20[k*4+2]=(uint8_t)(h[k] >> 16);
                        h20[k*4+3]=(uint8_t)(h[k] >> 24);
                    }
                    bool full = true;
                    for (int k=0;k<20;++k) if (h20[k]!=c_target_hash160[k]) full=false;
                    if (pref && full) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t sub=(uint64_t)(i+1);
                            for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = (int)batches_done;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    if (__any_sync(full_mask, pref && full)) {
                        __syncwarp(full_mask); DBG_EXIT("found-x"); WARP_FLUSH_HASHES(); return;
                    }
                }
            }

            uint64_t gxmi[4];
#pragma unroll
            for (int j=0;j<4;++j) gxmi[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(gxmi, gxmi, x1);
            _ModMult(inverse, inverse, gxmi);
        }

        // --- last: i = half-1, only - half*G side, then advance base by J=B*G ---
        {
            const int i = half - 1;
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
                ModNeg256(py_i, py_i);

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                fieldReduce(s);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint32_t h[5]; hash160_33_from_x_red(odd?0x03:0x02, px3, h);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool pref = (h[0] == target_prefix);
                if (__any_sync(full_mask, pref)) {
                    uint8_t h20[20];
                    for (int k=0;k<5;++k) {
                        h20[k*4+0]=(uint8_t)(h[k]      );
                        h20[k*4+1]=(uint8_t)(h[k] >> 8 );
                        h20[k*4+2]=(uint8_t)(h[k] >> 16);
                        h20[k*4+3]=(uint8_t)(h[k] >> 24);
                    }
                    bool full = true;
                    for (int k=0;k<20;++k) if (h20[k]!=c_target_hash160[k]) full=false;
                    if (pref && full) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t sub=(uint64_t)half;
                            for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = (int)batches_done;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    if (__any_sync(full_mask, pref && full)) {
                        __syncwarp(full_mask); DBG_EXIT("found-x"); WARP_FLUSH_HASHES(); return;
                    }
                }
            }

            uint64_t last_dx[4];
#pragma unroll
            for (int j=0;j<4;++j) last_dx[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(last_dx, last_dx, x1);
            _ModMult(inverse, inverse, last_dx);
        }

        // --- advance base: (x1,y1) = (x1,y1) + B*G  (J = B*G) ---
        {
            uint64_t lam[4], s[4], x3[4], y3[4];

            uint64_t Jy_minus_y1[4];
#pragma unroll
            for (int j=0;j<4;++j) Jy_minus_y1[j] = Jy_group[j];
            ModSub256(Jy_minus_y1, Jy_minus_y1, y1);

            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            uint64_t Jx_local[4]; for (int j=0;j<4;++j) Jx_local[j]=Jx_group[j];
            ModSub256(x3, x3, Jx_local);

            ModSub256(s, x1, x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y1);

#pragma unroll
            for (int j=0;j<4;++j) { x1[j] = x3[j]; y1[j] = y3[j]; }
        }

        {
            uint64_t addv = stride;
            if (dir > 0) {
                // forward: S += stride
                for (int k=0;k<4 && addv;++k){ uint64_t old=S[k]; S[k]=old+addv; addv=(S[k]<old)?1ull:0ull; }
            } else {
                // backward: S -= stride
                for (int k=0;k<4 && addv;++k){ uint64_t old=S[k]; S[k]=old-addv; addv=(old<addv)?1ull:0ull; }
            }
            sub256_u64_inplace(rem, (uint64_t)B);
        }
        ++batches_done;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = S[i];
    }
    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull) {
#ifdef SWEEP_KERNEL_DEBUG
        if (gid < 4 || gid == 4096) printf("[gid=%llu] any_left hit: rem=%016llx %016llx %016llx %016llx batches=%u\n",
          (unsigned long long)gid, (unsigned long long)rem[0], (unsigned long long)rem[1],
          (unsigned long long)rem[2], (unsigned long long)rem[3], batches_done);
#endif
        atomicAdd(d_any_left, 1u);
    }

#ifdef SWEEP_KERNEL_DEBUG
    if (gid == 30 || gid == 0) {
      printf("[gid=%llu] END: rem=%016llx S0=%016llx batches_done=%u\n",
        (unsigned long long)gid, (unsigned long long)rem[0],
        (unsigned long long)S[0], batches_done);
    }
#endif

    WARP_FLUSH_HASHES();
    DBG_EXIT("normal");
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}
