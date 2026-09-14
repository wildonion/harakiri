
<p align="center">
    <img src="https://github.com/wildonion/harakiri/blob/master/Harakiri/harakiri.jpg">
</p>

# Harakiri — CUDA secp256k1 Puzzle Sweeper

High-speed GPU private-key search for Bitcoin puzzle-style challenges.
Sweeps a scalar range, hashes each candidate public key
(SHA-256 + RIPEMD-160), and compares against a target hash160/address.

**Engine: asynchronous split-phase sweep.**
- One field inversion per batch of `B` points (suffix-product trick), so the
  per-key cost is a single point addition + one hash160, not a per-key inverse.
- **Bidirectional batch:** every batch of `B = 2*half` points covers
  `[S-half, S+half-1]` centered on the thread's current base `S` — the two
  `±(i+1)·G` branches are computed from the same inversion.
- **Two-front race (optional `--backward`):** each section can be swept from
  both ends at once — a forward half from the start plus a backward half from
  the end. On a single GPU this is expected-time-neutral (the fronts split the
  threads); its real payoff is on multi-GPU, where two devices sweep from
  opposite ends at full speed. See "Two-front race" below.
- **256-bit ranges:** full `[start, end]` hex ranges are supported (needed for
  puzzle 71, whose length is 2^70 ≈ 1.18e21 keys, far beyond 64-bit).
- **Multi-launch persistence:** work state (base point, scalar, remaining
  count) is kept on device between launches, so `--maxbatch` can bound each
  kernel launch without losing progress.
- **Live progress:** long sweeps are automatically split into ~10-second
  timeboxed launches, and a progress line is printed after each one
  (elapsed, Mkeys/s, keys done/total, ETA). No more silent 30-minute runs —
  you always see it working. Pass `--maxbatch` to disable timeboxing and do
  one giant launch instead.
- Prefix pre-filter: hash160 is compared against a 32-bit prefix first
  (__any_sync warp abort), full 20-byte compare only on prefix match.
- **Hash pipeline = Bitcoin's exact HASH160, fused in registers:** every
  candidate pubkey is compressed (0x02/0x03 + x), SHA-256'd, RIPEMD-160'd,
  then compared. The whole pipeline is `__forceinline__` + register-resident —
  SHA-256's digest feeds RIPEMD-160 through registers, never memory. The
  hot path uses the **register-reduced variant** `hash160_33_from_x_red`
  (`hash.h`): the SHA-256 message words W[0..7] are built **directly from the
  x limbs** and the RIPEMD-160 message words as byte-swaps of the SHA digest —
  no `msg[]`/`sha_bytes[]` byte arrays exist at all, which drops live-register
  pressure and every byte-shuffle instruction from the per-key loop (single
  specialized 33-byte SHA block, unrolled message schedule, 128 regs/thread).
  No shortcuts: any found key is a real address match.
- **Round-robin linear march:** thread `g` takes batches `g, g+T, g+2T, …`,
  so the whole grid sweeps the range in order and total-hashes-until-find ≈
  key position (not `threads × slab offset`).

---

## Requirements

- NVIDIA GPU: any CUDA-capable GPU. The build targets your GPU's architecture
  (SM) explicitly via a CLI flag — see Build below. Native SASS for the exact
  SM (sm_86 = RTX 30xx/Ampere, sm_89 = RTX 40xx/Ada, sm_120 = RTX 50xx/
  Blackwell) is embedded **plus** the matching compute PTX, which the driver
  JIT-compiles on any newer GPU.
- CUDA Toolkit **12.x** at
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\`
  (`build.bat` auto-detects the newest 12.x it finds).
- Visual Studio 2022 (MSVC) with C++ toolchain
  (`vswhere` must find it; `build.bat` locates the install automatically).
- Windows x64.

## Build

The project lives in `…\src\Harakiri\gpu\` (this README sits alongside the
source). Build from that directory:

```
cd src\Harakiri\gpu
build.bat                 auto: native SASS for the local GPU (-arch=native)
build.bat sm_86           native Ampere (RTX 30xx) + embedded compute_86 PTX
build.bat sm_120          native Blackwell (RTX 50xx) + embedded compute_120 PTX
build.bat sm_86 sm_120    multiple archs in one exe
```

The arch argument selects the target SM. Without an argument, `-arch=native`
compiles SASS for the local GPU; with explicit SM(s), the build embeds both
native SASS and the matching compute PTX so the exe also JIT-compiles on any
newer GPU. Requires **CUDA 12.x** (auto-detected; the sm_120/Blackwell targets
need CUDA 12.8+). Output is `harakiri_sweep.exe` either way.

Verify the install:

```
harakiri_sweep.exe --selftest --batch 1024 --range 0xdeadbeefcafef00d:500000
```

Expected output ends with `SELFTEST PASS` (and prints the detected GPU + SM count).

---

## Running for Puzzle 71

Puzzle 71:
- **Address:** `1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU`
- **Private key range:** `[0x400000000000000000, 0x7fffffffffffffffff]`
  (i.e. `2^70 .. 2^71-1`, length `2^70`)
- **hash160:** `f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8`
  (derived automatically from the address; no need to type it)

### One-line command (full puzzle-71 range)

```
harakiri_sweep.exe --batch 1024 --jobs 5120 --range 0x400000000000000000:0x7fffffffffffffffff --target 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU --maxbatch 1000000
```

That searches the **entire** puzzle-71 range, bidirectionally, until the key is
found or the range is exhausted. On an RTX 3050 at ~0.5 Gkeys/s the full 2^70
range is a ~140,000-year lottery — it is a long-running background job, not a
quick win. Run it with `--verbose` to watch progress, and read the ETA line.

### Bounded / progress runs

Cap each kernel launch with `--maxbatch` (work persists between launches, so
nothing is lost):

```
harakiri_sweep.exe --batch 1024 --jobs 5120 --range 0x400000000000000000:0x7fffffffffffffffff --target 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU --maxbatch 1000 --verbose
```

Stop anytime (Ctrl+C) and re-run the same command — it restarts at the range
start, not where you stopped. For resume support across restarts you must keep
the same `--range` and accept the restart from the beginning (device state is
persistent only within one process).

### Search a sub-range (first 2^32 keys for a quick sanity run)

```
harakiri_sweep.exe --batch 1024 --jobs 5120 --range 0x400000000000000000:0x4000000000ffffffff --target 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
```

---

## Range syntax

Three forms (all start values accept optional `0x` prefix):

| Form               | Meaning                                   | Example                                        |
|--------------------|-------------------------------------------|------------------------------------------------|
| `--range S`        | from `S` to infinity (2^96 cap)           | `--range 0x400000000000000000`                |
| `--range S:LEN`    | `S` hex + decimal key count               | `--range 0x123456789abcdef0:500000`           |
| `--range S:E`      | `S` to `E` **hex, inclusive**             | `--range 0x400000000000000000:0x7fffffffffffffffff` |

The `S:E` form is detected when the part after `:` contains hex letters
(`a-f`/`A-F`) or a `0x` prefix; pure-digit tails are treated as decimal lengths.
Length = `E - S + 1` (end inclusive).

---

## Options

| Option           | Default | Description                                              |
|------------------|---------|----------------------------------------------------------|
| `--batch N`      | 128     | Points per inversion batch (even, 2..1024). **1024 = the cap and the measured optimum.** `B>1024` is silently rejected by the kernel (0 keys). |
| `--threads N`    | 256     | Threads per block (kernel is `__launch_bounds__(256,2)`; max 256). |
| `--jobs N`       | auto    | Total worker threads; auto = `256 × SM count` (5120 on the 3050). **5120 = single-wave optimum** — see §6 below. |
| `--target A`     | —       | 40-hex hash160 **or** Base58Check address (`1PWo…`).    |
| `--iters N`      | 0       | Cap kernel launches (0 = until found/exhausted).        |
| `--maxbatch N`   | 0       | Cap batches per launch (unlimited by default).          |
| `--out FILE`     | —       | Write the found result to FILE (see below).             |
| `--warmup N`     | 0       | Run N seconds of GPU burn-in before the sweep (see below). |
| `--sections N`   | 1       | Split the range into N parallel sections (see below).   |
| `--backward`     | off     | Two-front race: forward half + backward half per section (see below). |
| `--verbose`      | off     | Per-launch progress lines.                              |
| `--selftest`     | off     | Verify engine on a random scalar inside the range.      |

`--jobs` defaults to `256 × detected SM count` (auto-detected from the GPU at
startup; you can still override it explicitly). The GPU name / SM count is
printed on every run.

### Sections / multi-GPU (`--sections N`)

`--sections N` splits the range into N contiguous slices that are swept in
parallel. With one GPU the slices run as **groups inside a single grid**
(each group of `jobs/N` threads strides independently through its own slice);
with N GPUs the slices are spread across the devices (one host thread per GPU,
each with its own grid, buffers and found flag). When any device finds the key,
a host-mapped stop flag halts every other device's kernel within microseconds
so no GPU wastes time after the answer is known.

```
harakiri_sweep.exe --batch 1024 --jobs 5120 --range 0x400000000000000000:0x7fffffffffffffffff --target 1PWo… --sections 2
```

On a multi-GPU box, `--sections 2` uses 2 GPUs; `--sections 4` uses up to 4
(min(N, GPUs)); on a single GPU the sections are parallel groups in one grid.
`--sections 1` (default) is the original single-frontier sweep. The selftest
forces `--sections 1` internally.

> **`--sections N` does NOT speed up a single GPU.** All sections share the same
> silicon, so the total work and the wave schedule are unchanged — measured:
> the same 2^32 key at `--sections 4` still took ~7 s, identical to 1 section.
> Sections exist purely to spread work across multiple physical GPUs.

### Two-front race (`--backward`)

By default every section is swept **forward only** (the frontier marches from the
section start to the end). `--backward` turns each section into **two racing
fronts**:

- the **lower half** `[A, A+len/2)` is marched **forward** from the start `A`;
- the **upper half** `[A+len/2, A+len)` is marched **backward** from the end `A+len`.

Think of a street with the key hidden somewhere along it. A single walker
(`--backward` off) has to walk the whole street. With `--backward` on, two
walkers start at opposite ends and race toward the middle — whichever reaches
the key first reports it.

> **Important — the expected-time math on a SINGLE GPU:** because the grid is
> fixed, the two fronts **split the threads** (`groupSize = jobs/2` each). A key
> at fraction `f` of the range is then found after
> `2·min(f, 1−f)·L/(T·B)` hashes instead of `f·L/(T·B)`:
>
> | key at fraction f | backward vs forward |
> |-------------------|---------------------|
> | `f < 0.5` (lower half) | **2× slower** — your key's front has half the threads |
> | `f = 0.662` | ~wash (measured: 5.36 s vs 5.54 s) |
> | `f > 0.667` | faster (at f=0.9, ~4.5× faster) |
>
> `E[min(f,1−f)] = 1/4`, so **expected time is identical** to forward: the
> two-front race is **expected-time-neutral on one GPU**. It only changes the
> *distribution* (capped worst case, but lower-half keys get 2× worse). The real
> payoff is on **multi-GPU**: two devices each run a full-speed front from one
> end, which halves expected time to `L/(4·T·B)`. On a single GPU, only use
> `--backward` if you believe the key sits in the upper third of the range.

Implementation: the grid doubles to `G = 2×sections` groups. Forward groups
upload `+J` (their base point advances upward); backward groups upload `-J`
(`(Jx, p−Jy)`) so the identical point-add marches the base point downward, and
each thread's start scalar is mirrored from the section end. Only the constant
memory `c_Jx`/`c_Jy`/`c_dir` per group differ — the inner kernel loop and the
batch key algebra are unchanged.

```
harakiri_sweep.exe --backward --batch 1024 --jobs 5120 --range 0x400000000000000000:0x7fffffffffffffffff --target 1PWo…
```

`--backward` composes with `--sections N` (`G = 2N` groups). It does **not**
change total throughput or wave scheduling; it only changes which keys each
frontier visits. On a single GPU the two fronts split the threads, so expected
time-to-first-key is unchanged (see the table above); the payoff is real only
with two GPUs sweeping from opposite ends. The selftest accepts
`--backward` and verifies the mirrored math still recovers a random scalar.

### Live progress (auto-timeboxing)

Without `--maxbatch`, a long range is swept in ~10-second timeboxed launches so
you get live feedback instead of a silent multi-minute kernel:

```
  [   12.6 s]  392.557 Mkeys/s | done 4.956e+09 / 1.100e+12 keys | ETA 46.5 min
  [   24.6 s]  403.991 Mkeys/s | done 9.935e+09 / 1.100e+12 keys | ETA 45.0 min
```

This costs only ~2–3% throughput (kernel relaunch + state persistence).
Pass `--maxbatch N` to disable timeboxing and run each launch as one big chunk.

### Warm-up (`--warmup N`)

Short ranges (2^30–2^40) finish in seconds — but a **cold GPU spends that entire
time at idle clocks**. Measured on the RTX 3050 6GB: the same 2^32 search ran at
**159 Mkeys/s cold** vs **~530 Mkeys/s warm** — a 3.3× wall-clock difference on
short runs (23.5 s → 7 s).

`--warmup N` runs N seconds of dense ALU burn-in before the sweep so boost
clocks kick in first:

```
harakiri_sweep.exe --batch 1024 --jobs 5120 --warmup 2 --range 100000000:1ffffffff --target 187swFM…
```

- Default is **off** (0 s) — long runs (puzzle 71, hours) don't need it; the GPU
  ramps during the first timeboxed launch anyway.
- Recommended for any range expected to finish in under ~1–2 minutes.
- Also adds `[warmup]` lines to stdout so you can see it working.

### Found-flag early stop

When a thread finds the key it claims the flag with `FOUND_LOCK`, writes the
result, then flips to `FOUND_READY`. The kernel used to only stop on
`FOUND_READY` — so every other warp kept hashing during that write window.
Now any nonzero flag stops the grid immediately (the host reads the result only
after `cudaDeviceSynchronize`, so this is race-free).

### Output to file (`--out`)

When a match is found, the result is written to the given file (in addition to
stdout):

```
Harakiri sweep result
=====================
range start : 0000000000000000000000000000000000000000000000400000000000000000
range end   : 00000000000000000000000000000000000000000000007fffffffffffffffff
target      : f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8
FOUND scalar: 0000000000000000000000000000000000000000000000xxxxxxxxxxxxxxxx
PRIVATE KEY : 0000000000000000000000000000000000000000000000xxxxxxxxxxxxxxxx
total keys  : ...
time        : ...
throughput  : ... Mkeys/s
```

Example:
```
harakiri_sweep.exe --batch 1024 --range 0x400000000000000000:0x7fffffffffffffffff --target 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU --out C:\puzzle71_found.txt
```

### Performance tuning

Measured on RTX 3050 6GB Laptop (20 SMs, auto-jobs = 5120):

| Config                       | Throughput          |
|------------------------------|---------------------|
| B=128,  jobs=5120 (auto)     | ~495 Mkeys/s        |
| B=512,  jobs=5120 (auto)     | ~498 Mkeys/s        |
| **B=1024, jobs=5120 (auto)** | **~500 Mkeys/s**    |
| B=1024, jobs=8192 (explicit) | ~582 Mkeys/s (peak, thermal-dependent) |

**Batch: 1024 is the cap and the optimum.** The kernel hard-caps
`MAX_BATCH_SIZE = 1024`; `--batch 2048/4096` are silently rejected (the kernel
returns with 0 keys). 1024 ≥ 512 ≥ 128 in every paired measurement.

**Jobs: auto (256 × SM) is the right default** — see §6. It is NOT about
throughput: on this card auto (5120) = 20 blocks = exactly one wave on 20 SMs,
while 8192 = 32 blocks = 1.6 waves, so the finder block's wave slot is a
lottery. 5120 wins on *expected time-to-find* because it never wastes a wave;
8192 only wins when the finder happens to land in wave 1 (measured: 77 s vs
59 s for the same key with a wave-1 finder at 6144 vs 8192 jobs — pure stride
effect, `t ∝ 1/jobs`).

Auto-jobs (`256 × SM count`) is the safe default; manual `--jobs` tuning
rarely gains more than a few percent and adds variance. `--threads` above 256
is rejected by the kernel's launch bounds. Sweep throughput is about 55% of
the pure-hash ceiling (~0.9 Gkeys/s) — the point-addition cost per key. On an
RTX 5090 expect roughly 10–20× this throughput (170 SMs vs 20), i.e.
multi-Gkeys/s; build native with `build.bat sm_120`.

---

## What happens when the key is found

The tool prints:

```
=== results ===
iterations : N
total keys : ...
time       : ...
throughput : ... Mkeys/s
FOUND scalar: 0000000000000000000000000000000000000000000000xxxxxxxxxxxxxxxx
PRIVATE KEY : 0000...xxxx (full 64-hex)
MATCH       : YES
```

The `FOUND scalar` (and `PRIVATE KEY`) is the private key in hex. Verify it
with any tool: `privkey → pubkey → hash160 → address` must equal
`1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU`.

---

## Files

| File                 | Purpose                                          |
|----------------------|--------------------------------------------------|
| `harakiri_sweep.cu`  | Host driver: CLI, Base58Check, 256-bit range math, self-test |
| `sweep_kernel.cu`    | CUDA kernel: bidirectional batch sweep, sections/backward group dispatch, FOUND protocol, MAX_BATCH_SIZE=1024 |
| `field.h`            | 256-bit field arithmetic (PTX, no `__uint128_t`)  |
| `hash.h`             | Fused register-resident SHA-256 + RIPEMD-160 (hash160) on device; hot path = register-reduced `hash160_33_from_x_red` |
| `build.bat`          | Build (CUDA 12.x, arch via CLI: `build.bat sm_86` / `sm_120` / auto) |
| `README.md`          | This document (usage + architecture)            |

## Verification quick-reference

```
:: sanity + regressions
harakiri_sweep.exe --selftest --batch 64   --range 0xdeadbeefcafef00d:500000
harakiri_sweep.exe --selftest --batch 512  --range 0xdeadbeefcafef00d:500000
harakiri_sweep.exe --selftest --batch 1024 --range 0xdeadbeefcafef00d:500000
:: multi-launch persistence
harakiri_sweep.exe --selftest --batch 1024 --maxbatch 2 --range 0xdeadbeefcafef00d:500000
:: full puzzle-71 range form (hex end, 256-bit length)
harakiri_sweep.exe --selftest --batch 1024 --range 0x400000000000000000:0x7fffffffffffffffff
:: puzzle-71 production (first 2^32 keys, bounded)
harakiri_sweep.exe --batch 1024 --jobs 5120 --range 0x400000000000000000:0x4000000000ffffffff --target 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU
```

## Single GPU Examples (for multi GPU use `--sections N` and `--backward` options where N is the number of GPU)  

> 33 bits key output:

```
./harakiri_sweep.exe --batch 1024 --jobs 10240 --range 100000000:1ffffffff --target 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu --maxbatch 1000000 --out found.txt

=== Harakiri Phase 2 sweep ===
range start : 0000000000000000000000000000000000000000000000000000000100000000
batch size  : 1024
selftest    : no
backward    : no
target      : 4e15e5189752d1eaf444dfd6bff399feb0443977  (from address)
GPU 0       : NVIDIA GeForce RTX 3050 6GB Laptop GPU (cc 8.6, 20 SMs)
config      : batch=1024 keys per inversion | threads/block=256 | jobs=10240 total threads | grid=40 blocks | sections=1 (groupSize=10240) | backward=no | max_batches=410 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000000066800

=== Harakiri Config Explained ===
- batch            keys hashed per one field-inversion batch (B)
- threads/block    block size (fixed 256, kernel launch bounds)
- jobs             total worker threads = threads/block x grid blocks
- grid             number of blocks = jobs / threads/block
- sections         range split into N parallel slices; groupSize = jobs/N
                   threads per slice (one slice per GPU on multi-GPU boxes)
- backward         two-front race per section: a forward half from the
                   section start + a backward half from the section end
- max_batches      batches per thread per kernel launch (auto-timeboxed to ~10s; add --maxbatch to override)
- max_chunk        keys each thread owns = range / jobs
[GPU 0][dbg] FOUND at launch 0: threadId=810 finder_batch=271

=== results ===
iterations : 0
total keys : 1664462400
sweep time : 2.855 s
wall time  : 2.866 s (incl. warmup/table-gen)
throughput : 583.013 Mkeys/s
ETA        : 4.5 s (range 4.295e+09 keys, done 1.664e+09)
FOUND scalar: 00000000000000000000000000000000000000000000000000000001a96ca8d8
PRIVATE KEY : 00000000000000000000000000000000000000000000000000000001a96ca8d8
target      : 4e15e5189752d1eaf444dfd6bff399feb0443977
MATCH       : YES
done
```

> 39 bits key output:

```
./harakiri_sweep.exe --batch 1024 --jobs 10240 --range 4000000000:7fffffffff --target 122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8 --maxbatch 100000 --out found.txt

=== Harakiri Phase 2 sweep ===
range start : 0000000000000000000000000000000000000000000000000000004000000000
batch size  : 1024
selftest    : no
backward    : no
target      : 0b304f2a79a027270276533fe1ed4eff30910876  (from address)
GPU 0       : NVIDIA GeForce RTX 3050 6GB Laptop GPU (cc 8.6, 20 SMs)
config      : batch=1024 keys per inversion | threads/block=256 | jobs=10240 total threads | grid=40 blocks | sections=1 (groupSize=10240) | backward=no | max_batches=26215 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000001999c00

=== Harakiri Config Explained ===
- batch            keys hashed per one field-inversion batch (B)
- threads/block    block size (fixed 256, kernel launch bounds)
- jobs             total worker threads = threads/block x grid blocks
- grid             number of blocks = jobs / threads/block
- sections         range split into N parallel slices; groupSize = jobs/N
                   threads per slice (one slice per GPU on multi-GPU boxes)
- backward         two-front race per section: a forward half from the
                   section start + a backward half from the section end
- max_batches      batches per thread per kernel launch (auto-timeboxed to ~10s; add --maxbatch to override)
- max_chunk        keys each thread owns = range / jobs
[GPU 0][dbg] FOUND at launch 0: threadId=4288 finder_batch=4658

=== results ===
iterations : 0
total keys : 28304004928
sweep time : 47.494 s
wall time  : 47.506 s (incl. warmup/table-gen)
throughput : 595.943 Mkeys/s
ETA        : 6.9 min (range 2.749e+11 keys, done 2.830e+10)
FOUND scalar: 0000000000000000000000000000000000000000000000000000004b5f8303e9
PRIVATE KEY : 0000000000000000000000000000000000000000000000000000004b5f8303e9
target      : 0b304f2a79a027270276533fe1ed4eff30910876
MATCH       : YES
done
```

> 39 bits key output no backward:

```
\harakiri_sweep.exe --batch 1024 --jobs 10240 --range 4000000000:4b9f8303e9 --target 122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8 --maxbatch 1000000 --warmup 2 2>&1 | Select-Object -Last 15
- max_chunk        keys each thread owns = range / jobs
[GPU 0][dbg] FOUND at launch 0: threadId=4288 finder_batch=4658

=== results ===
iterations : 0
total keys : 28283028256
sweep time : 47.505 s
wall time  : 51.122 s (incl. warmup/table-gen)
throughput : 595.367 Mkeys/s
ETA        : 36.3 s (range 4.992e+10 keys, done 2.828e+10)
FOUND scalar: 0000000000000000000000000000000000000000000000000000004b5f8303e9
PRIVATE KEY : 0000000000000000000000000000000000000000000000000000004b5f8303e9
target      : 0b304f2a79a027270276533fe1ed4eff30910876
MATCH       : YES
done
```

> 39 bits key output + backward:

```
.\harakiri_sweep.exe --batch 1024 --jobs 10240 --backward --range 4000000000:4b9f8303e9 --target 122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8 --maxbatch 1000000 --warmup 2 2>&1 | Select-Object -Last 15
- max_chunk        keys each thread owns = range / jobs
[GPU 0][dbg] FOUND at launch 0: threadId=9216 finder_batch=204

=== results ===
iterations : 0
total keys : 7776852064
sweep time : 13.049 s
wall time  : 16.623 s (incl. warmup/table-gen)
throughput : 595.985 Mkeys/s
ETA        : 1.2 min (range 4.992e+10 keys, done 7.777e+09)
FOUND scalar: 0000000000000000000000000000000000000000000000000000004b5f8303e9
PRIVATE KEY : 0000000000000000000000000000000000000000000000000000004b5f8303e9
target      : 0b304f2a79a027270276533fe1ed4eff30910876
MATCH       : YES
done
```

> 41 bits key output:

```
./harakiri_sweep.exe --batch 1024 --jobs 10240 --range 10000000000:1ffffffffff --target 1L5sU9qvJeuwQUdt4y1eiLmquFxKjtHr3E --maxbatch 1000000 --out found.txt

=== Harakiri Phase 2 sweep ===
range start : 0000000000000000000000000000000000000000000000000000010000000000
batch size  : 1024
selftest    : no
backward    : no
target      : d1562eb37357f9e6fc41cb2359f4d3eda4032329  (from address)
GPU 0       : NVIDIA GeForce RTX 3050 6GB Laptop GPU (cc 8.6, 20 SMs)
config      : batch=1024 keys per inversion | threads/block=256 | jobs=10240 total threads | grid=40 blocks | sections=1 (groupSize=10240) | backward=no | max_batches=104858 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000006666800

=== Harakiri Config Explained ===
- batch            keys hashed per one field-inversion batch (B)
- threads/block    block size (fixed 256, kernel launch bounds)
- jobs             total worker threads = threads/block x grid blocks
- grid             number of blocks = jobs / threads/block
- sections         range split into N parallel slices; groupSize = jobs/N
                   threads per slice (one slice per GPU on multi-GPU boxes)
- backward         two-front race per section: a forward half from the
                   section start + a backward half from the section end
- max_batches      batches per thread per kernel launch (auto-timeboxed to ~10s; add --maxbatch to override)
- max_chunk        keys each thread owns = range / jobs
[GPU 0][dbg] FOUND at launch 0: threadId=1715 finder_batch=34212

=== results ===
iterations : 0
total keys : 207541376448
sweep time : 416.588 s
wall time  : 416.600 s (incl. warmup/table-gen)
throughput : 498.194 Mkeys/s
ETA        : 29.8 min (range 1.100e+12 keys, done 2.075e+11)
FOUND scalar: 00000000000000000000000000000000000000000000000000000153869acc5b
PRIVATE KEY : 00000000000000000000000000000000000000000000000000000153869acc5b
target      : d1562eb37357f9e6fc41cb2359f4d3eda4032329
MATCH       : YES
result written to found.txt
done
```

# Harakiri — CUDA Architecture & Keyspace Splitting

This document explains how the Harakiri sweep engine maps onto the CUDA
execution model, and exactly how the keyspace is divided across the GPU.

---

## 1. The CUDA execution model

A GPU runs one **kernel** = one function executed by many **threads** in
parallel. Threads are organized in a strict hierarchy:

```
KERNEL  (one launch = one <<<...>>> call)
 └── GRID  (all blocks for this launch)
      └── BLOCK  (a group of threads that can share memory/sync)
           └── THREAD  (one worker, runs the kernel body)
                └── WARP (32 threads, executed together on one SM core)
```

Key hardware facts:

- An RTX 3050 6GB Laptop has **20 SMs** (streaming multiprocessors).
- Each SM can hold several blocks at once (up to 2048 resident threads/SM).
- The GPU scheduler hands blocks to SMs as they free up (occupancy).
- Warps (32 threads) are the real scheduling unit: one instruction is issued
  to an entire warp at a time.

The launch in Harakiri:

```cpp
kernel_point_add_and_check_oneinv<<<grid, threads_per_block>>>(...);
//                                  ^^^^    ^^^^^^^^^^^^^^^^
//                                  GRID        BLOCK
```

---

## 2. The four tunable terms, precisely defined

### 2.1 `--threads` → threads per block (`threads_per_block`, default 256)

How many threads execute together in one block. We use **256** because the
kernel carries the compile-time contract:

```cpp
__launch_bounds__(256, 2)
```

This tells the compiler: "compile for 256 threads/block, and target 2 blocks
resident per SM." It lets the compiler safely use up to 128 registers per
thread without blowing the SM register file.

### 2.2 `--jobs` → total worker threads (`threadsTotal`)

How many independent workers exist across the whole grid. The grid size is
derived on the host:

```cpp
int grid = (int)((threadsTotal + threads_per_block - 1) / threads_per_block);
```

So `--jobs 8192 --threads 256` → `grid = 32 blocks`, each 256 threads, total
8192 threads.

| `--jobs` | blocks | threads/block | total threads | vs. 20 SMs |
|----------|--------|---------------|---------------|------------|
| 5120 (auto) | 20  | 256 | 5120 | 1 block/SM wave |
| 8192        | 32  | 256 | 8192 | ~1.6 waves      |

Without `--jobs`, the host auto-detects the GPU and defaults to
`256 × SM count` (e.g., 5120 on this card, ~43k on an RTX 5090).

### 2.3 `--batch B` → points per inversion batch

This is the **batch-inversion** size — the heart of the design. Every thread
sweeps keys **B at a time**. Within one batch of B=512 keys, the thread:

1. Builds a suffix-product tree over `(Gx[i] - x1)` — the x-differences
   between the thread's base point and the precomputed table points `(i+1)·G`.
2. Computes **ONE** modular inverse (the expensive 256-bit `_ModInv`) for the
   whole batch.
3. Derives all B per-key inverses from that single one via cheap
   multiplications (`dx_inv_i = subp[i] · inverse`).

Why: a field inversion is ~10–50× more expensive than a multiply. Batching
means each key effectively pays `1/512` of an inversion, so the per-key cost
is ~1 point-add + 1 hash instead of 1 inversion + 1 hash.

A batch of size B = 2·half covers B consecutive keys centered on S:

```
batch at base S covers:   [S - half, S + half - 1]
                           ├─ + branch: S + (i+1)·G      (half - 1 keys)
                           ├─ - branch: S - (i+1)·G      (half - 1 keys)
                           ├─ base:     S
                           └─ last:     S - half·G
```

This is the **bidirectional** sweep: both `+(i+1)·G` and `-(i+1)·G` are
computed from the same batch inversion.

### 2.4 `--maxbatch` → batches per launch

How many batches ONE kernel launch may process before returning (0 =
unlimited). With auto-timeboxing it is sized so each launch runs ~10 seconds,
and the host prints live progress (Mkeys/s, ETA) after every launch.

Between launches, thread state (base point, scalar, remaining count) is kept
on device — **multi-launch persistence** — so no keys are skipped and no
progress is lost.

---

## 3. How the keyspace is split (round-robin / linear march)

Keys are linear indices:

```
range:   [start, start + len)
key at index k  =  start + k        (k = 0, 1, 2, ...)
```

**Batch `b`** covers keys `[start + b·B, start + (b+1)·B)` — B consecutive
keys. The grid must cover all batches `b = 0, 1, 2, …, len/B`.

### 3.1 The assignment (round-robin stride)

```
thread g handles batches:   g,  g + T,  g + 2T,  g + 3T, ...
                            where T = threadsTotal (--jobs)
```

Example with `--jobs 8192 --batch 512`:

```
thread 0     → batch 0,   batch 8192,   batch 16384, ...
               keys [start, start+512), [start+8192·512, ...) ...
thread 1     → batch 1,   batch 8193,   batch 16385, ...
thread 2     → batch 2,   batch 8194,   batch 16386, ...
...
thread 8191  → batch 8191, batch 16383, batch 24575, ...
```

- **Round 0**: all 8192 threads together cover batches 0..8191 = keys
  `[start, start + 8192·512)` — a contiguous 4.19M-key frontier.
- **Round 1**: batches 8192..16383 — the next 4.19M keys. And so on.

So the grid **marches through the range in linear order**. A key at position
48.8B is found after ~48.8B hashes, not after `T × (offset within a slab)`
the way a contiguous per-thread chunking would behave.

### 3.2 The per-thread loop

```cpp
// thread g:
S   = range_start + g·B + half        // first batch starts at its own offset
x1  = S·G                             // base point (scalarMulBaseKernel)
rem = chunk_len                       // keys left for this thread

while (batches_done < max_batches && rem >= B):
    process batch at S:               // [S - half, S + half) = B keys
    S   += T·B                        // jump to next assigned batch (stride)
    x1  += (T·B)·G                    // advance base point by the stride point
    rem -= B
```

The `(T·B)·G` stride point is uploaded to the kernel constants as
`c_Jx/c_Jy` (precomputed on the host via `scalar_mul_g`). It serves double
duty:

1. The suffix-product tree's dummy point (it cancels algebraically, so its
   value is irrelevant to correctness of the batch inversion).
2. The per-batch base-point advance (which is why it must equal `(T·B)·G`).

### 3.3 Sections (`--sections N`)

`--sections N` splits `[start, start+len)` into N contiguous slices and sweeps
them in parallel:

- **Single GPU**: the grid is split into N groups of `groupSize = jobs/N`
  threads each. Group `g` owns section `g`; its threads stride by
  `groupSize·B` (not `T·B`) inside its slice, and its stride point is
  `c_Jx[g] = (groupSize·B)·G`. The group index is `gid / groupSize`; the
  local index is `gid % groupSize`. All groups share the grid's found flag.
- **Multi GPU**: the sections are spread round-robin across devices
  (`min(sections, GPUs)` devices are used, one host thread per device, each
  with its own grid, buffers and found flag). A **host-mapped stop flag**
  (`cudaHostAlloc` + `cudaHostGetDevicePointer`) is passed to every kernel;
  the moment any device finds the key its host sets the flag and every other
  device's kernel halts (polled once per 64 batches to avoid PCIe reads on the
  hot path — the device-local found flag is still checked every batch).

```
--sections 1 (default): one front, T = jobs,   stride = T·B,   c_Jx[0] = (T·B)·G
--sections 2 on 1 GPU : two fronts, G = jobs/2, stride = G·B,   c_Jx[0..1] = (G·B)·G
--sections 2 on 2 GPUs: one front per device, each with its own T, flag, buffers
```

Per device the per-thread loop is unchanged except `T → groupSize` and the
start scalar is `section_start[group] + local·B + half` instead of
`range_start + g·B + half`. Everything else (batch inversion, prefix compare,
multi-launch persistence, found flag) is identical.

### 3.4 Two-front race (`--backward`) — the group dispatch

`--backward` doubles the grid: `G = 2 × sections`. Each section `s` becomes
**two groups** that share the section's keys but march from opposite ends:

| Group | Owns | Direction | Start scalar (thread `local`) | Stride point |
|-------|------|-----------|-------------------------------|--------------|
| `g = 2s`   | lower half `[A, A+len/2)`   | forward (`dir=+1`)  | `A + local·B + half` | `+J = (Jx, Jy)` |
| `g = 2s+1` | upper half `[A+len/2, A+len)`| backward (`dir=-1`) | `A+len − local·B − half` | `−J = (Jx, p−Jy)` |

where `J = (groupSize·B)·G`. Both groups use the identical inner batch loop and
key algebra; only two things differ per group, both carried in constant memory:

- **`c_dir[g]`** — `+1` or `-1`. The kernel hoists it once per thread and the
  scalar advance becomes `S += dir·(groupSize·B)` each batch.
- **`c_Jy[g]`** — forward groups upload `+Jy`, backward groups upload
  `p − Jy` (the negated y-coordinate), so the same point-add walks the base
  point downward for backward groups.

The forward group covers `[A, A+half)`, the backward group covers
`[half, A+len)` marching down from `A+len` — together the whole section, no
gaps, with the two fronts racing to the middle.

```
--backward --sections 1 : 2 groups  (lower forward + upper backward)
--backward --sections N : 2N groups (N forward halves + N backward halves)
```

The per-device machinery (round-robin sections, host-mapped stop flag,
multi-launch persistence) is unchanged; backward just changes the *group count*
and which keys each group's front visits. Because the two fronts share the same
silicon, total throughput and wave scheduling are unchanged — and because each
front gets half the threads, the expected time-to-find on a single GPU is
**unchanged** (see the table in "Two-front race (`--backward`)" above).

---

## 4. The full pipeline per key

```
load base point x1 (S·G), scalar S
┌────────────────────────────────────────────────────────────┐
│ for each batch (B keys):                                   │
│   1. suffix-product tree over (Gx[i] - x1)                 │
│   2. ONE _ModInv on the total product                      │
│   3. for i in 0..half-1:                                   │
│        dx_inv = subp[i] · inverse       ← per-key inversion│
│        P_plus  = S + (i+1)·G → hash160  ← + branch         │
│        P_minus = S - (i+1)·G → hash160  ← - branch         │
│   4. compare hash prefix against target 32-bit prefix      │
│        (warp-wide __any_sync abort on prefix match)        │
│   5. advance base by stride point (T·B)·G                  │
└────────────────────────────────────────────────────────────┘
```

Every `FLUSH_THRESHOLD` (65536) hashes, the warp reduces its local counter
and `atomicAdd`s it to `d_hashes`, so the host can report live throughput and
ETA.

---

## 5. Tunables and why the defaults are what they are

| Flag | CUDA concept | Meaning | Typical value |
|------|--------------|---------|---------------|
| `--threads` | blockDim.x | threads per block | 256 |
| `--jobs`    | threadsTotal | total threads (grid = jobs/threads) | 5120–8192 |
| `--batch`   | inversion batch | keys per batch sharing one inverse | 512 |
| `--maxbatch`| — | batches per kernel launch (persistence) | auto ~10s, or huge for one shot |
| grid        | gridDim | `ceil(jobs / 256)` blocks | 20–32 |

- **B=1024** is the optimum (and the hard cap, `MAX_BATCH_SIZE = 1024`):
  larger B amortizes the inversion further, but the suffix-product tree costs
  `O(B)` sequential multiplications per batch.
- **jobs = 256/SM**: full occupancy without oversubscription (see §6 for why
  exceeding it is a lottery, not a speedup).
- **Round-robin stride**: total-hashes-until-find ≈ key position — the same
  property that makes KeyHunt fast, but with a cheaper per-key cost
  (1 point-add + 1 hash vs. 1 inversion + 1 hash per key).

---

## 6. Wave scheduling & the stride lottery (choosing `--jobs` and `--batch`)

### 6.1 Wave scheduling on 20 SMs

The launch uses `__launch_bounds__(256, 2)` — the compiler budgets **2 blocks
resident per SM** (that is why it can use up to 128 regs/thread). On the RTX
3050's 20 SMs the resident capacity is **40 blocks**. The scheduler hands
blocks out in **waves**: a grid of 20 blocks fills all 20 SMs in wave 1; a
grid of 32 blocks runs 20 in wave 1 and the remaining 12 in wave 2 (behind
the first wave, sharing the same silicon).

| `--jobs` | blocks | waves | time-to-find (measured, same key) |
|----------|--------|-------|-----------------------------------|
| 5120 (auto) | 20 | **1 wave** | baseline |
| 6144 | 24 | 2 waves | 77 s |
| 8192 | 32 | 2 waves | 59 s (finder in wave 1) |

### 6.2 The stride effect: `t ∝ 1/jobs`

Because the march is round-robin with stride `jobs × B`, the grid covers
`jobs × B` keys per round. A key found by a wave-1 thread is reached after
≈ `position / (jobs × B)` rounds — **more jobs = fewer rounds = faster find**,
*provided the finder is in wave 1*. Measured on the same big-range key with the
finder in block 0 (wave 1) both times: 8192 jobs → 59.2 s vs 6144 jobs →
77.3 s. Ratio 77.3/59.2 ≈ 1.31 ≈ 8192/6144. ✓

### 6.3 The lottery

If the finder lands in wave 2, it does a full extra wave of wasted work before
it ever starts hashing — a fixed penalty of ~one grid pass. On short ranges
(seconds) that penalty dominates: 5120 jobs (single wave) never pays it. That
is why **5120 (= 256 × 20 SMs) is the expected-time optimum on this card**,
even though peak throughput (Mkeys/s) is higher at 8192 jobs. More jobs is a
gamble on wave-1 placement; auto-jobs avoids the gamble entirely. (Same story
for `--sections N` on one GPU — all sections share the same waves, so no
speedup; see above.)

### 6.4 Batch: 1024 is the cap and the optimum

`MAX_BATCH_SIZE = 1024` (sweep_kernel.cu:27). Any `--batch` above it makes the
kernel return with **0 keys** (silent, so it just looks like an instant
"exhausted" run). 1024 is also the measured throughput optimum on this GPU
(≥512 ≥128 in every paired run). Use `--batch 1024` for production sweeps.

### 6.5 Benchmarking `--jobs` / `--backward` with `--selftest`

To pick `--jobs` (or test `--backward`) against a real sweep, you want a key at
a *known* position so you can measure time-to-find. The built-in `--selftest`
picks a deterministic scalar inside the range, hashes it to a target, and
sweeps until it is found — no external helper needed:

```
harakiri_sweep.exe --selftest --batch 1024 --range 100000000:1ffffffff
harakiri_sweep.exe --selftest --backward --batch 1024 --range 100000000:1ffffffff
```

The selftest scalar is derived from `range_start` (deterministic per range), so
the same `--range` yields the same key position across runs. Run the same
selftest at a few `--jobs` values and compare wall-clock time-to-find — the
wave-lottery pattern from §6.2/§6.3 reproduces every time. And `--backward`
composes with `--selftest`, so you can A/B the two-front race on the same
range/key: on a single GPU expect it to *lose* for lower-half keys, wash near
f=2/3, and win only for keys in the upper third (see the table in "Two-front
race (`--backward`)").

If you need a *specific* key at an exact position (e.g. 25%, 50%, 75%), the
address→hash160 path is built into `--target`, so you can convert any known
private key's address with any standard tool and hand it straight to the
sweeper.

---

# The Bidirectional Sweep, Step by Step

> **TL;DR — "bidirectional" is used in two distinct senses, don't conflate them:**
>
> 1. **Per-batch algebra (always on):** inside a single batch, keys are produced
>    on **both sides of the batch center** (`S-1, S-2, …` and `S+1, S+2, …`)
>    from the **same field inversion**. This is the "bidirectional batch sweep"
>    — one inversion buys 2× the key coverage. It has nothing to do with search
>    direction.
> 2. **Search direction (only with `--backward`):** with `--backward`, the march
>    itself splits into a forward front (from the section start) and a backward
>    front (from the section end), racing toward the middle. Without `--backward`
>    the march is **strictly one-way** — every thread moves forward from
>    `range_start` toward `range_end` in order (see "Round-robin linear march").
>
> The rest of this section explains sense **1** (the always-on per-batch ±
> algebra). Sense **2** is covered in "Two-front race (`--backward`)" above.

This section walks through exactly what "bidirectional" means in Harakiri and
why it costs almost nothing extra.

## The problem being solved

A thread owns a base scalar `S` and its public key `P = S·G`. To sweep the
range it must visit every key `S + d` for `d = -half … +half-1` and hash the
x-coordinate of `(S+d)·G`.

Naive approach: for each `d`, compute `(S+d)·G` from scratch. That needs a
full scalar multiplication per key — far too slow.

Cheaper idea: `(S+d)·G = S·G + d·G`. If we precompute the small table
`d·G` for `d = 1..half` once (constant memory `c_Gx/c_Gy`), then each key is
just **one point addition** `P + (d·G)`. Point addition needs an inversion of
the x-difference — but that's exactly what the batch trick removes (see
§2.3 above).

## Why two branches from one inversion

Point addition `P + Q` requires `1/(x_Q - x_P)` — one field inversion per
addition. Inversion is ~10–50× costlier than a multiply, so doing one per key
kills the budget.

Harakiri instead computes, for one base point `x1`:

```
prod = (Gx[0]-x1) · (Gx[1]-x1) · … · (Gx[half-1]-x1) · (Jx-x1)
inv  = 1 / prod                       (ONE inversion for the whole batch)
```

Then for each `i`, the per-key inverse is recovered with a single multiply:

```
dx_inv_i = subp[i] · inv              (subp = suffix products, precomputed)
          = 1 / (Gx[i] - x1)
```

So **one `_ModInv` serves all B keys** of the batch. That is the
"asynchronous split-phase" in the engine description: split the batch
(products), invert once, then fan the inverse back out.

## The ± branches in detail

For a batch centered at base `S`, the kernel emits B hashes:

```
d = -(half)      →  P - half·G        (last point, - branch only)
d = -(i+1)       →  P - (i+1)·G       (- branch, i = 0..half-2)
d = 0            →  P = S·G           (base point check)
d = +(i+1)       →  P + (i+1)·G       (+ branch, i = 0..half-2)
```

Consecutive keys covered: `[S-half, S+half-1]` — exactly B keys, no gaps, no
overlap within the batch, and the next batch (after the stride jump) continues
the march.

Both `+` and `-` branches reuse the **same** `dx_inv_i`:

```
+ branch:  λ = (Gy[i] - y1) · dx_inv_i          → P + (i+1)·G
- branch:  λ = (-Gy[i] - y1) · dx_inv_i         → P - (i+1)·G
```

The only extra work for the second branch is one `ModNeg` of `Gy[i]` — a
negation, not an inversion. So bidirectional coverage costs almost nothing
on top of unidirectional: **the same single inversion buys you 2× the key
coverage per batch.**

## Concrete numeric example (puzzle 40, found key)

Run: `--batch 512 --jobs 8192 --range 4000000000:7fffffffff --target 122AJ…`

```
range start : 0x4000000000            (2^38)
found key   : 0x4b5f8303e9            = range_start + 0xb5f8303e9
position    : 0xb5f8303e9 = 48,802,830,825 keys into the range
```

- The key is in batch `b = 48,802,830,825 / 512 = 95,318,028.96… → 95,318,028`.
- Round-robin assigns that batch to thread `g = b mod 8192 = 4108`, at round
  `k = b / 8192 = 11,635`.
- When thread 4108 reaches its batch, base `S` satisfies
  `key = S ± (i+1)` for the exact `i` that matches the target's hash160.
- Total hashes before FOUND ≈ key position ≈ 48.8e9, not
  `8192 × (offset within a slab)` — the linear-march property.

---

# Hash Functions Used

Every candidate key is compressed to a 33-byte public key
(`0x02/0x03 || x`), then hashed exactly the way Bitcoin addresses are built.

## Pipeline

```
private key d  →  public key P = d·G   (secp256k1 point)
P → 33-byte compressed encoding        (prefix 0x02/0x03 + 32-byte x)
       │
       ▼
   SHA-256(33B)      → 32 bytes
       │
       ▼
   RIPEMD-160(32B)   → 20 bytes  =  HASH160
       │
       ▼
  compare 20 bytes vs target (after a 32-bit prefix pre-filter)
```

## The functions (all implemented in `hash.h`, device-side)

| Function | Input → Output | Used for |
|----------|----------------|----------|
| `sha256_33bytes` | 33-byte pubkey → 32 B | SHA-256 stage, specialized single-block |
| `ripemd160_32bytes` | 32 B → 20 B | RIPEMD-160 stage |
| `hash160_33_from_x` | x-limb + parity → 20 B | fused pipeline, candidate → target (byte-array reference version) |
| `hash160_33_from_x_red` | x-limb + parity → 20 B | **hot-path fused pipeline** — register-reduced: message words built directly from the x limbs, no byte arrays |
| `hash160_prefix4` | x-limb + parity → 4 B | 32-bit prefix pre-filter |
| double-SHA256 | `sha256(sha256(x))` | Base58Check checksum (4 bytes) |
| Base58Check decode | address → 21 B (version+hash160) | `--target` address input |

## The fused-register design (why it's fast)

The headline feature of the hash path: **the whole HASH160 is register-resident
and fused — SHA-256's output feeds RIPEMD-160's input directly through
registers, never through memory.** (`hash.h` header comment:
*"register-resident fused SHA-256 + RIPEMD-160"*.)

Three techniques work together:

### 1. `__forceinline__` everywhere — zero call overhead

Every stage (`sha256_33bytes`, `ripemd160_32bytes`, `hash160_33_from_x`,
`hash160_33_from_x_red`, `hash160_prefix4`) is `__forceinline__`. The compiler
inlines the entire pipeline into the sweep kernel's hot loop as one
straight-line register program — no function prologue/epilogue, no spills for
arguments, no `call/ret`.

### 2. Specialized single-block SHA-256 — no generic machinery

`sha256_33bytes` is hand-tuned for exactly one 33-byte message (compressed
pubkey). There is no generic message-length loop, no padding calculation. The
register-reduced variant goes one step further and **never materializes the
message bytes at all** — the big-endian message words W[0..7] are built
directly from the x limbs with a few shifts/ORs:

```c
// msg = prefix(1) || xr3(8 BE) || xr2(8 BE) || xr1(8 BE) || xr0(8 BE)
W[0] = ((uint32_t)prefix << 24) | (uint32_t)((xr3 >> 40) & 0xFFFFFFULL);
W[1] = (uint32_t)(xr3 >> 8);
...
W[8] = ((uint32_t)(xr0 & 0xFFULL) << 24) | 0x00800000U;   // last x byte + 0x80 pad
W[15] = 0x00000108U;     // bit length 33*8 = 264, precomputed constant
```

W[8..15] are **compile-time constants** — the message schedule's tail is
free. Only W[0..7] come from the actual x-coordinate. The 64-round loop is
fully `#pragma unroll`ed, so the message schedule `W[64]` lives in **registers
the whole time** (a generic SHA-256 that handles arbitrary lengths would need
runtime padding branches and memory traffic).

### 3. Fused stage-to-stage — SHA-256 out → RIPEMD-160 in, in registers

In `hash160_33_from_x_red` there is no intermediate byte array either way:

```c
// SHA-256: h0..h7 digest built from the W[] words above
// RIPEMD-160 message words = byte-swap of the SHA digest words:
X[i] = ((v & 0x000000FFU) << 24) | ((v & 0x0000FF00U) << 8) |
       ((v & 0x00FF0000U) >> 8)  | ((v & 0xFF000000U) >> 24);   // v = h0..h7
```

The 32-byte SHA-256 result is byte-swapped and consumed by RIPEMD-160 in the
same register stream. A naive implementation would write the SHA-256 digest
to global memory (or even to a local `sha_bytes[]` array), then load it back
for RIPEMD-160 — memory round-trips and byte-shuffle instructions per
candidate key. At 0.5–0.9 Gkeys/s that's a huge tax. Here it's pure
register-to-register flow, and the byte arrays themselves don't exist.

### Register-pressure budget

The fused hash plus the field math compiles to a kernel with:
- **128 registers/thread** (enabled by `__launch_bounds__(256, 2)`)
- a **~16.7 KB stack frame** (the batch `subp[]` suffix-product array lives
  in local memory — unavoidable, it's B/2 × 32 bytes of live state)
- minimal actual register spill — the register-reduced hash keeps the hot
  path almost entirely in registers (the byte-array reference version needed
  a `sha_bytes[]` buffer and its shuffling on top).

## Verification

- SHA-256 device kernel validated against the official NIST FIPS 180-4 test
  vectors and `hashlib` on the host.
- RIPEMD-160 validated against known vectors.
- Hash160 of the puzzle-71 address confirmed:
  `1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU` → `f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8`
- **Register-reduced variant validated on real Bitcoin puzzle keys** — full
  sweep runs recovered the known private keys and reported `MATCH: YES`:
  - puzzle 33: `0x1a96ca8d8` → `187swFMjz1G54ycVU56B7jZFHFTNVQFDiu`
    (`4e15e5189752d1eaf444dfd6bff399feb0443977`)
  - puzzle 39: `0x4b5f8303e9` → `122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8`
    (`0b304f2a79a027270276533fe1ed4eff30910876`)
  so `hash160_33_from_x_red` is bit-exact against Bitcoin's HASH160 on
  real-world data (no false positives in either sweep).
- Base58Check decode validates the 4-byte checksum over the 21-byte
  `version || hash160` payload (a classic bug source — fixed here).

## Prefix pre-filter (why it's fast)

Hashing is the hot loop, so the full 20-byte compare is avoided:

1. Compare only `h[0]` (first 4 bytes = 32-bit prefix) against the target.
2. `__any_sync` warp reduction: if **no** lane matches the prefix, skip the
   full compare entirely.
3. Only on a prefix hit does the thread do the 20-byte compare and (if full
   match) claim the FOUND flag via `atomicCAS`.

Random keys match a 32-bit prefix with probability 2^-32 — so the expensive
20-byte path runs essentially never until the real key is hit.

---

# Example Flow: From Range to Found Key

This is the whole pipeline, bottom-up, on a real run so you can trace every
number. It answers three questions:

1. **How are keys built and spread across threads?** (keyspace → batches → threads)
2. **What does one thread do per batch inside the sweep kernel?** (ECDSA math + hashing)
3. **Where are the bottlenecks and what have we already done about them?**

The trace below uses a plain **forward** sweep (`--backward` off, `--batch 512`)
so the round-robin math is the clearest. With `--backward` the per-thread batch
algebra is identical — only the start scalar and the march direction flip for
the backward half (see "Two-front race" above).

All examples use this actual run:

```
harakiri_sweep.exe --batch 512 --jobs 8192 --warmup 2 \
  --range 100000000:1ffffffff --target 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu
```

```
range start : 0x100000000            (2^32)
range size  : 0x100000000 = 4,294,967,296 keys   (2^32 keys, "33-bit puzzle")
found key   : 0x1a96ca8d8            = range_start + 0xa96ca8d8
position    : 2,842,470,616          = 66.2 % through the range
```

---

## 1. How the keyspace is built and spread

### 1.1 Three levels: range → chunk → batch

| Level | Size (this run) | What it is |
|-------|-----------------|------------|
| **range** | 2^32 keys | the whole search space |
| **chunk** | 524,288 keys (`0x80000`) | one thread's share = range / T |
| **batch** | 512 keys (`B`) | the unit of work: one inversion serves B keys |

8192 threads (`--jobs`), 512 keys per batch:

```
chunk = range / T  =  4,294,967,296 / 8192  =  524,288 keys per thread
max_batches/thread = chunk / B              =  524,288 / 512 = 1024
```

### 1.2 Round-robin assignment (the "linear march")

Thread `g` (g = 0..8191) does not take batch `g`, then `g+1`, … no — that
would give every thread an equal *count* but leave threads idle while others
finish their chunk. Instead, thread `g` takes **strided** batches:

```
thread g processes batches:  g, g+T, g+2T, g+3T, …   (T = 8192)
```

- Round `k`: thread `g` processes batch number `b = g + k·T`.
- **Stride** = T·B = 8192 × 512 = 4,194,304 keys per round — the whole grid
  advances 4,194,304 keys between "rounds".
- Every thread is always busy, and the grid as a whole sweeps the range
  **in order** — the key at position P is found after ≈ P total hashes, no
  matter which thread actually hits it.

### 1.3 Trace the winning key

```
position      = 2,842,470,616
batch index b = 2,842,470,616 / 512      = 5,551,700
thread  g     = 5,551,700 mod 8192       = 5,716
round   k     = 5,551,700 / 8192         = 677
offset in batch = 2,842,470,616 - 5,551,700·512 = 216   (key #216 of 512)
```

Check: `b = g + k·T = 5,716 + 677 × 8192 = 5,551,700` ✓

So **thread 5,716, at round 677, found the key as the 216th key of its batch.**

---

## 2. Kernel launch: grid, blocks, threads

The sweep kernel is launched with a **1-D grid**:

```
blockDim  = 256 threads/block   (fixed by __launch_bounds__(256, 2))
gridDim   = T / 256 = 8192 / 256 = 32 blocks
```

| CUDA concept | This run | Role |
|--------------|----------|------|
| grid | 32 blocks | all 8192 threads march the range together |
| block | 256 threads | max occupancy on the GPU (2 blocks/SM × 20 SMs = 40 blocks resident) |
| thread | 8192 workers | each owns the strided batch sequence from §1.2 |
| batch | 512 keys | one inversion → 512 candidate public keys |
| round | 1024/thread | every thread marches ~1024 batches of its chunk |

The kernel is one launch **per round-set of batches** — actually it stays
resident and loops internally per thread, persisting its marching state in
registers/local memory between batches (see the state diagram below). With
auto-timeboxing the host relaunches it every ~10 s (for live progress), and
each thread's position is saved/restored so the march is seamless.

---

## 3. What a thread does per batch (the sweep kernel, step by step)

A thread's per-batch work splits into the **ECDSA math half** and the
**hashing half**. Using B = 512, one thread processes 512 keys per batch:

### Step 0 — persistent state (registers)

Each thread carries its marching position as a **base point**:
`S·G` = (x1, y1), plus a precomputed jump `(T·B)·G` to advance to the next
round. It *never* recomputes scalars — state survives across batches via
`__launch_bounds__`-enabled register residency (the stack frame is
local-memory, see Register-pressure budget above).

### Step 1 — suffix products (ECDSA math)

For the batch centered at S, we need `1/(Gx[i] - x1)` for every i. Instead of
512 inversions, compute the **suffix product array** once:

```
prod   = (Gx[0]-x1)·(Gx[1]-x1)·…·(Gx[255]-x1)·(Jx-x1)
subp[i]= (Gx[i+1]-x1)·…·(Gx[255]-x1)·(Jx-x1)     ← B/2+1 entries
```

### Step 2 — one batch inversion (ECDSA math)

```
inv = 1 / prod          ← ONE Montgomery exponentiation (~255 squarings)
```

This is the single most expensive *single* operation in the batch — but it
replaces 512 of them.

### Step 3 — fan out the inverse (ECDSA math)

For each i, recover the per-key inverse with one multiply:

```
dx_inv_i = subp[i] · inv = 1 / (Gx[i] - x1)
```

### Step 4 — bidirectional point additions (± branches)

Each i produces **two** candidate keys (that's the bidirectional trick):

```
+ branch:  λ = (Gy[i] - y1) · dx_inv_i        → P + (i+1)·G
- branch:  λ = (-Gy[i] - y1) · dx_inv_i       → P - (i+1)·G
```

Each is a standard affine point addition: one field multiply for λ, one for
x, one for y. The second branch reuses the *same* dx_inv_i (only a `ModNeg`
of Gy[i] is extra), so **one inversion buys 512 × 2 key checks**.

The batch covers keys `[S-256, S+255]` — 512 consecutive keys, no gaps.

### Step 5 — fused HASH160 (hashing half)

For each of the two branches, the point's x-coordinate + parity → compressed
33-byte pubkey → SHA-256 → RIPEMD-160, all in registers (see
"Hash Functions Used" above). The hot path is the register-reduced
`hash160_33_from_x_red` — the SHA-256 message words are built directly from
the x limbs and the RIPEMD-160 words are byte-swaps of the SHA digest, so no
byte arrays exist in the loop:

```
x + parity → hash160_33_from_x_red → 20-byte h160   (register-resident)
```

### Step 6 — prefix pre-filter + compare (hashing half)

1. Compare only the first 4 bytes of h160 against the target's prefix.
2. `__any_sync` warp reduction — if no lane matches, skip step 3 for the
   whole warp (probability 2^-32 of a hit, so this is almost always skipped).
3. On a hit: full 20-byte compare, and on true match claim the result via
   `atomicCAS` on the FOUND flag.

### Step 7 — advance

`S·G += (T·B)·G` (one point addition with a precomputed constant) and repeat
from Step 1 for the next round's batch.

---

## 4. Where the time goes — bottlenecks and what we fixed

### 4.1 The per-key cost budget (rough, per 2-key branch pair)

| Operation | per batch (512 keys) | per key | bottleneck? |
|-----------|----------------------|---------|-------------|
| suffix products | 256 mul | 0.5 | no |
| batch inversion | 1 pow (≈255 sq + mul) | ~1 mul-equiv | **amortized to ~nothing** |
| point additions | 512 λ/x/y sequences | ~6 field mul | **yes — the ECDSA math half** |
| fused HASH160 | 512 × (64+80 rounds) | ~144 rounds | **yes — the hashing half** |

The two real hotspots are **point addition** (field arithmetic, all in PTX
`mad.hi.cc.u64` fused multiply-add) and **HASH160** (144 hash rounds). Both
are pure ALU — no memory traffic in the hot loop, which is exactly how we
get ~530 Mkeys/s on a laptop 3050.

### 4.2 What we've already done (search-side engineering)

| Problem | Fix | Measured effect |
|---------|-----|-----------------|
| 1 inversion/key was fatal | batch inversion: 1 per 512 keys | the key enablement of the whole design |
| 1 point add per key | bidirectional ± branches reuse dx_inv_i | 2× coverage per batch for ~1 `ModNeg` extra |
| non-linear traversal wasted work | round-robin linear march | key at 66% found after ≈66% of hashes, not random |
| hashing in memory | fused register-resident SHA-256+RIPEMD-160 | no memory round-trips at 0.5 Gkeys/s |
| byte-array hashing | register-reduced hash `hash160_33_from_x_red` (message words built from x limbs, no `msg[]`/`sha_bytes[]`) | fewer live registers + zero byte shuffles in the hot loop; measured **~585–590 Mk/s** on the RTX 3050 6GB (33/39-bit real-key runs) |
| full 20-byte compare every key | 32-bit prefix pre-filter + warp reduction | compare cost ≈ 0 until the real key |
| cold GPU on short runs | `--warmup N` burn-in | 159 → 532 Mkeys/s (23.5 s → 7 s on 2^32) |
| threads kept hashing during found-write | stop on FOUND_LOCK, not just FOUND_READY | closes the finder's write window |

---

# The Four Searches

Harakiri has **four** ordered levels of "search strategy". They compose: each
one builds on the one below it. From smallest to largest:

```
1. Bidi batch        — always on, per-inversion algebra
2. Round-robin march — always on, how the grid sweeps a range
3. Sections          — --sections N, split the range into chunks (also per GPU)
4. Backward          — --backward, two-front race (split the range in half)
```

## 1. Bidi batch (always on)

Inside **one inversion batch** of B = 2·half keys, the thread checks the base
scalar S plus keys on both sides of it, all from a **single field inversion**
(`sweep_kernel.cu:303-432`):

```
batch centered on S covers keys:   [S - half, S + half - 1]
   d = -half     →  S - half·G              (last, - branch only)
   d = -(i+1)    →  S - (i+1)·G             (- branch, i = 0..half-2)
   d = 0         →  S·G                     (base point check)
   d = +(i+1)    →  S + (i+1)·G             (+ branch, i = 0..half-2)
```

One inversion → B hashes, B consecutive keys, no gaps. This is the
"bidirectional" in "bidirectional batch sweep" — it is **search-direction
independent** and always on, even with `--backward` off.

## 2. Round-robin march (always on)

The grid sweeps a range **in linear order**. Thread `g` takes batches
`g, g+T, g+2T, …` where `T = jobs` (or `groupSize` once sections split the
grid). So the grid covers keys `[start + k·B, …)` for the first T batches in
one "round", then jumps `T·B` keys for the next round (`sweep_kernel.cu:123-131`).

```
round 0: threads 0..T-1  cover batches 0..T-1       = keys 1 .. T·B
round 1: threads 0..T-1  cover batches T..2T-1       = keys T·B+1 .. 2T·B
...
```

Meaning a key at position P is found after ≈ P hashes, regardless of which
thread hits it. Example: `--jobs 8192 --batch 512` → stride = 8192·512 =
4,194,304 keys/round; the frontier advances linearly through the range.

## 3. Sections — split the range into chunks (`--sections N`)

`partition_range()` slices `[start, start+len)` into **N contiguous chunks**
(`harakiri_sweep.cu:264-291`), and each chunk is swept by its own group of
`groupSize = jobs/N` threads, round-robin *within* that chunk.

On `--sections N` with N GPUs, the chunks are spread round-robin across the
devices (`harakiri_sweep.cu:858-861`): device d owns sections d, d+N, d+2N, …
Each device runs its chunks as groups in one grid, and a host-mapped stop flag
halts every other device the instant any GPU finds the key.

```
--sections 4 on 2 GPUs:

range [start, start+len)
├─ section 0 → GPU 0 ╮
│  section 2 → GPU 0 ├─ each = one group marching forward
│  section 1 → GPU 1 │
└─ section 3 → GPU 1 ╯
```

```
harakiri_sweep.exe --batch 1024 --jobs 5120 --sections 4 \
  --range 0x400000000000000000:0x7fffffffffffffffff --target 1PWo…
```

## 4. Backward — split the range in half, search from both sides (`--backward`)

`--backward` doubles the group count (`G = 2·Sections`). Each chunk is split
in two and **raced from both ends** toward the middle (`harakiri_sweep.cu:371-395`):

```
one section [A, A+len):
├─ group 2s   : [A, A+len/2)        forward  (dir +1, marches up from A)
└─ group 2s+1 : [A+len/2, A+len)    backward (dir -1, marches down from A+len)
```

The backward half uploads the **negated** J point `(Jx, p−Jy)` so the identical
point-add walks the base point downward; only `c_Jx`/`c_Jy`/`c_dir` differ per
group (`sweep_kernel.cu:145-154, 527-535`). Think "two walkers start at the
two ends of the street and race to the middle."

```
harakiri_sweep.exe --backward --batch 1024 --jobs 5120 --sections 2 \
  --range 0x400000000000000000:0x7fffffffffffffffff --target 1PWo…
```

```
range [start, start+len)
├─ section 0 → GPU 0 ── group 0 : [A, A+len/2)        forward  ╮ two-front
│                    └─ group 1 : (A+len, A+len/2]    backward ╯ race
├─ section 1 → GPU 1 ── group 2 : [B, B+len/2)        forward  ╮ two-front
│                    └─ group 3 : (B+len, B+len/2]    backward ╯ race
```

**Recap — how they combine.** `--sections N` splits the range into N chunks
(per GPU), each chunk marches round-robin (search 2), each march checks keys
bidirectionally inside every batch (search 1), and with `--backward` each
chunk additionally halves itself and races both ends (search 4).
`--backward --sections N` = 2N groups total.
