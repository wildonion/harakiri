

# GPU realtime monitoring 

```powershell
vidia-smi -i 0,1,2 -pl 489 # limit the power 
nvidia-smi --query-gpu=timestamp,index,name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active --format=csv -l 5 -f gpu_monitor.csv
nvidia-smi dmon -s pucvmet -d 2 -o DT -f gpu_dmon.csv
```

## 41 bits key output:

```powershell
PS C:\Users\Administrator\Desktop\wildonion> ./harakiri_sweep.exe --global --batch 1024 --jobs 87040 --range 10000000000:1ffffffffff --target 1L5sU9qvJeuwQUdt4y1eiLmquFxKjtHr3E --maxbatch 200 --out found.txt --sections 3 --gpui 0,1,2

=== Harakiri Phase 2 sweep ===
range start : 0000000000000000000000000000000000000000000000000000010000000000
batch size  : 1024
selftest    : no
backward    : no
target      : d1562eb37357f9e6fc41cb2359f4d3eda4032329  (from address)
CUDA driver/runtime: driver 13040, 3 device(s) visible
  device 0: NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, canMapHostMemory=1, unifiedAddressing=1)
  device 1: NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, canMapHostMemory=1, unifiedAddressing=1)
  device 2: NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, canMapHostMemory=1, unifiedAddressing=1)
GPU plan    : using 3 of 3 device(s): 0 1 2  | 3 section(s) spread across them
stride      : global shared frontier, T=261120 threads x B=1024 across 3 GPU(s)
GPU 0       : NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs)
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=261120, global stride) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000000404400

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
GPU 2       : NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs)
GPU 1       : NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs)
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=261120, global stride) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000000404400
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=261120, global stride) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000000404400
[GPU 2][dbg] FOUND at launch 6: threadId=56891 finder_batch=141

=== results ===
devices     : 3 GPUs
iterations : 21
found on    : launch 6
total keys : 371086211008
sweep time : 14.112 s
wall time  : 14.145 s (incl. warmup/table-gen)
throughput : 26295.370 Mkeys/s
FOUND scalar: 00000000000000000000000000000000000000000000000000000153869acc5b
PRIVATE KEY : 00000000000000000000000000000000000000000000000000000153869acc5b
target      : d1562eb37357f9e6fc41cb2359f4d3eda4032329
MATCH       : YES
searched    : 3.711e+11 / 1.100e+12 keys (33.75% of range)
result written to found.txt
done
```

## 44 bits key output:

```powershell
S C:\Users\Administrator\Desktop\wildonion> ./harakiri_sweep.exe --global --batch 1024 --jobs 87040 --range 80000000000:fffffffffff --target 1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF --maxbatch 200 --out found.txt --sections 3 --gpui 0,1,2

=== Harakiri Phase 2 sweep ===
range start : 0000000000000000000000000000000000000000000000000000080000000000
batch size  : 1024
selftest    : no
backward    : no
target      : 80df54e1f612f2fc5bdc05c9d21a83aa8d20791e  (from address)
CUDA driver/runtime: driver 13040, 3 device(s) visible
  device 0: NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, canMapHostMemory=1, unifiedAddressing=1)
  device 1: NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, canMapHostMemory=1, unifiedAddressing=1)
  device 2: NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, canMapHostMemory=1, unifiedAddressing=1)
GPU plan    : using 3 of 3 device(s): 0 1 2  | 3 section(s) spread across them
stride      : global shared frontier, T=261120 threads x B=1024 across 3 GPU(s)
GPU 0       : NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs)
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=261120, global stride) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000002020400

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
GPU 2       : NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs)
GPU 1       : NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs)
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=261120, global stride) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000002020400
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=261120, global stride) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000002020400
[GPU 2][dbg] FOUND at launch 123: threadId=65753 finder_batch=115

=== results ===
devices     : 3 GPUs
iterations : 376
found on    : launch 123
total keys : 6689052361120
sweep time : 252.300 s
wall time  : 252.348 s (incl. warmup/table-gen)
throughput : 26512.301 Mkeys/s
FOUND scalar: 00000000000000000000000000000000000000000000000000000e02b35a358f
PRIVATE KEY : 00000000000000000000000000000000000000000000000000000e02b35a358f
target      : 80df54e1f612f2fc5bdc05c9d21a83aa8d20791e
MATCH       : YES
searched    : 6.689e+12 / 8.796e+12 keys (76.05% of range)
result written to found.txt
done
```