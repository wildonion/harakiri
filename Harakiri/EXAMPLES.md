

# GPU realtime monitoring 

```powershell
nvidia-smi -i 0,1,2 -pl 489 # limit the power 
nvidia-smi --query-gpu=timestamp,index,name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active --format=csv -l 5 -f gpu_monitor.csv
nvidia-smi dmon -s pucvmet -d 2 -o DT -f gpu_dmon.csv
```

## 45 bits key output (`--global`):

```powershell
PS C:\Users\Administrator\Desktop\wildonion> ./harakiri_sweep.exe --global --batch 1024 --jobs 87040 --range 200000000000:3fffffffffff --target 1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP --maxbatch 200 --out found.txt --sections 3 --gpui 0,1,2

=== Harakiri Phase 2 sweep ===
range start : 0000000000000000000000000000000000000000000000000000200000000000
batch size  : 1024
selftest    : no
backward    : no
target      : 9a012260d01c5113df66c8a8438c9f7a1e3d5dac  (from address)
CUDA driver/runtime: driver 13040, 3 device(s) visible
  device 0: NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, canMapHostMemory=1, unifiedAddressing=1)
  device 1: NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, canMapHostMemory=1, unifiedAddressing=1)
  device 2: NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, canMapHostMemory=1, unifiedAddressing=1)
GPU plan    : using 3 of 3 device(s): 0 1 2  | 3 section(s) spread across them
stride      : global shared frontier, T=261120 threads x B=1024 across 3 GPU(s)
GPU 0       : NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs)
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=261120, global stride) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000008080c00

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
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=261120, global stride) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000008080c00
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=261120, global stride) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000008080c00
[GPU 2][dbg] FOUND at launch 303: threadId=21009 finder_batch=77

=== results ===
devices     : 3 GPUs
iterations : 922
found on    : launch 303
total keys : 16406891189216
sweep time : 620.332 s
wall time  : 620.401 s (incl. warmup/table-gen)
throughput : 26448.569 Mkeys/s
FOUND scalar: 00000000000000000000000000000000000000000000000000002ec18388d544
PRIVATE KEY : 00000000000000000000000000000000000000000000000000002ec18388d544
target      : 9a012260d01c5113df66c8a8438c9f7a1e3d5dac
MATCH       : YES
searched    : 1.641e+13 / 3.518e+13 keys (46.63% of range)
result written to found.txt
done
```

## 45 bits key output (`--chunks`):

```powershell
PS C:\Users\Administrator\Desktop\wildonion> ./harakiri_sweep.exe --chunks --batch 1024 --jobs 87040 --range 200000000000:3fffffffffff --target 1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP --maxbatch 200 --out found.txt --sections 3 --gpui 0,1,2

=== Harakiri Phase 2 sweep ===
range start : 0000000000000000000000000000000000000000000000000000200000000000
batch size  : 1024
selftest    : no
backward    : no
target      : 9a012260d01c5113df66c8a8438c9f7a1e3d5dac  (from address)
CUDA driver/runtime: driver 13040, 3 device(s) visible
  device 0: NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, canMapHostMemory=1, unifiedAddressing=1)
  device 1: NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, canMapHostMemory=1, unifiedAddressing=1)
  device 2: NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs, canMapHostMemory=1, unifiedAddressing=1)
GPU plan    : using 3 of 3 device(s): 0 1 2  | 3 section(s) spread across them
stride      : per-thread contiguous blocks, 3 GPU slice(s) x jobs threads x B=1024
GPU 0       : NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs)
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=87040, block chunks) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000008081400

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
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=87040, block chunks) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000008081400
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=87040, block chunks) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000008081400
[GPU 1][dbg] FOUND at launch 166: threadId=33368 finder_batch=37

=== results ===
devices     : 3 GPUs
iterations : 499
found on    : launch 166
total keys : 8867050573056
sweep time : 335.979 s
wall time  : 336.058 s (incl. warmup/table-gen)
throughput : 26391.684 Mkeys/s
FOUND scalar: 00000000000000000000000000000000000000000000000000002ec18388d544
PRIVATE KEY : 00000000000000000000000000000000000000000000000000002ec18388d544
target      : 9a012260d01c5113df66c8a8438c9f7a1e3d5dac
MATCH       : YES
searched    : 8.867e+12 / 3.518e+13 keys (25.20% of range)
result written to found.txt
done
```