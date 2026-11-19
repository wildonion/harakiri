

# GPU realtime monitoring 

```powershell
nvidia-smi -i 0,1,2 -pl 489 # limit the power 
nvidia-smi --query-gpu=timestamp,index,name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active --format=csv -l 5 -f gpu_monitor.csv
nvidia-smi dmon -s pucvmet -d 2 -o DT -f gpu_dmon.csv
```

## 45 bits key output (`--chunks`) - 575.00 W all GPUs:

```powershell
PS C:\Users\Administrator\Desktop\7> .\harakiri_sweep.exe --chunks --batch 1024 --jobs 87040 --range 200000000000:3fffffffffff --target 1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP --maxbatch 200 --out found.txt --sections 3 --gpui 0,1,2 --ckpt-every 30 --checkpoint C:\Users\Administrator\Desktop\7\

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
checkpoint : every 30 s -> C:\Users\Administrator\Desktop\7\\gpu<N>.bin (auto-resume on restart)
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
GPU 1       : NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs)
GPU 2       : NVIDIA GeForce RTX 5090 (cc 12.0, 170 SMs)
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=87040, block chunks) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000008081400
config      : batch=1024 keys per inversion | threads/block=256 | jobs=87040 total threads | grid=340 blocks | sections=1 (groupSize=87040, block chunks) | backward=no | max_batches=200 per launch | max_chunk=0000000000000000000000000000000000000000000000000000000008081400
[GPU 2][ckpt] t=   30.6 s |  9.31 GK/s | section   2.43% (2.852e+11/1.173e+13 keys) | ETA 20.5 min
[GPU 1][ckpt] t=   31.2 s |  9.13 GK/s | section   2.43% (2.852e+11/1.173e+13 keys) | ETA 20.9 min
[GPU 0][ckpt] t=   31.7 s |  8.99 GK/s | section   2.43% (2.852e+11/1.173e+13 keys) | ETA 21.2 min
[GPU 2][ckpt] t=   61.4 s |  9.29 GK/s | section   4.86% (5.704e+11/1.173e+13 keys) | ETA 20.0 min
[GPU 1][ckpt] t=   62.7 s |  9.09 GK/s | section   4.86% (5.704e+11/1.173e+13 keys) | ETA 20.5 min
[GPU 0][ckpt] t=   63.5 s |  8.98 GK/s | section   4.86% (5.704e+11/1.173e+13 keys) | ETA 20.7 min
[GPU 2][ckpt] t=   92.5 s |  9.25 GK/s | section   7.30% (8.556e+11/1.173e+13 keys) | ETA 19.6 min
[GPU 0][ckpt] t=   93.5 s |  8.96 GK/s | section   7.14% (8.378e+11/1.173e+13 keys) | ETA 20.3 min
[GPU 1][ckpt] t=   94.6 s |  9.05 GK/s | section   7.30% (8.556e+11/1.173e+13 keys) | ETA 20.0 min
[GPU 0][ckpt] t=  123.6 s |  8.95 GK/s | section   9.42% (1.105e+12/1.173e+13 keys) | ETA 19.8 min
[GPU 2][ckpt] t=  123.7 s |  9.22 GK/s | section   9.73% (1.141e+12/1.173e+13 keys) | ETA 19.1 min
[GPU 1][ckpt] t=  126.5 s |  9.02 GK/s | section   9.73% (1.141e+12/1.173e+13 keys) | ETA 19.6 min
[GPU 0][ckpt] t=  153.6 s |  8.94 GK/s | section  11.70% (1.373e+12/1.173e+13 keys) | ETA 19.3 min
[GPU 2][ckpt] t=  155.0 s |  9.20 GK/s | section  12.16% (1.426e+12/1.173e+13 keys) | ETA 18.7 min
[GPU 1][ckpt] t=  158.4 s |  9.00 GK/s | section  12.16% (1.426e+12/1.173e+13 keys) | ETA 19.1 min
[GPU 0][ckpt] t=  183.6 s |  8.93 GK/s | section  13.98% (1.640e+12/1.173e+13 keys) | ETA 18.8 min
[GPU 2][ckpt] t=  186.2 s |  9.19 GK/s | section  14.59% (1.711e+12/1.173e+13 keys) | ETA 18.2 min
[GPU 1][ckpt] t=  190.4 s |  8.99 GK/s | section  14.59% (1.711e+12/1.173e+13 keys) | ETA 18.6 min
[GPU 0][ckpt] t=  213.7 s |  8.93 GK/s | section  16.26% (1.907e+12/1.173e+13 keys) | ETA 18.3 min
[GPU 2][ckpt] t=  217.4 s |  9.18 GK/s | section  17.02% (1.996e+12/1.173e+13 keys) | ETA 17.7 min
[GPU 1][ckpt] t=  220.4 s |  8.98 GK/s | section  16.87% (1.979e+12/1.173e+13 keys) | ETA 18.1 min
[GPU 0][ckpt] t=  243.7 s |  8.92 GK/s | section  18.54% (2.175e+12/1.173e+13 keys) | ETA 17.8 min
[GPU 2][ckpt] t=  248.7 s |  9.17 GK/s | section  19.45% (2.282e+12/1.173e+13 keys) | ETA 17.2 min
[GPU 1][ckpt] t=  252.4 s |  8.97 GK/s | section  19.30% (2.264e+12/1.173e+13 keys) | ETA 17.6 min
[GPU 0][ckpt] t=  273.8 s |  8.92 GK/s | section  20.82% (2.442e+12/1.173e+13 keys) | ETA 17.4 min
[GPU 2][ckpt] t=  280.0 s |  9.17 GK/s | section  21.89% (2.567e+12/1.173e+13 keys) | ETA 16.7 min
[GPU 1][ckpt] t=  282.4 s |  8.96 GK/s | section  21.58% (2.531e+12/1.173e+13 keys) | ETA 17.1 min
[GPU 0][ckpt] t=  303.8 s |  8.92 GK/s | section  23.10% (2.710e+12/1.173e+13 keys) | ETA 16.9 min
[GPU 2][ckpt] t=  311.3 s |  9.16 GK/s | section  24.32% (2.852e+12/1.173e+13 keys) | ETA 16.1 min
[GPU 1][ckpt] t=  312.4 s |  8.96 GK/s | section  23.86% (2.799e+12/1.173e+13 keys) | ETA 16.6 min
[GPU 1][dbg] FOUND at launch 166: threadId=33368 finder_batch=37
[GPU 1][ckpt] final snapshot saved (in progress)
[GPU 2][ckpt] final snapshot saved (in progress)
[GPU 0][ckpt] final snapshot saved (section complete)

=== results ===
devices     : 3 GPUs
iterations : 503
found on    : launch 166
total keys : 8944011396864
sweep time : 331.160 s
wall time  : 331.254 s (incl. warmup/table-gen)
throughput : 27008.092 Mkeys/s
FOUND scalar: 00000000000000000000000000000000000000000000000000002ec18388d544
PRIVATE KEY : 00000000000000000000000000000000000000000000000000002ec18388d544
target      : 9a012260d01c5113df66c8a8438c9f7a1e3d5dac
MATCH       : YES
searched    : 8.944e+12 / 3.518e+13 keys (25.42% of range)
result written to found.txt
done
```
