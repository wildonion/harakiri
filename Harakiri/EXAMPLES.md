

# GPU realtime monitoring 

```powershell
nvidia-smi -i 0,1,2 -pl 489 # limit the power 
nvidia-smi --query-gpu=timestamp,index,name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,memory.used,pcie.link.gen.current,pcie.link.width.current,clocks_throttle_reasons.active --format=csv -l 5 -f gpu_monitor.csv
nvidia-smi dmon -s pucvmet -d 2 -o DT -f gpu_dmon.csv
```

## 45 bits key output (`--chunks`) - 575.00 W all GPUs:

```powershell
PS C:\Users\Administrator\Desktop\wildonion> .\harakiri_sweep.exe --chunks --batch 1024 --jobs 87040 --range 200000000000:3fffffffffff --target 1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP --maxbatch 200 --out found.txt --sections 3 --gpui 0,1,2 --ckpt-every 30 --checkpoint C:\Users\Administrator\Desktop\wildonion\

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
checkpoint : every 30 s -> C:\Users\Administrator\Desktop\wildonion\\gpu<N>.bin (auto-resume on restart)
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
[GPU 0][ckpt] t=   30.1 s |  8.28 GK/s | section   2.13% (2.496e+11/1.173e+13 keys) | ETA 23.1 min
[GPU 1][ckpt] t=   30.3 s |  8.24 GK/s | section   2.13% (2.496e+11/1.173e+13 keys) | ETA 23.2 min
[GPU 2][ckpt] t=   30.6 s |  8.15 GK/s | section   2.13% (2.496e+11/1.173e+13 keys) | ETA 23.5 min
[GPU 0][ckpt] t=   60.1 s |  8.30 GK/s | section   4.26% (4.991e+11/1.173e+13 keys) | ETA 22.6 min
[GPU 1][ckpt] t=   60.8 s |  8.20 GK/s | section   4.26% (4.991e+11/1.173e+13 keys) | ETA 22.8 min
[GPU 2][ckpt] t=   61.5 s |  8.11 GK/s | section   4.26% (4.991e+11/1.173e+13 keys) | ETA 23.1 min
[GPU 0][ckpt] t=   90.4 s |  8.28 GK/s | section   6.38% (7.487e+11/1.173e+13 keys) | ETA 22.1 min
[GPU 1][ckpt] t=   91.7 s |  8.17 GK/s | section   6.38% (7.487e+11/1.173e+13 keys) | ETA 22.4 min
[GPU 2][ckpt] t=   92.7 s |  8.07 GK/s | section   6.38% (7.487e+11/1.173e+13 keys) | ETA 22.7 min
[GPU 0][ckpt] t=  120.8 s |  8.27 GK/s | section   8.51% (9.982e+11/1.173e+13 keys) | ETA 21.6 min
[GPU 1][ckpt] t=  122.5 s |  8.15 GK/s | section   8.51% (9.982e+11/1.173e+13 keys) | ETA 22.0 min
[GPU 2][ckpt] t=  124.0 s |  8.05 GK/s | section   8.51% (9.982e+11/1.173e+13 keys) | ETA 22.2 min
[GPU 0][ckpt] t=  151.0 s |  8.26 GK/s | section  10.64% (1.248e+12/1.173e+13 keys) | ETA 21.1 min
[GPU 1][ckpt] t=  153.5 s |  8.13 GK/s | section  10.64% (1.248e+12/1.173e+13 keys) | ETA 21.5 min
[GPU 2][ckpt] t=  155.3 s |  8.03 GK/s | section  10.64% (1.248e+12/1.173e+13 keys) | ETA 21.7 min
[GPU 0][ckpt] t=  181.2 s |  8.26 GK/s | section  12.77% (1.497e+12/1.173e+13 keys) | ETA 20.6 min
[GPU 1][ckpt] t=  184.5 s |  8.12 GK/s | section  12.77% (1.497e+12/1.173e+13 keys) | ETA 21.0 min
[GPU 2][ckpt] t=  186.7 s |  8.02 GK/s | section  12.77% (1.497e+12/1.173e+13 keys) | ETA 21.3 min
[GPU 0][ckpt] t=  211.6 s |  8.25 GK/s | section  14.90% (1.747e+12/1.173e+13 keys) | ETA 20.2 min
[GPU 1][ckpt] t=  215.5 s |  8.11 GK/s | section  14.90% (1.747e+12/1.173e+13 keys) | ETA 20.5 min
[GPU 2][ckpt] t=  218.0 s |  8.01 GK/s | section  14.90% (1.747e+12/1.173e+13 keys) | ETA 20.8 min
[GPU 0][ckpt] t=  241.9 s |  8.25 GK/s | section  17.02% (1.996e+12/1.173e+13 keys) | ETA 19.7 min
[GPU 1][ckpt] t=  246.5 s |  8.10 GK/s | section  17.02% (1.996e+12/1.173e+13 keys) | ETA 20.0 min
[GPU 2][ckpt] t=  249.4 s |  8.01 GK/s | section  17.02% (1.996e+12/1.173e+13 keys) | ETA 20.3 min
[GPU 0][ckpt] t=  272.1 s |  8.25 GK/s | section  19.15% (2.246e+12/1.173e+13 keys) | ETA 19.1 min
[GPU 1][ckpt] t=  277.6 s |  8.09 GK/s | section  19.15% (2.246e+12/1.173e+13 keys) | ETA 19.5 min
[GPU 2][ckpt] t=  280.7 s |  8.00 GK/s | section  19.15% (2.246e+12/1.173e+13 keys) | ETA 19.7 min
[GPU 0][ckpt] t=  302.7 s |  8.25 GK/s | section  21.28% (2.496e+12/1.173e+13 keys) | ETA 18.7 min
[GPU 1][ckpt] t=  308.6 s |  8.09 GK/s | section  21.28% (2.496e+12/1.173e+13 keys) | ETA 19.0 min
[GPU 2][ckpt] t=  312.1 s |  8.00 GK/s | section  21.28% (2.496e+12/1.173e+13 keys) | ETA 19.2 min
[GPU 0][ckpt] t=  333.1 s |  8.24 GK/s | section  23.41% (2.745e+12/1.173e+13 keys) | ETA 18.2 min
[GPU 1][ckpt] t=  339.6 s |  8.08 GK/s | section  23.41% (2.745e+12/1.173e+13 keys) | ETA 18.5 min
[GPU 2][ckpt] t=  343.4 s |  7.99 GK/s | section  23.41% (2.745e+12/1.173e+13 keys) | ETA 18.7 min
[GPU 0][ckpt] t=  363.3 s |  8.24 GK/s | section  25.53% (2.995e+12/1.173e+13 keys) | ETA 17.7 min
[GPU 1][dbg] FOUND at launch 166: threadId=33368 finder_batch=37
[GPU 1][ckpt] final snapshot saved (in progress)
[GPU 0][ckpt] final snapshot saved (section complete)
[GPU 2][ckpt] final snapshot saved (section complete)

=== results ===
devices     : 3 GPUs
iterations : 502
found on    : launch 166
total keys : 8917285744096
sweep time : 367.042 s
wall time  : 367.138 s (incl. warmup/table-gen)
throughput : 24295.029 Mkeys/s
FOUND scalar: 00000000000000000000000000000000000000000000000000002ec18388d544
PRIVATE KEY : 00000000000000000000000000000000000000000000000000002ec18388d544
target      : 9a012260d01c5113df66c8a8438c9f7a1e3d5dac
MATCH       : YES
searched    : 8.917e+12 / 3.518e+13 keys (25.34% of range)
result written to found.txt
done
```

## 70 bits - 4 packs of 2^50 bits each, example commands:

```
.\harakiri_sweep.exe --chunks --batch 1024 --jobs 87040 --range 7FF7F4000000000000:+1125899906842624 --target 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU --maxbatch 200 --sections 3 --gpui 0,1,2
.\harakiri_sweep.exe --chunks --batch 1024 --jobs 87040 --range 7FF7F8000000000000:+1125899906842624 --target 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU --maxbatch 200 --sections 3 --gpui 0,1,2
.\harakiri_sweep.exe --chunks --batch 1024 --jobs 87040 --range 7FF7FC000000000000:+1125899906842624 --target 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU --maxbatch 200 --sections 3 --gpui 0,1,2
.\harakiri_sweep.exe --chunks --batch 1024 --jobs 87040 --range 7FF800000000000000:+1125899906842624 --target 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU --maxbatch 200 --sections 3 --gpui 0,1,2
```
