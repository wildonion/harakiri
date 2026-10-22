


## 44 bits key outpu (KeyHunt Cuda)

```powershell
PS C:\Users\Administrator\Desktop\wildonion> .\khc.exe -g --gpui 0,1,2 --gpux 1024,256,1024,256,1024,256 -m address --coin BTC --range 80000000000:fffffffffff 1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF -o found.txt

KeyHunt-Cuda v1.07

COMP MODE    : COMPRESSED
COIN TYPE    : BITCOIN
SEARCH MODE  : Single Address
DEVICE       : GPU
CPU THREAD   : 0
GPU IDS      : 0, 1, 2
GPU GRIDSIZE : 1024x256, 1024x256, 1024x256
SSE          : YES
RKEY         : 0 Mkeys
MAX FOUND    : 65536
BTC ADDRESS  : 1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF
OUTPUT FILE  : found.txt

Start Time   : Thu Oct 22 10:51:00 2026
Global start : 80000000000 (44 bit)
Global end   : FFFFFFFFFFF (44 bit)
Global range : 7FFFFFFFFFF (43 bit)

GPU          : GPU #2 NVIDIA GeForce RTX 5090 (170x0 cores) Grid(1024x256)

GPU          : GPU #0 NVIDIA GeForce RTX 5090 (170x0 cores) Grid(1024x256)

GPU          : GPU #1 NVIDIA GeForce RTX 5090 (170x0 cores) Grid(1024x256)

[00:00:12] [CPU+GPU: 20329.04 Mk/s] [GPU: 20329.04 Mk/s] [C: 2.813721 %] [R: 0] [T: 247,497,490,432 (38 bit)] [F: 0]
=================================================================================
PubAddress: 1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF
Priv (WIF): p2pkh:KwDiBf89QgGbjEhKnhXJuH7LrciVrZi3qYjgdCpdGyxm7PWoNQdr
Priv (HEX): E02B35A358F
PubK (HEX): 025E466E97ED0E7910D3D90CEB0332DF48DDF67D456B9E7303B50A3D89DE357336
=================================================================================
[00:00:14] [CPU+GPU: 20350.18 Mk/s] [GPU: 20350.18 Mk/s] [C: 3.289795 %] [R: 0] [T: 289,373,421,568 (39 bit)] [F: 1]

BYE
```