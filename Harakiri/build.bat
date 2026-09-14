@echo off
REM ============================================================
REM  Harakiri - CUDA secp256k1 puzzle solver (Phase 2 sweep engine)
REM  CUDA 12.x build script with selectable GPU architecture.
REM
REM  Usage:
REM    build.bat                      auto-detect local GPU arch (-arch=native)
REM    build.bat sm_86                native Ampere  (RTX 30xx) + embedded PTX
REM    build.bat sm_89                native Ada     (RTX 40xx) + embedded PTX
REM    build.bat sm_120               native Blackwell (RTX 50xx) + embedded PTX
REM    build.bat sm_89 sm_86 sm_120   multiple archs in one exe
REM
REM  The script auto-picks the newest installed CUDA 12.x toolkit under
REM  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA
REM  (set CUDA_TOOLKIT below if installed elsewhere).
REM  Output: harakiri_sweep.exe
REM ============================================================
setlocal EnableDelayedExpansion
set SM_ARGS=%*

REM ---------- locate CUDA 12.x toolkit (newest first) ----------
set "CUDA_TOOLKIT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
set "CUDA_VER="
for %%v in (v13.3 v13.0 v12.9 v12.8 v12.7 v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0) do (
  if not defined CUDA_VER if exist "%CUDA_TOOLKIT%\%%v\bin\nvcc.exe" set "CUDA_VER=%%v"
)
if not defined CUDA_VER (echo ERROR: no CUDA 12.x found under %CUDA_TOOLKIT% & exit /b 1)
set "NVCC=%CUDA_TOOLKIT%\%CUDA_VER%\bin\nvcc.exe"
echo Using CUDA %CUDA_VER% : %NVCC%

REM ---------- locate MSVC via vswhere ----------
set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VSDIR=%%i"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set cl=

REM ---------- arch selection ----------
if "%SM_ARGS%"=="" (
  echo arch      : auto (native SASS for the local GPU, -arch=native)
  "%NVCC%" -O3 -rdc=true -use_fast_math --ptxas-options=-O3 -arch=native ^
    -std=c++17 -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH ^
    -lcudadevrt -cudart=static ^
    -o harakiri_sweep.exe harakiri_sweep.cu
  if errorlevel 1 (echo BUILD FAILED & exit /b 1)
  echo BUILD OK  (native SASS for local GPU, CUDA %CUDA_VER%)
  goto :done
)

REM explicit arch(s): build a -gencode list: SASS + embedded compute PTX per arch
set "GENCODE="
for %%a in (%SM_ARGS%) do (
  for /f "tokens=1,2 delims=_" %%x in ("%%a") do set "COMPUTE=compute_%%y"
  set "GENCODE=!GENCODE! -gencode arch=!COMPUTE!,code=%%a -gencode arch=!COMPUTE!,code=!COMPUTE!"
)
echo arch      : %SM_ARGS% (native SASS + embedded PTX)
"%NVCC%" -O3 -rdc=true -use_fast_math --ptxas-options=-O3 !GENCODE! ^
  -std=c++17 -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH ^
  -lcudadevrt -cudart=static ^
  -o harakiri_sweep.exe harakiri_sweep.cu
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK  (%SM_ARGS% native + PTX, CUDA %CUDA_VER%)
:done
endlocal
