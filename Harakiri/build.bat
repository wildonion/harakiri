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
for %%v in (v13.4 v13.3 v13.2 v13.1 v13.0 v12.9 v12.8 v12.7 v12.6 v12.5 v12.4 v12.3 v12.2 v12.1 v12.0) do (
  if not defined CUDA_VER if exist "%CUDA_TOOLKIT%\%%v\bin\nvcc.exe" set "CUDA_VER=%%v"
)
if not defined CUDA_VER (echo ERROR: no CUDA 12.x found under %CUDA_TOOLKIT% & exit /b 1)
set "NVCC=%CUDA_TOOLKIT%\%CUDA_VER%\bin\nvcc.exe"
echo Using CUDA %CUDA_VER% : %NVCC%

REM ---------- locate MSVC via vswhere ----------
REM NOTE: vswhere lives under "...\(x86)\...". Its parenthesized path cannot be
REM passed through 'for /f' (cmd mangles it -> "\Microsoft was unexpected"), so
REM the result is written to a temp file and read back with 'set /p'.
set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSOUT=%TEMP%\harakiri_vswhere.txt"
set "VSDIR="
if exist "%VSWHERE%" (
  "%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%VSOUT%" 2>nul
  set /p VSDIR=<"%VSOUT%"
)
del /q "%VSOUT%" 2>nul
if not defined VSDIR (echo ERROR: no Visual Studio with C++ tools found (install VS 2022 with "Desktop development with C++") & exit /b 1)
echo MSVC      : %VSDIR%
if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" (echo ERROR: vcvars64.bat not found under "%VSDIR%" & exit /b 1)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl.exe >nul 2>&1
if errorlevel 1 (echo ERROR: cl.exe not in PATH after vcvars64.bat & exit /b 1)
REM A stale 'cl' env var pointing at a removed MSVC version makes nvcc abort
REM with exit code 2 and no diagnostic at all. Clear it.
set "cl="


REM ---------- arch selection ----------
REM NOTE: do NOT pass -rdc=true. This program is a single translation unit, so
REM relocatable device code buys nothing, and on newer toolkits it leaves the
REM kernels unresolved in the linked fatbin: the first launch then fails with
REM "named symbol not found" (cudaErrorNamedSymbolNotFound=801) and the sweep
REM reports 0 keys. Whole-program compilation is what we want here.
set "NVCCFLAGS=-O3 -use_fast_math --ptxas-options=-O3"

if "%SM_ARGS%"=="" (
  REM auto: ask the driver for the local GPU's compute capability, then emit an
  REM explicit -gencode with SASS + embedded PTX. More predictable than
  REM -arch=native, which silently depends on nvcc's own GPU detection.
  set "CC="
  for /f "usebackq delims=" %%c in (`nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul`) do if not defined CC set "CC=%%c"
  if defined CC (
    set "MAJOR=!CC:~0,1!"
    set "MINOR=!CC:~2,1!"
    set "SM=sm_!MAJOR!!MINOR!"
  )
  if not defined SM set "SM=sm_120"
  for /f "tokens=1,2 delims=_" %%x in ("!SM!") do set "COMPUTE=compute_%%y"
  set "GENCODE=-gencode arch=!COMPUTE!,code=!SM! -gencode arch=!COMPUTE!,code=!COMPUTE!"
  echo arch      : auto-detected !SM! from nvidia-smi - SASS + embedded PTX
  if not defined CC echo            nvidia-smi unavailable - defaulted to !SM!
  goto :compile
)

REM explicit arch(s): build a -gencode list: SASS + embedded compute PTX per arch
set "GENCODE="
for %%a in (%SM_ARGS%) do (
  for /f "tokens=1,2 delims=_" %%x in ("%%a") do set "COMPUTE=compute_%%y"
  set "GENCODE=!GENCODE! -gencode arch=!COMPUTE!,code=%%a -gencode arch=!COMPUTE!,code=!COMPUTE!"
)
echo arch      : %SM_ARGS% (native SASS + embedded PTX)

:compile
"%NVCC%" %NVCCFLAGS% !GENCODE! ^
  -std=c++17 -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH ^
  -lcudadevrt -cudart=static ^
  -o harakiri_sweep.exe harakiri_sweep.cu
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK  (CUDA %CUDA_VER%)
:done
endlocal
