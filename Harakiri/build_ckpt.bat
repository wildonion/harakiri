@echo off
REM ============================================================
REM  Build harakiri_sweep_ckpt.exe (checkpoint/snapshot feature)
REM  Same recipe as build.bat, but:
REM    - output name is harakiri_sweep_ckpt.exe (never touches the
REM      production harakiri_sweep.exe)
REM    - fixed sm_86 SASS + embedded PTX so the SAME exe runs on
REM      both boxes (RTX 5090 = sm_120, driver JITs from sm_86)
REM ============================================================
setlocal EnableDelayedExpansion

set "CUDA_TOOLKIT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
set "NVCC=%CUDA_TOOLKIT%\v13.4\bin\nvcc.exe"
if not exist "%NVCC%" (echo ERROR: %NVCC% not found & exit /b 1)

set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSOUT=%TEMP%\harakiri_vswhere_ckpt.txt"
set "VSDIR="
if exist "%VSWHERE%" (
  "%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%VSOUT%" 2>nul
  set /p VSDIR=<"%VSOUT%"
)
del /q "%VSOUT%" 2>nul
if not defined VSDIR (echo ERROR: no Visual Studio with C++ tools found & exit /b 1)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl.exe >nul 2>&1
if errorlevel 1 (echo ERROR: cl.exe not in PATH after vcvars64.bat & exit /b 1)
set "cl="

"%NVCC%" -O3 -use_fast_math --ptxas-options=-O3,-v ^
  -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 ^
  -std=c++17 -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH ^
  -lcudadevrt -cudart=static ^
  -o harakiri_sweep_ckpt.exe harakiri_sweep.cu
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo BUILD OK: harakiri_sweep_ckpt.exe
endlocal
