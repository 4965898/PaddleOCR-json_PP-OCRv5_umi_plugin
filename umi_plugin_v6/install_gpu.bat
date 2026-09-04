@echo off
title PP-OCRv6 ONNX Runtime Plugin - GPU Setup
cd /d "%~dp0"
echo ========================================
echo  PP-OCRv6 ONNX Runtime Plugin - GPU Setup
echo ========================================
echo.

REM ---- Python check: venv must exist (created by install.bat) ----
if not exist "ppocr_v6_env\Scripts\python.exe" goto :no_python
REM venv version gate (same as install.bat: paddleocr needs 3.10-3.13)
ppocr_v6_env\Scripts\python.exe -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul
if errorlevel 1 goto :venv_badver

REM ---- NVIDIA check ----
echo [CHECK] NVIDIA GPU and driver...
nvidia-smi >nul 2>nul
if errorlevel 1 goto :no_nvidia
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>nul
echo.

REM ---- RTX 50 (Blackwell sm_120) routing ----
REM Official onnxruntime-gpu 1.27+ on PyPI is built with CUDA 13 (includes
REM sm_120 kernels); versions <=1.26 are CUDA 12.8 without sm_120.
REM Blackwell GPUs MUST use the CUDA 13 build -> install_gpu_rtx50.bat
set "GPU_CCAP="
for /f "tokens=1 delims=." %%C in ('nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2^>nul') do (
    if %%C GEQ 12 set "GPU_CCAP=%%C"
)
if defined GPU_CCAP goto :rtx50
nvidia-smi --query-gpu=name --format=csv,noheader 2>nul | findstr /i /c:"RTX 50" >nul 2>nul
if not errorlevel 1 goto :rtx50

REM ================================================
REM  Standard path: CUDA 12 build (driver R525+)
REM ================================================
echo [1/3] Removing CPU-only onnxruntime (if installed) to avoid conflicts...
ppocr_v6_env\Scripts\pip uninstall -y onnxruntime 2>nul
REM Also remove CUDA 13 runtime packages if a previous script installed them
ppocr_v6_env\Scripts\pip uninstall -y nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-cufft nvidia-curand nvidia-cudnn-cu13 2>nul
echo Done.
echo.

echo [2/3] Installing paddleocr + onnxruntime-gpu (CUDA 12) + CUDA + cuDNN ...
echo This may take 5-15 minutes (downloads ~1.6GB)...
echo Components installed by pip:
echo   - onnxruntime-gpu <1.27 (CUDA 12.8 build, supports NVIDIA driver R525+)
echo   - nvidia-cuda-runtime-cu12 (CUDA 12.x Runtime)
echo   - nvidia-cudnn-cu12 (cuDNN 9.x)
echo   - nvidia-cufft-cu12, nvidia-curand-cu12, etc.
echo No manual CUDA/cuDNN installation is needed.
echo.
ppocr_v6_env\Scripts\pip install paddleocr "onnxruntime-gpu[cuda,cudnn]>=1.23,<1.27" --upgrade
if errorlevel 1 goto :install_fail

echo.
echo [3/3] Verifying GPU support...
ppocr_v6_env\Scripts\python verify_gpu.py
if errorlevel 1 goto :cuda_fail
echo [OK] CUDAExecutionProvider is available!

echo.
echo ========================================
echo  GPU Setup Complete!
echo ========================================
echo.
echo Enable "GPU Acceleration" in Umi-OCR plugin settings.
echo First recognition may be slower (GPU init), then ~17x faster.
echo.
echo To verify GPU is working, check Umi-OCR logs for:
echo   [ppocr_v6] engine=onnxruntime, gpu_backend=cuda
echo   [ppocr_v6] GPU verified: det model session uses ['CUDAExecutionProvider', ...]
echo.
echo If you see "gpu_backend=None" or a CPU fallback warning,
echo GPU acceleration is NOT active. See README for troubleshooting.
echo.
echo Please restart Umi-OCR.
echo.
pause
exit /b 0

REM ================================================
REM  RTX 50 path: CUDA 13 build (driver R580+)
REM ================================================
:rtx50
echo [INFO] RTX 50 series (Blackwell, compute capability %GPU_CCAP%) detected.
echo        Blackwell GPUs need the CUDA 13 build of onnxruntime-gpu,
echo        which requires NVIDIA driver R580+ and Python 3.11+.
echo        Switching to install_gpu_rtx50.bat ...
echo.
call install_gpu_rtx50.bat
exit /b %errorlevel%

:no_python
echo [ERROR] Virtual environment ppocr_v6_env does not exist.
echo         Please run install.bat first to complete the base installation,
echo         then run this script to upgrade GPU support.
echo.
pause
exit /b 1

:venv_badver
echo [ERROR] The virtual environment's Python version is not supported (need 3.10-3.13).
echo         Please delete the ppocr_v6_env folder and re-run install.bat,
echo         which will automatically set up a compatible Python.
echo.
pause
exit /b 1

:no_nvidia
echo [ERROR] No NVIDIA GPU or driver detected.
echo GPU acceleration requires an NVIDIA GPU with up-to-date drivers.
echo.
echo Please install the latest NVIDIA driver from:
echo   https://www.nvidia.com/Download/index.aspx
echo CUDA 12.x runtime (bundled by this script) requires driver version
echo R525+ on Windows. Old drivers will cause silent CPU fallback.
echo.
echo For CPU-only, please run install.bat instead.
echo.
pause
exit /b 1

:install_fail
echo.
echo [WARNING] Auto-install failed. Please run manually:
echo   ppocr_v6_env\Scripts\pip install paddleocr "onnxruntime-gpu[cuda,cudnn]>=1.23,<1.27"
echo.
pause
exit /b 1

:cuda_fail
echo.
echo [ERROR] CUDAExecutionProvider is NOT available!
echo onnxruntime-gpu was installed but cannot load CUDA/cuDNN DLLs.
echo Common causes:
echo   1. NVIDIA driver is too old - update to latest version
echo   2. CUDA/cuDNN packages failed to download - check network
echo   3. Conflicting onnxruntime CPU version still installed
echo.
echo Please update your NVIDIA driver and re-run this script.
echo Driver download: https://www.nvidia.com/Download/index.aspx
echo.
pause
exit /b 1
