@echo off
title PP-OCRv6 ONNX Runtime Plugin - GPU Setup
cd /d "%~dp0"
echo ========================================
echo  PP-OCRv6 ONNX Runtime Plugin - GPU Setup
echo ========================================
echo.

REM Check Python: venv must exist (created by install.bat) or system Python available
if exist "ppocr_v6_env\Scripts\python.exe" goto :check_nvidia
where python >nul 2>nul
if errorlevel 1 goto :no_python

:check_nvidia
echo [CHECK] NVIDIA GPU and driver...
nvidia-smi >nul 2>nul
if errorlevel 1 goto :no_nvidia
echo [OK] NVIDIA GPU detected
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>nul
echo.

REM Create venv if not exists
if exist "ppocr_v6_env\Scripts\python.exe" goto :venv_exists
echo [1/4] Creating virtual environment ppocr_v6_env ...
python -m venv ppocr_v6_env
if errorlevel 1 goto :venv_fail
goto :venv_done

:venv_exists
echo [1/4] Virtual environment already exists, skipping.

:venv_done
echo.
echo [2/4] Removing CPU-only onnxruntime (if installed) to avoid conflicts...
ppocr_v6_env\Scripts\pip uninstall -y onnxruntime 2>nul
echo Done.
echo.

echo [3/4] Installing paddleocr + onnxruntime-gpu + CUDA + cuDNN ...
echo This may take 5-15 minutes (downloads ~1.6GB)...
echo Components installed by pip:
echo   - onnxruntime-gpu (GPU inference engine)
echo   - nvidia-cuda-runtime-cu12 (CUDA 12.x Runtime)
echo   - nvidia-cudnn-cu12 (cuDNN 9.x)
echo   - nvidia-cublas-cu12, nvidia-cufft-cu12, etc.
echo No manual CUDA/cuDNN installation is needed.
echo.
ppocr_v6_env\Scripts\pip install paddleocr "onnxruntime-gpu[cuda,cudnn]" --upgrade
if errorlevel 1 goto :install_fail

echo.
echo [4/4] Verifying GPU support (creating test CUDA session)...
ppocr_v6_env\Scripts\python -c "import sysconfig,os;[os.add_dll_directory(os.path.join(sysconfig.get_paths()['purelib'],'nvidia',s,'bin')) for s in os.listdir(os.path.join(sysconfig.get_paths()['purelib'],'nvidia')) if os.path.isdir(os.path.join(sysconfig.get_paths()['purelib'],'nvidia',s,'bin'))] if os.path.isdir(os.path.join(sysconfig.get_paths()['purelib'],'nvidia')) else None;import onnxruntime as ort;ps=ort.get_available_providers();print('Available providers:',ps);exit(0 if 'CUDAExecutionProvider' in ps else 1)"
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

:no_python
echo [ERROR] 未检测到 Python，且虚拟环境不存在�?
echo         请先运行 install.bat 完成基础安装（开箱即用，无需预装 Python），
echo         再运行本脚本升级 GPU 支持�?
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
echo R525+ (Windows). Old drivers will cause silent CPU fallback.
echo.
echo For CPU-only, please run install.bat instead.
echo.
pause
exit /b 1

:venv_fail
echo [ERROR] Failed to create virtual environment.
pause
exit /b 1

:install_fail
echo.
echo [WARNING] Auto-install failed. Please run manually:
echo   ppocr_v6_env\Scripts\pip uninstall -y onnxruntime
echo   ppocr_v6_env\Scripts\pip install paddleocr "onnxruntime-gpu[cuda,cudnn]"
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
echo   3. Conflicting onnxruntime (CPU) still installed
echo.
echo Please update your NVIDIA driver and re-run this script.
echo Driver download: https://www.nvidia.com/Download/index.aspx
echo.
pause
exit /b 1