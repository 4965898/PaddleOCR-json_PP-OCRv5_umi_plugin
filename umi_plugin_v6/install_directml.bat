@echo off
title PP-OCRv6 ONNX Runtime Plugin - DirectML Setup
cd /d "%~dp0"
echo ========================================
echo  PP-OCRv6 ONNX Runtime Plugin - DirectML Setup
echo  (Intel Arc / AMD / any DirectX 12 GPU)
echo ========================================
echo.

REM ---- Python check: venv must exist (created by install.bat) ----
if not exist "ppocr_v6_env\Scripts\python.exe" goto :no_python
REM venv version gate (same as install.bat: paddleocr needs 3.10-3.13)
ppocr_v6_env\Scripts\python.exe -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul
if errorlevel 1 goto :venv_badver

echo [CHECK] DirectX 12 GPU...
echo [INFO] DirectML supports Intel Arc, AMD, and NVIDIA DX12 GPUs.
echo        NVIDIA users may prefer install_gpu.bat (CUDA) for best performance.
echo.

echo [1/2] Removing conflicting onnxruntime variants (if installed)...
ppocr_v6_env\Scripts\pip uninstall -y onnxruntime onnxruntime-gpu 2>nul
REM Also remove NVIDIA runtime packages pulled in by install_gpu*.bat
ppocr_v6_env\Scripts\pip uninstall -y nvidia-cuda-runtime nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc nvidia-cuda-nvrtc-cu12 nvidia-cufft nvidia-cufft-cu12 nvidia-curand nvidia-curand-cu12 nvidia-cudnn nvidia-cudnn-cu12 nvidia-cudnn-cu13 2>nul
echo Done.
echo.

echo [2/2] Installing paddleocr + onnxruntime-directml ...
echo This may take 1-3 minutes...
echo.
ppocr_v6_env\Scripts\pip install paddleocr onnxruntime-directml --upgrade
if errorlevel 1 goto :install_fail

echo.
echo [VERIFY] Checking DirectML provider...
ppocr_v6_env\Scripts\python -c "import onnxruntime as ort; ps=ort.get_available_providers(); print('Available providers:', ps); print('DirectML OK!' if 'DmlExecutionProvider' in ps else 'DirectML NOT available')"

echo.
echo ========================================
echo  DirectML Setup Complete!
echo ========================================
echo.
echo Enable "GPU Acceleration" in Umi-OCR plugin settings.
echo The plugin auto-selects the backend: CUDA (if installed) > DirectML > CPU.
echo DirectML works with Intel Arc, AMD, and any DirectX 12 GPU.
echo.
echo Please restart Umi-OCR.
echo.
pause
exit /b 0

:no_python
echo [ERROR] Virtual environment ppocr_v6_env does not exist.
echo         Please run install.bat first to complete the base installation
echo         (works out of the box, no Python pre-installation required),
echo         then run this script to add DirectML support.
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

:install_fail
echo.
echo [WARNING] Auto-install failed. Please run manually:
echo   ppocr_v6_env\Scripts\pip install paddleocr onnxruntime-directml
echo.
echo NOTE: onnxruntime-directml conflicts with onnxruntime / onnxruntime-gpu.
echo       This script uninstalls them automatically before installing;
echo       if you still see issues, uninstall manually:
echo         ppocr_v6_env\Scripts\pip uninstall -y onnxruntime onnxruntime-gpu
echo       then re-run this script.
echo.
pause
exit /b 1
