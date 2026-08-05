@echo off
chcp 65001 >nul
echo ========================================
echo  PP-OCRv6 ONNX Runtime Plugin - DirectML Setup
echo  (Intel Arc / AMD / any DirectX 12 GPU)
echo ========================================
echo.

REM Check Python: venv must exist (created by install.bat) or system Python available
if not exist "ppocr_v6_env\Scripts\python.exe" (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] 未检测到 Python，且虚拟环境不存在。
        echo         请先运行 install.bat 完成基础安装（开箱即用，无需预装 Python），
        echo         再运行本脚本升级 DirectML 支持。
        echo.
        pause
        exit /b 1
    )
)

REM Check for a DirectX 12 GPU (Intel Arc / AMD / NVIDIA all supported)
echo [CHECK] DirectX 12 GPU...
REM DirectML runs on any DX12 GPU. We cannot reliably detect DX12 from a batch
REM script, so we only warn here; the real check is whether DmlExecutionProvider
REM is available after install (verified at the end of this script).
echo [INFO] DirectML supports Intel Arc, AMD, and NVIDIA DX12 GPUs.
echo        NVIDIA users may prefer install_gpu.bat (CUDA) for best performance.
echo.

cd /d "%~dp0"

if not exist "ppocr_v6_env\Scripts\python.exe" (
    echo [1/2] Creating virtual environment ppocr_v6_env ...
    python -m venv ppocr_v6_env
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/2] Virtual environment already exists, skipping.
)

echo.
echo [2/2] Installing paddleocr + onnxruntime-directml ...
echo This may take 1-3 minutes...
echo.
ppocr_v6_env\Scripts\pip install paddleocr onnxruntime-directml --upgrade
if errorlevel 1 (
    echo.
    echo [WARNING] Auto-install failed. Please run manually:
    echo   ppocr_v6_env\Scripts\pip install paddleocr onnxruntime-directml
    echo.
    echo NOTE: onnxruntime-directml conflicts with onnxruntime / onnxruntime-gpu.
    echo       If you previously ran install.bat or install_gpu.bat, uninstall first:
    echo         ppocr_v6_env\Scripts\pip uninstall -y onnxruntime onnxruntime-gpu
    echo       then re-run this script.
    echo.
    pause
    exit /b 1
)

echo.
echo [VERIFY] Checking DirectML provider...
ppocr_v6_env\Scripts\python -c "import onnxruntime as ort; ps=ort.get_available_providers(); print('Available providers:', ps); print('DirectML OK!' if 'DmlExecutionProvider' in ps else 'DirectML NOT available')"

echo.
echo ========================================
echo  DirectML Setup Complete!
echo ========================================
echo.
echo Enable "GPU Acceleration" in Umi-OCR plugin settings.
echo The plugin auto-selects the backend: CUDA (if installed) ^> DirectML ^> CPU.
echo DirectML works with Intel Arc, AMD, and any DirectX 12 GPU.
echo.
echo Please restart Umi-OCR.
echo.
pause
