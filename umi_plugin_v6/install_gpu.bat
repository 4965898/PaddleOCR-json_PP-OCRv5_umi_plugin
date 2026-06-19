@echo off
chcp 65001 >nul
echo ========================================
echo  PP-OCRv6 ONNX Runtime Plugin - GPU Setup
echo ========================================
echo.

REM Check Python
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add to PATH.
    echo Download: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM Check NVIDIA GPU
echo [CHECK] NVIDIA GPU...
nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [ERROR] No NVIDIA GPU or driver detected.
    echo GPU acceleration requires an NVIDIA GPU with up-to-date drivers.
    echo For CPU-only, please run install.bat instead.
    echo.
    pause
    exit /b 1
)
echo [OK] NVIDIA GPU detected
echo.

cd /d "%~dp0"

if not exist "ppocr_v6_env\Scripts\python.exe" (
    echo [1/3] Creating virtual environment ppocr_v6_env ...
    python -m venv ppocr_v6_env
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment already exists, skipping.
)

echo.
echo [2/3] Installing paddleocr + onnxruntime-gpu + CUDA + cuDNN ...
echo This may take 5-15 minutes (downloads ~1.6GB)...
echo.
ppocr_v6_env\Scripts\pip install paddleocr "onnxruntime-gpu[cuda,cudnn]" --upgrade
if errorlevel 1 (
    echo.
    echo [WARNING] Auto-install failed. Please run manually:
    echo   ppocr_v6_env\Scripts\pip install paddleocr "onnxruntime-gpu[cuda,cudnn]"
    echo.
    pause
    exit /b 1
)

echo.
echo [3/3] Verifying GPU support...
ppocr_v6_env\Scripts\python -c "import onnxruntime as ort; ps=ort.get_available_providers(); print('Available providers:', ps); print('CUDA OK!' if 'CUDAExecutionProvider' in ps else 'CUDA NOT available')"

echo.
echo ========================================
echo  GPU Setup Complete!
echo ========================================
echo.
echo Enable "GPU Acceleration" in Umi-OCR plugin settings.
echo First recognition may be slower (GPU init), then ~17x faster.
echo.
echo Please restart Umi-OCR.
echo.
pause
