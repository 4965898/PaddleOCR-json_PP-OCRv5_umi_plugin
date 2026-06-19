@echo off
chcp 65001 >nul
echo ========================================
echo  PP-OCRv6 ONNX Runtime Plugin - Setup
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

cd /d "%~dp0"

echo [1/2] Creating virtual environment ppocr_v6_env ...
python -m venv ppocr_v6_env
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo [2/2] Installing paddleocr + onnxruntime ...
echo This may take 1-3 minutes...
echo.
ppocr_v6_env\Scripts\pip install paddleocr onnxruntime --upgrade
if errorlevel 1 (
    echo.
    echo [WARNING] Auto-install failed. Please run manually:
    echo   ppocr_v6_env\Scripts\pip install paddleocr onnxruntime
    echo.
    echo For GPU acceleration, also install:
    echo   ppocr_v6_env\Scripts\pip install onnxruntime-gpu
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Setup complete!
echo ========================================
echo.
echo Models will be auto-downloaded on first use.
echo Please restart Umi-OCR.
echo.
pause
