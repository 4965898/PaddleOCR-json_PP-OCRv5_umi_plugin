@echo off
chcp 65001 >nul
echo ========================================
echo  PP-OCRv6 ONNX Runtime 插件 - 环境安装
echo ========================================
echo.

REM 检查 Python
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 Python。请先安装 Python 3.10+ 并添加到系统 PATH。
    echo 下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

cd /d "%~dp0"

echo [1/2] 创建虚拟环境 ppocr_v6_env ...
python -m venv ppocr_v6_env
if errorlevel 1 (
    echo [错误] 创建虚拟环境失败。
    pause
    exit /b 1
)

echo.
echo [2/2] 安装依赖 paddleocr + onnxruntime ...
echo 正在安装，请耐心等待（约 1-3 分钟）...
echo.
ppocr_v6_env\Scripts\pip install paddleocr onnxruntime --upgrade
if errorlevel 1 (
    echo.
    echo [警告] 自动安装失败，请手动运行以下命令:
    echo   ppocr_v6_env\Scripts\pip install paddleocr onnxruntime
    echo.
    echo 若需 GPU 加速，可额外安装: ppocr_v6_env\Scripts\pip install onnxruntime-gpu
    echo （需要 CUDA 12 + cuDNN 9 运行库）
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  安装完成！
echo ========================================
echo.
echo 首次使用时，插件会自动下载所选尺寸的 PP-OCRv6 ONNX 模型到 models 目录。
echo 请重启 Umi-OCR 以加载插件。
echo.
pause
