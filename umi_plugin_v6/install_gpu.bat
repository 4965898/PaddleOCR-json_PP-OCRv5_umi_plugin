@echo off
chcp 65001 >nul
echo ========================================
echo  PP-OCRv6 ONNX Runtime 插件 - GPU 加速安装
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

REM 检查 NVIDIA GPU
echo [检查] NVIDIA GPU...
nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 NVIDIA GPU 或驱动。
    echo GPU 加速需要 NVIDIA 显卡及最新驱动。
    echo 如仅需 CPU 加速，请运行 install.bat
    echo.
    pause
    exit /b 1
)
echo [OK] 检测到 NVIDIA GPU
echo.

cd /d "%~dp0"

if not exist "ppocr_v6_env\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境 ppocr_v6_env ...
    python -m venv ppocr_v6_env
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败。
        pause
        exit /b 1
    )
) else (
    echo [1/3] 虚拟环境已存在，跳过创建
)

echo.
echo [2/3] 安装 paddleocr + onnxruntime-gpu + CUDA + cuDNN ...
echo 正在安装，请耐心等待（约 5-15 分钟，需下载约 1.6GB）...
echo.
ppocr_v6_env\Scripts\pip install paddleocr "onnxruntime-gpu[cuda,cudnn]" --upgrade
if errorlevel 1 (
    echo.
    echo [警告] 自动安装失败，请手动运行以下命令:
    echo   ppocr_v6_env\Scripts\pip install paddleocr "onnxruntime-gpu[cuda,cudnn]"
    echo.
    pause
    exit /b 1
)

echo.
echo [3/3] 验证 GPU 支持...
ppocr_v6_env\Scripts\python -c "import onnxruntime as ort; ps=ort.get_available_providers(); print('Available providers:', ps); print('CUDA OK!' if 'CUDAExecutionProvider' in ps else 'CUDA NOT available')"

echo.
echo ========================================
echo  GPU 安装完成！
echo ========================================
echo.
echo 在 Umi-OCR 插件设置中勾选「启用GPU」即可使用 GPU 加速。
echo 首次识别会稍慢（GPU 初始化），后续识别速度大幅提升（约 17 倍）。
echo.
echo 请重启 Umi-OCR 以加载插件。
echo.
pause
