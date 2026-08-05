@echo off
title PP-OCRv6 ONNX 插件 - 开箱即用安装
cd /d "%~dp0"

echo ========================================
echo  PP-OCRv6 ONNX Runtime Plugin - 安装
echo  开箱即用：无需预装 Python
echo ========================================
echo.

REM 已有虚拟环境：跳过创建，直接升级依赖
if exist "ppocr_v6_env\Scripts\python.exe" (
    echo [提示] 已检测到虚拟环境 ppocr_v6_env，跳过创建。
    echo        如需全新安装，请先删除 ppocr_v6_env 文件夹再运行本脚本。
    goto :install_deps
)

REM ---- 检测系统 Python ----
set "SYS_PYTHON="
for %%P in (python python3) do (
    where %%P >nul 2>nul && ( set "SYS_PYTHON=%%P" & goto :got_python )
)

REM ---- 无系统 Python：通过 uv 自动下载便携 Python 3.11 ----
echo 未检测到系统 Python。
echo 将自动下载便携 Python 3.11（via uv，首次约 30MB）...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $url='https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'; $z=\"$env:TEMP\ppocr_uv.zip\"; $d=\"$env:TEMP\ppocr_uv\"; Write-Host '[1/3] 下载 uv 工具...'; try { Invoke-WebRequest -Uri $url -OutFile $z -UseBasicParsing } catch { Write-Host '[ERROR] uv 下载失败，请检查网络' -ForegroundColor Red; exit 1 }; if(Test-Path $d){Remove-Item $d -Recurse -Force}; Expand-Archive -Path $z -DestinationPath $d -Force; $uv=\"$d\uv.exe\"; if(-not(Test-Path $uv)){Write-Host '[ERROR] uv.exe 未找到' -ForegroundColor Red; exit 1}; Write-Host '[2/3] 下载便携 Python 3.11...'; & $uv python install 3.11; if($LASTEXITCODE -ne 0){Write-Host '[ERROR] Python 下载失败' -ForegroundColor Red; exit 1}; Write-Host '[3/3] 创建虚拟环境 ppocr_v6_env...'; & $uv venv ppocr_v6_env --python 3.11; if($LASTEXITCODE -ne 0 -or -not(Test-Path 'ppocr_v6_env\Scripts\python.exe')){Write-Host '[ERROR] 虚拟环境创建失败' -ForegroundColor Red; exit 1}; Write-Host '[OK] 便携 Python 环境就绪' -ForegroundColor Green"
if errorlevel 1 (
    echo.
    echo [ERROR] 便携 Python 自动下载失败。
    echo         请检查网络连接，或手动安装 Python 3.10+ 后重试：
    echo         https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
goto :install_deps

:got_python
echo [OK] 检测到系统 Python，创建虚拟环境...
%SYS_PYTHON% -m venv ppocr_v6_env
if errorlevel 1 (
    echo [ERROR] 虚拟环境创建失败
    pause
    exit /b 1
)

:install_deps
echo.
echo 安装依赖 paddleocr + onnxruntime（约 200MB，需 1-3 分钟）...
echo.
ppocr_v6_env\Scripts\pip install paddleocr onnxruntime --upgrade
if errorlevel 1 (
    echo.
    echo [ERROR] 依赖安装失败。请检查网络后重试，或手动执行：
    echo   ppocr_v6_env\Scripts\pip install paddleocr onnxruntime
    echo.
    pause
    exit /b 1
)

echo.
echo 验证安装...
ppocr_v6_env\Scripts\python -c "import paddleocr, onnxruntime; print('[OK] paddleocr + onnxruntime 导入成功')" 2>nul
if errorlevel 1 (
    echo [WARNING] 验证未完全通过，但安装脚本已完成。请重启 Umi-OCR 测试。
) else (
    echo.
    echo ========================================
    echo  安装完成！
    echo ========================================
    echo.
    echo 首次识别会自动下载 PP-OCRv6 模型（10-50MB）。
    echo 如需 GPU 加速，另行运行：
    echo   install_gpu.bat      NVIDIA 显卡（CUDA）
    echo   install_directml.bat Intel Arc / AMD（DirectML）
    echo.
    echo 请重启 Umi-OCR。
)
echo.
pause
