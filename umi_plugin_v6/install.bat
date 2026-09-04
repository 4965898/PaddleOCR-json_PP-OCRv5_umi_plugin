@echo off
title PP-OCRv6 ONNX 插件 - 开箱即用安装
cd /d "%~dp0"

echo ========================================
echo  PP-OCRv6 ONNX Runtime Plugin - 安装
echo  开箱即用：无需预装 Python
echo ========================================
echo.

REM 支持的 Python 版本：3.10 - 3.13
REM 原因：paddleocr 依赖链（PyYAML 等）在 3.9- / 3.14+ 尚无预编译
REM wheel，会触发源码编译并失败（issue #14）。
REM 版本不符时本脚本自动改用 uv 下载便携 Python 3.11，用户无需任何操作。

REM ---- 情况一：已有虚拟环境，校验其 Python 版本 ----
if not exist "ppocr_v6_env\Scripts\python.exe" goto :find_python
ppocr_v6_env\Scripts\python.exe -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul
if errorlevel 1 goto :venv_badver
echo [提示] 已检测到虚拟环境 ppocr_v6_env（版本兼容），直接安装/升级依赖。
echo        如需全新安装，请先删除 ppocr_v6_env 文件夹再运行本脚本。
goto :install_deps

:venv_badver
echo [提示] 已有虚拟环境的 Python 版本不受支持（需 3.10-3.13），
echo        paddleocr 依赖链在该版本缺少预编译包，安装会失败。
echo        将自动改用便携 Python 3.11 重建环境，无需手动操作...
set "OLDENV=ppocr_v6_env_backup"
if exist "%OLDENV%" rmdir /s /q "%OLDENV%" >nul 2>nul
ren ppocr_v6_env "%OLDENV%" >nul 2>nul
if exist "ppocr_v6_env" goto :venv_locked
echo [OK] 旧环境已移至 %OLDENV%（新环境确认可用后可手动删除）。
goto :uv_setup

:venv_locked
echo [ERROR] 无法移动旧环境 ppocr_v6_env（可能被占用）。
echo         请关闭 Umi-OCR 后重试，或手动删除该文件夹。
pause
exit /b 1

REM ---- 情况二：无虚拟环境，检测系统 Python ----
:find_python
set "SYS_PYTHON="
for %%P in (python python3) do (
    where %%P >nul 2>nul && ( set "SYS_PYTHON=%%P" & goto :check_pyver )
)

REM ---- 无系统 Python：通过 uv 自动下载便携 Python 3.11 ----
echo 未检测到系统 Python。
goto :uv_setup

:check_pyver
REM 排除 Windows Store 占位 python（能 where 到但无法运行）
%SYS_PYTHON% -c "import sys" >nul 2>nul
if errorlevel 1 goto :py_stub
%SYS_PYTHON% -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul
if errorlevel 1 goto :py_unsupported
echo [OK] 检测到系统 Python，创建虚拟环境...
%SYS_PYTHON% -m venv ppocr_v6_env
if errorlevel 1 goto :venv_fail
goto :install_deps

:py_stub
echo 检测到 python 命令但无法运行（可能为 Microsoft Store 占位程序）。
goto :uv_setup

:py_unsupported
set "SYSVER=未知"
for /f "delims=" %%V in ('%SYS_PYTHON% -c "import sys; print(sys.version.split()[0])" 2^>nul') do set "SYSVER=%%V"
echo [提示] 系统 Python 版本为 %SYSVER%，不在支持范围（3.10-3.13）。
echo        paddleocr 依赖链（PyYAML 等）在该版本缺少预编译包，安装会失败。
echo        将自动改用便携 Python 3.11（无需手动安装任何东西）...
goto :uv_setup

:uv_setup
echo.
echo 通过 uv 自动下载便携 Python 3.11（首次约 30MB）...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $url='https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'; $z=\"$env:TEMP\ppocr_uv.zip\"; $d=\"$env:TEMP\ppocr_uv\"; Write-Host '[1/3] 下载 uv 工具...'; $px=$null; $ie=Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction SilentlyContinue; if($ie -and $ie.ProxyEnable -eq 1 -and $ie.ProxyServer){ $px=$ie.ProxyServer; if($px -notmatch '://'){$px='http://'+$px} }; $ok=$false; foreach($i in 1..3){ try{ Invoke-WebRequest -Uri $url -OutFile $z -UseBasicParsing; $ok=$true; break }catch{ if($px){ try{ Invoke-WebRequest -Uri $url -OutFile $z -UseBasicParsing -Proxy $px; $ok=$true; break }catch{} }; Start-Sleep -Seconds 3 } }; if(-not $ok){ Write-Host '[ERROR] uv 下载失败，请检查网络' -ForegroundColor Red; exit 1 }; if(Test-Path $d){Remove-Item $d -Recurse -Force}; Expand-Archive -Path $z -DestinationPath $d -Force; $uv=\"$d\uv.exe\"; if(-not(Test-Path $uv)){Write-Host '[ERROR] uv.exe 未找到' -ForegroundColor Red; exit 1}; Write-Host '[2/3] 下载便携 Python 3.11...'; & $uv python install 3.11; if($LASTEXITCODE -ne 0){Write-Host '[ERROR] Python 下载失败' -ForegroundColor Red; exit 1}; Write-Host '[3/3] 创建虚拟环境 ppocr_v6_env（含 pip）...'; & $uv venv ppocr_v6_env --python 3.11 --seed; if($LASTEXITCODE -ne 0 -or -not(Test-Path 'ppocr_v6_env\Scripts\python.exe')){Write-Host '[ERROR] 虚拟环境创建失败' -ForegroundColor Red; exit 1}; Write-Host '[OK] 便携 Python 3.11 环境就绪' -ForegroundColor Green"
if errorlevel 1 goto :uv_fail
goto :install_deps

:uv_fail
echo.
echo [ERROR] 便携 Python 自动下载失败。
echo         请检查网络连接，或手动安装 Python 3.10-3.13 后重试：
echo         https://www.python.org/downloads/
echo.
pause
exit /b 1

:venv_fail
echo [WARNING] 系统 Python 创建虚拟环境失败，改用便携 Python 3.11...
if exist "ppocr_v6_env" rmdir /s /q "ppocr_v6_env" >nul 2>nul
goto :uv_setup

:install_deps
echo.
echo 安装依赖 paddleocr + onnxruntime（约 200MB，需 1-3 分钟）...
echo.
ppocr_v6_env\Scripts\python -m pip install paddleocr onnxruntime --upgrade
if errorlevel 1 goto :deps_fail

echo.
echo 验证安装...
ppocr_v6_env\Scripts\python -c "import paddleocr, onnxruntime; print('[OK] paddleocr + onnxruntime 导入成功')" 2>nul
if errorlevel 1 goto :verify_warn
set "VENVVER="
for /f "delims=" %%V in ('ppocr_v6_env\Scripts\python -c "import sys; print(sys.version.split()[0])" 2^>nul') do set "VENVVER=%%V"
echo.
echo ========================================
echo  安装完成！Python %VENVVER%
echo ========================================
goto :done

:verify_warn
echo [WARNING] 验证未完全通过，但安装脚本已完成。请重启 Umi-OCR 测试。
goto :done

:deps_fail
echo.
echo [ERROR] 依赖安装失败。请检查网络后重试，或手动执行：
echo   ppocr_v6_env\Scripts\python -m pip install paddleocr onnxruntime
echo.
pause
exit /b 1

:done
echo.
echo 首次识别会自动下载 PP-OCRv6 模型（10-50MB）。
echo 如需 GPU 加速，另行运行：
echo   install_gpu.bat        NVIDIA 显卡（CUDA）
echo   install_gpu_rtx50.bat  NVIDIA RTX 50 系（Blackwell）
echo   install_directml.bat   Intel Arc / AMD（DirectML）
echo.
echo 请重启 Umi-OCR。
echo.
pause
