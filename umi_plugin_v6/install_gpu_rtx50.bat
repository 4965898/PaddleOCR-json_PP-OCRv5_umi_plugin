@echo off
title PP-OCRv6 ONNX Runtime Plugin - GPU Setup (RTX 50 / CUDA 13)
cd /d "%~dp0"
echo ========================================
echo  PP-OCRv6 ONNX Runtime Plugin - GPU Setup
echo  (RTX 50 series / Blackwell - CUDA 13)
echo ========================================
echo.

REM Why this script exists:
REM   Official onnxruntime-gpu <=1.26 (CUDA 12.8 build) has no sm_120
REM   (Blackwell) kernels, so RTX 50 cards silently fall back to CPU.
REM   Since onnxruntime-gpu 1.27+, the PyPI package is built with CUDA 13
REM   and includes sm_120 kernels. This script installs that CUDA 13 build.
REM Requirements: NVIDIA driver R580+ and Python 3.11+.

REM ---- venv must exist (created by install.bat) ----
if not exist "ppocr_v6_env\Scripts\python.exe" goto :no_python
ppocr_v6_env\Scripts\python.exe -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul
if errorlevel 1 goto :venv_badver

REM ---- NVIDIA driver check: CUDA 13 needs R580+ ----
nvidia-smi >nul 2>nul
if errorlevel 1 goto :no_nvidia
set "DRIVER_MAJOR=0"
for /f "tokens=1 delims=. " %%D in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader 2^>nul') do set "DRIVER_MAJOR=%%D"
echo [INFO] Detected NVIDIA driver version: %DRIVER_MAJOR%.x
if %DRIVER_MAJOR% LSS 580 goto :driver_old

REM ---- Python 3.11+ required (onnxruntime-gpu CUDA 13 builds need it) ----
ppocr_v6_env\Scripts\python.exe -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,11) else 1)" >nul 2>nul
if errorlevel 1 goto :rebuild_venv
goto :install

REM ================================================
REM  Rebuild venv with portable Python 3.12 (via uv)
REM  (only when current venv is Python 3.10)
REM ================================================
:rebuild_venv
echo [INFO] RTX 50 support requires Python 3.11+, but the current virtual
echo        environment uses an older Python. It will be rebuilt with
echo        portable Python 3.12 (downloaded via uv, ~30MB)...
set "OLDENV=ppocr_v6_env_backup"
if exist "%OLDENV%" rmdir /s /q "%OLDENV%" >nul 2>nul
ren ppocr_v6_env "%OLDENV%" >nul 2>nul
if exist "ppocr_v6_env" goto :venv_locked
echo [OK] Old environment moved to %OLDENV% (delete it manually once the new one works).
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $url='https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'; $z=\"$env:TEMP\ppocr_uv.zip\"; $d=\"$env:TEMP\ppocr_uv\"; Write-Host '[1/3] Downloading uv ...'; $px=$null; $ie=Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -ErrorAction SilentlyContinue; if($ie -and $ie.ProxyEnable -eq 1 -and $ie.ProxyServer){ $px=$ie.ProxyServer; if($px -notmatch '://'){$px='http://'+$px} }; $ok=$false; foreach($i in 1..3){ try{ Invoke-WebRequest -Uri $url -OutFile $z -UseBasicParsing; $ok=$true; break }catch{ if($px){ try{ Invoke-WebRequest -Uri $url -OutFile $z -UseBasicParsing -Proxy $px; $ok=$true; break }catch{} }; Start-Sleep -Seconds 3 } }; if(-not $ok){ Write-Host '[ERROR] uv download failed, check network' -ForegroundColor Red; exit 1 }; if(Test-Path $d){Remove-Item $d -Recurse -Force}; Expand-Archive -Path $z -DestinationPath $d -Force; $uv=\"$d\uv.exe\"; if(-not(Test-Path $uv)){Write-Host '[ERROR] uv.exe not found' -ForegroundColor Red; exit 1}; Write-Host '[2/3] Downloading portable Python 3.12 ...'; & $uv python install 3.12; if($LASTEXITCODE -ne 0){Write-Host '[ERROR] Python download failed' -ForegroundColor Red; exit 1}; Write-Host '[3/3] Creating virtual environment ppocr_v6_env ...'; & $uv venv ppocr_v6_env --python 3.12 --seed; if($LASTEXITCODE -ne 0 -or -not(Test-Path 'ppocr_v6_env\Scripts\python.exe')){Write-Host '[ERROR] Failed to create virtual environment' -ForegroundColor Red; exit 1}; Write-Host '[OK] Portable Python 3.12 environment ready' -ForegroundColor Green"
if errorlevel 1 goto :uv_fail
echo.
echo [INFO] Reinstalling base dependencies (paddleocr + onnxruntime)...
ppocr_v6_env\Scripts\python -m pip install paddleocr onnxruntime
if errorlevel 1 goto :deps_fail
goto :install

REM ================================================
REM  Install CUDA 13 build of onnxruntime-gpu
REM ================================================
:install
echo.
echo [1/3] Removing CPU-only onnxruntime and CUDA 12 packages (if installed)...
ppocr_v6_env\Scripts\pip uninstall -y onnxruntime onnxruntime-gpu 2>nul
ppocr_v6_env\Scripts\pip uninstall -y nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cudnn-cu12 2>nul
echo Done.
echo.

echo [2/3] Installing onnxruntime-gpu 1.28+ (CUDA 13, Blackwell sm_120) + CUDA + cuDNN ...
echo This may take 5-15 minutes (downloads ~2GB)...
echo Components installed by pip:
echo   - onnxruntime-gpu >=1.28 (CUDA 13 build, includes sm_120 kernels)
echo   - nvidia-cuda-runtime 13.x (CUDA 13 Runtime)
echo   - nvidia-cudnn-cu13 (cuDNN 9.x for CUDA 13)
echo   - nvidia-cufft, nvidia-curand, etc.
echo No manual CUDA/cuDNN installation is needed.
echo.
ppocr_v6_env\Scripts\pip install paddleocr "onnxruntime-gpu[cuda,cudnn]>=1.28" --upgrade
if errorlevel 1 goto :install_fail

echo.
echo [3/3] Verifying GPU support...
ppocr_v6_env\Scripts\python verify_gpu.py
if errorlevel 1 goto :cuda_fail
echo [OK] CUDAExecutionProvider is available!

echo.
echo ========================================
echo  RTX 50 GPU Setup Complete!
echo ========================================
echo.
echo Enable "GPU Acceleration" in Umi-OCR plugin settings.
echo First recognition may be slower (GPU init), then ~17x faster.
echo.
echo Note: some Conv ops may log "running in Fallback mode" warnings on
echo Blackwell - they still run on GPU via a generic cuDNN codepath.
echo See README (RTX 50 section) for details.
echo.
echo Please restart Umi-OCR.
echo.
pause
exit /b 0

:no_python
echo [ERROR] Virtual environment ppocr_v6_env does not exist.
echo         Please run install.bat first to complete the base installation,
echo         then run this script.
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

:venv_locked
echo [ERROR] Cannot move the old environment ppocr_v6_env (files in use?).
echo         Close Umi-OCR and retry, or delete the folder manually.
echo.
pause
exit /b 1

:uv_fail
echo.
echo [ERROR] Portable Python download failed.
echo         Check your network, or install Python 3.11-3.13 manually and
echo         re-run install.bat, then this script:
echo         https://www.python.org/downloads/
echo.
pause
exit /b 1

:deps_fail
echo.
echo [ERROR] Failed to install base dependencies. Check network and retry, or run:
echo   ppocr_v6_env\Scripts\python -m pip install paddleocr onnxruntime
echo.
pause
exit /b 1

:no_nvidia
echo [ERROR] No NVIDIA GPU or driver detected.
echo GPU acceleration requires an NVIDIA GPU with up-to-date drivers.
echo.
echo Please install the latest NVIDIA driver from:
echo   https://www.nvidia.com/Download/index.aspx
echo.
pause
exit /b 1

:driver_old
echo [ERROR] NVIDIA driver too old for CUDA 13 (driver %DRIVER_MAJOR%.x, need R580+).
echo.
echo RTX 50 series (Blackwell) requires the CUDA 13 build of
echo onnxruntime-gpu, which needs NVIDIA driver R580 or newer.
echo Please update your driver, then re-run this script:
echo   https://www.nvidia.com/Download/index.aspx
echo.
pause
exit /b 1

:install_fail
echo.
echo [WARNING] Auto-install failed. Please run manually:
echo   ppocr_v6_env\Scripts\pip install paddleocr "onnxruntime-gpu[cuda,cudnn]>=1.28"
echo.
pause
exit /b 1

:cuda_fail
echo.
echo [ERROR] CUDAExecutionProvider is NOT available!
echo onnxruntime-gpu was installed but cannot load CUDA/cuDNN DLLs.
echo Common causes:
echo   1. NVIDIA driver is too old (need R580+) - update to latest version
echo   2. CUDA/cuDNN packages failed to download - check network
echo   3. Conflicting onnxruntime CPU version still installed
echo.
echo Driver download: https://www.nvidia.com/Download/index.aspx
echo.
pause
exit /b 1
