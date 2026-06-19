@echo off
REM PP-OCRv6 ONNX Runtime 插件启动入口
REM 按优先级查找 Python 环境：
REM   1. 插件自带的虚拟环境（ppocr_v6_env）
REM   2. 源码仓库的虚拟环境（../../ppocr_v6_env，开发用）
REM   3. 系统 PATH 中的 python
set "SCRIPT_DIR=%~dp0"
set "PYTHON="

if exist "%SCRIPT_DIR%ppocr_v6_env\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%ppocr_v6_env\Scripts\python.exe"
) else if exist "%SCRIPT_DIR%..\..\ppocr_v6_env\Scripts\python.exe" (
    set "PYTHON=%SCRIPT_DIR%..\..\ppocr_v6_env\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

"%PYTHON%" "%SCRIPT_DIR%ppocr_v6_server.py" %*
