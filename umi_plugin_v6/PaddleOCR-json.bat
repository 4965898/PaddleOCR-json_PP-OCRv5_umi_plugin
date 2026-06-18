@echo off
REM PP-OCRv6 ONNX Runtime 插件启动入口
REM 优先使用插件自带的虚拟环境（ppocr_v6_env），其次使用系统 Python
if exist "%~dp0ppocr_v6_env\Scripts\python.exe" (
    "%~dp0ppocr_v6_env\Scripts\python.exe" "%~dp0ppocr_v6_server.py" %*
) else (
    python "%~dp0ppocr_v6_server.py" %*
)
