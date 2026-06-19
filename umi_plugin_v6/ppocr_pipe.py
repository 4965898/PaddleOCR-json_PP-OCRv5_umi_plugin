import os
import atexit
import subprocess
import threading
from json import loads as jsonLoads, dumps as jsonDumps
from sys import platform as sysPlatform
from base64 import b64encode


class PPOCR_pipe:
    def __init__(self, exePath: str, argument: dict = None):
        exePath = os.path.abspath(exePath)
        cwd = os.path.abspath(os.path.join(exePath, os.pardir))
        cmds = [exePath]
        if isinstance(argument, dict):
            for key, value in argument.items():
                if isinstance(value, bool):
                    cmds += [f"--{key}={value}"]
                elif isinstance(value, str):
                    cmds += [f"--{key}", value]
                else:
                    cmds += [f"--{key}", str(value)]
        self.ret = None
        self._stderr_lines = []
        startupinfo = None
        if "win32" in str(sysPlatform).lower():
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags = (
                subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
            )
            startupinfo.wShowWindow = subprocess.SW_HIDE
        self.ret = subprocess.Popen(
            cmds,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
        )
        # 守护线程持续读取 stderr，防止缓冲区满阻塞子进程，并保留最近日志便于排错
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        while True:
            if not self.ret.poll() == None:
                self._stderr_thread.join(timeout=1.0)
                err_msg = "".join(self._stderr_lines).strip()
                raise Exception(f"OCR init fail. stderr: {err_msg}")
            initStr = self.ret.stdout.readline().decode("utf-8", errors="ignore")
            if "OCR init completed." in initStr:
                break
        atexit.register(self.exit)

    def _drain_stderr(self):
        """持续读取子进程 stderr，防止缓冲区满导致子进程阻塞"""
        try:
            for line in iter(self.ret.stderr.readline, b""):
                self._stderr_lines.append(line.decode("utf-8", errors="ignore"))
                if len(self._stderr_lines) > 50:
                    self._stderr_lines.pop(0)
        except Exception:
            pass

    def runDict(self, writeDict: dict):
        if not self.ret:
            return {"code": 901, "data": f"引擎实例不存在。"}
        if not self.ret.poll() == None:
            err_msg = "".join(self._stderr_lines).strip()
            return {"code": 902, "data": f"子进程已崩溃。stderr: {err_msg}"}
        writeStr = jsonDumps(writeDict, ensure_ascii=True, indent=None) + "\n"
        try:
            self.ret.stdin.write(writeStr.encode("utf-8"))
            self.ret.stdin.flush()
        except Exception as e:
            return {"code": 902, "data": f"向识别器进程传入指令失败，疑似子进程已崩溃。{e}"}
        try:
            getStr = self.ret.stdout.readline().decode("utf-8", errors="ignore")
        except Exception as e:
            return {"code": 903, "data": f"读取识别器进程输出值失败。异常信息：[{e}]"}
        try:
            return jsonLoads(getStr)
        except Exception as e:
            return {"code": 904, "data": f"识别器输出值反序列化JSON失败。异常信息：[{e}]。原始内容：[{getStr}]"}

    def run(self, imgPath: str):
        writeDict = {"image_path": imgPath}
        return self.runDict(writeDict)

    def runBase64(self, imageBase64: str):
        writeDict = {"image_base64": imageBase64}
        return self.runDict(writeDict)

    def runBytes(self, imageBytes):
        imageBase64 = b64encode(imageBytes).decode("utf-8")
        return self.runBase64(imageBase64)

    def exit(self):
        if hasattr(self, "ret"):
            if not self.ret:
                return
            try:
                self.ret.kill()
            except Exception as e:
                print(f"[Error] ret.kill() {e}")
        self.ret = None
        atexit.unregister(self.exit)
        print("###  PPOCR引擎子进程关闭！")

    def __del__(self):
        self.exit()
