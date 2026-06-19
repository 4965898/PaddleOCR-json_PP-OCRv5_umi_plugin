import os
import atexit
import subprocess
from json import loads as jsonLoads, dumps as jsonDumps
from sys import platform as sysPlatform
from base64 import b64encode


class PPOCR_pipe:
    def __init__(self, exePath: str, argument: dict = None):
        """初始化识别器（管道模式）。\n
        `exePath`: 识别器`PaddleOCR_json.exe`的路径。\n
        `argument`: 启动参数，字典`{"键":值}`。参数说明见 https://github.com/hiroi-sora/PaddleOCR-json
        """
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
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo,
        )
        while True:
            if not self.ret.poll() == None:
                raise Exception(f"OCR init fail.")
            initStr = self.ret.stdout.readline().decode("utf-8", errors="ignore")
            if "OCR init completed." in initStr:
                break
        atexit.register(self.exit)

    def runDict(self, writeDict: dict):
        """传入指令字典，发送给引擎进程。\n
        `writeDict`: 指令字典。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        if not self.ret:
            return {"code": 901, "data": f"引擎实例不存在。"}
        if not self.ret.poll() == None:
            return {"code": 902, "data": f"子进程已崩溃。"}
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
        """对一张本地图片进行文字识别。\n
        `exePath`: 图片路径。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        writeDict = {"image_path": imgPath}
        return self.runDict(writeDict)

    def runClipboard(self):
        """立刻对剪贴板第一位的图片进行文字识别。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        return self.run("clipboard")

    def runBase64(self, imageBase64: str):
        """对一张编码为base64字符串的图片进行文字识别。\n
        `imageBase64`: 图片base64字符串。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        writeDict = {"image_base64": imageBase64}
        return self.runDict(writeDict)

    def runBytes(self, imageBytes):
        """对一张图片的字节流信息进行文字识别。\n
        `imageBytes`: 图片字节流。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        imageBase64 = b64encode(imageBytes).decode("utf-8")
        return self.runBase64(imageBase64)

    def exit(self):
        """关闭引擎子进程"""
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

    @staticmethod
    def printResult(res: dict):
        """用于调试，格式化打印识别结果。\n
        `res`: OCR识别结果。"""
        if res["code"] == 100:
            index = 1
            for line in res["data"]:
                print(f"{index}-置信度：{round(line['score'], 2)}，文本：{line['text']}")
                index += 1
        elif res["code"] == 100:
            print("图片中未识别出文字。")
        else:
            print(f"图片识别失败。错误码：{res['code']}，错误信息：{res['data']}")

    def __del__(self):
        self.exit()