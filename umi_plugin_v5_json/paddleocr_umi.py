from call_func import CallFunc

from .paddleocr import PPOCR_pipe

import os
import logging
import psutil

logger = logging.getLogger("Umi-OCR")

ExePath = os.path.dirname(os.path.abspath(__file__)) + "/PaddleOCR-json.exe"

ExeConfigs = [
    ("enable_mkldnn", "enable_mkldnn"),
    ("config_path", "language"),
    ("det", "det"),
    ("cls", "cls"),
    ("use_angle_cls", "cls"),
    ("cpu_threads", "cpu_threads"),
    ("rec_batch_num", "rec_batch_num"),
    ("use_tensorrt", "use_tensorrt"),
    ("precision", "precision"),
    ("gpu_id", "gpu_id"),
]


def _boxCenter(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _reorderVertical(data):
    if not isinstance(data, list) or len(data) <= 1:
        return data
    itemsWithCenter = []
    for item in data:
        if not isinstance(item, dict) or "box" not in item:
            continue
        try:
            cx, cy = _boxCenter(item["box"])
            itemsWithCenter.append((cx, cy, item))
        except Exception:
            continue
    if len(itemsWithCenter) <= 1:
        return data
    widths = []
    for cx, cy, item in itemsWithCenter:
        box = item["box"]
        w = abs(box[1][0] - box[0][0])
        if w > 0:
            widths.append(w)
    if not widths:
        return data
    avgWidth = sum(widths) / len(widths)
    threshold = avgWidth * 0.6
    itemsWithCenter.sort(key=lambda x: x[0])
    columns = []
    currentCol = [itemsWithCenter[0]]
    for i in range(1, len(itemsWithCenter)):
        if abs(itemsWithCenter[i][0] - currentCol[0][0]) < threshold:
            currentCol.append(itemsWithCenter[i])
        else:
            columns.append(currentCol)
            currentCol = [itemsWithCenter[i]]
    columns.append(currentCol)
    columns.sort(key=lambda col: col[0][0], reverse=True)
    result = []
    for col in columns:
        col.sort(key=lambda x: x[1])
        for cx, cy, item in col:
            result.append(item)
    logger.info(f"[VerticalText] 重排序: {len(data)}项 -> {len(result)}项, {len(columns)}列")
    return result


class Api:
    def __init__(self, globalArgd):
        if not os.path.exists(ExePath):
            raise ValueError(f'[Error] Exe path "{ExePath}" does not exist.')
        self.api = None
        self.exeConfigs = {}
        self.launchConfigs = {}
        self.engineSign = None
        self.runDeviceMode = "cpu"
        self.deviceMode = {"infer_mode": "cpu", "gpu_id": 0}
        self.verticalText = False
        self._updateExeConfigs(self.exeConfigs, globalArgd)
        self._updateDeviceMode(self.deviceMode, globalArgd)
        if "vertical_text" in globalArgd:
            self.verticalText = bool(globalArgd["vertical_text"])
        self.ramInfo = {"max": -1, "time": -1, "timerID": ""}
        m = globalArgd["ram_max"]
        if isinstance(m, (int, float)):
            self.ramInfo["max"] = m
        m = globalArgd["ram_time"]
        if isinstance(m, (int, float)):
            self.ramInfo["time"] = m
        self.isInit = True

    def _updateExeConfigs(self, target, data):
        for c in ExeConfigs:
            if c[1] in data:
                target[c[0]] = data[c[1]]
        self._updateLimitSideLen(target, data)

    def _updateLimitSideLen(self, target, data):
        if "limit_side_len" not in data:
            return
        sideLen = data["limit_side_len"]
        if sideLen == "custom":
            custom = data.get("limit_side_len_custom")
            if isinstance(custom, int) and custom >= 32:
                target["limit_side_len"] = custom
            else:
                target["limit_side_len"] = 960
        else:
            target["limit_side_len"] = sideLen

    def _updateDeviceMode(self, target, data):
        mode = data.get("infer_mode")
        if isinstance(mode, str):
            mode = mode.lower()
            if mode in ("cpu", "gpu", "hybrid"):
                target["infer_mode"] = mode
        gpuID = data.get("gpu_id")
        if isinstance(gpuID, int) and gpuID >= 0:
            target["gpu_id"] = gpuID

    def _buildLaunchConfigs(self, baseConfigs, useGPU):
        cfg = baseConfigs.copy()
        cfg["use_gpu"] = bool(useGPU)
        if useGPU:
            cfg["gpu_id"] = self.deviceMode["gpu_id"]
        else:
            cfg.pop("gpu_id", None)
            cfg.pop("use_tensorrt", None)
            cfg.pop("precision", None)
        return cfg

    def _makeEngineSign(self, exeConfigs, deviceMode):
        return (
            tuple(sorted(exeConfigs.items())),
            deviceMode["infer_mode"],
            deviceMode["gpu_id"],
        )

    def _postProcess(self, res):
        if not self.verticalText:
            return res
        if res.get("code") != 100:
            return res
        if not isinstance(res.get("data"), list):
            return res
        logger.info(f"[VerticalText] 启用竖排重排序, {len(res['data'])}个文本块")
        res["data"] = _reorderVertical(res["data"])
        return res

    def start(self, argd):
        tempConfigs = self.exeConfigs.copy()
        self._updateExeConfigs(tempConfigs, argd)
        tempMode = self.deviceMode.copy()
        self._updateDeviceMode(tempMode, argd)
        if "vertical_text" in argd:
            self.verticalText = bool(argd["vertical_text"])
        logger.info(f"[VerticalText] start: vertical_text={self.verticalText}")
        newSign = self._makeEngineSign(tempConfigs, tempMode)
        if not self.api == None:
            if newSign == self.engineSign:
                return ""
            self.stop()
        self.exeConfigs = tempConfigs
        self.deviceMode = tempMode
        mode = self.deviceMode["infer_mode"]
        try:
            if mode == "gpu":
                launch = self._buildLaunchConfigs(tempConfigs, True)
                self.api = PPOCR_pipe(ExePath, launch)
                self.launchConfigs = launch
                self.runDeviceMode = "gpu"
            elif mode == "hybrid":
                try:
                    launch = self._buildLaunchConfigs(tempConfigs, True)
                    self.api = PPOCR_pipe(ExePath, launch)
                    self.launchConfigs = launch
                    self.runDeviceMode = "gpu"
                except Exception:
                    logger.warning("[VerticalText] GPU初始化失败，自动回退到CPU模式。")
                    launch = self._buildLaunchConfigs(tempConfigs, False)
                    self.api = PPOCR_pipe(ExePath, launch)
                    self.launchConfigs = launch
                    self.runDeviceMode = "cpu"
            else:
                launch = self._buildLaunchConfigs(tempConfigs, False)
                self.api = PPOCR_pipe(ExePath, launch)
                self.launchConfigs = launch
                self.runDeviceMode = "cpu"
        except Exception as e:
            self.api = None
            return f"[Error] OCR init fail. Argd: {tempConfigs}\n{e}"
        self.engineSign = newSign
        return ""

    def stop(self):
        if self.api == None:
            return
        self.api.exit()
        self.api = None

    def runPath(self, imgPath: str):
        self.__runBefore()
        res = self.api.run(imgPath)
        res = self._postProcess(res)
        self.__ramClear()
        return res

    def runBytes(self, imageBytes):
        self.__runBefore()
        res = self.api.runBytes(imageBytes)
        res = self._postProcess(res)
        self.__ramClear()
        return res

    def runBase64(self, imageBase64):
        self.__runBefore()
        res = self.api.runBase64(imageBase64)
        res = self._postProcess(res)
        self.__ramClear()
        return res

    def __runBefore(self):
        CallFunc.delayStop(self.ramInfo["timerID"])

    def _restart(self):
        self.stop()
        try:
            self.api = PPOCR_pipe(ExePath, self.launchConfigs)
        except Exception as e:
            self.api = None
            logger.error(f"[VerticalText] 重启引擎失败: {e}")

    def __ramClear(self):
        if self.ramInfo["max"] > 0:
            pid = self.api.ret.pid
            rss = psutil.Process(pid).memory_info().rss
            rss /= 1048576
            if rss > self.ramInfo["max"]:
                self._restart()
        if self.ramInfo["time"] > 0:
            self.ramInfo["timerID"] = CallFunc.delay(
                self._restart, self.ramInfo["time"]
            )