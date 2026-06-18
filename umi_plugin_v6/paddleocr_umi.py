from call_func import CallFunc

from .ppocr_pipe import PPOCR_pipe

import os
import logging
import psutil

logger = logging.getLogger("Umi-OCR")

# 关键：使用 .bat 代替 .exe
ExePath = os.path.dirname(os.path.abspath(__file__)) + "/PaddleOCR-json.bat"

ExeConfigs = [
    ("config_path", "language"),
    ("det", "det"),
    ("cls", "cls"),
    ("rec_batch_num", "rec_batch_num"),
    ("limit_side_len", "limit_side_len"),
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
    heights = []
    for cx, cy, item in itemsWithCenter:
        box = item["box"]
        h = abs(box[2][1] - box[0][1])
        if h > 0:
            heights.append(h)
    if not heights:
        return data
    avgHeight = sum(heights) / len(heights)
    threshold = avgHeight * 0.5
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
    logger.info(f"[VerticalText] 重排序: {len(data)}项 -> {len(result)}项, {len(columns)}列, 阈值={threshold:.1f}")
    return result


class Api:
    def __init__(self, globalArgd):
        if not os.path.exists(ExePath):
            raise ValueError(f'[Error] Exe path "{ExePath}" does not exist.')
        self.api = None
        self.exeConfigs = {}
        self.launchConfigs = {}
        self.engineSign = None
        self.verticalText = False
        self._updateExeConfigs(self.exeConfigs, globalArgd)
        if "vertical_text" in globalArgd:
            self.verticalText = bool(globalArgd["vertical_text"])
        self.ramInfo = {"max": -1, "time": -1, "timerID": ""}
        m = globalArgd.get("ram_max", -1)
        if isinstance(m, (int, float)):
            self.ramInfo["max"] = m
        m = globalArgd.get("ram_time", -1)
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

    def _makeEngineSign(self, exeConfigs):
        return tuple(sorted(exeConfigs.items()))

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
        if "vertical_text" in argd:
            self.verticalText = bool(argd["vertical_text"])
        newSign = self._makeEngineSign(tempConfigs)
        if not self.api == None:
            if newSign == self.engineSign:
                return ""
            self.stop()
        self.exeConfigs = tempConfigs
        try:
            self.api = PPOCR_pipe(ExePath, tempConfigs)
            self.launchConfigs = tempConfigs
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
            logger.error(f"重启引擎失败: {e}")

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
