from call_func import CallFunc
from plugin_i18n import Translator

from .ppocr_pipe import PPOCR_pipe

import os
import logging
import psutil
from base64 import b64encode

logger = logging.getLogger("Umi-OCR")

# i18n：与 paddleocr_config.py 共用同一份 i18n.csv
tr = Translator(__file__, "i18n.csv")

# 关键：使用 .bat 代替 .exe
ExePath = os.path.dirname(os.path.abspath(__file__)) + "/PaddleOCR-json.bat"

ExeConfigs = [
    ("config_path", "language"),
    ("det", "det"),
    ("cls", "cls"),
    ("rec_batch_num", "rec_batch_num"),
    ("limit_side_len", "limit_side_len"),
    ("use_gpu", "use_gpu"),
    # CPU 推理线程数：0=自动（全部核心），调小可降低内存占用
    ("cpu_threads", "cpu_threads"),
    # A1: det 框内缩比例，传给 server.py，0=关闭
    ("shrink_poly_ratio", "shrink_poly_ratio"),
    # 表格识别：开关 + 输出格式（html / tsv / off）
    # 键名用 w1_/w2_ 前缀让字母序排在 vertical_text 之后（QVariantMap 按键名排序）
    ("w1_table_mode", "table_mode"),
    ("w2_table_format", "table_format"),
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
    logger.info(tr("[VerticalText] 重排序: {0}项 -> {1}项, {2}列, 阈值={3:.1f}").format(len(data), len(result), len(columns), threshold))
    return result


class Api:
    def __init__(self, globalArgd):
        if not os.path.exists(ExePath):
            raise ValueError(tr('[Error] 引擎路径不存在: "{0}"').format(ExePath))
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
        self._updateShrinkPolyRatio(target, data)

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

    def _updateShrinkPolyRatio(self, target, data):
        # 处理 "PDF文本层精对齐" 选择 "自定义" 的情况：
        # optionsList 里 "custom" 字符串先被 ExeConfigs 循环填入 target["shrink_poly_ratio"]，
        # 这里再用 shrink_poly_ratio_custom 的浮点值覆盖。与 _updateLimitSideLen 同模式。
        if "shrink_poly_ratio" not in data:
            return
        ratio = data["shrink_poly_ratio"]
        if ratio == "custom":
            custom = data.get("shrink_poly_ratio_custom")
            if isinstance(custom, (int, float)) and custom >= 0.0:
                target["shrink_poly_ratio"] = float(custom)
            else:
                target["shrink_poly_ratio"] = 0.08
        else:
            target["shrink_poly_ratio"] = ratio

    def _makeEngineSign(self, exeConfigs):
        return tuple(sorted(exeConfigs.items()))

    def _postProcess(self, res):
        if not self.verticalText:
            return res
        if res.get("code") != 100:
            return res
        if not isinstance(res.get("data"), list):
            return res
        logger.info(tr("[VerticalText] 启用竖排重排序, {0}个文本块").format(len(res['data'])))
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
            return tr("[Error] OCR 初始化失败。配置: {0}; {1}").format(tempConfigs, e)
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

    # --- 表格识别（可选高级功能）---
    # 返回结构：{"code": 100, "data": {"html": "<table>...</table>", "tables": [...]}}
    # 表格识别不使用竖排重排后处理；模型按需懒加载，首次调用约 10~30 秒。
    def runTablePath(self, imgPath: str):
        self.__runBefore()
        res = self.api.runDict({"image_path": imgPath, "table": True})
        self.__ramClear()
        return res

    def runTableBytes(self, imageBytes):
        self.__runBefore()
        imageBase64 = b64encode(imageBytes).decode("utf-8")
        res = self.api.runDict({"image_base64": imageBase64, "table": True})
        self.__ramClear()
        return res

    def runTableBase64(self, imageBase64):
        self.__runBefore()
        res = self.api.runDict({"image_base64": imageBase64, "table": True})
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
            logger.error(tr("重启引擎失败: {0}").format(e))

    def __ramClear(self):
        if self.ramInfo["max"] > 0:
            # 子进程可能已崩溃被 exit() 置 None（用户日志中的 AttributeError 来源）
            if self.api is None or getattr(self.api, "ret", None) is None:
                return
            pid = self.api.ret.pid
            rss = psutil.Process(pid).memory_info().rss
            rss /= 1048576
            if rss > self.ramInfo["max"]:
                self._restart()
        if self.ramInfo["time"] > 0:
            self.ramInfo["timerID"] = CallFunc.delay(
                self._restart, self.ramInfo["time"]
            )