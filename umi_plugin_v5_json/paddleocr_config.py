import os
import psutil
from plugin_i18n import Translator

tr = Translator(__file__, "i18n.csv")

MODELS_CONFIGS = "/models/configs.txt"


def _getlanguageList():
    """configs.txt 格式示例：
    config_chinese.txt 简体中文
    config_en.txt English
    """
    optionsList = []
    configsPath = os.path.dirname(os.path.abspath(__file__)) + MODELS_CONFIGS
    try:
        with open(configsPath, "r", encoding="utf-8") as file:
            content = file.read()
            lines = content.split("\n")
            for l in lines:
                parts = l.split(" ", 1)
                if len(parts) == 2 and parts[0].strip():
                    optionsList.append([f"models/{parts[0].strip()}", parts[1].strip()])
        return optionsList
    except FileNotFoundError:
        print("[Error] PPOCR配置文件configs不存在，请检查文件路径是否正确。", configsPath)
    except IOError:
        print("[Error] PPOCR配置文件configs无法打开或读取。")
    return []


_LanguageList = _getlanguageList()


def _getThreads():
    try:
        phyCore = psutil.cpu_count(logical=False)
        lgiCore = psutil.cpu_count(logical=True)
        if (
            not isinstance(phyCore, int)
            or not isinstance(lgiCore, int)
            or lgiCore < phyCore
        ):
            raise ValueError("核心数计算异常")
        if phyCore * 2 == lgiCore or phyCore == lgiCore:
            threads = lgiCore
            if threads > 16:
                return 16
            return threads
        big = lgiCore - phyCore
        threads = big * 2
        if threads > 16:
            return 16
        return threads
    except Exception as e:
        print("[Warning] 无法获取CPU核心数！", e)
        return 4


_threads = _getThreads()


def _getRamMax():
    ramMax = 1024
    try:
        totalMemoryBytes = psutil.virtual_memory().total
        ramMax *= 0.5
        ramMax = totalMemoryBytes / 1048576
        ramMax = int(ramMax)
    except Exception as e:
        print("[Warning] 无法获取系统总内存数！", e)
    if ramMax < 512:
        ramMax = 512
    elif ramMax > 8192:
        ramMax = 8192
    return ramMax


_ramMax = _getRamMax()

globalOptions = {
    "title": tr("PaddleOCR（PP-OCRv5）"),
    "type": "group",
    "infer_mode": {
        "title": tr("推理设备模式"),
        "optionsList": [
            ["cpu", tr("仅CPU")],
            ["gpu", tr("仅GPU")],
            ["hybrid", tr("CPU+GPU混合（推荐）")],
        ],
        "toolTip": tr("混合模式会优先尝试GPU，若启动失败则自动回退CPU，兼顾速度与稳定性。仅GPU模式需要GPU版构建的PaddleOCR-json.exe。"),
    },
    "gpu_id": {
        "title": tr("GPU编号"),
        "default": 0,
        "min": 0,
        "isInt": True,
    },
    "enable_mkldnn": {
        "title": tr("启用MKL-DNN加速"),
        "default": True,
        "toolTip": tr("使用MKL-DNN数学库提高神经网络的计算速度。能大幅加快CPU推理速度，但也会增加内存占用。"),
    },
    "cpu_threads": {
        "title": tr("线程数"),
        "default": _threads,
        "min": 1,
        "isInt": True,
        "toolTip": tr("CPU推理线程数。不要盲目拉满，建议8~16间测试最优值。"),
    },
    "use_tensorrt": {
        "title": tr("启用TensorRT加速"),
        "default": False,
        "toolTip": tr("使用TensorRT加速GPU推理，可显著提升GPU推理速度。需要GPU版构建的PaddleOCR-json.exe及TensorRT环境。"),
    },
    "precision": {
        "title": tr("推理精度"),
        "optionsList": [
            ["fp32", tr("FP32（默认）")],
            ["fp16", tr("FP16（更快）")],
        ],
        "toolTip": tr("FP16精度可加速GPU推理，但可能略微降低精度。仅GPU模式有效。"),
    },
    "ram_max": {
        "title": tr("内存占用限制"),
        "default": _ramMax,
        "min": -1,
        "unit": "MB",
        "isInt": True,
        "toolTip": tr("值>0时启用。引擎内存占用超过该值时，执行内存清理。"),
    },
    "ram_time": {
        "title": tr("内存闲时清理"),
        "default": 60,
        "min": -1,
        "unit": tr("秒"),
        "isInt": True,
        "toolTip": tr("值>0时启用。引擎空闲时间超过该值时，执行内存清理。"),
    },
}

localOptions = {
    "title": tr("PaddleOCR PP-OCRv5"),
    "type": "group",
    "language": {
        "title": tr("语言/模型库"),
        "optionsList": _LanguageList,
    },
    "det": {
        "title": tr("启用文本检测"),
        "default": True,
        "toolTip": tr("启用det目标检测。若图片中只含一行文本且无空白区域，可关闭det以加快速度。"),
    },
    "cls": {
        "title": tr("纠正文本方向"),
        "default": False,
        "toolTip": tr("启用方向分类，识别倾斜或倒置的文本。可能降低识别速度。"),
    },
    "rec_batch_num": {
        "title": tr("识别批处理数"),
        "default": 6,
        "min": 1,
        "isInt": True,
        "toolTip": tr("识别模型批处理大小。增大可提高吞吐量，但会增加内存/显存占用。"),
    },
    "vertical_text": {
        "title": tr("竖排文字模式"),
        "default": False,
        "toolTip": tr("启用后，识别结果将按竖排阅读顺序重排：从右到左逐列，每列从上到下。适用于竖排繁体中文等场景。"),
    },
    "limit_side_len": {
        "title": tr("限制图像边长"),
        "optionsList": [
            [960, "960 " + tr("（默认）")],
            [2880, "2880"],
            [4320, "4320"],
            [999999, tr("无限制")],
            ["custom", tr("自定义")],
        ],
        "toolTip": tr("将边长大于该值的图片进行压缩，可以提高识别速度。可能降低识别精度。"),
    },
    "limit_side_len_custom": {
        "title": tr("自定义图像边长"),
        "default": 960,
        "min": 32,
        "isInt": True,
        "toolTip": tr('当"限制图像边长"选择"自定义"时生效。建议填写32或48的公倍数。'),
    },
}