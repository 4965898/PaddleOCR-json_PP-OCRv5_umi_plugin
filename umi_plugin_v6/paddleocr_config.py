from plugin_i18n import Translator

tr = Translator(__file__, "i18n.csv")

globalOptions = {
    "title": tr("PaddleOCR（PP-OCRv6）"),
    "type": "group",
    "ram_max": {
        "title": tr("内存占用限制"),
        "default": 8192,
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
    "title": tr("PaddleOCR PP-OCRv6"),
    "type": "group",
    "language": {
        "title": tr("模型尺寸"),
        "optionsList": [
            ["models/config_medium.txt", "高精度（medium）"],
            ["models/config_small.txt", "快速（small）"],
        ],
        "toolTip": tr("PP-OCRv6 识别模型为多语言模型，可识别中英日韩等，无需按语言切换。medium 精度高，small 速度快。首次使用时自动下载对应模型到插件目录。"),
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
        "toolTip": tr("识别模型批处理大小。CPU 下：medium 建议保持默认 6（调大可能变慢），small 可调大到 16-32。GPU 下：可大幅调高以提升速度，但受显存限制，爆显存时报错请调小此值。"),
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
    "use_gpu": {
        "title": tr("启用GPU加速"),
        "default": False,
        "toolTip": tr("启用 NVIDIA GPU 加速（需先运行 install_gpu.bat 安装 GPU 组件）。GPU 模式比 CPU 快约 17 倍。无 GPU 或未安装组件时自动降级到 CPU。"),
    },
}
