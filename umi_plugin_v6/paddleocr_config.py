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
            ["models/config_medium.txt", tr("高精度（medium）")],
            ["models/config_small.txt", tr("快速（small）")],
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
    "cpu_threads": {
        "title": tr("CPU线程数"),
        "default": 0,
        "min": 0,
        "isInt": True,
        "toolTip": tr("ONNX Runtime 推理线程数。0=自动（使用全部 CPU 核心，默认）。调小可降低内存占用（每个线程会分配独立工作区缓冲区），内存紧张时建议设为 4~8；不影响识别精度。GPU 模式下控制 CPU 侧算子线程数。"),
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
        "toolTip": tr("启用 GPU 加速。NVIDIA 显卡请先运行 install_gpu.bat（CUDA，最快）；Intel Arc / AMD 等显卡请运行 install_directml.bat（DirectML）。插件按已安装组件自动选择后端：CUDA 优先，其次 DirectML，无组件时降级到 CPU。"),
    },
    "shrink_poly_ratio": {
        "title": tr("PDF文本层精对齐"),
        "optionsList": [
            [0.0, tr("关闭")],
            [0.08, tr("0.08（推荐）")],
            [0.05, tr("0.05（轻微）")],
            [0.12, tr("0.12（较强）")],
            ["custom", tr("自定义")],
        ],
        "default": 0.0,
        "toolTip": tr("将检测框向内收缩，抵消 DBNet 的 expand_ratio，让 box 更贴合真实文字范围。可改善扫描 PDF 生成的双层 layered.pdf 中文本层与图像的对齐（det 框默认会比真实文字外扩一圈，导致字号被高估、行末字符超出图像文字）。仅 PDF 双层文档场景需要，其他场景可保持关闭。"),
    },
    "shrink_poly_ratio_custom": {
        "title": tr("自定义精对齐比例"),
        "default": 0.08,
        "min": 0.0,
        "max": 0.5,
        "isInt": False,
        "toolTip": tr('当"PDF文本层精对齐"选择"自定义"时生效。典型值 0.08，范围 0.02~0.20。0=关闭。'),
    },
}
