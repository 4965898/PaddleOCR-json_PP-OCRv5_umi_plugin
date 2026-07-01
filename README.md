# PaddleOCR PP-OCRv6 Umi-OCR 插件（ONNX Runtime 版）v1.3

基于 [PaddleOCR 3.7.0](https://github.com/PaddlePaddle/PaddleOCR) + [ONNX Runtime](https://onnxruntime.ai/) 的 Umi-OCR 插件，使用最新的 **PP-OCRv6** 模型。

## 特性

- **PP-OCRv6 模型**：基于 PPLCNetV4 统一骨干网络，识别精度大幅提升
- **ONNX Runtime 引擎**：轻量、易部署，绕过 paddlepaddle 的 oneDNN 兼容性问题
- **自动下载模型**：首次使用时自动下载所选尺寸的 ONNX 模型到插件目录，无需手动下载
- **两档模型**：medium（高精度）/ small（快速），可随时切换
- **多语言识别**：PP-OCRv6 识别模型为多语言模型，可识别中英日韩等，无需按语言切换
- **性能优化**：开启 ONNX Runtime 图优化最高级 + 内存模式，充分利用 CPU 多核
- **GPU 显存动态分配**：按显卡总显存自适应分配 ORT CUDA arena 上限（small 40% / medium 65%），8GB 显卡稳定在 5.8GB，不再吃满显存
- **显存碎片防护**：每页识别后自动清理 paddle + torch CUDA 缓存，防止多页 PDF 显存累积导致 bad allocation
- **UTF-8 编码**：修复 Windows 下中文识别乱码问题

## 环境要求

- **Umi-OCR**：Paddle v2.1.5 及以上
- **Python**：3.10+（需添加到系统 PATH，用于创建虚拟环境）
- **操作系统**：Windows 10/11 x64
- **磁盘空间**：约 500MB（虚拟环境 + 模型文件）

## 安装步骤

### 第 1 步：放置插件

将整个 `umi_plugin_v6` 文件夹复制到 Umi-OCR 的插件目录：

```
Umi-OCR/
└── UmiOCR-data/
    └── plugins/
        └── umi_plugin_v6/    ← 复制到这里
            ├── install.bat
            ├── PaddleOCR-json.bat
            ├── ppocr_v6_server.py
            ├── ...
            └── models/
                ├── config_medium.txt
                └── config_small.txt
```

### 第 2 步：安装环境

双击运行 `install.bat`，脚本会自动：
1. 创建 Python 虚拟环境 `ppocr_v6_env`
2. 安装 `paddleocr` + `onnxruntime` 依赖

安装约需 1-3 分钟（取决于网速）。

> **GPU 加速**（可选，推荐 NVIDIA 显卡用户使用）：
>
> **显卡要求**：CUDA 加速仅支持 **GTX 10 系列及之后**的 NVIDIA 显卡（如 GTX 1050/1060/1070/1080、RTX 20/30/40/50 系列等）。GTX 10 之前的显卡（如 GTX 9xx、7xx、6xx 等）不支持本插件依赖的现代 CUDA/cuDNN 运行库，无法启用 GPU 加速。此类老显卡用户建议使用 CPU 模式，或改用旧版 **PP-OCRv5** 插件（兼容性更好）。
>
> 如需 GPU 加速，双击运行 `install_gpu.bat`，脚本会自动安装 `onnxruntime-gpu` + CUDA Runtime + cuDNN（约 1.6GB），无需手动下载任何文件。
>
> 安装完成后，在 Umi-OCR 插件设置中勾选「启用GPU」即可。
>
> **性能对比**（RTX 3070 Ti Laptop，medium 模型，4 行中文）：
>
> | 模式 | 平均识别耗时 | 加速比 |
> |------|-------------|--------|
> | CPU | 9.5s | 1x |
> | GPU | 0.55s | **17x** |
>
> 首次识别会稍慢（GPU 内核初始化），后续识别速度大幅提升。无 GPU 或缺少运行库时会自动降级到 CPU。
>
> **显存自适应分配**（v1.3 新增）：插件会自动检测显卡总显存，并按模型尺寸动态分配 ORT CUDA arena 上限：
>
> | 模型尺寸 | 显存占比 | 8GB 显卡示例 | 12GB 显卡示例 |
> |---------|---------|-------------|--------------|
> | small（快速） | 40% | 3.2GB | 4.8GB |
> | medium（高精度） | 65% | 5.2GB | 7.8GB |
>
> 留出的显存给 cuDNN workspace、CUDA context、paddle 缓存等使用，避免显存吃满导致 bad allocation 或 CUDA error 999。每页识别后还会自动清理 paddle + torch 的 CUDA 缓存，防止多页 PDF 显存碎片累积。

### 第 3 步：重启 Umi-OCR

重启 Umi-OCR，在「设置 → 当前接口」选择 **PaddleOCR（PP-OCRv6）** 即可使用。

首次识别时会自动下载所选尺寸的 ONNX 模型（约 10-50MB），下载后缓存到插件 `models/` 目录，后续无需重复下载。

## 使用说明

### 模型尺寸选择

在插件设置中选择模型尺寸：

| 选项 | 模型 | 精度 | 速度 | 适用场景 |
|------|------|------|------|----------|
| 高精度（medium） | PP-OCRv6_medium | 最高 | 较慢 | 高精度需求 |
| 快速（small） | PP-OCRv6_small | 较高 | 快（约 3 倍） | 日常使用、低配电脑 |

> PP-OCRv6 识别模型为多语言模型，可识别中英日韩等，无需按语言切换。

### 性能参数

| 设置项 | 默认值 | 说明 |
|--------|--------|------|
| 限制图像边长 | 960 | 调小（如 640）可显著提速，但可能降低小字识别精度 |
| 识别批处理数 | 6 | 调大（如 16）可提高多行文本吞吐量，不影响精度 |
| 启用文本检测 | 开启 | 单行纯文本图片可关闭以跳过检测，显著加速 |
| 纠正文本方向 | 关闭 | 识别倾斜/倒置文本时开启，会降低速度 |

### 性能优化建议

- **日常截图文字**：选 small + 限制图像边长 640 + 批处理数 16，速度最快
- **高精度需求**：选 medium + 限制图像边长 960 + 批处理数 6
- **单行文本**：关闭「启用文本检测」可跳过检测阶段
- 代码层面已开启 ONNX Runtime 图优化最高级 + 内存模式，无需额外配置

## 架构说明

本插件采用子进程架构，绕过 Umi-OCR 自带 Python 3.8 版本过低（paddleocr 3.7.0 需 Python 3.9+）的问题：

```
Umi-OCR (Python 3.8)
  └─ PaddleOCR-json.bat
       └─ ppocr_v6_env (Python 3.10)
            └─ ppocr_v6_server.py
                 └─ PaddleOCR 3.7.0 + ONNX Runtime
                      └─ JSON stdin/stdout 通信（UTF-8 编码）
```

- **引擎选择**：自动检测 onnxruntime 是否安装，优先使用 ONNX Runtime（轻量），未安装则回退 paddlepaddle
- **模型管理**：ONNX 模型统一存放在插件 `models/` 目录，不污染 Umi-OCR 其他插件
- **编码处理**：server 强制 stdin/stdout 使用 UTF-8，避免 Windows 下中文乱码

## 模型存放位置

自动下载的模型存放在插件自己的 `models/official_models/` 目录：

```
umi_plugin_v6/
└── models/
    ├── config_medium.txt
    ├── config_small.txt
    ├── configs.txt
    └── official_models/              ← 自动下载的模型在这里
        ├── PP-OCRv6_medium_det_onnx/
        │   └── inference.onnx
        ├── PP-OCRv6_medium_rec_onnx/
        │   └── inference.onnx
        ├── PP-OCRv6_small_det_onnx/
        │   └── inference.onnx
        └── PP-OCRv6_small_rec_onnx/
            └── inference.onnx
```

> 只下载用户选择的尺寸的模型，不会一次下载两种。

## 关于 mkldnn 加速

**mkldnn（oneDNN）对本插件无效。** mkldnn 是 paddlepaddle 的 CPU 加速后端，而本插件使用 ONNX Runtime 引擎绕过了 paddlepaddle。ONNX Runtime 使用自带的 MLAS 优化库做 CPU 加速，并已开启图优化最高级 + 内存模式，无需 mkldnn。

## 常见问题

### Q: 首次识别很慢？
A: 首次使用时需要下载模型（约 10-50MB），下载后缓存到本地，后续无需重复下载。模型下载源默认为 HuggingFace，国内较慢时可设置环境变量 `PADDLE_PDX_MODEL_SOURCE=bos` 使用百度云源。

### Q: 中文识别乱码？
A: 本插件已修复 Windows 下中文乱码问题（server 强制 UTF-8 编码）。如仍出现乱码，请确认使用的是最新版 `ppocr_v6_server.py`。

### Q: GPU 不生效？
A: 运行 `install_gpu.bat` 一键安装 GPU 所需组件（onnxruntime-gpu + CUDA Runtime + cuDNN）。安装后 onnxruntime 会自动加载 CUDA provider。如仍不生效，检查显卡驱动是否为最新版本。无 GPU 时会自动降级到 CPU。

> **显卡兼容性**：CUDA 加速仅支持 GTX 10 系列及之后的 NVIDIA 显卡。GTX 10 之前的显卡（如 GTX 9xx、7xx 等）不支持现代 CUDA/cuDNN，GPU 加速无法生效，请使用 CPU 模式，或改用旧版 PP-OCRv5 插件。

### Q: 如何切换模型尺寸？
A: 在 Umi-OCR 的插件设置中切换「模型尺寸」。切换后会重新加载引擎，首次使用新尺寸时需下载对应模型。

## 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 百度飞桨 OCR
- [ONNX Runtime](https://onnxruntime.ai/) - 微软跨平台推理引擎
- [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) - 免费开源的 OCR 软件

## 更新日志

### v1.3

- **GPU 显存动态分配**：按显卡总显存自适应分配 ORT CUDA arena 上限（small 40% / medium 65%），替代原先硬编码的固定上限。8GB 显卡实测 medium 模型 + rec_batch_num=20 稳定在 5.8GB。
- **paddle 显存清理**：`_cleanup_gpu_memory()` 新增 `paddle.device.cuda.empty_cache()` 调用。原实现只清理 torch 缓存，对 paddleocr 推理时的 paddle CUDA 缓存无效，导致多页 PDF 显存从 1GB 逐渐累积到 7.8GB。
- **`__ramClear` 崩溃修复**：子进程崩溃后 `exit()` 会把 `self.api.ret` 置为 None，原 `__ramClear` 未判空直接访问 `.pid` 导致 `AttributeError`。新增 `if self.api is None or getattr(self.api, "ret", None) is None: return` 保护。
- **GPU 显存检测**：新增 `_get_gpu_total_memory_gb()`，三级 fallback（paddle → torch → nvidia-smi）准确识别显卡总显存。

### v1.2

- 修复 CUDNN_FE failure 11 错误（移除 workspace 限制 + 改用 DEFAULT 算法）
- 新增 PDF 文本层精对齐选项（推荐比例 0.08）
- 修复 CPU 模式 numpy 数组真值判断崩溃
- 修复 GPU cuDNN FE 执行失败

### v1.1

- 修复 cuDNN 加载失败
- GPU 显存优化（首次引入 `gpu_mem_limit` + `arena_extend_strategy`）
