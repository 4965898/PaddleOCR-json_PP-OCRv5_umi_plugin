# UmiOCR PP-OCRv6 ONNX Plugin（v1.7）

本仓库包含两个版本的 Umi-OCR PaddleOCR 插件：

| 版本 | 目录 | 引擎 | 模型 | 推荐度 |
|------|------|------|------|--------|
| **PP-OCRv6** | `umi_plugin_v6/` | Python + ONNX Runtime | PP-OCRv6（自动下载） | ⭐ 推荐 |
| PP-OCRv5 | 仓库根目录 | C++（PaddleOCR-json.exe） | PP-OCRv5（手动下载） | 历史版本 |

---

# PP-OCRv6 插件（ONNX Runtime 版）⭐ 推荐

基于 [PaddleOCR 3.7.0](https://github.com/PaddlePaddle/PaddleOCR) + [ONNX Runtime](https://onnxruntime.ai/) 的 Umi-OCR 插件，使用最新的 **PP-OCRv6** 模型。

## 特性

- **PP-OCRv6 模型**：基于 PPLCNetV4 统一骨干网络，识别精度大幅提升
- **ONNX Runtime 引擎**：轻量、易部署，绕过 paddlepaddle 的 oneDNN 兼容性问题
- **自动下载模型**：首次使用时自动下载所选尺寸的 ONNX 模型到插件目录，无需手动下载
- **两档模型**：medium（高精度）/ small（快速），可随时切换
- **多语言识别**：PP-OCRv6 识别模型为多语言模型，可识别中英日韩等，无需按语言切换
- **多后端 GPU 加速**：NVIDIA 显卡走 CUDA（最快，约 17 倍）；Intel Arc / AMD 等任意 DirectX 12 显卡走 DirectML；插件按已安装组件自动选择后端
- **CPU 线程数可调**：暴露 ONNX Runtime 推理线程数设置，调小可显著降低内存占用（每线程分配独立工作区缓冲区），不影响识别精度
- **性能优化**：开启 ONNX Runtime 图优化最高级 + 内存模式，充分利用 CPU 多核
- **GPU 显存动态分配**：按显卡总显存自适应分配 ORT CUDA arena 上限（small 50% / medium 65%），8GB 显卡稳定在 5.8GB，不再吃满显存
- **显存碎片防护**：每页识别后自动清理 GPU 缓存（paddle/torch），每 50 页自动重建 ORT session 释放 BFC arena 碎片，防止长 PDF 末尾报 `BFCArena::AllocateRawInternal` 错误；遇到该错误时自动重建 session 并重试当前页
- **UTF-8 编码**：修复 Windows 下中文识别乱码问题

## 环境要求

- **Umi-OCR**：Paddle v2.1.5 及以上
- **Python**：**无需预装**（install.bat 会自动下载便携 Python）；如已安装 Python 3.10+ 则直接使用
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

### 第 2 步：安装环境（开箱即用，无需预装 Python）

双击运行 `install.bat`，脚本会自动完成全部安装：
1. **检测系统 Python**：有则直接使用
2. **无 Python 时自动下载**：通过 [uv](https://github.com/astral-sh/uv) 自动下载便携 Python 3.11（约 30MB），无需手动安装
3. 创建虚拟环境 `ppocr_v6_env`
4. 安装 `paddleocr` + `onnxruntime` 依赖（约 200MB）

安装约需 1-5 分钟（取决于网速；无 Python 时首次多下 30MB）。

> 新手只需双击 `install.bat` 等待完成即可，全程无需命令行操作。

> **GPU 加速**（可选，推荐 NVIDIA 显卡用户使用）：
>
> **GPU 加速所需环境（NVIDIA CUDA 路径）**：
>
> | 组件 | 要求 | 说明 |
> |------|------|------|
> | **NVIDIA 显卡** | GTX 10 系列及之后 | GTX 1050+、RTX 20/30/40/50 系列等。GTX 10 之前（如 GTX 9xx/7xx）不支持现代 CUDA/cuDNN |
> | **NVIDIA 显卡驱动** | **R525 或更高**（Windows） | 驱动版本过低是 GPU 不生效的最常见原因！[驱动下载](https://www.nvidia.com/Download/index.aspx) |
> | CUDA Runtime 12.x | 自动安装 | 由 `install_gpu.bat` 通过 pip 安装 `nvidia-cuda-runtime-cu12`，**无需手动安装 CUDA Toolkit** |
> | cuDNN 9.x | 自动安装 | 由 `install_gpu.bat` 通过 pip 安装 `nvidia-cudnn-cu12`，**无需手动下载 cuDNN** |
> | onnxruntime-gpu | 自动安装 | 由 `install_gpu.bat` 安装，包含 CUDA Execution Provider |
>
> **重要**：`install_gpu.bat` 会自动通过 pip 安装 CUDA Runtime 和 cuDNN 运行库（约 1.6GB），**无需手动安装 NVIDIA CUDA Toolkit 或 cuDNN**。只需确保显卡驱动为最新版本（R525+）。
>
> 双击运行 `install_gpu.bat`，脚本会自动：
> 1. 卸载可能冲突的 CPU 版 `onnxruntime`
> 2. 安装 `onnxruntime-gpu` + CUDA Runtime + cuDNN（约 1.6GB）
> 3. 验证 CUDAExecutionProvider 是否可用
>
> 安装完成后，在 Umi-OCR 插件设置中勾选「启用GPU加速」即可。
>
> **如何确认 GPU 正在工作**：启动后查看 Umi-OCR 日志，应看到：
> ```
> [ppocr_v6] engine=onnxruntime, gpu_backend=cuda
> [ppocr_v6] GPU verified: det model session uses ['CUDAExecutionProvider', ...]
> [ppocr_v6] GPU verified: rec model session uses ['CUDAExecutionProvider', ...]
> ```
> 如果看到 `gpu_backend=None` 或 CPU fallback 警告，说明 GPU 未生效，请按下方「常见问题」排查。
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
> **显存自适应分配**（v1.3 新增，v1.4 优化）：插件会自动检测显卡总显存，并按模型尺寸动态分配 ORT CUDA arena 上限：
>
> | 模型尺寸 | 显存占比 | 8GB 显卡示例 | 12GB 显卡示例 |
> |---------|---------|-------------|--------------|
> | small（快速） | 50% | 4.0GB | 6.0GB |
> | medium（高精度） | 65% | 5.2GB | 7.8GB |
>
> 留出的显存给 cuDNN workspace、CUDA context、paddle 缓存等使用，避免显存吃满导致 bad allocation 或 CUDA error 999。每页识别后还会自动清理 GPU 缓存，防止多页 PDF 显存碎片累积。

> **DirectML 加速**（v1.6 新增，适用于 Intel Arc / AMD 等非 NVIDIA 显卡）：
>
> 没有 NVIDIA 显卡也能用 GPU 加速。DirectML 是微软的 DirectX 12 推理后端，支持 **Intel Arc 核显/独显**（如 Intel Core Ultra 5/7 125H/155H 自带 Arc Graphics）、**AMD 显卡**，以及任意 DirectX 12 GPU。
>
> 双击运行 `install_directml.bat`，脚本会自动安装 `onnxruntime-directml`（约 200MB，无需 CUDA/cuDNN）。
>
> 安装完成后，在 Umi-OCR 插件设置中勾选「启用GPU加速」即可。插件自动识别已安装的后端：**CUDA 优先 → 其次 DirectML → 无则降级 CPU**。启动后可在日志中看到 `[ppocr_v6] engine=onnxruntime, gpu_backend=directml` 确认生效。
>
> **注意**：`onnxruntime-directml` 与 `onnxruntime` / `onnxruntime-gpu` 互斥，同一虚拟环境只能装一个。若之前跑过 `install.bat` 或 `install_gpu.bat`，请先 `ppocr_v6_env\Scripts\pip uninstall -y onnxruntime onnxruntime-gpu`，再运行 `install_directml.bat`。NVIDIA 用户仍推荐 `install_gpu.bat`（CUDA 比 DirectML 更快）。

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
| CPU线程数 | 0（自动） | ONNX Runtime 推理线程数。0=用全部 CPU 核心；调小（如 4~8）可降低内存占用，不影响精度 |
| 启用文本检测 | 开启 | 单行纯文本图片可关闭以跳过检测，显著加速 |
| 纠正文本方向 | 关闭 | 识别倾斜/倒置文本时开启，会降低速度 |
| PDF文本层精对齐 | 关闭 | 仅 PDF 双层文档场景需要。提供 0.05/0.08/0.12 预设与「自定义」浮点数输入（范围 0~0.5），推荐 0.08 |

### 性能优化建议

- **日常截图文字**：选 small + 限制图像边长 640 + 批处理数 16，速度最快
- **高精度需求**：选 medium + 限制图像边长 960 + 批处理数 6
- **单行文本**：关闭「启用文本检测」可跳过检测阶段
- 代码层面已开启 ONNX Runtime 图优化最高级 + 内存模式，无需额外配置

### 内存占用优化

本插件以常驻子进程方式运行，模型加载后内存会持续占用。若觉得内存偏高，可按以下方式调优（均不影响识别精度）：

- **调小「CPU线程数」**（最有效）：ONNX Runtime 默认使用全部 CPU 核心，**每个线程都会分配独立的工作区缓冲区**。在核心数多的 CPU 上（如 Intel Core Ultra 5 125H 有 14 核 20 线程），这部分内存开销可观。将「CPU线程数」从默认 0（自动）改为 4~8，可显著降低内存占用，速度损失通常很小。
- **调小「识别批处理数」**：批处理数越大，单次推理的中间张量越大。CPU 模式建议保持默认 6。
- **调低「内存占用限制」**（全局设置）：默认 8192MB，引擎子进程 RSS 超过该值时会自动重启释放内存。内存紧张的机器可调小到 2048~4096MB。
- **选 small 模型**：small 模型本体比 medium 小，内存占用更低，速度快约 3 倍。
- **周期性重建**：插件每 50 页自动重建一次 ORT session 释放内存碎片（CUDA 释放 BFC arena、CPU 释放 ORT arena），长 PDF 不会内存持续上涨。DirectML 后端显存由 DX12 驱动按需分配，无需重建。

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

- **引擎选择**：自动检测 onnxruntime 是否安装，优先使用 ONNX Runtime（轻量），未安装则回退 paddlepaddle；GPU 后端按 CUDA → DirectML → CPU 自动选择
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

### Q: GPU 不生效？打开硬件加速后速度没有明显提升？
A: 这通常是因为 CUDA/cuDNN 运行库未正确加载，ORT 静默回退到了 CPU 模式。请按以下步骤排查：

1. **检查日志**：启动后查看 Umi-OCR 日志（或 stderr 输出），找到 `[ppocr_v6] engine=..., gpu_backend=...` 行：
   - `gpu_backend=cuda` + `GPU verified: ... session uses ['CUDAExecutionProvider', ...]` → GPU 正在工作
   - `gpu_backend=None` + WARNING → **GPU 未生效**，按以下步骤修复

2. **更新显卡驱动**（最常见原因）：CUDA 12.x 运行库需要 NVIDIA 驱动 **R525 或更高版本**。旧驱动会导致 CUDA DLL 加载失败，ORT 静默回退到 CPU。请到 [NVIDIA 驱动下载页](https://www.nvidia.com/Download/index.aspx) 更新到最新驱动。

3. **重新运行 `install_gpu.bat`**：脚本会自动卸载冲突的 CPU 版 `onnxruntime`，重新安装 `onnxruntime-gpu` + CUDA Runtime + cuDNN，并验证 CUDAExecutionProvider 是否可用。

4. **检查 onnxruntime 版本**：在 `ppocr_v6_env` 中运行 `pip list | findstr onnxruntime`，确认安装的是 `onnxruntime-gpu`（而非 `onnxruntime`）。两者互斥，同时安装会导致冲突。如需手动修复：
   ```
   ppocr_v6_env\Scripts\pip uninstall -y onnxruntime onnxruntime-gpu
   ppocr_v6_env\Scripts\pip install "onnxruntime-gpu[cuda,cudnn]"
   ```

5. **验证 CUDA 可用性**：
   ```
   ppocr_v6_env\Scripts\python -c "import onnxruntime as ort; print(ort.get_available_providers())"
   ```
   输出应包含 `CUDAExecutionProvider`。如果没有，说明 CUDA/cuDNN DLL 加载失败，请更新驱动后重试。

> **注意**：即使 GPU 正常工作，OCR 模型的 GPU 利用率也可能较低（1-10%），因为 OCR 推理的计算量相对较小，大部分时间花在 CPU 端的图像预处理和后处理上。这是正常现象——GPU 加速的效果体现在总耗时减少，而非 GPU 占用率高。

> **显卡兼容性**：CUDA 加速仅支持 GTX 10 系列及之后的 NVIDIA 显卡。GTX 10 之前的显卡（如 GTX 9xx、7xx 等）不支持现代 CUDA/cuDNN，请改用 `install_directml.bat`（DirectML，支持任意 DX12 GPU）或 CPU 模式。

### Q: 没有 NVIDIA 显卡能用 GPU 加速吗？
A: 可以。运行 `install_directml.bat` 安装 DirectML 后端，支持 Intel Arc（含 Intel Core Ultra 核显）、AMD、以及任意 DirectX 12 GPU。在插件设置中勾选「启用GPU加速」即可，插件会自动选用 DirectML 后端。

### Q: 内存占用太高怎么办？
A: 见上文「内存占用优化」一节。最有效的办法是调小「CPU线程数」（默认 0=用全部核心，改为 4~8 可显著降低内存，不影响精度），其次调小「识别批处理数」和全局的「内存占用限制」。

### Q: 「CPU线程数」有上限吗？支持多核 CPU 吗？
A: 没有上限，支持任意核心数的 CPU（32 核、64 核均可）。该设置仅校验为非负整数，输入框无硬编码上限。但**不建议设大于物理核心数**——ONNX Runtime 的 `intra_op_num_threads` 超过物理核心数会因线程上下文切换开销而变慢，不会提速。默认 `0=自动` 已用满全部物理核心，对多核 CPU 已是吞吐最优；该设置的设计方向是「调小降内存」，而非「调大加速」。

### Q: 如何切换模型尺寸？
A: 在 Umi-OCR 的插件设置中切换「模型尺寸」。切换后会重新加载引擎，首次使用新尺寸时需下载对应模型。

## 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 百度飞桨 OCR
- [ONNX Runtime](https://onnxruntime.ai/) - 微软跨平台推理引擎
- [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) - 免费开源的 OCR 软件

## 更新日志

### v1.7

- **开箱即用：无需预装 Python**（降低新手使用门槛）：`install.bat` 增强为开箱即用模式——检测到系统无 Python 时，自动通过 [uv](https://github.com/astral-sh/uv) 下载便携 Python 3.11（约 30MB）并创建虚拟环境，用户无需手动安装 Python。已有系统 Python 时直接使用。新手只需双击 `install.bat` 等待完成即可，全程零命令行操作。
- **修复 GPU 加速静默回退到 CPU 的问题**（回应 issue #10：用户反馈"打开硬件加速速度也不明显"，CPU 占用 77% 而 GPU 占用仅 1-3%，根因是 CUDA/cuDNN 运行库未正确加载，ORT 静默回退到 CPU 但用户无从知晓）：
  - **`_select_engine()` 新增 GPU 不可用警告**：当 `use_gpu=True` 但 `CUDAExecutionProvider` 和 `DmlExecutionProvider` 均不可用时，输出醒目的多行 WARNING 到 stderr，列出 available providers、回退到 CPU 的事实、以及修复方法（运行 install_gpu.bat / install_directml.bat + 更新显卡驱动）。
  - **新增 `_verify_gpu_session()`**：在 `init_ocr()` 末尾 best-effort 访问 paddlex 内部 `ONNXRuntimeRunner.session.get_providers()`，验证 ORT session **实际**使用的 providers（而非全局可用列表）。即使 `get_available_providers()` 报告 CUDA 可用，session 创建时也可能因 DLL 版本不匹配等原因静默回退到 CPU——此检查让回退变得可见。验证成功输出 `GPU verified: ... session uses ['CUDAExecutionProvider', ...]`，失败输出醒目 WARNING。
- **改进 `install_gpu.bat`**：
  - 安装前自动卸载冲突的 CPU 版 `onnxruntime`（`pip uninstall -y onnxruntime`），避免两者同时存在导致 ORT 加载错误的 DLL。
  - 显示显卡驱动版本和型号（`nvidia-smi --query-gpu=driver_version,name`），便于用户确认驱动是否满足 CUDA 12.x 的 R525+ 要求。
  - 安装后验证 CUDAExecutionProvider 真正可用（`exit(0 if 'CUDAExecutionProvider' in ps else 1)`），不可用时给出明确的错误提示和修复建议，而非静默继续。
  - 提示用户在 Umi-OCR 日志中查找 `[ppocr_v6] GPU verified: ...` 确认 GPU 生效。
- **更新 GPU 加速文档**：README 新增「GPU 加速所需环境」表格，明确列出 NVIDIA 显卡、显卡驱动（R525+）、CUDA Runtime 12.x、cuDNN 9.x、onnxruntime-gpu 五项要求及来源；FAQ「GPU 不生效？」改为分步骤排查指南（检查日志 → 更新驱动 → 重新安装 → 检查 pip 包 → 验证 CUDA）。

### v1.6

- **新增 DirectML 后端（Intel Arc / AMD 等 GPU 加速）**：此前 GPU 加速仅支持 NVIDIA CUDA，Intel Arc 核显（如 Intel Core Ultra 5/7 125H/155H 自带 Arc Graphics）、AMD 等显卡无法加速。新增 DirectML 后端，支持任意 DirectX 12 GPU。运行 `install_directml.bat` 安装 `onnxruntime-directml` 后，插件自动选用（CUDA 优先 → DirectML → CPU）。
  - `_select_engine()` 新增 `DmlExecutionProvider` 分支：通过显式 `providers` 列表 + `device_type="cpu"` 绕过 paddlex `_check_device_support` 对 CUDAExecutionProvider 的强制要求（DirectML 无 CUDA EP 会被拒）；`init_ocr()` 对 DirectML 路径额外传 `device="cpu"` 确定性触发绕过。
  - 新增全局 `_gpu_backend`（`"cuda"` / `"directml"` / `None`）区分后端。DirectML 显存由 DX12 驱动按需分配，无 BFC arena，因此跳过周期性 session 重建（避免无谓的 2~3 秒重建开销）。
- **新增「CPU线程数」设置**（回应社区反馈"没有多线程功能设置"）：暴露 ONNX Runtime 推理线程数（`intra_op_num_threads` / `inter_op_num_threads`），0=自动（用全部 CPU 核心，默认）。调小可降低内存占用——每线程会分配独立工作区缓冲区，核心数多的 CPU（如 14 核 20 线程）开销可观。server 早已支持 `--cpu_threads` 参数，本次补齐 UI 配置（`paddleocr_config.py`）与子进程传参（`paddleocr_umi.py` ExeConfigs）。
- **内存占用优化**（回应社区反馈"内存占用高"）：
  - `_cleanup_gpu_memory()` 将 `gc.collect()` 前移到 `if not _use_gpu` 早退之前，CPU 模式下也回收 Python 层循环引用垃圾（numpy 结果数组等），避免长 PDF 累积导致内存上涨。
  - 周期性 session 重建对 CPU 模式同样生效（释放 ORT CPU arena 碎片），长 PDF 内存不再持续上涨。
  - 新增「内存占用优化」文档章节与 FAQ，给出调小 CPU 线程数 / 批处理数 / 内存占用限制等可行建议。
- **后端可观测性**：`init_ocr()` 启动时输出 `[ppocr_v6] engine=onnxruntime, gpu_backend=...` 到 stderr（被 `ppocr_pipe` 守护线程捕获），便于用户确认 GPU 加速是否生效及实际后端。
- 更新「启用GPU加速」设置说明，区分 NVIDIA（install_gpu.bat）与 Intel Arc/AMD（install_directml.bat）安装路径。

### v1.5

- **修复长 PDF 末尾 BFC arena 失败**：跑几百页英文 PDF 时，ORT 的 BFC arena 会累积前面页的 buffer 不释放（设计如此，为复用），到最后几页 `FusedMatMul` / `BiasSoftmax` 等大块节点申请不到连续显存就报 `BFCArena::AllocateRawInternal: Available memory of X is smaller than requested bytes of Y`。原 `_cleanup_gpu_memory()` 只能 `gc.collect()` + `torch.cuda.empty_cache()`，对 ORT 的 arena 无效——必须销毁 session 才能让 arena 归还 CUDA。
  - 新增 `_rebuild_ocr()`：销毁并重建 ORT session，强制释放整个 arena（代价 2~3 秒重新加载模型）
  - **周期性重建**：每 50 页自动重建一次（8GB 显卡阈值；更小显存可调小至 30，更大可调到 100）
  - **错误自动恢复**：检测到 BFC arena 错误时自动重建 session 并重试当前页一次
  - `init_ocr()` 新增 `_init_args` 保存，供重建复用
- **BFC arena 错误识别**：异常分支新增 `bfcarena` / `allocaterawinternal` / `available memory of` 关键字检测，给出"降低识别批处理数"的中文提示
- **高 rec_batch_num 风险说明**：batch=30 时 FusedMatMul 单次申请 ~556MB，8GB 显卡建议保持默认 6 或最高调到 10
- **PDF文本层精对齐支持自定义数值**：「PDF文本层精对齐」下拉框新增「自定义」选项，选中后显示浮点数输入框（范围 0~0.5，默认 0.08），可输入任意精度比例值，不再局限于 0.05/0.08/0.12 三个预设

### v1.4

- **修复 rec-only 模式崩溃**（det=False）：`TextRecognition` 初始化时未传 `model_name`，paddlex 默认用 `PP-OCRv6_medium_rec`，与 small 本地目录不匹配导致 init 失败（错误码 803）。补传 `model_name = rec_model`。
- **修复 small 模型空白页卡死**（det=True）：small 模型原先 `cudnn_conv_use_max_workspace="0"`，在几近空白页（仅竖线）上 cuDNN 找不到有效卷积算法，触发 native 崩溃绕过 Python try/except。统一改为 `"1"`，同时将 small 显存占比从 40% 提到 50% 以覆盖 workspace 增量。
- **修复 rec-only 两页后停止**（det=False）：`_cleanup_gpu_memory()` 在 ORT 引擎下调 `paddle.device.cuda.empty_cache()` 会触发 paddle 延迟初始化 CUDA 上下文，与 ORT 的 CUDAExecutionProvider 冲突导致上下文损坏。新增 `_engine` 全局变量，ORT 引擎时跳过 paddle 清理（det=True 路径不受影响，paddle 上下文已由 pipeline 正常初始化）。

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

### v1.0

- 基于 PaddleOCR 3.7.0 + ONNX Runtime 的 PP-OCRv6 插件初始版本

---

# PP-OCRv5 插件（历史版本）

> 以下为 PP-OCRv5 版本的说明，基于 C++ 的 PaddleOCR-json 可执行文件。v6 版本已改用 Python + ONNX Runtime，推荐使用 v6。v5 代码保留在仓库根目录。

**修改自 [win7_x64_PaddleOCR-json](https://github.com/hiroi-sora/Umi-OCR_plugins/tree/2.0.0/win7_x64_PaddleOCR-json)**

兼容 `Windows 7/10/11 x64`

**下载预编译好的插件: [Releases](https://github.com/OneDongua/PaddleOCR-json_PP-OCRv5_umi_plugin/releases/latest)**

## 相比原版插件的改进

- **模型升级**：将 PaddleOCR 从 v2.6/v2.8 升级至 v3.1（PP-OCRv5），识别精度大幅提升。
- **快速模型**：新增 PP-OCRv5 mobile_rec 轻量识别模型，速度提升 3~5 倍，精度仍高于旧版 v3。
- **多语言分离**：语言选项分为简体中文、繁體中文、English、日本語，各有高精度/快速两种模式。
- **推理设备模式**：新增仅CPU、仅GPU、CPU+GPU混合三种推理模式。
- **TensorRT 加速**：支持启用 TensorRT 加速 GPU 推理。
- **FP16 精度**：支持 FP16 推理精度，可加速 GPU 推理。
- **文本检测开关**：可关闭 det 检测以加速单行文本识别。
- **识别批处理数**：可调整 rec_batch_num 提高吞吐量。
- **竖排文字模式**：可按竖排阅读顺序重排识别结果（从右到左逐列，每列从上到下）。
- **参数传递修复**：修复启动参数传递方式，避免含空格路径的解析错误。
- **字典文件修复**：使用正确的 PP-OCRv5 字典（18383 字符），替代旧版 v1 字典（245 字符）。

## 部署步骤

### 第1步：克隆插件源码

```sh
git clone https://github.com/OneDongua/PaddleOCR-json_PP-OCRv5_umi_plugin.git
```

### 第2步：准备 PaddleOCR-json 可执行文件

#### 方式1：直接下载

- 浏览器访问 [PaddleOCR-json 发布页](https://github.com/OneDongua/PaddleOCR-json/releases) ，获取最新的 Windows 发行包 `PaddleOCR-json_v1.4.1-ext_windows_x64.7z` 的链接，下载压缩包并解压。
- 解压出来的文件夹，改名为 `win7_x64_PaddleOCR-json` 。

#### 方式2：从源码构建

- 见 [PaddleOCR-json Windows 构建指南](https://github.com/OneDongua/PaddleOCR-json/blob/main/cpp/README.md) 。

### 第3步：组装插件，放置插件

- 将仓库根目录中的所有文件，复制到 `win7_x64_PaddleOCR-json` 。
- 在 `win7_x64_PaddleOCR-json` 中，双击 `PaddleOCR-json.exe` 测试。正常情况下，应该打开一个控制台窗口，显示 `OCR init completed.` 。
- 将 `win7_x64_PaddleOCR-json` 整个文件夹，复制到 `UmiOCR-data\plugins` 中。

### 第4步：下载快速模型（可选）

如需使用"快速"模式，需额外下载 PP-OCRv5 mobile_rec 模型：

1. 从 HuggingFace 下载以下 3 个文件：
   - [inference.pdiparams](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec/resolve/main/inference.pdiparams)（约 16 MB，模型权重）
   - [inference.json](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec/resolve/main/inference.json)（模型结构）
   - [inference.yml](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec/resolve/main/inference.yml)（模型配置）
2. 在插件的 `models/` 目录下创建 `PP-OCRv5_mobile_rec_infer` 文件夹。
3. 将下载的 3 个文件放入该文件夹中。

## 全局设置说明

**注：v5 版本目前仅支持CPU模式，GPU模型现不可用，请在设置中设置为"仅CPU"模式使用！！！**

| 设置项 | 默认值 | 说明 |
|--------|--------|------|
| 推理设备模式 | 仅CPU | 仅CPU / 仅GPU / CPU+GPU混合（推荐） |
| GPU编号 | 0 | 多卡环境下选择 GPU 序号 |
| 启用MKL-DNN加速 | 开启 | 大幅加快 CPU 推理速度，但增加内存占用 |
| 线程数 | 自动 | CPU 推理线程数，建议 8~16 间测试最优值 |
| 启用TensorRT加速 | 关闭 | 加速 GPU 推理，需 GPU 版 exe 及 TensorRT 环境 |
| 推理精度 | FP32 | FP16 可加速 GPU 推理，可能略微降低精度 |
| 内存占用限制 | 自动 | 引擎内存超限时执行清理 |
| 内存闲时清理 | 60秒 | 引擎空闲超时后执行清理 |

## 局部设置说明

| 设置项 | 默认值 | 说明 |
|--------|--------|------|
| 语言/模型库 | 简体中文（高精度） | 高精度用 server_rec，快速用 mobile_rec |
| 启用文本检测 | 开启 | 单行文本可关闭以加速 |
| 纠正文本方向 | 关闭 | 识别倾斜/倒置文本，可能降低速度 |
| 识别批处理数 | 6 | 增大可提高吞吐量，增加内存/显存占用 |
| 竖排文字模式 | 关闭 | 按竖排阅读顺序重排结果（从右到左逐列） |
| 限制图像边长 | 960 | 压缩大图加速，可能降低精度 |

## v5 性能优化建议

- 保持 `启用MKL-DNN加速` 为开启。
- `线程数` 不要盲目拉满，建议从 `8~16` 间测试最优值（本插件默认已限制上限 16，避免过多线程争抢）。
- 使用"快速"模型（mobile_rec），速度提升 3~5 倍，精度仍高于旧版 v3。
- 大图较多时可优先使用 `限制图像边长=960`（更快）或按精度需求改为 2880/4320；也支持 `自定义` 输入任意边长。
- 仅在确有旋转文本时开启 `纠正文本方向`，否则保持关闭以减少额外开销。
- 单行文本可关闭 `启用文本检测` 以跳过检测阶段，显著加速。
- 长时间批量识别时，按机器内存情况设置 `内存占用限制` 和 `内存闲时清理`，减少内存膨胀引起的性能抖动。
