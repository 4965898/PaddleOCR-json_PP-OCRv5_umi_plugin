# UmiOCR PP-OCRv6 ONNX Plugin（v1.1）

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
- **性能优化**：开启 ONNX Runtime 图优化最高级 + 内存模式，充分利用 CPU 多核
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

### Q: 如何切换模型尺寸？
A: 在 Umi-OCR 的插件设置中切换「模型尺寸」。切换后会重新加载引擎，首次使用新尺寸时需下载对应模型。

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

---

## 更新日志

### v1.1（2026-06-20）

**GPU cuDNN 加载修复**：
- 修复 GPU 模式下 `Invalid handle. Cannot load symbol cudnnCreate` 错误
- 根因：`from paddleocr import` 在 `_setup_nvidia_dlls()` 之前执行，paddleocr 导入过程干扰了后续 ORT CUDA 加载 cuDNN
- 修复：在 `import paddleocr` 之前先调用 `_setup_nvidia_dlls()` 添加 NVIDIA DLL 路径

**GPU 显存优化**：
- 新增 `cudnn_conv_algo_search=HEURISTIC`：避免默认 EXHAUSTIVE 策略搜索所有卷积算法时分配大量临时 workspace，减少约 1-2G 空闲显存占用
- 新增 `cudnn_conv_use_max_workspace=False`：不预分配最大 workspace，进一步减少显存
- 速度损失极小（卷积算法差异通常在 5% 以内）

### v1.0

- 基于 PaddleOCR 3.7.0 + ONNX Runtime 的 PP-OCRv6 插件初始版本
- 修复 GPU 多页 PDF `bad allocation`（arena_extend_strategy=kSameAsRequested + 每页显存清理）
- 修复 small 模型 `Model name mismatch`（use_local 时同时传 model_name 和 model_dir）
- 修复 stderr 被丢弃导致初始化错误不可见（DEVNULL → PIPE + 守护线程）
- 修复 base64 临时文件异常路径泄漏（清理移到 finally 块）
- 删除无效的 `sess.run_options.free()` 清理代码（RunOptions 无 free() 方法）
- 修复 bat 文件 LF/CRLF 编码问题（改为 CRLF + 无 BOM + 纯 ASCII）
- 将 v5 代码整理到 `umi_plugin_v5_json/` 子文件夹

---

## 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 百度飞桨 OCR
- [ONNX Runtime](https://onnxruntime.ai/) - 微软跨平台推理引擎
- [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) - 免费开源的 OCR 软件
- [PaddleOCR-json](https://github.com/hiroi-sora/PaddleOCR-json) - v5 版本基于的 C++ OCR 工具
