# UmiOCR PP-OCRv6 ONNX Plugin (v1.6)

This repository contains two versions of the Umi-OCR PaddleOCR plugin:

| Version | Directory | Engine | Model | Recommendation |
|------|------|------|------|--------|
| **PP-OCRv6** | `umi_plugin_v6/` | Python + ONNX Runtime | PP-OCRv6 (Auto-download) | ⭐ Recommended |
| PP-OCRv5 | Root Directory | C++ (PaddleOCR-json.exe) | PP-OCRv5 (Manual download) | Legacy Version |

---

# PP-OCRv6 Plugin (ONNX Runtime Version) ⭐ Recommended

A Umi-OCR plugin based on [PaddleOCR 3.7.0](https://github.com/PaddlePaddle/PaddleOCR) + [ONNX Runtime](https://onnxruntime.ai/), utilizing the latest **PP-OCRv6** model.

## Features

- **PP-OCRv6 Model**: Based on the PPLCNetV4 unified backbone, significantly improving recognition accuracy.
- **ONNX Runtime Engine**: Lightweight and easy to deploy; bypasses oneDNN compatibility issues associated with `paddlepaddle`.
- **Automatic Model Download**: Automatically downloads the ONNX model of the selected size to the plugin directory upon first use; no manual downloading required.
- **Two Model Tiers**: Medium (High Accuracy) / Small (Fast), switchable at any time.
- **Multilingual Recognition**: The PP-OCRv6 recognition model is multilingual, supporting Chinese, English, Japanese, Korean, etc., without needing to switch languages.
- **Multi-backend GPU Acceleration**: NVIDIA GPUs use CUDA (fastest, ~17x); Intel Arc / AMD or any DirectX 12 GPU uses DirectML. The plugin auto-selects the backend based on installed components.
- **Adjustable CPU Threads**: Exposes the ONNX Runtime inference thread count. Lowering it significantly reduces memory usage (each thread allocates an independent workspace buffer) without affecting accuracy.
- **Performance Optimization**: Enables the highest level of ONNX Runtime graph optimization + memory mode to fully utilize multi-core CPUs.
- **Dynamic GPU VRAM Allocation**: Adaptively allocates the ORT CUDA arena limit based on total GPU VRAM (Small 50% / Medium 65%). For 8GB cards, it stabilizes around 5.8GB, preventing VRAM saturation.
- **VRAM Fragmentation Protection**: Automatically clears GPU cache (paddle/torch) after each page. Rebuilds the ORT session every 50 pages to release BFC arena fragments, preventing `BFCArena::AllocateRawInternal` errors at the end of long PDFs; automatically rebuilds the session and retries the current page upon encountering this error.
- **UTF-8 Encoding**: Fixes Chinese character encoding issues on Windows.

## Environment Requirements

- **Umi-OCR**: Paddle v2.1.5 or higher.
- **Python**: 3.10+ (Must be added to system PATH to create the virtual environment).
- **OS**: Windows 10/11 x64.
- **Disk Space**: Approx. 500MB (Virtual environment + model files).

## Installation Steps

### Step 1: Place the Plugin

Copy the entire `umi_plugin_v6` folder into the Umi-OCR plugins directory:

```
Umi-OCR/
└── UmiOCR-data/
    └── plugins/
        └── umi_plugin_v6/    ← Copy here
            ├── install.bat
            ├── PaddleOCR-json.bat
            ├── ppocr_v6_server.py
            ├── ...
            └── models/
                ├── config_medium.txt
                └── config_small.txt
```

### Step 2: Install Environment

Double-click `install.bat`. The script will automatically:
1. Create a Python virtual environment named `ppocr_v6_env`.
2. Install `paddleocr` + `onnxruntime` dependencies.

Installation takes approximately 1-3 minutes (depending on network speed).

> **GPU Acceleration** (Optional, recommended for NVIDIA GPU users):
>
> **Hardware Requirements**: CUDA acceleration only supports NVIDIA GPUs from the **GTX 10 series and later** (e.g., GTX 1050/1060/1070/1080, RTX 20/30/40/50 series). GPUs prior to the GTX 10 series (e.g., GTX 9xx, 7xx, 6xx) do not support the modern CUDA/cuDNN runtime libraries required by this plugin and cannot enable GPU acceleration. Users with older cards are advised to use CPU mode, or use `install_directml.bat` (DirectML supports any DX12 GPU), or the legacy **PP-OCRv5** plugin (which has better compatibility).
>
> To enable GPU acceleration, double-click `install_gpu.bat`. The script will automatically install `onnxruntime-gpu` + CUDA Runtime + cuDNN (approx. 1.6GB) without requiring manual downloads.
>
> After installation, check "Enable GPU Acceleration" in the Umi-OCR plugin settings.
>
> **Performance Comparison** (RTX 3070 Ti Laptop, medium model, 4 lines of Chinese):
>
> | Mode | Avg. Recognition Time | Speedup |
> |------|-------------|--------|
> | CPU | 9.5s | 1x |
> | GPU | 0.55s | **17x** |
>
> The first recognition will be slightly slower (due to GPU kernel initialization), but subsequent speeds will increase significantly. It will automatically fallback to CPU if no GPU or runtime libraries are found.
>
> **Adaptive VRAM Allocation** (Added v1.3, Optimized v1.4): The plugin automatically detects total GPU VRAM and dynamically allocates the ORT CUDA arena limit:
>
> | Model Size | VRAM % | 8GB Card Example | 12GB Card Example |
> |---------|---------|-------------|--------------|
> | Small (Fast) | 50% | 4.0GB | 6.0GB |
> | Medium (High Acc) | 65% | 5.2GB | 7.8GB |
>
> The remaining VRAM is reserved for cuDNN workspace, CUDA context, paddle cache, etc., to avoid "bad allocation" or CUDA error 999. GPU cache is also cleared after each page to prevent VRAM fragmentation in multi-page PDFs.

> **DirectML Acceleration** (Added v1.6, for Intel Arc / AMD and other non-NVIDIA GPUs):
>
> GPU acceleration is available even without an NVIDIA GPU. DirectML is Microsoft's DirectX 12 inference backend, supporting **Intel Arc integrated/discrete GPUs** (e.g., the Arc Graphics built into Intel Core Ultra 5/7 125H/155H), **AMD GPUs**, and any DirectX 12 GPU.
>
> Double-click `install_directml.bat`. The script will automatically install `onnxruntime-directml` (approx. 200MB; no CUDA/cuDNN required).
>
> After installation, check "Enable GPU Acceleration" in the Umi-OCR plugin settings. The plugin auto-detects the installed backend: **CUDA first → then DirectML → fallback to CPU if none**. After startup, you can confirm it took effect via the log line `[ppocr_v6] engine=onnxruntime, gpu_backend=directml`.
>
> **Note**: `onnxruntime-directml` is mutually exclusive with `onnxruntime` / `onnxruntime-gpu`; only one can be installed in the same virtual environment. If you previously ran `install.bat` or `install_gpu.bat`, run `ppocr_v6_env\Scripts\pip uninstall -y onnxruntime onnxruntime-gpu` first, then run `install_directml.bat`. NVIDIA users should still prefer `install_gpu.bat` (CUDA is faster than DirectML).

### Step 3: Restart Umi-OCR

Restart Umi-OCR and select **PaddleOCR (PP-OCRv6)** in "Settings → Current Interface".

The ONNX model of the selected size (approx. 10-50MB) will be automatically downloaded during the first recognition and cached in the plugin's `models/` directory.

## Usage Instructions

### Model Size Selection

Select the model size in the plugin settings:

| Option | Model | Accuracy | Speed | Use Case |
|------|------|------|------|----------|
| High Accuracy (medium) | PP-OCRv6_medium | Highest | Slower | High accuracy needs |
| Fast (small) | PP-OCRv6_small | High | Fast (approx 3x) | Daily use, low-spec PCs |

> The PP-OCRv6 recognition model is multilingual and supports Chinese, English, Japanese, and Korean without switching languages.

### Performance Parameters

| Setting | Default | Description |
|--------|--------|------|
| Limit Image Side Length | 960 | Reducing this (e.g., to 640) can significantly increase speed but may lower accuracy for small text. |
| Recognition Batch Size | 6 | Increasing this (e.g., to 16) can improve multi-line text throughput without affecting accuracy. |
| CPU Threads | 0 (Auto) | ONNX Runtime inference thread count. 0 = use all CPU cores; lowering (e.g., to 4~8) reduces memory usage without affecting accuracy. |
| Enable Text Detection | On | Can be disabled for single-line plain text images to skip detection and accelerate process. |
| Correct Text Orientation | Off | Enable when recognizing tilted or upside-down text; will reduce speed. |
| PDF Text Layer Fine Align | Off | Only needed for PDF dual-layer documents. Provides 0.05/0.08/0.12 presets and "Custom" float input (0~0.5), recommended 0.08. |

### Performance Optimization Tips

- **Daily screenshots**: Select `small` + `Limit image side length 640` + `Batch size 16` for maximum speed.
- **High accuracy**: Select `medium` + `Limit image side length 960` + `Batch size 6`.
- **Single line text**: Disable "Enable Text Detection" to skip the detection phase.
- Graph optimization (highest level) and memory mode are already enabled in the code; no extra configuration is needed.

### Memory Usage Optimization

This plugin runs as a resident subprocess; memory stays occupied after the model is loaded. If memory usage feels high, try these tweaks (none affect accuracy):

- **Lower "CPU Threads"** (most effective): ONNX Runtime uses all CPU cores by default, and **each thread allocates an independent workspace buffer**. On CPUs with many cores (e.g., Intel Core Ultra 5 125H has 14 cores / 20 threads), this overhead is significant. Changing "CPU Threads" from the default 0 (Auto) to 4~8 can noticeably reduce memory usage with minimal speed loss.
- **Lower "Recognition Batch Size"**: A larger batch means larger intermediate tensors per inference. CPU mode recommends keeping the default 6.
- **Lower "Memory Limit"** (global setting): Default 8192MB; the engine subprocess auto-restarts to release memory when RSS exceeds this value. On memory-constrained machines, lower it to 2048~4096MB.
- **Choose the small model**: The small model is smaller than medium, uses less memory, and is ~3x faster.
- **Periodic Rebuild**: The plugin rebuilds the ORT session every 50 pages to release memory fragments (CUDA releases the BFC arena; CPU releases the ORT arena), so memory does not keep growing on long PDFs. The DirectML backend's VRAM is allocated on demand by the DX12 driver and does not need rebuilding.

## Architecture Overview

This plugin uses a subprocess architecture to bypass the issue where Umi-OCR's built-in Python 3.8 is too old (paddleocr 3.7.0 requires Python 3.9+):

```
Umi-OCR (Python 3.8)
  └─ PaddleOCR-json.bat
       └─ ppocr_v6_env (Python 3.10)
            └─ ppocr_v6_server.py
                 └─ PaddleOCR 3.7.0 + ONNX Runtime
                      └─ JSON stdin/stdout communication (UTF-8 encoding)
```

- **Engine Selection**: Automatically detects if `onnxruntime` is installed, prioritizing it (lightweight). If not found, it falls back to `paddlepaddle`. The GPU backend is auto-selected as CUDA → DirectML → CPU.
- **Model Management**: ONNX models are stored within the plugin's `models/` directory to avoid polluting other Umi-OCR plugins.
- **Encoding**: The server forces stdin/stdout to use UTF-8 to prevent Chinese character corruption on Windows.

## Model Storage Location

Automatically downloaded models are stored in the plugin's `models/official_models/` directory:

```
umi_plugin_v6/
└── models/
    ├── config_medium.txt
    ├── config_small.txt
    ├── configs.txt
    └── official_models/              ← Auto-downloaded models are here
        ├── PP-OCRv6_medium_det_onnx/
        │   └── inference.onnx
        ├── PP-OCRv6_medium_rec_onnx/
        │   └── inference.onnx
        ├── PP-OCRv6_small_det_onnx/
        │   └── inference.onnx
        └── PP-OCRv6_small_rec_onnx/
            └── inference.onnx
```

> Only the models for the user's selected size are downloaded; both sizes are not downloaded simultaneously.

## Regarding mkldnn Acceleration

**mkldnn (oneDNN) has no effect on this plugin.** mkldnn is the CPU acceleration backend for `paddlepaddle`, but this plugin uses the ONNX Runtime engine, bypassing `paddlepaddle`. ONNX Runtime uses its own MLAS optimization library for CPU acceleration, with high-level graph optimization and memory mode already enabled.

## FAQ

### Q: Why is the first recognition so slow?
A: The first use requires downloading the model (approx. 10-50MB). Once cached locally, it is not needed again. The default source is HuggingFace; if downloads are slow in China, set the environment variable `PADDLE_PDX_MODEL_SOURCE=bos` to use the Baidu Cloud source.

### Q: Why are there garbled Chinese characters?
A: This plugin fixes Windows encoding issues (server forces UTF-8). If you still see garbled text, please ensure you are using the latest `ppocr_v6_server.py`.

### Q: Why is GPU not working?
A: For NVIDIA GPUs, run `install_gpu.bat` to install the CUDA components; for Intel Arc / AMD and other non-NVIDIA GPUs, run `install_directml.bat` to install the DirectML components. After installation, check "Enable GPU Acceleration" in the plugin settings. The plugin auto-selects the backend based on installed components (CUDA first → DirectML → CPU). If it still doesn't work, check whether your GPU drivers are up to date. After startup, check the log line `[ppocr_v6] engine=..., gpu_backend=...` to confirm the actual backend.

> **GPU Compatibility**: CUDA acceleration is only supported on NVIDIA GPUs from the GTX 10 series onwards. GPUs before GTX 10 (e.g., GTX 9xx, 7xx) do not support modern CUDA/cuDNN; please use `install_directml.bat` (DirectML supports any DX12 GPU) or CPU mode.

### Q: Can I use GPU acceleration without an NVIDIA GPU?
A: Yes. Run `install_directml.bat` to install the DirectML backend, which supports Intel Arc (including Intel Core Ultra integrated graphics), AMD, and any DirectX 12 GPU. Check "Enable GPU Acceleration" in the plugin settings and the plugin will auto-select the DirectML backend.

### Q: What if memory usage is too high?
A: See the "Memory Usage Optimization" section above. The most effective method is lowering "CPU Threads" (default 0 = all cores; changing to 4~8 significantly reduces memory without affecting accuracy), followed by lowering "Recognition Batch Size" and the global "Memory Limit".

### Q: Is there an upper limit on "CPU Threads"? Does it support multi-core CPUs?
A: There is no upper limit; it supports CPUs with any number of cores (32-core, 64-core all work). This setting only validates as a non-negative integer, with no hardcoded upper bound in the input box. However, **it is not recommended to set it higher than the physical core count**—ONNX Runtime's `intra_op_num_threads` exceeding the physical core count will slow down due to thread context-switching overhead, with no speedup. The default `0 = Auto` already uses all physical cores, which is optimal throughput for multi-core CPUs; this setting is designed to "lower to save memory", not "raise to accelerate".

### Q: How do I switch model sizes?
A: Switch the "Model Size" in Umi-OCR's plugin settings. The engine will reload, and the corresponding model will be downloaded if it's the first time using that size.

---

# PP-OCRv5 Plugin (Legacy Version)

> The following instructions are for the PP-OCRv5 version based on the C++ PaddleOCR-json executable. The v6 version uses Python + ONNX Runtime and is recommended. v5 code is preserved in the root directory of the repository.

**Modified from [win7_x64_PaddleOCR-json](https://github.com/hiroi-sora/Umi-OCR_plugins/tree/2.0.0/win7_x64_PaddleOCR-json)**

Compatible with `Windows 7/10/11 x64`.

**Download pre-compiled plugin: [Releases](https://github.com/OneDongua/PaddleOCR-json_PP-OCRv5_umi_plugin/releases/latest)**

## Improvements over the original plugin

- **Model Upgrade**: Upgraded PaddleOCR from v2.6/v2.8 to v3.1 (PP-OCRv5), greatly improving accuracy.
- **Fast Model**: Added PP-OCRv5 mobile_rec lightweight recognition model (3-5x faster than original, accuracy higher than v3).
- **Language Separation**: Language options split into Simplified Chinese, Traditional Chinese, English, and Japanese, each with "High Accuracy" and "Fast" modes.
- **Inference Device Modes**: Added CPU-only, GPU-only, and CPU+GPU Hybrid modes.
- **TensorRT Acceleration**: Supports enabling TensorRT for GPU inference.
- **FP16 Precision**: Supports FP16 inference to accelerate GPU processing.
- **Text Detection Switch**: Ability to disable detection (`det`) to accelerate single-line text recognition.
- **Recognition Batch Size**: Adjustable `rec_batch_num` to increase throughput.
- **Vertical Text Mode**: Can rearrange recognition results in vertical reading order (right-to-left columns, top-to-bottom).
- **Parameter Passing Fix**: Fixed startup parameter passing to avoid parsing errors with paths containing spaces.
- **Dictionary File Fix**: Uses the correct PP-OCRv5 dictionary (18383 characters) instead of the old v1 dictionary (245 characters).

## Deployment Steps

### Step 1: Clone Plugin Source

```sh
git clone https://github.com/OneDongua/PaddleOCR-json_PP-OCRv5_umi_plugin.git
```

### Step 2: Prepare PaddleOCR-json Executable

#### Method 1: Direct Download
- Visit the [PaddleOCR-json Releases page](https://github.com/OneDongua/PaddleOCR-json/releases), download the latest Windows package `PaddleOCR-json_v1.4.1-ext_windows_x64.7z`, and extract it.
- Rename the extracted folder to `win7_x64_PaddleOCR-json`.

#### Method 2: Build from Source
- See the [PaddleOCR-json Windows Build Guide](https://github.com/OneDongua/PaddleOCR-json/blob/main/cpp/README.md).

### Step 3: Assemble and Place Plugin

- Copy all files from the root of this repository into `win7_x64_PaddleOCR-json`.
- Double-click `PaddleOCR-json.exe` inside `win7_x64_PaddleOCR-json` to test. A console window should open and display `OCR init completed.`.
- Copy the entire `win7_x64_PaddleOCR-json` folder into `UmiOCR-data\plugins`.

### Step 4: Download Fast Model (Optional)

To use "Fast" mode, you must download the PP-OCRv5 mobile_rec model:

1. Download the following 3 files from HuggingFace:
   - [inference.pdiparams](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec/resolve/main/inference.pdiparams) (approx. 16 MB, weights)
   - [inference.json](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec/resolve/main/inference.json) (structure)
   - [inference.yml](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec/resolve/main/inference.yml) (config)
2. Create a folder named `PP-OCRv5_mobile_rec_infer` inside the plugin's `models/` directory.
3. Place the 3 downloaded files into that folder.

## Global Settings

**Note: v5 currently only supports CPU mode; GPU models are unavailable. Please set to "CPU Only" in settings!!!**

| Setting | Default | Description |
|--------|--------|------|
| Inference Device Mode | CPU Only | CPU Only / GPU Only / CPU+GPU Hybrid (Recommended) |
| GPU ID | 0 | Select GPU index in multi-GPU environments |
| Enable MKL-DNN | On | Significantly speeds up CPU inference but increases memory usage |
| Threads | Auto | CPU inference threads; test between 8-16 for optimal value |
| Enable TensorRT | Off | Accelerates GPU inference; requires GPU exe and TensorRT environment |
| Inference Precision | FP32 | FP16 can accelerate GPU inference but may slightly lower accuracy |
| Memory Limit | Auto | Performs cleanup when engine memory exceeds limit |
| Idle Memory Cleanup | 60s | Performs cleanup after engine is idle for the specified timeout |

## Local Settings

| Setting | Default | Description |
|--------|--------|------|
| Language/Model Lib | Simplified Chinese (High Acc) | server_rec for High Acc, mobile_rec for Fast |
| Enable Text Detection | On | Can be disabled for single-line text to speed up |
| Correct Text Orientation | Off | For tilted/upside-down text; may reduce speed |
| Recognition Batch Size | 6 | Increase for throughput, but increases memory/VRAM usage |
| Vertical Text Mode | Off | Rearranges results in vertical reading order (right-to-left) |
| Limit Image Side Length | 960 | Compresses large images for speed; may lower accuracy |

## v5 Performance Optimization Tips

- Keep `Enable MKL-DNN` turned on.
- Do not maximize `Threads` blindly; test between `8-16` for the best value (this plugin limits the upper bound to 16 to avoid thread contention).
- Use the "Fast" model (`mobile_rec`) for 3-5x speed increase while maintaining higher accuracy than v3.
- For many large images, use `Limit image side length = 960` (faster) or adjust to 2880/4320 based on accuracy needs; `Custom` input is also supported.
- Only enable `Correct Text Orientation` when dealing with rotated text to avoid unnecessary overhead.
- Disable `Enable Text Detection` for single-line text to skip the detection phase and significantly accelerate.
- For long-term batch recognition, configure `Memory Limit` and `Idle Memory Cleanup` based on your machine's RAM to reduce performance jitter caused by memory bloat.

---

## Changelog

### v1.6

- **Added DirectML backend (Intel Arc / AMD GPU acceleration)**: Previously GPU acceleration only supported NVIDIA CUDA; Intel Arc integrated graphics (e.g., the Arc Graphics built into Intel Core Ultra 5/7 125H/155H), AMD, and other GPUs could not be accelerated. Added a DirectML backend supporting any DirectX 12 GPU. Run `install_directml.bat` to install `onnxruntime-directml`, then the plugin auto-selects it (CUDA first → DirectML → CPU).
  - `_select_engine()` adds a `DmlExecutionProvider` branch: bypasses paddlex `_check_device_support`'s hard requirement on `CUDAExecutionProvider` (DirectML without CUDA EP would be rejected) via an explicit `providers` list + `device_type="cpu"`; `init_ocr()` additionally passes `device="cpu"` for the DirectML path to deterministically trigger the bypass.
  - Added global `_gpu_backend` (`"cuda"` / `"directml"` / `None`) to distinguish backends. DirectML VRAM is allocated on demand by the DX12 driver with no BFC arena, so periodic session rebuilds are skipped (avoiding the needless 2~3s rebuild cost).
- **Added "CPU Threads" setting** (addressing community feedback "no multi-thread setting"): Exposes ONNX Runtime inference threads (`intra_op_num_threads` / `inter_op_num_threads`); 0 = Auto (use all CPU cores, default). Lowering it reduces memory usage—each thread allocates an independent workspace buffer, which adds up on high-core CPUs (e.g., 14-core/20-thread). The server already supported `--cpu_threads`; this release adds the UI config (`paddleocr_config.py`) and subprocess parameter wiring (`paddleocr_umi.py` ExeConfigs).
- **Memory usage optimization** (addressing community feedback "high memory usage"):
  - `_cleanup_gpu_memory()` moves `gc.collect()` ahead of the `if not _use_gpu` early return, so CPU mode also reclaims Python-level cyclic garbage (numpy result arrays, etc.), preventing memory growth on long PDFs.
  - Periodic session rebuild now also applies to CPU mode (releasing ORT CPU arena fragments); long-PDF memory no longer keeps growing.
  - Added a "Memory Usage Optimization" doc section and FAQ with actionable advice (lower CPU threads / batch size / memory limit).
- **Backend observability**: `init_ocr()` prints `[ppocr_v6] engine=onnxruntime, gpu_backend=...` to stderr at startup (captured by the `ppocr_pipe` guardian thread), so users can confirm whether GPU acceleration is in effect and which backend is used.
- Updated the "Enable GPU Acceleration" setting description to distinguish the NVIDIA (`install_gpu.bat`) and Intel Arc/AMD (`install_directml.bat`) install paths.

### v1.5

- **Fixed BFC arena failure at the end of long PDFs**: When processing hundreds of pages of English PDFs, the ORT BFC arena accumulates buffers from previous pages without releasing them (by design for reuse). On the last few pages, large nodes like `FusedMatMul` / `BiasSoftmax` fail to allocate contiguous VRAM, resulting in `BFCArena::AllocateRawInternal: Available memory of X is smaller than requested bytes of Y`. The original `_cleanup_gpu_memory()` only called `gc.collect()` + `torch.cuda.empty_cache()`, which doesn't affect the ORT arena—the session must be destroyed to return memory to CUDA.
  - Added `_rebuild_ocr()`: Destroys and rebuilds the ORT session, forcing the entire arena to release (cost: 2-3 seconds to reload model).
  - **Periodic Rebuild**: Automatically rebuilds every 50 pages (threshold for 8GB cards; can be lowered to 30 for smaller VRAM or raised to 100 for larger VRAM).
  - **Automatic Error Recovery**: Detects BFC arena errors, rebuilds the session, and retries the current page once.
  - `init_ocr()` now saves `_init_args` for reuse during rebuilds.
- **BFC arena Error Recognition**: Added keyword detection for `bfcarena` / `allocaterawinternal` / `available memory of` to provide a Chinese prompt suggesting "reducing recognition batch size".
- **High rec_batch_num Risk Warning**: At batch=30, `FusedMatMul` requests ~556MB at once; 8GB cards are recommended to keep the default 6 or max 10.
- **PDF Text Layer Fine Align Custom Values**: Added a "Custom" option to the "PDF Text Layer Fine Align" dropdown, displaying a float input box (range 0~0.5, default 0.08) for precise control beyond the 0.05/0.08/0.12 presets.

### v1.4

- **Fixed crash in rec-only mode** (`det=False`): `TextRecognition` initialization failed because `model_name` wasn't passed, causing paddlex to default to `PP-OCRv6_medium_rec`, which mismatched the local small model directory (Error code 803). Fixed by passing `model_name = rec_model`.
- **Fixed small model freeze on blank pages** (`det=True`): The small model originally used `cudnn_conv_use_max_workspace="0"`. On nearly blank pages (e.g., just a vertical line), cuDNN couldn't find a valid convolution algorithm, triggering a native crash that bypassed Python try/except. Unified to `"1"` and increased small model VRAM allocation from 40% to 50%.
- **Fixed rec-only stopping after two pages** (`det=False`): Calling `paddle.device.cuda.empty_cache()` in `_cleanup_gpu_memory()` triggered lazy initialization of the paddle CUDA context, which conflicted with the ORT `CUDAExecutionProvider` and corrupted the context. Added global `_engine` variable to skip paddle cleanup when using the ORT engine.

### v1.3

- **Dynamic GPU VRAM Allocation**: Adaptively allocates the ORT CUDA arena limit based on total VRAM (Small 50% / Medium 65%) instead of hardcoded limits.
- **Paddle VRAM Cleanup**: Added `paddle.device.cuda.empty_cache()` to `_cleanup_gpu_memory()` (optimized in v1.4 to skip for ORT engine).
- **`__ramClear` Crash Fix**: After a subprocess crash, `exit()` set `self.api.ret` to None; `__ramClear` accessed `.pid` without a null check, causing an `AttributeError`.
- **GPU VRAM Detection**: Added `_get_gpu_total_memory_gb()` with three-tier fallback (paddle → torch → nvidia-smi).

### v1.2 (2026-06-20)

**Fixed GPU mode `CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED` error**:
- `cudnn_conv_use_max_workspace=False` in v1.1 caused insufficient workspace, preventing cuDNN FE from executing convolutions.
- `cudnn_conv_algo_search=HEURISTIC` in v1.1 could also trigger cuDNN FE bugs.
- Fix: Removed `cudnn_conv_use_max_workspace` and changed `cudnn_conv_algo_search` to `DEFAULT` (uses cuDNN default algorithm without searching).
- Trade-off: VRAM usage is slightly higher than v1.1 (workspace is no longer restricted) but lower than v1.0 (still controlled by `arena_extend_strategy`). Stability is prioritized over VRAM optimization.

### v1.1 (2026-06-20)

**GPU cuDNN Loading Fix**:
- Fixed `Invalid handle. Cannot load symbol cudnnCreate` error in GPU mode.
- Root cause: `from paddleocr import` was executed before `_setup_nvidia_dlls()`, and the import process interfered with the subsequent ORT CUDA cuDNN loading.
- Fix: Call `_setup_nvidia_dlls()` to add NVIDIA DLL paths before `import paddleocr`.

**GPU VRAM Optimization**:
- Added `cudnn_conv_algo_search=HEURISTIC`: Avoids allocating large temporary workspaces when the default EXHAUSTIVE strategy searches all convolution algorithms, reducing idle VRAM usage by ~1-2GB.
- Added `cudnn_conv_use_max_workspace=False`: Prevents pre-allocation of maximum workspace, further reducing VRAM.
- Speed loss is minimal (algorithm differences are typically within 5%).

### v1.0

- Initial version of PP-OCRv6 plugin based on PaddleOCR 3.7.0 + ONNX Runtime.
- Fixed `bad allocation` in GPU multi-page PDFs (`arena_extend_strategy=kSameAsRequested` + per-page VRAM cleanup).
- Fixed `Model name mismatch` for small model (passed both `model_name` and `model_dir` when using `use_local`).
- Fixed invisible initialization errors caused by stderr being discarded (DEVNULL → PIPE + guardian thread).
- Fixed base64 temporary file path leaks (cleanup moved to `finally` block).
- Removed invalid `sess.run_options.free()` cleanup code (`RunOptions` has no `free()` method).
- Fixed bat file LF/CRLF encoding issues (changed to CRLF + No BOM + Pure ASCII).
- Organized v5 code into `umi_plugin_v5_json/` subfolder.

---

## Acknowledgements

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Baidu PaddlePaddle OCR
- [ONNX Runtime](https://onnxruntime.ai/) - Microsoft Cross-Platform Inference Engine
- [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) - Free and Open Source OCR Software
