# UmiOCR PP-OCRv6 ONNX Plugin (v2.0)

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
- **Table Recognition (v2.0 New)**: Settings UI toggle + output format dropdown (HTML/TSV); auto-detects tables in images and outputs table-formatted text (paste into Excel to form a table instantly); also available as a standalone API. Based on PP-DocLayout_plus-L layout detection + SLANet_plus table structure recognition + PP-OCRv6 cell text recognition, all ONNX models with **zero new dependencies** (no extra pip packages required).
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
- **Python**: **No pre-installation needed** (`install.bat` auto-downloads a portable Python); if Python **3.10 - 3.13** is already installed, it will be used directly; **3.9 or below / 3.14 or above** will automatically fall back to the portable version (see below).
- **OS**: Windows 10/11 x64.
- **Disk Space**: Approx. 500MB (virtual environment) + model files (default small ~30MB, optional medium ~130MB, table recognition ~131MB, downloaded on demand).

## Installation Steps

### Step 1: Place the Plugin

Copy the entire `umi_plugin_v6` folder into the Umi-OCR plugins directory:

```
Umi-OCR/
└── UmiOCR-data/
    └── plugins/
        └── umi_plugin_v6/            ← Copy here
            ├── install.bat            ← CPU installation script
            ├── install_gpu.bat        ← NVIDIA GPU acceleration script (non-RTX 50)
            ├── install_gpu_rtx50.bat  ← NVIDIA RTX 50 series only (CUDA 13)
            ├── install_directml.bat   ← DirectML (Intel Arc/AMD) script
            ├── PaddleOCR-json.bat     ← Launch script
            ├── ppocr_v6_server.py     ← OCR + table recognition server
            ├── ppocr_pipe.py          ← Recognition pipeline
            ├── paddleocr_umi.py       ← Umi-OCR plugin interface
            ├── paddleocr_config.py    ← Settings UI config (includes table recognition toggle)
            ├── verify_gpu.py          ← GPU diagnostic tool
            ├── i18n.csv               ← Multilingual text
            ├── __init__.py
            ├── models/                ← Model storage directory (auto-created)
            │   ├── config_small.txt
            │   ├── config_medium.txt
            │   └── official_models/   ← Auto-downloaded ONNX models go here
            └── ppocr_v6_env/          ← Virtual environment (created after install)
```

### Step 2: Install Environment (Out-of-the-box, no Python pre-install needed)

Double-click `install.bat`. The script will automatically:
1. **Detect an existing virtual environment**: If `ppocr_v6_env` already exists with a compatible Python version (3.10 - 3.13), dependencies are upgraded in place.
2. **Detect system Python**: Used directly if the version is within 3.10 - 3.13.
3. **Auto-download when incompatible or missing**: If system Python is 3.9 or below / 3.14 or above (some paddleocr dependencies such as PyYAML lack prebuilt wheels on those versions and source builds fail), or no Python is installed, a portable Python 3.11 (~30MB) is downloaded via [uv](https://github.com/astral-sh/uv) to rebuild the virtual environment — **nothing needs to be installed manually**.
4. Create a Python virtual environment named `ppocr_v6_env`.
5. Install `paddleocr` + `onnxruntime` dependencies (~200MB).

Installation takes approximately 1-5 minutes (depending on network speed; first run without Python adds ~30MB).

> Beginners just need to double-click `install.bat` and wait — no command-line interaction required. An old incompatible virtual environment is automatically backed up as `ppocr_v6_env_backup`; delete it manually once the new one works.

> **GPU Acceleration** (Optional, recommended for NVIDIA GPU users):
>
> **GPU Acceleration Requirements (NVIDIA CUDA path)**:
>
> | Component | Requirement | Notes |
> |------|------|------|
> | **NVIDIA GPU** | GTX 10 series or later | GTX 1050+, RTX 20/30/40 series, etc. (**RTX 50 series: see dedicated section below**). GPUs before GTX 10 (e.g., GTX 9xx/7xx) do not support modern CUDA/cuDNN |
> | **NVIDIA GPU Driver** | **R525 or higher** (Windows) | Outdated drivers are the most common cause of GPU not working! [Driver Download](https://www.nvidia.com/Download/index.aspx) |
> | CUDA Runtime 12.x | Auto-installed | Installed via pip `nvidia-cuda-runtime-cu12` by `install_gpu.bat`; **no manual CUDA Toolkit install needed** |
> | cuDNN 9.x | Auto-installed | Installed via pip `nvidia-cudnn-cu12` by `install_gpu.bat`; **no manual cuDNN download needed** |
> | onnxruntime-gpu | Auto-installed | Installed by `install_gpu.bat` (pinned to the CUDA 12 build, <1.27); includes CUDA Execution Provider |
>
> **Important**: `install_gpu.bat` automatically installs CUDA Runtime and cuDNN runtime libraries via pip (~1.6GB). **No manual NVIDIA CUDA Toolkit or cuDNN installation required.** Just ensure your GPU driver is up-to-date (R525+).
>
> Double-click `install_gpu.bat`. The script will automatically:
> 1. Uninstall any conflicting CPU version of `onnxruntime` and CUDA 13 runtime packages
> 2. Install `onnxruntime-gpu` (CUDA 12 build) + CUDA Runtime + cuDNN (~1.6GB)
> 3. Verify that CUDAExecutionProvider is available (by directly loading the ORT CUDA provider DLL, validating the full dependency chain)
>
> **When an RTX 50 series GPU is detected, the script automatically delegates to `install_gpu_rtx50.bat`** (see dedicated section below) — no manual selection needed.
>
> After installation, check "Enable GPU Acceleration" in the Umi-OCR plugin settings.
>
> **How to confirm GPU is working**: After startup, check the Umi-OCR log for:
> ```
> [ppocr_v6] engine=onnxruntime, gpu_backend=cuda
> [ppocr_v6] GPU verified: det model session uses ['CUDAExecutionProvider', ...]
> [ppocr_v6] GPU verified: rec model session uses ['CUDAExecutionProvider', ...]
> ```
> If you see `gpu_backend=None` or a CPU fallback warning, GPU is not active — follow the troubleshooting steps in the FAQ below.
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
> **Note**: `onnxruntime-directml` is mutually exclusive with `onnxruntime` / `onnxruntime-gpu`; only one can be installed in the same virtual environment. `install_directml.bat` uninstalls conflicting packages automatically before installing (since v2.1, no manual uninstall needed). NVIDIA users should still prefer `install_gpu.bat` (CUDA is faster than DirectML).
>
> **About Vulkan**: ONNX Runtime does not officially provide a Vulkan Execution Provider, so this plugin cannot support Vulkan like the ncnn engine. However, on Windows, **DirectML already achieves the same cross-vendor GPU acceleration goal as Vulkan** — both support NVIDIA/AMD/Intel Arc/integrated GPUs. The difference is that Vulkan is cross-platform (Linux/macOS) and zero-dependency, while DirectML is Windows-only and requires installing `onnxruntime-directml`. This plugin is Windows-only, so DirectML is sufficient.

### Step 3: Restart Umi-OCR

Restart Umi-OCR and select **PaddleOCR (PP-OCRv6)** in "Settings → Current Interface".

The ONNX model of the selected size (default small ~30MB) will be automatically downloaded during the first recognition and cached in the plugin's `models/` directory.

## Usage Instructions

### Model Size Selection

Select the model size in the plugin settings:

| Option | Model | Accuracy | Speed | Use Case |
|------|------|------|------|----------|
| Fast (small) | PP-OCRv6_small | High (default, sufficient for daily use) | Fast (approx 3x) | Daily use, low-spec PCs |
| High Accuracy (medium) | PP-OCRv6_medium | Highest | Slower | High accuracy needs |

> **Default small**: The entire PP-OCRv6 series has significantly improved accuracy over older versions (PP-OCRv5 and earlier). The small model is sufficient for daily scenarios. First use of small auto-downloads ~30MB; switching to medium manually downloads ~130MB. Only the selected size is downloaded.

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
| Table Recognition | Off | When enabled, auto-detects tables and outputs table-formatted text (HTML/TSV). First use auto-downloads table models (~131MB). |
| Table Output Format | HTML | Table text format: HTML (table source) / TSV (tab-separated, paste into Excel to form a table) / Off. Only effective when "Table Recognition" is enabled. |

### Table Recognition (v1.9 New)

The plugin additionally provides a table recognition entry point that converts structured tables in images into **HTML table source**. Normal OCR is unaffected; table functionality is lazy-loaded (first call takes ~10-30 seconds to load models and download missing models, subsequent recognition runs at normal speed).

**API Methods** (additional API methods similar to `runPath` / `runBytes` / `runBase64`):

| Method | Parameter | Description |
|------|------|------|
| `runTablePath(imgPath)` | Image file path | Table recognition |
| `runTableBytes(imageBytes)` | Image bytes | Table recognition |
| `runTableBase64(imageBase64)` | Image Base64 string | Table recognition |

**Return Structure**: `{"code": 100, "data": {"html": "...", "tables": [...]}}`

- `data.html`: HTML source of all detected tables in the image concatenated together
- `data.tables`: Detailed results for each table, containing:
  - `html`: HTML source of that table
  - `box`: Table position coordinates
  - `cells`: List of cells, each containing `box` ([x1,y1,x2,y2] coordinates) and `text` (recognized text for that cell)

> Table recognition does not use the "Vertical Text Mode" post-processing. The table model is affected by the "Enable GPU Acceleration" setting: when enabled, table recognition also uses the CUDA / DirectML backend.
>
> **Example**: Recognizing a table image containing `Item/Qty/Price + Apple/3/1.5...`, the `html` output is `<table><tbody><tr><td>Item</td><td>Qty</td><td>Price</td></tr>...`, and each cell in `cells` has coordinates matching the text 1:1.

### Table Recognition Toggle (v2.0 New)

Two new settings have been added below "Vertical Text Mode" in the plugin settings, allowing table-formatted text to be output directly in the standard OCR flow without calling the API separately:

| Setting | Options | Description |
|--------|------|------|
| Table Recognition | Off / On | Toggle. When enabled, tables detected in images are output as table-formatted text; plain text remains as-is; images without tables behave unchanged. Off by default; does not affect normal recognition speed. First use auto-downloads table models (~131MB). |
| Table Output Format | HTML (table source) / TSV (tab-separated) / Off | Dropdown. Effective when "Table Recognition" is enabled: HTML outputs `<table>` source for embedding in web/rich text; TSV is tab-separated text that pastes directly into Excel/WPS as a table; Off is equivalent to disabling table recognition. |

When enabled, detected tables in the recognition results are output as a **single text block** (`is_table: true`); cell text within the table region is not output again; text outside tables retains the original line-by-line output. In Vertical Text Mode, table blocks participate in reordering as a whole.

### Performance Optimization Tips

- **Daily screenshots**: Select `small` + `Limit image side length 640` + `Batch size 16` for maximum speed.
- **High accuracy**: Select `medium` + `Limit image side length 960` + `Batch size 6`.
- **Single line text**: Disable "Enable Text Detection" to skip the detection phase.
- **Plain text scenarios**: Keep "Table Recognition" off to avoid extra table detection overhead.
- **Table scenarios**: Enable "Table Recognition" and select TSV output; paste into Excel/WPS to form a table instantly. Table models are lazy-loaded; first table recognition takes ~10-30 seconds, subsequent runs are normal speed.
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
        ├── PP-OCRv6_small_rec_onnx/
        │   └── inference.onnx
        ├── PP-DocLayout_plus-L_onnx/ ← Table recognition: layout analysis (~124MB, downloaded on first table recognition)
        │   └── inference.onnx
        └── SLANet_plus_onnx/         ← Table recognition: cell structure (~7.4MB)
            └── inference.onnx
```

> Only the models for the user's selected size are downloaded; both sizes are not downloaded simultaneously.
> Table recognition models (DocLayout + SLANet) are auto-downloaded on the **first table recognition**; normal OCR is unaffected.

### Manually Downloading Table Recognition Models

If auto-download fails on first use of table recognition (network timeout, HuggingFace blocked, etc.), you can manually download and place the model files.

**Models to download (2 files, ~131MB total)**:

| Model | Purpose | Size | Baidu Cloud Direct Download URL |
|-------|---------|------|-------------------------------|
| PP-DocLayout_plus-L_onnx | Table layout detection | ~124MB | `https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_onnx_infer.tar` |
| SLANet_plus_onnx | Table structure recognition | ~7.4MB | `https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_onnx_infer.tar` |

**Manual Installation Steps**:

1. Download the two `.tar` files above (open the links in a browser, or use a download manager)
2. Extract each tar file; you will get two directories with the same names:
   - `PP-DocLayout_plus-L_onnx/` (contains `inference.onnx` and other files)
   - `SLANet_plus_onnx/` (contains `inference.onnx` and other files)
3. Place these two directories into the plugin's model directory:

```
umi_plugin_v6/
└── models/
    └── official_models/
        ├── PP-DocLayout_plus-L_onnx/   ← Place here
        │   └── inference.onnx
        └── SLANet_plus_onnx/           ← Place here
            └── inference.onnx
```

4. Restart Umi-OCR, enable the "Table Recognition" toggle, and it will work without triggering auto-download.

> **Alternative download source**: The Baidu Cloud direct links above are fast in China. If you prefer HuggingFace, you can also download from `https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L_onnx` and `https://huggingface.co/PaddlePaddle/SLANet_plus_onnx` (use `https://hf-mirror.com` as a mirror in China). Place them in the same location as above.

## Regarding mkldnn Acceleration

**mkldnn (oneDNN) has no effect on this plugin.** mkldnn is the CPU acceleration backend for `paddlepaddle`, but this plugin uses the ONNX Runtime engine, bypassing `paddlepaddle`. ONNX Runtime uses its own MLAS optimization library for CPU acceleration, with high-level graph optimization and memory mode already enabled.

## FAQ

### Q: Why is the first recognition so slow?
A: The first use requires downloading the model (default small ~30MB). Once cached locally, it is not needed again. The default source is HuggingFace; if downloads are slow in China, set the environment variable `PADDLE_PDX_MODEL_SOURCE=bos` to use the Baidu Cloud source. After the model is loaded, the plugin automatically performs a warmup inference (log line `[ppocr_v6] warmup completed`), so the first real recognition has no cold-start latency.

### Q: Why are there garbled Chinese characters?
A: This plugin fixes Windows encoding issues (server forces UTF-8). If you still see garbled text, please ensure you are using the latest `ppocr_v6_server.py`.

### Q: Why is GPU not working? No speedup after enabling hardware acceleration?
A: This is usually because the CUDA/cuDNN runtime libraries are not loaded correctly, causing ORT to silently fall back to CPU mode. Follow these steps to troubleshoot:

1. **Check the log**: After startup, check the Umi-OCR log (or stderr output) for the `[ppocr_v6] engine=..., gpu_backend=...` line:
   - `gpu_backend=cuda` + `GPU verified: ... session uses ['CUDAExecutionProvider', ...]` → GPU is working
   - `gpu_backend=None` + WARNING → **GPU is not active**, follow the steps below to fix

2. **Update GPU driver** (most common cause): CUDA 12.x runtime requires NVIDIA driver **R525 or higher**. Old drivers cause CUDA DLL loading failures, making ORT silently fall back to CPU. Go to the [NVIDIA Driver Download page](https://www.nvidia.com/Download/index.aspx) and update to the latest driver.

3. **Re-run `install_gpu.bat`**: The script automatically uninstalls conflicting CPU `onnxruntime`, reinstalls `onnxruntime-gpu` + CUDA Runtime + cuDNN, and verifies CUDAExecutionProvider availability.

4. **Check onnxruntime version**: Run `pip list | findstr onnxruntime` in `ppocr_v6_env` to confirm `onnxruntime-gpu` (not `onnxruntime`) is installed. The two are mutually exclusive; having both causes conflicts. To manually fix (RTX 50 series uses `>=1.28`, other GPUs use `<1.27`):
   ```
   ppocr_v6_env\Scripts\pip uninstall -y onnxruntime onnxruntime-gpu
   ppocr_v6_env\Scripts\pip install "onnxruntime-gpu[cuda,cudnn]>=1.23,<1.27"
   ```

5. **Verify CUDA availability**:
   ```
   ppocr_v6_env\Scripts\python -c "import onnxruntime as ort; print(ort.get_available_providers())"
   ```
   The output should include `CUDAExecutionProvider`. Note: this only proves the EP is compiled into onnxruntime — **it does NOT prove the CUDA DLLs can actually load** (ORT silently falls back to CPU). Continue with `verify_gpu.py` for the definitive check.

6. **Definitive verification + diagnostics**: Run `verify_gpu.py`:
   ```
   ppocr_v6_env\Scripts\python verify_gpu.py
   ```
   It directly loads the ORT CUDA provider DLL (CUDA major-version agnostic, works with both 12 and 13), validating the entire cudart/cublas/cudnn dependency chain, and lists the DLLs found. Exit code 0 = GPU available; on failure it reports the missing DLL groups and fix suggestions.
   - Different versions of nvidia pip packages may have different directory structures (e.g., `cudart64_12.dll` might be in `nvidia\cu13\bin\x86_64\` instead of `nvidia\cuda_runtime\bin\`, `cublas64_12.dll` might be in `nvidia\cublas\` root). v1.8 changed to recursive scanning of all subdirectories under `nvidia\` to auto-adapt to these differences.
   - Since v2.1, the script no longer hardcodes filenames like `cudart64_12.dll` (the CUDA 13 DLL is named `cudart64_13.dll`, which would cause false failures); it uses `cudart64_*.dll` wildcard matching and validates by loading the provider DLL directly.

7. **RTX 50 series specifics**: See "RTX 50 series GPU not working?" below — RTX 50 must use the CUDA 13 build (1.27+); the plain `install_gpu.bat` (CUDA 12 build) silently falls back to CPU.

> **Note**: Even when GPU is working properly, OCR model GPU utilization may be low (1-10%) because OCR inference is relatively light computation, with most time spent on CPU-side image pre/post-processing. This is normal — GPU acceleration shows up as reduced total time, not high GPU utilization.

> **GPU Compatibility**: CUDA acceleration is only supported on NVIDIA GPUs from the GTX 10 series onwards. GPUs before GTX 10 (e.g., GTX 9xx, 7xx) do not support modern CUDA/cuDNN; please use `install_directml.bat` (DirectML supports any DX12 GPU) or CPU mode.

### Q: RTX 50 series GPU not working? (issue #15)
A: RTX 50 series (Blackwell, compute capability 12.0) **must use the CUDA 13 build of onnxruntime-gpu (1.27+)**. The older CUDA 12.8 build (<=1.26) does not include sm_120 kernels, and the CUDA EP silently falls back to CPU (symptom: `install_gpu.bat` reports success and the GPU is present, but no speedup). Fix:

1. Run `install_gpu_rtx50.bat` (running `install_gpu.bat` also auto-delegates upon detecting RTX 50)
2. Ensure the NVIDIA driver is **R580+** (CUDA 13 runtime requirement)
3. The virtual environment needs Python 3.11+ — if it is 3.10, the script automatically rebuilds it with portable Python 3.12 via uv

Also, if the old `verify_gpu.py` reported "CUDA unavailable" while the GPU was actually working: the old script hardcoded the `cudart64_12.dll` filename, but in a CUDA 13 environment the DLL is named `cudart64_13.dll` — a false negative (one of the root causes of issue #15). v2.1 switched to wildcard matching + loading the provider DLL directly, making it version-agnostic.

### Q: "Python version not supported" or PyYAML build failure during installation? (issue #14)
A: Some paddleocr dependencies (e.g., PyYAML) have no prebuilt wheels for Python 3.9- / 3.14+, so pip falls back to a source build which fails. Since v2.1, `install.bat` has built-in version gating:

- **System Python within 3.10 - 3.13**: Used directly, identical to the old behavior
- **System Python 3.9 or below / 3.14 or above / not installed / a Microsoft Store stub**: Automatically downloads a portable Python 3.11 (~30MB) via [uv](https://github.com/astral-sh/uv) to create the virtual environment — **nothing needs to be installed manually**
- **Existing virtual environment with an incompatible Python** (e.g., created with Python 3.14 by an older script): Automatically backed up as `ppocr_v6_env_backup`, then rebuilt with portable 3.11

If the uv download fails, the script retries 3 times and automatically uses system proxy settings; only if it still fails does it prompt for a manual Python 3.10-3.13 installation.

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

### v2.1

- **Added Python version gating; completely fixes the Python 3.14 installation failure** (issue #14: with system Python 3.14, `install.bat` failed because PyYAML has no prebuilt wheel and the source build failed):
  - **Root cause**: PyYAML and other packages in the paddleocr dependency chain have no prebuilt wheels for Python 3.14 yet, so pip falls back to a source build (requires MSVC compiler) which fails; 3.9 and below are likewise unsupported by paddleocr 3.7.0.
  - **Fix**: `install.bat` now restricts Python to **3.10 - 3.13**. System Python within the range is used directly; otherwise (not installed, Microsoft Store stub, or out of range), a portable Python 3.11 is auto-downloaded via [uv](https://github.com/astral-sh/uv) to rebuild the virtual environment — **keeping it out-of-the-box, no manual Python installation at any point**. The uv download retries 3 times with automatic system-proxy detection; incompatible old virtual environments are backed up as `ppocr_v6_env_backup`.
  - `install_gpu.bat` / `install_gpu_rtx50.bat` / `install_directml.bat` add the same venv version gate, guiding users to re-run `install.bat` when incompatible.
- **Added RTX 50 series (Blackwell sm_120) GPU support** (issue #15):
  - **Root cause**: onnxruntime-gpu <=1.26 on PyPI is the CUDA 12.8 build, which **does not include sm_120 kernels** — the CUDA EP silently falls back to CPU on RTX 50; 1.27+ switched to the CUDA 13 build which includes sm_120 (onnxruntime#29711).
  - **New `install_gpu_rtx50.bat`**: installs `onnxruntime-gpu >=1.28` (CUDA 13 build) + cu13-series runtimes; requires driver R580+ and Python 3.11+ (a 3.10 venv is auto-rebuilt with portable 3.12 via uv).
  - **`install_gpu.bat` auto-routing**: scans the compute capability of all GPUs via `nvidia-smi` (any >= 12.0 counts as Blackwell in multi-GPU setups, or GPU name contains "RTX 50") and auto-delegates to `install_gpu_rtx50.bat`; the standard path pins `onnxruntime-gpu <1.27` (CUDA 12 build, compatible with older R525+ drivers) and cleans up leftover CUDA 13 packages.
- **Rewrote `verify_gpu.py`, fixing the CUDA 13 DLL false-negative** (issue #15, second root cause: the old script hardcoded `cudart64_12.dll`, but the CUDA 13 DLL is named `cudart64_13.dll`, causing "GPU actually works but verification fails"):
  - DLL diagnostics now use `cudart64_*.dll` wildcard matching — CUDA major-version agnostic (12/13 and future versions).
  - **The definitive check now loads `onnxruntime_providers_cuda.dll` directly**: the Windows loader resolves the whole cudart/cublas/cudnn dependency chain; if loading succeeds, the CUDA EP can be created and will not silently fall back — far more reliable than `get_available_providers()` (which only reflects the compiled-in EP list).
  - Added Blackwell detection with version-matching hints: RTX 50 + ORT <1.27 warns about missing sm_120 kernels and points to `install_gpu_rtx50.bat`.
- **`install_directml.bat` improvements**: auto-uninstalls conflicting `onnxruntime` / `onnxruntime-gpu` and NVIDIA runtime packages before installing (previously manual); unified venv check (run `install.bat` first); fixed garbled Chinese text.

### v2.0

- **Added Table Recognition switch in settings UI** (addressing issue #12): located below Vertical Text Mode in plugin settings (same area as Enable GPU Acceleration). When enabled, the standard OCR flow auto-detects tables in images and outputs the table as a **single text block** (is_table: true); cell text inside the table region is deduplicated (not output twice); plain text and images without tables behave exactly as before. Off by default; normal recognition speed unaffected.
- **Added Table Output Format dropdown** (independent of the switch): options are **HTML (table source)** / **TSV (tab-separated)** / **Off**. HTML is suitable for embedding in web/rich text; TSV pastes directly into Excel/WPS as a table. Choosing Off is equivalent to disabling table recognition.
- Table block coordinates come from the layout-detected table region (4-point polygon), interleaved with plain text by center-y; the standalone runTablePath etc. APIs remain unchanged.

### v1.9


- **Added Table Recognition**: Recognize structured tables in images as HTML table source. Based on `PP-DocLayout_plus-L` (layout analysis) + `SLANet_plus` (table structure) + PP-OCRv6 (cell text recognition), all using **ONNX models + onnxruntime engine**, **zero new dependencies** (no extra pip packages required; ~131MB of table models auto-downloaded on first use).
  - New plugin API entries `runTablePath` / `runTableBytes` / `runTableBase64`, input identical to normal OCR; returns `{code, data: {html, tables}}`, where `tables[].cells[]` contains cell coordinates `box` and text `text`.
  - Table recognition is lazy-loaded: the pipeline is created only on first call (~10-30s); normal OCR startup and runtime are completely unaffected, no extra memory usage.
  - Bypasses paddlex's mandatory `paddlex[ocr]` full-extra dependency check during `table_recognition` pipeline init via `_patch_table_deps()` (the required components aren't actually needed at runtime), guaranteeing zero extra installs.
  - Reuses `_select_engine()` engine_config: table recognition uses CUDA / DirectML backend too when "Enable GPU Acceleration" is on.
  - `_collect_table_results()` normalizes output: strips residual `<html><body>` wrappers from HTML; spatial-matches cell boxes by detection-box centroids so `cells[].text` maps 1:1 to cells.
- **Default model changed to small**: "Model Size" now defaults to "Fast (small)"—small's accuracy is already higher than the old PP-OCRv6_medium lineage (PP-OCRv6 series accuracy is significantly improved), sufficient for daily use. Switch to medium manually only when higher accuracy is needed. First use of small auto-downloads ~30MB of models.
- **Startup warmup**: after model init, immediately runs one inference on a synthetic image to pre-complete ONNX Runtime / CUDA arena allocation and kernel compilation, eliminating cold-start latency on the first real recognition. Warmup failure doesn't affect usage (logged only).

### v1.8

- **Fix CUDA/cuDNN DLL path search defect** (thoroughly resolves issue #10: user reported "no speedup with hardware acceleration", GPU usage only 1-3%, CPU 80%):
  - **Root cause**: `_setup_nvidia_dlls()` and `verify_gpu.py`'s `setup_nvidia_dlls()` only scanned the single `nvidia\<sub>\bin\` layout, failing to adapt to directory differences across nvidia pip package versions. Actual abnormal structures encountered: `nvidia\cu13\bin\x86_64\cudart64_12.dll` (cu13 package has an extra `x86_64` layer) and `nvidia\cublas\cublas64_12.dll` (cublas package has no `bin` subdir; DLLs live in the package root). Missing DLLs → ORT silently falls back to CPU when creating sessions → GPU inactive. After the user manually fixed paths, GPU usage jumped to 80-90%.
  - **Fix**: changed "scan only `nvidia\<sub>\bin\`" to recursive `os.walk` over `nvidia\`, adding every directory containing `.dll` files to `os.add_dll_directory()` and `PATH`. Works regardless of whether DLLs are in `bin\`, `bin\x86_64\`, or the package root; users no longer need to move DLLs manually.
  - **Affected files**: `_setup_nvidia_dlls()` in `ppocr_v6_server.py` (runtime) and `setup_nvidia_dlls()` + `find_nvidia_bin_dirs()` in `verify_gpu.py` (install-time verification). Both updated in sync.
  - **FAQ update**: "GPU not working?" troubleshooting adds step 6 "Check CUDA/cuDNN DLL paths", explaining v1.8 auto-adapts to different directory structures and pointing to `verify_gpu.py` for diagnosis.

### v1.7

- **Out-of-the-box: no Python preinstall needed** (lowers the barrier for beginners): `install.bat` upgraded to out-of-the-box mode—if no system Python is detected, it auto-downloads a portable Python 3.11 (~30MB) via [uv](https://github.com/astral-sh/uv) and creates the virtual environment, with no manual Python install needed. Uses the system Python directly if present. Beginners just double-click `install.bat` and wait; zero command-line interaction.
- **Fix GPU acceleration silently falling back to CPU** (addresses issue #10: user reported "no speedup with hardware acceleration", CPU 77% while GPU only 1-3%; root cause: CUDA/cuDNN runtime libs not loaded correctly, ORT silently fell back to CPU with no user visibility):
  - **`_select_engine()` adds a GPU-unavailable warning**: when `use_gpu=True` but neither `CUDAExecutionProvider` nor `DmlExecutionProvider` is available, prints a prominent multi-line WARNING to stderr listing available providers, the CPU fallback fact, and fix steps (run install_gpu.bat / install_directml.bat + update GPU driver).
  - **New `_verify_gpu_session()`**: best-effort reads paddlex-internal `ONNXRuntimeRunner.session.get_providers()` at the end of `init_ocr()` to verify the providers the ORT session *actually* uses (not the global availability list). Even if `get_available_providers()` reports CUDA available, session creation can silently fall back to CPU due to DLL version mismatches etc.—this check makes the fallback visible. Prints `GPU verified: ... session uses ['CUDAExecutionProvider', ...]` on success, a prominent WARNING on failure.
- **Improved `install_gpu.bat`**:
  - Auto-uninstalls the conflicting CPU `onnxruntime` before installing (`pip uninstall -y onnxruntime`), avoiding wrong-DLL loading when both are present.
  - Displays GPU driver version and model (`nvidia-smi --query-gpu=driver_version,name`) so users can confirm the driver meets CUDA 12.x's R525+ requirement.
  - Verifies CUDAExecutionProvider is actually usable after install (`exit(0 if 'CUDAExecutionProvider' in ps else 1)`), with explicit error message and fix advice instead of silently continuing.
  - Reminds users to check `[ppocr_v6] GPU verified: ...` in the Umi-OCR log to confirm GPU is active.
- **Updated GPU acceleration docs**: README adds a "GPU Acceleration Requirements" table clearly listing the five requirements and their sources: NVIDIA GPU, GPU driver (R525+), CUDA Runtime 12.x, cuDNN 9.x, onnxruntime-gpu; FAQ "GPU not working?" is now a step-by-step troubleshooting guide (check log → update driver → reinstall → check pip packages → verify CUDA).

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
