"""PP-OCRv6 Python 管道服务端 - 兼容 PaddleOCR-json 协议
用法：python ppocr_v6_server.py --config_path=models/config_simplified_medium.txt

基于 PaddleOCR 3.7.0 API，自动兼容 onnxruntime（优先）与 paddle（回退）两种引擎。
"""
import sys
import os
import json
import argparse
import base64
import tempfile
import gc

# 强制 stdin/stdout 使用 UTF-8 编码，避免 Windows 下中文乱码
sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

# 关键修复：移除可能包含本地 paddleocr.py 的路径，避免遮蔽系统 paddleocr 包
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _script_dir and p != '' and p != '.']

# 让 paddleocr 自动下载的模型存放到插件目录（便携），而非用户主目录 ~/.paddlex
# 必须在 import paddleocr/paddlex 之前设置
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", os.path.join(_script_dir, "models"))

_ocr = None  # PaddleOCR 实例（det=True）
_recognizer = None  # TextRecognition 实例（det=False）
_det = True
# det 框内缩比例（抵消 DBNet expand_ratio，让 box 更贴合真实文字，改善 PDF 文本层对齐）
# 0 表示不收缩；典型值 0.08~0.12（各点向重心移动该比例，框宽高约收缩 2 倍该值）
_shrink_ratio = 0.0

# 表格识别（懒加载）：首次收到表格命令时才创建 paddlex table_recognition 管道
_table_pipe = None
_table_model_size = "small"
_patched_table_deps = False
# 表格识别开关（设置界面）：True 时 run_ocr 自动检测表格并输出格式文本
_table_mode = False
# 表格输出格式：html / tsv
_table_format = "html"


def _setup_nvidia_dlls():
    """把 pip 安装的 nvidia CUDA/cuDNN DLL 路径加入搜索路径（GPU 加速所需）。

    递归扫描 nvidia\\ 下所有子目录，把包含 .dll 文件的目录加入 DLL 搜索路径。
    兼容不同版本 nvidia pip 包的目录结构差异（issue #10 根因）：
      - 正常结构：nvidia\\<sub>\\bin\\*.dll
      - cu13 包结构：nvidia\\cu13\\bin\\x86_64\\*.dll
      - 旧版 cublas：nvidia\\cublas\\*.dll（无 bin 子目录）
    之前版本只扫描 nvidia\\<sub>\\bin\\，导致后两种结构的 DLL 无法被找到，
    ORT 创建 session 时静默回退到 CPU。
    """
    try:
        import sysconfig
        site_dir = sysconfig.get_paths()["purelib"]
        nvidia_base = os.path.join(site_dir, "nvidia")
        if not os.path.isdir(nvidia_base):
            return
        added = set()
        for root, _dirs, files in os.walk(nvidia_base):
            # 只把包含 .dll 文件的目录加入搜索路径（跳过 include/lib 等无 dll 目录）
            if any(f.lower().endswith(".dll") for f in files):
                if root not in added:
                    try:
                        os.add_dll_directory(root)
                    except Exception:
                        pass
                    os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
                    added.add(root)
    except Exception:
        pass


_use_gpu = False
_engine = None  # 推理引擎名称（"onnxruntime" / "paddle" / None）
_gpu_backend = None  # GPU 后端类型："cuda" / "directml" / None（CPU 或无引擎）
_init_args = None  # 保存 init_ocr 的 args，供周期性重建 ORT session 使用
_page_count = 0  # 已处理页数计数器，达阈值时重建 session 释放 BFC arena 碎片
# 每 N 页重建一次 ORT session，强制释放 BFC arena 累积的显存碎片。
# 8GB 显卡建议 50；更小显存可调小至 30，更大显存可调大到 100。
_REBUILD_PAGE_THRESHOLD = 50


def _cleanup_gpu_memory():
    """释放 GPU 显存缓存，防止多页 PDF 识别时显存碎片累积导致 bad allocation"""
    # gc.collect() 对长驻子进程始终有用（回收 Python 层循环引用垃圾，如 numpy
    # 结果数组）。CPU 模式下也执行，避免多页 PDF 累积 Python 层垃圾导致内存上涨。
    # ONNX Runtime session 由 Python 引用计数管理，gc.collect() 会回收无引用的
    # session 及其显存。不要尝试手动调用 session.run_options.free()——RunOptions
    # 没有公开的 free() 方法，调用会抛 AttributeError 被静默吞掉，实际无效。
    gc.collect()
    if not _use_gpu:
        return
    # 释放 PyTorch 的 CUDA 缓存（paddleocr 部分模块可能用 torch）
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    # 释放 PaddlePaddle 的 CUDA 缓存（paddleocr 推理时 paddle 会缓存显存，
    # torch.empty_cache 对 paddle 无效，必须调 paddle 自己的释放 API）
    # 注意：当引擎是 onnxruntime 时，paddle 未参与推理，其 CUDA 上下文可能未
    # 完全初始化。此时调 paddle.device.cuda.empty_cache() 会触发 paddle 延迟
    # 初始化 CUDA 上下文，与 ORT 的 CUDAExecutionProvider 上下文冲突，导致
    # CUDA 上下文损坏。后续 ORT session.run() 会因上下文损坏而 native 崩溃
    # （绕过 Python try/except，表现为 rec-only 模式识别两页后进程死亡）。
    # det=True 路径下 PaddleOCR pipeline 会正常初始化 paddle CUDA 上下文，
    # 所以不受影响；仅 onnxruntime 引擎需要跳过。
    if _engine == "onnxruntime":
        return
    try:
        import paddle
        if hasattr(paddle, "device") and hasattr(paddle.device, "cuda"):
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


def _get_gpu_total_memory_gb():
    """获取 GPU 总显存（GB），用于动态计算 gpu_mem_limit。
    检测顺序：paddle → torch → nvidia-smi 命令 → 默认 8GB。
    """
    # 1. paddle
    try:
        import paddle
        if hasattr(paddle, "device") and hasattr(paddle.device, "cuda"):
            props = paddle.device.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
    except Exception:
        pass
    # 2. torch
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    # 3. nvidia-smi 命令
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip().split("\n")[0]) / 1024
    except Exception:
        pass
    return 8.0  # 默认 8GB


def _rebuild_ocr():
    """重建 ORT session，强制释放 BFC arena 累积的显存碎片。

    ORT 的 BFC arena 设计上持有显存不释放（为下次 run 复用 buffer），
    被 gpu_mem_limit 封顶后，跑几百页 PDF 会被前面页的各种形状分配占满，
    最后几页找不到连续空间就报 BFCArena::AllocateRawInternal 错误
    （FusedMatMul / BiasSoftmax 等大块节点申请失败）。

    _cleanup_gpu_memory() 只能 gc + torch.empty_cache，对 ORT 的 arena
    无效——必须销毁 session 才能让 arena 归还 CUDA。代价是 2~3 秒重新加载模型。

    返回 True 表示重建成功，False 表示失败（调用方应给出错误提示）。
    """
    global _ocr, _recognizer, _page_count
    if _init_args is None:
        return False
    try:
        _ocr = None
        _recognizer = None
        gc.collect()
        # ORT session 销毁后其 arena 才真正归还 CUDA，再清 torch 缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        init_ocr(_init_args)
        _page_count = 0
        return True
    except Exception as e:
        print(f"ORT session rebuild failed: {e}", file=sys.stderr, flush=True)
        return False


def _select_engine(use_gpu, cpu_threads=None, model_size="medium"):
    """自动选择推理引擎。优先 onnxruntime（轻量），未安装则回退 paddle。

    GPU 后端选择顺序：CUDA（NVIDIA，最快）→ DirectML（Intel Arc / AMD 等任意
    DirectX 12 GPU）→ CPU。三者均通过显式 providers 列表指定，paddlex 的
    engine_config 会原样传给 ORT InferenceSession。

    注意：mkldnn（oneDNN）是 paddlepaddle 的 CPU 加速后端，对 onnxruntime 无效。
    onnxruntime CPU 使用自有的 MLAS 优化库；这里通过开启图优化最高级、内存模式
    与显式线程数来提升 CPU 推理速度（无需下载任何额外文件）。
    """
    global _gpu_backend
    _setup_nvidia_dlls()
    try:
        import onnxruntime as ort
    except ImportError:
        _gpu_backend = None
        if use_gpu:
            print("[ppocr_v6] WARNING: use_gpu=True but onnxruntime is not installed. "
                  "GPU acceleration unavailable, falling back to CPU. "
                  "Please run install_gpu.bat (NVIDIA) or install_directml.bat (Intel/AMD).",
                  file=sys.stderr, flush=True)
        return None, None
    providers = ort.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" not in providers and "DmlExecutionProvider" not in providers:
        # 用户请求 GPU 但没有任何 GPU EP 可用——给出醒目警告，避免用户以为
        # GPU 已生效却看不到加速效果（issue #10：用户反馈"打开硬件加速速度也不明显"，
        # 根因是 CUDA/cuDNN 运行库未正确安装，ORT 静默回退到 CPU）。
        print("=" * 70, file=sys.stderr, flush=True)
        print("[ppocr_v6] WARNING: GPU acceleration requested (use_gpu=True) but "
              "NO GPU ExecutionProvider is available!", file=sys.stderr, flush=True)
        print(f"[ppocr_v6] Available providers: {providers}", file=sys.stderr, flush=True)
        print("[ppocr_v6] Falling back to CPU mode. GPU will NOT be used.", file=sys.stderr, flush=True)
        print("[ppocr_v6] To fix this:", file=sys.stderr, flush=True)
        print("[ppocr_v6]   NVIDIA GPU: run install_gpu.bat to install onnxruntime-gpu + CUDA + cuDNN",
              file=sys.stderr, flush=True)
        print("[ppocr_v6]   Intel/AMD GPU: run install_directml.bat to install onnxruntime-directml",
              file=sys.stderr, flush=True)
        print("[ppocr_v6]   Also ensure your GPU driver supports CUDA 12.x (NVIDIA) or DirectX 12 (Intel/AMD)",
              file=sys.stderr, flush=True)
        print("=" * 70, file=sys.stderr, flush=True)
    if use_gpu and "CUDAExecutionProvider" in providers:
        # GPU 模式 —— NVIDIA CUDA 后端
        _gpu_backend = "cuda"
        # arena_extend_strategy=kSameAsRequested：按实际需求分配显存，避免默认
        #   kNextPowerOfTwo 按2的幂分配导致显存碎片化（多页PDF第2页起报 bad allocation）
        # enable_mem_pattern=False：避免不同页 batch size 变化导致内存模式不匹配
        # cudnn_conv_algo_search=DEFAULT：不搜索卷积算法，用cuDNN默认算法
        #   避免EXHAUSTIVE/HEURISTIC搜索路径触发cuDNN FE的CUDNN_BACKEND_API_FAILED错误
        # cudnn_conv_use_max_workspace：cuDNN 卷积 workspace 模式（显存大户）
        #   统一用 "1"：给 cuDNN 充足 workspace 以找到合法卷积算法。
        #   small 模型在几近空白页（仅竖线）上，卷积激活值分布异常，
        #   workspace="0" 会导致 cuDNN 找不到有效算法而触发 native 崩溃
        #   （绕过 Python try/except，表现为进程死亡、GPU 占用归零）。
        use_max_ws = "1"
        # gpu_mem_limit：按显卡总显存动态分配 ORT CUDA arena 上限（字节），
        #   充分发挥不同显卡性能，而非硬编码固定值。
        #   small 模型：50%（workspace="1" 需更多显存，原 40% 偏紧）
        #   medium 模型：65%（留 35% 给 cuDNN、CUDA context、paddle 缓存等）
        #   8GB 显卡示例：small 4.0GB / medium 5.2GB
        #   12GB 显卡示例：small 6.0GB / medium 7.8GB
        gpu_total_gb = _get_gpu_total_memory_gb()
        ratio = 0.65 if model_size == "medium" else 0.50
        gpu_mem_limit = int(gpu_total_gb * ratio * 1024 * 1024 * 1024)
        cfg = {
            "device_type": "gpu",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "provider_options": [
                {
                    "device_id": 0,
                    "gpu_mem_limit": gpu_mem_limit,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "cudnn_conv_use_max_workspace": use_max_ws,
                },
                {},
            ],
            "graph_optimization_level": 99,  # ORT_ENABLE_ALL
            "enable_mem_pattern": False,
        }
    elif use_gpu and "DmlExecutionProvider" in providers:
        # GPU 模式 —— DirectML 后端（Intel Arc / AMD 等任意 DirectX 12 GPU）
        # 适用场景：没有 NVIDIA 显卡的用户（如 Intel Core Ultra 集成 Arc 核显、
        # AMD 独显）。需先运行 install_directml.bat 安装 onnxruntime-directml。
        #
        # 关键绕过：paddlex 的 ONNXRuntimeEngine._check_device_support 在
        # device_type="gpu" 时强制要求 CUDAExecutionProvider（DirectML 无该 EP
        # 会被拒）。这里 device_type 设为 "cpu"，使 _check_device_support 提前
        # 返回；真正的执行后端由 providers 列表中的 DmlExecutionProvider 决定。
        # init_ocr 还会向 PaddleOCR 传 device="cpu"，确定性触发该绕过（即使
        # 用户误装了 paddlepaddle-gpu 也能正常工作）。
        # device_type 无论如何都会被 paddlex._apply_device 按 get_default_device()
        # 覆盖为 "cpu"（本插件安装的是 CPU 版 paddlepaddle，is_compiled_with_cuda()
        # 为 False），因此 DirectML 推理不受影响。
        _gpu_backend = "directml"
        # DirectML 显存由 DX12 驱动管理，无 BFC arena / gpu_mem_limit 概念，
        # 也不需要 cuDNN 相关参数。enable_mem_pattern=False：与 CUDA 同理，
        # 避免不同页 batch size 变化导致内存模式不匹配。
        cfg = {
            "device_type": "cpu",
            "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
            "provider_options": [
                {"device_id": 0},
                {},
            ],
            "graph_optimization_level": 99,  # ORT_ENABLE_ALL
            "enable_mem_pattern": False,
        }
    else:
        # CPU 模式：图优化最高级 + 内存模式（CPU 推理稳定提速，无副作用）
        _gpu_backend = None
        cfg = {
            "device_type": "cpu",
            "providers": ["CPUExecutionProvider"],
            "graph_optimization_level": 99,  # ORT_ENABLE_ALL
            "enable_mem_pattern": True,
            "enable_cpu_mem_arena": True,
        }
    # CPU 线程数：显式设置时生效，否则用 onnxruntime 默认（全部逻辑核）
    # 0=自动。调小可降低内存占用（每线程分配独立工作区缓冲区），不影响精度。
    if cpu_threads and cpu_threads > 0:
        cfg["intra_op_num_threads"] = cpu_threads
        cfg["inter_op_num_threads"] = cpu_threads
    return "onnxruntime", cfg


def _verify_gpu_session(ocr_or_recognizer, det=True):
    """初始化后验证 ORT session 是否真正使用了 GPU ExecutionProvider。

    paddlex 内部的 ONNXRuntimeRunner 会持有 InferenceSession 对象，其
    get_providers() 返回 session 实际使用的 providers（而非全局可用列表）。
    若 CUDA EP 在 import 时可用但 session 创建时静默回退到 CPU（如 DLL
    版本不匹配、驱动过旧），此处会检测到并给出醒目警告。

    访问路径（paddlex 3.7.x 内部结构，best-effort，失败则静默跳过）：
      PaddleOCR.paddlex_pipeline._pipeline.text_det_model._predictor._runner.session
      PaddleOCR.paddlex_pipeline._pipeline.text_rec_model._predictor._runner.session
      TextRecognition.paddlex_pipeline._pipeline.text_rec_model._predictor._runner.session

    issue #10：用户反馈"打开硬件加速速度也不明显"，根因是 CUDA/cuDNN 未正确
    安装，ORT 静默回退到 CPU。此函数让回退变得可见。
    """
    try:
        pipeline = ocr_or_recognizer.paddlex_pipeline
        # _OCRPipeline 才有 text_det_model / text_rec_model；外层包装器可能没有
        inner = getattr(pipeline, "_pipeline", pipeline)
        model_attrs = []
        if det and hasattr(inner, "text_det_model"):
            model_attrs.append(("det", inner.text_det_model))
        if hasattr(inner, "text_rec_model"):
            model_attrs.append(("rec", inner.text_rec_model))
        for label, model in model_attrs:
            pred = getattr(model, "_predictor", None)
            if pred is None:
                continue
            runner = getattr(pred, "_runner", None)
            if runner is None:
                continue
            session = getattr(runner, "session", None)
            if session is None or not hasattr(session, "get_providers"):
                continue
            actual = session.get_providers()
            if _gpu_backend in ("cuda", "directml"):
                expected = ("CUDAExecutionProvider" if _gpu_backend == "cuda"
                            else "DmlExecutionProvider")
                if expected in actual:
                    print(f"[ppocr_v6] GPU verified: {label} model session uses {actual}",
                          file=sys.stderr, flush=True)
                else:
                    print("=" * 70, file=sys.stderr, flush=True)
                    print(f"[ppocr_v6] WARNING: GPU backend={_gpu_backend} was requested, "
                          f"but {label} model session fell back to CPU!", file=sys.stderr, flush=True)
                    print(f"[ppocr_v6] Session providers: {actual}", file=sys.stderr, flush=True)
                    print("[ppocr_v6] This usually means CUDA/cuDNN DLLs failed to load at "
                          "session creation time.", file=sys.stderr, flush=True)
                    print("[ppocr_v6] Please check:", file=sys.stderr, flush=True)
                    print("[ppocr_v6]   1. GPU driver is up-to-date (NVIDIA: supports CUDA 12.x)",
                          file=sys.stderr, flush=True)
                    print("[ppocr_v6]   2. Re-run install_gpu.bat to reinstall CUDA/cuDNN runtime",
                          file=sys.stderr, flush=True)
                    print("=" * 70, file=sys.stderr, flush=True)
    except Exception:
        # 内部结构访问失败时静默跳过——不影响正常推理流程
        pass


def parse_config(args):
    config = {"model_size": "small", "lang": "ch"}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.config_path:
        config_file = args.config_path
        if not os.path.isabs(config_file):
            config_file = os.path.join(script_dir, config_file)
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        key, val = parts[0], parts[1]
                        if key in ("model_size", "lang"):
                            config[key] = val
    return config


def init_ocr(args):
    global _ocr, _recognizer, _det, _use_gpu, _shrink_ratio, _engine, _gpu_backend, _init_args
    _init_args = args  # 保存以供周期性重建 ORT session 使用
    # 必须在 import paddleocr 之前添加 NVIDIA DLL 路径。
    # 否则 paddleocr/paddlex 导入过程会干扰后续 ORT CUDA 加载 cuDNN，
    # 导致 "Invalid handle. Cannot load symbol cudnnCreate" 错误。
    _setup_nvidia_dlls()
    from paddleocr import PaddleOCR, TextRecognition

    config = parse_config(args)
    model_size = config["model_size"]
    lang = config["lang"]
    global _table_model_size
    _table_model_size = model_size
    global _table_mode, _table_format
    _table_mode = bool(getattr(args, "table_mode", False))
    _table_format = str(getattr(args, "table_format", "html") or "html").lower()
    det = args.det if args.det is not None else True
    cls = args.cls if args.cls is not None else False
    rec_batch_num = args.rec_batch_num or 6
    limit_side_len = args.limit_side_len or 960
    use_gpu = bool(args.use_gpu)
    _use_gpu = use_gpu
    # A1: det 框内缩比例（0=关闭，典型 0.08~0.12）
    _shrink_ratio = max(0.0, float(getattr(args, "shrink_poly_ratio", 0.0) or 0.0))

    det_model = f"PP-OCRv6_{model_size}_det"
    rec_model = f"PP-OCRv6_{model_size}_rec"
    engine, engine_config = _select_engine(use_gpu, args.cpu_threads, model_size)
    _engine = engine  # 记录引擎类型，供 _cleanup_gpu_memory 判断是否跳过 paddle 清理
    # _gpu_backend 已在 _select_engine 中设置（"cuda" / "directml" / None）
    # 输出后端信息到 stderr，便于用户确认 GPU 加速是否生效（被 ppocr_pipe 守护线程捕获）
    print(f"[ppocr_v6] engine={engine}, gpu_backend={_gpu_backend}", file=sys.stderr, flush=True)

    engine_kwargs = {}
    if engine:
        engine_kwargs["engine"] = engine
        engine_kwargs["engine_config"] = engine_config

    # DirectML 绕过 paddlex 的 CUDA 强制检查：传 device="cpu" 让 _apply_device 把
    # engine_config 的 device_type 置为 "cpu"，_check_device_support 提前返回；
    # 真正使用 DmlExecutionProvider（由 providers 列表决定）。确定性触发，即使
    # 用户误装 paddlepaddle-gpu 也能正常工作。CUDA/CPU 路径不传 device，保持原行为。
    if _gpu_backend == "directml":
        engine_kwargs["device"] = "cpu"

    # 优先使用本地 ONNX 模型目录（统一管理），不存在则回退自动下载
    script_dir = os.path.dirname(os.path.abspath(__file__))
    det_onnx_dir = os.path.join(script_dir, "models", f"PP-OCRv6_{model_size}_det_onnx")
    rec_onnx_dir = os.path.join(script_dir, "models", f"PP-OCRv6_{model_size}_rec_onnx")
    use_local = os.path.isdir(det_onnx_dir) and os.path.isdir(rec_onnx_dir)

    _det = det
    if det:
        # 完整管道：文本检测 + 识别
        # cls（纠正文本方向）映射到 use_textline_orientation
        ocr_args = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": cls,
            "lang": lang,
            "text_det_limit_side_len": limit_side_len,
            "text_recognition_batch_size": rec_batch_num,
        }
        if use_local:
            # 本地目录 + 模型名都要传，否则 paddlex 默认用 medium，与 small 目录不匹配
            ocr_args["text_detection_model_name"] = det_model
            ocr_args["text_detection_model_dir"] = det_onnx_dir
            ocr_args["text_recognition_model_name"] = rec_model
            ocr_args["text_recognition_model_dir"] = rec_onnx_dir
        else:
            ocr_args["text_detection_model_name"] = det_model
            ocr_args["text_recognition_model_name"] = rec_model
        ocr_args.update(engine_kwargs)
        _ocr = PaddleOCR(**ocr_args)
    else:
        # 仅识别：跳过检测，适合单行文本
        # TextRecognition 不接受 text_recognition_batch_size / lang 等参数
        rec_args = {}
        if use_local:
            # 本地目录 + 模型名都要传，否则 paddlex 默认用 medium，与 small 目录不匹配
            rec_args["model_name"] = rec_model
            rec_args["model_dir"] = rec_onnx_dir
        else:
            rec_args["model_name"] = rec_model
        rec_args.update(engine_kwargs)
        _recognizer = TextRecognition(**rec_args)

    # 验证 ORT session 是否真正使用了 GPU EP（issue #10）。
    # _select_engine 已检查 get_available_providers()，但 session 创建时可能
    # 因 DLL 版本不匹配等原因静默回退到 CPU。此处 best-effort 检查实际 providers，
    # 让用户能从日志确认 GPU 是否真正生效。
    if _engine == "onnxruntime" and _gpu_backend:
        _verify_gpu_session(_ocr if _ocr else _recognizer, det=det)

    # 预热：对刚创建的模型跑一次最简推理，提前完成 ONNX Runtime 的
    # ORT heap arena / CUDA BFC arena 分配与 kernel 编译。否则首次真实识别
    # 会包含这部分开销（冷启动第一张图明显偏慢）。失败仅记日志，不影响使用。
    try:
        import time as _time
        import numpy as _np
        _t0 = _time.time()
        # 640x192 白底 + 几行黑条纹近似文本，足够触发 det + rec 完整推理链
        _warm = _np.full((192, 640, 3), 255, dtype=_np.uint8)
        for _y in (60, 90, 120, 150):
            _warm[_y:_y + 12, 20:620, :] = 0
        if det:
            list(_ocr.predict(_warm))
        else:
            list(_recognizer.predict(_warm))
        print(f"[ppocr_v6] warmup completed in {_time.time() - _t0:.2f}s",
              file=sys.stderr, flush=True)
    except Exception as _e:
        print(f"[ppocr_v6] warmup skipped: {_e}", file=sys.stderr, flush=True)


def _get(page, key, default=None):
    if isinstance(page, dict):
        return page.get(key, default)
    return getattr(page, key, default)


def _shrink_poly(poly, ratio):
    """将 4 点检测框向重心收缩 ratio 比例，抵消 DBNet 后处理的 expand_ratio。

    用于让返回的 box 更贴合真实文字范围，改善 Umi-OCR 生成 layered.pdf 时
    文本层与图像的对齐（det 框默认会比真实文字外扩一圈，导致字号被高估、
    行末字符超出图像文字）。

    ratio=0.1 表示各点向重心移动 10%，框宽高约收缩 20%。
    ratio<=0 时原样返回（整型化）。

    注意：poly 可能是 numpy 数组，不能直接用 `if not poly` 判空
    （会触发 ValueError: ambiguous truth value），须用 len() 或 is None。
    """
    if poly is None or ratio <= 0:
        return [[int(p[0]), int(p[1])] for p in poly] if poly is not None else []
    try:
        import numpy as np
        p = np.asarray(poly, dtype=np.float32)
        if p.ndim != 2 or p.shape[0] < 3:
            return [[int(pt[0]), int(pt[1])] for pt in p]
        cx = float(p[:, 0].mean())
        cy = float(p[:, 1].mean())
        shrunk = p + (np.array([cx, cy], dtype=np.float32) - p) * ratio
        return [[int(round(pt[0])), int(round(pt[1]))] for pt in shrunk]
    except Exception:
        return [[int(p[0]), int(p[1])] for p in poly]


def _validate_rec_polys(rec_polys, dt_polys, texts):
    """校验 rec_polys 是否为原图坐标，决定是否优先使用。

    paddleocr 主流版本中 rec_polys 与 dt_polys 同为原图坐标且数值一致，
    应优先使用 rec_polys（识别阶段对齐后的框，更贴字）。
    但少数版本可能把 rec_polys 输出为裁剪/透视变换后的局部框，此时坐标
    范围与 dt_polys 差异显著，必须回退 dt_polys，否则 box 会指向错误位置。

    校验规则：
      1. 三者长度一致；
      2. rec_polys / dt_polys 形状一致；
      3. 二者整体 bbox 的宽高比在 [0.8, 1.2] 区间（同一坐标系）。

    注意：rec_polys / dt_polys 可能是 numpy 数组，不能直接 `if not rec_polys`
    判空（触发 ValueError: ambiguous truth value），须用 is None / len() 判断。
    """
    if rec_polys is None or dt_polys is None or not texts:
        return False
    if len(rec_polys) == 0 or len(dt_polys) == 0:
        return False
    n = len(texts)
    if len(rec_polys) != n or len(dt_polys) != n:
        return False
    try:
        import numpy as np
        rp = np.asarray(rec_polys, dtype=np.float32)
        dp = np.asarray(dt_polys, dtype=np.float32)
        if rp.shape != dp.shape or rp.ndim != 3 or rp.shape[1:] != (4, 2):
            return False
        r_range = rp.max(axis=(0, 1)) - rp.min(axis=(0, 1))
        d_range = dp.max(axis=(0, 1)) - dp.min(axis=(0, 1))
        if np.any(d_range < 1):  # dt_polys 退化，无法比对
            return False
        scale = r_range / d_range
        if np.any(scale < 0.8) or np.any(scale > 1.2):
            return False
        return True
    except Exception:
        return False


def run_ocr(cmd):
    global _ocr, _recognizer, _page_count
    if _ocr is None and _recognizer is None:
        return {"code": 901, "data": "引擎未初始化"}

    # 周期性重建 ORT session：每 N 页重建一次，强制释放 BFC arena 累积的显存碎片。
    # 否则跑长 PDF 时 arena 会被前面页的 buffer 占满，最后几页找不到连续空间报
    # BFCArena::AllocateRawInternal 错误（FusedMatMul / BiasSoftmax 节点申请失败）。
    # DirectML 无 BFC arena（显存由 DX12 驱动按需分配），无需重建；CUDA 重建释放
    # BFC 碎片，CPU 重建释放 ORT CPU arena 碎片，二者保留。
    _page_count += 1
    if _gpu_backend != "directml" and _page_count >= _REBUILD_PAGE_THRESHOLD:
        _rebuild_ocr()

    # BFC arena 失败时自动重建并重试一次。
    # attempt=1 失败 → 重建 session → attempt=2 重跑；attempt=2 再失败则报错。
    for attempt in (1, 2):
        tmp_path = None
        try:
            if "image_path" in cmd:
                input_data = cmd["image_path"]
            elif "image_base64" in cmd:
                img_bytes = base64.b64decode(cmd["image_base64"])
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(img_bytes)
                tmp.close()
                input_data = tmp.name
                tmp_path = tmp.name
            else:
                return {"code": 403, "data": "No valid tasks."}

            if _det:
                result = list(_ocr.predict(input_data))
            else:
                result = list(_recognizer.predict(input_data))

            # 解析结果，兼容 det（复数字段）与 rec-only（单数字段）两种输出
            data_list = []
            for page in result:
                if page is None:
                    continue
                texts = _get(page, "rec_texts", None)
                scores = _get(page, "rec_scores", None)
                rec_polys = _get(page, "rec_polys", None)
                dt_polys = _get(page, "dt_polys", None)
                if texts is None:
                    # rec-only 模式：单数标量
                    text = _get(page, "rec_text", None)
                    if text is None:
                        continue
                    texts = [text]
                    scores = [_get(page, "rec_score", 0.0)]
                    polys = None
                else:
                    scores = scores or []
                    # A2: 优先 rec_polys（识别阶段对齐后的框，更贴字），但需校验
                    #     其与 dt_polys 处于同一原图坐标系；不一致则回退 dt_polys，
                    #     防止某些 paddleocr 版本把 rec_polys 输出为局部裁剪框
                    # 注意：rec_polys / dt_polys 可能是 numpy 数组，不能用 `if rec_polys`
                    # 判空（触发 ValueError），须用 is not None + len() 判断
                    if rec_polys is not None and _validate_rec_polys(rec_polys, dt_polys, texts):
                        polys = rec_polys
                    else:
                        polys = dt_polys
                for i, text in enumerate(texts):
                    score = float(scores[i]) if i < len(scores) else 0.0
                    if polys is not None and i < len(polys):
                        poly = polys[i]
                        # A1: 向重心内缩，抵消 DBNet expand_ratio，让 box 更贴字
                        box = _shrink_poly(poly, _shrink_ratio)
                    else:
                        box = [[0, 0], [0, 0], [0, 0], [0, 0]]
                    data_list.append({"box": box, "text": text, "score": score})

            # 表格识别开关开启且格式非"关闭"：检测表格，输出格式文本并过滤表格区内重复文本
            if _table_mode and _table_format != "off":
                table_blocks, table_polys = _run_ocr_with_table(input_data)
                if table_blocks:
                    import numpy as _np
                    for item in data_list:
                        if not table_polys:
                            break
                        try:
                            pts = _np.asarray(item.get("box"), dtype=float)
                            if pts.ndim == 2 and pts.shape[0] >= 3:
                                cx = float(pts[:, 0].mean())
                                cy = float(pts[:, 1].mean())
                                if any(_point_in_poly(cx, cy, p) for p in table_polys):
                                    item["_drop"] = True
                        except Exception:
                            pass
                    data_list = [i for i in data_list if not i.get("_drop")]
                    # 表格块按位置插入（按 bbox 中心 y 排序），与普通文本混排
                    data_list.extend(table_blocks)
                    data_list.sort(key=lambda i: _block_center_y(i))

            if not data_list:
                return {"code": 101, "data": f'No text found in image. Path:"{input_data}"'}
            return {"code": 100, "data": data_list}
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            # BFC arena 显存池碎片化：rec 模型的 MatMul/Softmax 等大块节点
            # 需要连续显存，但 arena 被前面页的 buffer 占满找不到空间。
            # 关键字：BFCArena / AllocateRawInternal / "Available memory of"
            is_bfc = ("bfcarena" in low or "allocaterawinternal" in low
                      or "available memory of" in low)
            if is_bfc:
                if attempt == 1:
                    # 重建 session 释放整个 arena，然后重试当前页
                    if _rebuild_ocr():
                        continue  # 进入 attempt=2 重跑
                    return {"code": 902,
                            "data": "ORT 显存池碎片化，引擎重建失败。请降低'识别批处理数'后重试。"}
                return {"code": 902,
                        "data": "ORT 显存池碎片化，已重建引擎仍失败。请降低'识别批处理数'后重试。"}
            if "bad allocation" in low or "out of memory" in low:
                return {"code": 902, "data": "GPU 显存不足（bad allocation）。请降低'识别批处理数'后重试。"}
            if "cudnn" in low or "cudaexecutionprovider" in low or "conv node" in low:
                # cuDNN 执行失败：通常是 cuDNN/CUDA 版本与显卡驱动不兼容，
                # 或 cuDNN 9.x FE 图 API 在特定 GPU 架构上不支持。
                # 建议用户关闭 GPU 加速改用 CPU 模式（稳定），或更新显卡驱动。
                return {
                    "code": 902,
                    "data": "GPU cuDNN 执行失败。请在插件设置中关闭'GPU加速'后重试，"
                    "或更新显卡驱动到最新版。详情：" + e.__class__.__name__,
                }
            return {"code": 902, "data": f"OCR 异常: {e}"}
        finally:
            # 无论成功还是异常，都清理 base64 临时文件，避免泄漏到 %TEMP%
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            _cleanup_gpu_memory()


def _patch_table_deps():
    """绕过 paddlex 的 `paddlex[ocr]` 全量 extra 依赖检查（表格识别仅需已在环境中的
    onnxruntime + 官方 ONNX 模型，无需补齐 pip 组件大全家桶）。

    paddlex 的 table_recognition 管道在 __init__ 时执行 require_extra("ocr")，
    要求 EXTRAS['ocr'] 全部 ~20 个包都可用（含 scikit-learn/scipy/lxml/openpyxl 等），
    否则直接抛 DependencyError。这些包与表格识别（PP-DocLayout_plus-L 布局 +
    SLANet_plus 结构识别 + PP-OCRv6 det/rec）的实际推理无关。

    这里仅把 "ocr" 这个 extra 的检查结果豁免为 True，其余 extra（如 "genai"）
    保持原逻辑。仅在 first 表格调用前执行一次，且幂等。
    """
    global _patched_table_deps
    if _patched_table_deps:
        return
    try:
        import paddlex.utils.deps as _deps
        _orig = _deps.is_extra_available.__wrapped__  # 解开 lru_cache 拿到原函数
        def _is_extra_available(extra):
            if extra == "ocr":
                return True
            return _orig(extra)
        _deps.is_extra_available = _is_extra_available
        _patched_table_deps = True
        print("[ppocr_v6] table pipeline: paddlex[ocr] extra check bypassed "
              "(deps verified at import time)", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[ppocr_v6] WARNING: failed to patch paddlex deps check: {e}",
              file=sys.stderr, flush=True)


def _get_table_pipeline():
    """懒加载 paddlex table_recognition 管道（v1：布局 + SLANet_plus + PP-OCRv6）。

    模型组合（全部为官方 ONNX 包，无额外 pip 依赖）：
      - LayoutDetection:   PP-DocLayout_plus-L（页面布局，含 table 区域检测）
      - TableStructure:    SLANet_plus（表格结构识别 → HTML）
      - GeneralOCR 子管道: PP-OCRv6 det + rec（识别单元格文本，尺寸跟随 OCR 配置）
    关闭 doc_preprocessor（弯曲矫正/方向分类，v1 已确认不实例化其 LCNet 模型），
    与现有的轻量 OCR 路径一致。
    复用 init_ocr 时的 engine_config（CUDA/DirectML/CPU 全兼容）。
    """
    global _table_pipe
    if _table_pipe is not None:
        return _table_pipe
    _patch_table_deps()
    from paddlex import create_pipeline
    from paddlex.inference import load_pipeline_config

    cfg = load_pipeline_config("table_recognition")
    cfg["use_doc_preprocessor"] = False
    cfg["use_doc_orientation_classify"] = False
    cfg["use_doc_unwarping"] = False
    cfg["SubModules"]["LayoutDetection"]["model_name"] = "PP-DocLayout_plus-L"
    det_model = f"PP-OCRv6_{_table_model_size}_det"
    rec_model = f"PP-OCRv6_{_table_model_size}_rec"
    cfg["SubPipelines"]["GeneralOCR"]["SubModules"]["TextDetection"]["model_name"] = det_model
    cfg["SubPipelines"]["GeneralOCR"]["SubModules"]["TextRecognition"]["model_name"] = rec_model

    # 复用 OCR 路径的引擎配置（onnxruntime + 当前 device/GPU 后端）
    engine, engine_config = _select_engine(_use_gpu, _getattr(_init_args, "cpu_threads", None), _table_model_size)
    kwargs = {"config": cfg, "engine": engine}
    if engine == "onnxruntime":
        # DirectML 场景：_select_engine 已把 device_type 置为 "cpu"，直接复用即可
        kwargs["engine_config"] = engine_config
        kwargs["device"] = "cpu" if _gpu_backend == "directml" else None
    print(f"[ppocr_v6] creating table pipeline (layout+SLANet_plus+{det_model}/{rec_model}, "
          f"engine={engine})...", file=sys.stderr, flush=True)
    _table_pipe = create_pipeline(**kwargs)
    print("[ppocr_v6] table pipeline ready.", file=sys.stderr, flush=True)
    return _table_pipe


def _getattr(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default


def _collect_table_results(page):
    """把 paddlex page 结果整理为统一 JSON：{code, data:{html, tables}}。

    paddlex 输出结构（SDKResult.json 的 res 字段）：
      layout_det_res:  {boxes: [{label, score, coordinate}]}
      table_res_list:  [{pred_html, cell_box_list, table_ocr_pred}]
    这里输出与 PaddleOCR-json 兼容的外层协议，data 内含：
      html  : 所有表格的 <table>...</table> 片段拼接（不含 <html><body> 包装）
      tables: [{html, box, cells}] 每个表格的原图坐标 box 与单元格文字列表
    """
    res = page.json.get("res", {}) if hasattr(page, "json") else page
    layout = res.get("layout_det_res", {}) or {}
    table_list = res.get("table_res_list", []) or []
    tables = []
    full_html = ""
    for t in table_list:
        pred_html = t.get("pred_html", "")
        # 去掉 paddlex 的 <html><body> 外包装，仅保留表格本体，便于直接嵌入
        body = pred_html
        for _ in range(4):  # <html><body> 最多两层包装，循环剥到不再变化为止
            changed = False
            for tag in ("body", "html"):
                open_tag = f"<{tag}>"
                close_tag = f"</{tag}>"
                if body.lower().startswith(open_tag) and body.lower().endswith(close_tag):
                    body = body[len(open_tag):-len(close_tag)]
                    changed = True
            if not changed:
                break
        cells = []
        cell_boxes = t.get("cell_box_list", []) or []
        ocr_pred = t.get("table_ocr_pred", {}) or {}
        rec_texts = ocr_pred.get("rec_texts", []) if isinstance(ocr_pred, dict) else []
        rec_polys = ocr_pred.get("rec_polys", []) if isinstance(ocr_pred, dict) else []
        # 按空间匹配：把每个 OCR 文本归入其中心点所在的单元格（cell 为 [x1,y1,x2,y2]）
        for cb in cell_boxes:
            cells.append({"box": list(cb), "text": ""})
        if cells and rec_texts:
            import numpy as np
            for i, txt in enumerate(rec_texts):
                poly = rec_polys[i] if i < len(rec_polys) else None
                if poly is None or not len(poly):
                    continue
                pts = np.asarray(poly, dtype=np.float32)
                if pts.ndim != 2 or pts.shape[0] < 2:
                    continue
                cx = float(pts[:, 0].mean())
                cy = float(pts[:, 1].mean())
                for cell in cells:
                    b = cell["box"]
                    if len(b) == 4 and b[0] - 1 <= cx <= b[2] + 1 and b[1] - 1 <= cy <= b[3] + 1:
                        if cell["text"]:
                            cell["text"] += "\n"
                        cell["text"] += txt
                        break
        tables.append({
            "html": body,
            "box": t.get("bbox", []),
            "cells": cells,
        })
        full_html += body
    return {
        "html": full_html,
        "tables": tables,
    }


def _html_to_tsv(html):
    """把 <table> HTML 源码转成 TSV（制表符分隔）文本。

    每行一个 <tr>，行内单元格用制表符分隔；剥离单元格内嵌标签与 HTML 实体。
    供「表格输出格式=TSV」时输出，粘贴到 Excel/WPS 可直接成表。
    """
    import re
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I)
    lines = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)
        line = []
        for c in cells:
            c = re.sub(r"<[^>]+>", "", c)
            for ent, rep in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                             ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")):
                c = c.replace(ent, rep)
            line.append(c.strip())
        lines.append("\t".join(line))
    return "\n".join(lines)


def _point_in_poly(px, py, poly):
    """射线法判断点是否在多边形内（表格区域过滤用）。"""
    if poly is None or len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i][0], poly[i][1]
        x2, y2 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        if (y1 > py) != (y2 > py):
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_cross:
                inside = not inside
    return inside


def _block_center_y(item):
    """文本块 bbox 中心 y（按行序混排表格块与普通文本时排序用）。"""
    box = item.get("box") or []
    try:
        if len(box) == 4 and isinstance(box[0], (list, tuple)) and len(box[0]) == 2:
            return sum(p[1] for p in box) / 4.0
    except Exception:
        pass
    return 0.0


def _run_ocr_with_table(input_data):
    """表格识别开关开启时：对图片做表格检测。

    若检测到表格，返回 (表格文本块列表, 表格区域多边形列表)；否则返回 (None, None)。
    表格块 text 为所选格式（html/tsv）拼接文本，box 为该表格区域 4 点坐标。
    调用方负责过滤落在表格区域内的普通 OCR 文本块，避免重复输出。
    """
    global _table_pipe
    try:
        try:
            pipe = _get_table_pipeline()
        except Exception as e:
            print(f"[ppocr_v6] table pipeline init failed: {e}", file=sys.stderr, flush=True)
            return None, None
        t_result = list(pipe.predict(input_data))
        if not t_result:
            return None, None
        data = _collect_table_results(t_result[0])
        tables = data.get("tables", [])
        if not tables:
            return None, None
        # 表格区域坐标来自 layout_det_res（label 含 "table"），table_res_list 本身无 bbox
        page = t_result[0]
        res = page.json.get("res", {}) if hasattr(page, "json") else page
        layout = res.get("layout_det_res", {}) or {}
        table_boxes = [
            b.get("coordinate", []) for b in (layout.get("boxes", []) or [])
            if str(b.get("label", "")).lower().startswith("table")
        ]
        blocks = []
        table_polys = []
        for i, t in enumerate(tables):
            fmt = _table_format
            if fmt == "tsv":
                text = _html_to_tsv(t.get("html", ""))
            else:
                text = t.get("html", "")
            if not text.strip():
                continue
            # 优先 layout 表格区域坐标；转成 4 点多边形供协议输出
            box = t.get("box") or []
            if i < len(table_boxes):
                box = table_boxes[i] or box
            pts = []
            if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                x1, y1, x2, y2 = (float(v) for v in box)
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            elif len(box) == 4 and isinstance(box[0], (list, tuple)) and len(box[0]) == 2:
                pts = [[int(p[0]), int(p[1])] for p in box]
            blocks.append({"box": pts, "text": text, "score": 1.0, "is_table": True})
            if pts:
                table_polys.append(pts)
        if not blocks:
            return None, None
        print(f"[ppocr_v6] table mode: {len(blocks)} table(s) detected, format={_table_format}",
              file=sys.stderr, flush=True)
        return blocks, table_polys
    except Exception as e:
        print(f"[ppocr_v6] table mode skipped: {e}", file=sys.stderr, flush=True)
        return None, None


def run_table(cmd):
    """执行表格识别命令：{"image_path": "...", "table": true}。

    输入与 run_ocr 兼容（image_path / image_base64 二选一），输出：
      {"code": 100, "data": {"html": ..., "tables": [...]}}
    """
    global _table_pipe
    try:
        try:
            pipe = _get_table_pipeline()
        except Exception as e:
            return {"code": 902, "data": f"表格识别初始化失败: {e}"}
        tmp_path = None
        try:
            if "image_path" in cmd:
                input_data = cmd["image_path"]
            elif "image_base64" in cmd:
                img_bytes = base64.b64decode(cmd["image_base64"])
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(img_bytes)
                tmp.close()
                input_data = tmp.name
                tmp_path = tmp.name
            else:
                return {"code": 403, "data": "No valid tasks."}
            result = list(pipe.predict(input_data))
            if not result:
                return {"code": 101, "data": f'No table found in image. Path:"{input_data}"'}
            data = _collect_table_results(result[0])
            if not data["html"]:
                return {"code": 101, "data": f'No table found in image. Path:"{input_data}"'}
            return {"code": 100, "data": data}
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            _cleanup_gpu_memory()
    except Exception as e:
        return {"code": 902, "data": f"表格识别异常: {e}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument("--det", type=lambda x: x.lower() == "true" if isinstance(x, str) else bool(x), default=None)
    parser.add_argument("--cls", type=lambda x: x.lower() == "true" if isinstance(x, str) else bool(x), default=None)
    parser.add_argument("--rec_batch_num", type=int, default=None)
    parser.add_argument("--limit_side_len", type=int, default=None)
    parser.add_argument("--use_gpu", type=lambda x: x.lower() == "true" if isinstance(x, str) else bool(x), default=False)
    parser.add_argument("--enable_mkldnn", default=None)
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--use_tensorrt", default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--use_angle_cls", default=None)
    # A1: det 框内缩比例，抵消 DBNet expand_ratio，改善 PDF 文本层对齐。0=关闭
    parser.add_argument("--shrink_poly_ratio", type=float, default=0.0)
    # 表格识别（设置界面开关）：0/1 开启后 run_ocr 自动输出表格格式文本
    parser.add_argument("--table_mode", type=lambda x: str(x).lower() in ("1", "true", "yes", "on"), default=False)
    # 表格输出格式：html / tsv / off（off 等效于未开启）
    parser.add_argument("--table_format", type=str, default="html")

    args, _ = parser.parse_known_args()

    try:
        init_ocr(args)
    except Exception as e:
        print(f"OCR init fail: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    print("OCR init completed.", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"code": 400, "data": "Invalid JSON"}, ensure_ascii=False), flush=True)
            continue
        if cmd.get("table"):
            result = run_table(cmd)
        else:
            result = run_ocr(cmd)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
