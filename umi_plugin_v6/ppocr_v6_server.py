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


def _setup_nvidia_dlls():
    """把 pip 安装的 nvidia CUDA/cuDNN DLL 路径加入搜索路径（GPU 加速所需）"""
    try:
        import sysconfig
        site_dir = sysconfig.get_paths()["purelib"]
        nvidia_base = os.path.join(site_dir, "nvidia")
        if os.path.isdir(nvidia_base):
            for sub in os.listdir(nvidia_base):
                dll_dir = os.path.join(nvidia_base, sub, "bin")
                if os.path.isdir(dll_dir):
                    os.add_dll_directory(dll_dir)
                    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


_use_gpu = False
_engine = None  # 推理引擎名称（"onnxruntime" / "paddle" / None）
_init_args = None  # 保存 init_ocr 的 args，供周期性重建 ORT session 使用
_page_count = 0  # 已处理页数计数器，达阈值时重建 session 释放 BFC arena 碎片
# 每 N 页重建一次 ORT session，强制释放 BFC arena 累积的显存碎片。
# 8GB 显卡建议 50；更小显存可调小至 30，更大显存可调大到 100。
_REBUILD_PAGE_THRESHOLD = 50


def _cleanup_gpu_memory():
    """释放 GPU 显存缓存，防止多页 PDF 识别时显存碎片累积导致 bad allocation"""
    if not _use_gpu:
        return
    # ONNX Runtime session 由 Python 引用计数管理，gc.collect() 会回收无引用的
    # session 及其显存。不要尝试手动调用 session.run_options.free()——RunOptions
    # 没有公开的 free() 方法，调用会抛 AttributeError 被静默吞掉，实际无效。
    gc.collect()
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

    注意：mkldnn（oneDNN）是 paddlepaddle 的 CPU 加速后端，对 onnxruntime 无效。
    onnxruntime CPU 使用自带的 MLAS 优化库；这里通过开启图优化最高级、内存模式
    与显式线程数来提升 CPU 推理速度（无需下载任何额外文件）。
    """
    _setup_nvidia_dlls()
    try:
        import onnxruntime as ort
    except ImportError:
        return None, None
    providers = ort.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" in providers:
        # GPU 模式
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
    else:
        # CPU 模式：图优化最高级 + 内存模式（CPU 推理稳定提速，无副作用）
        cfg = {
            "device_type": "cpu",
            "providers": ["CPUExecutionProvider"],
            "graph_optimization_level": 99,  # ORT_ENABLE_ALL
            "enable_mem_pattern": True,
            "enable_cpu_mem_arena": True,
        }
    # CPU 线程数：显式设置时生效，否则用 onnxruntime 默认（全部逻辑核）
    if cpu_threads and cpu_threads > 0:
        cfg["intra_op_num_threads"] = cpu_threads
        cfg["inter_op_num_threads"] = cpu_threads
    return "onnxruntime", cfg


def parse_config(args):
    config = {"model_size": "medium", "lang": "ch"}
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
    global _ocr, _recognizer, _det, _use_gpu, _shrink_ratio, _engine, _init_args
    _init_args = args  # 保存以供周期性重建 ORT session 使用
    # 必须在 import paddleocr 之前添加 NVIDIA DLL 路径。
    # 否则 paddleocr/paddlex 导入过程会干扰后续 ORT CUDA 加载 cuDNN，
    # 导致 "Invalid handle. Cannot load symbol cudnnCreate" 错误。
    _setup_nvidia_dlls()
    from paddleocr import PaddleOCR, TextRecognition

    config = parse_config(args)
    model_size = config["model_size"]
    lang = config["lang"]
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

    engine_kwargs = {}
    if engine:
        engine_kwargs["engine"] = engine
        engine_kwargs["engine_config"] = engine_config

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
    _page_count += 1
    if _page_count >= _REBUILD_PAGE_THRESHOLD:
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
        result = run_ocr(cmd)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
