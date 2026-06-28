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


def _cleanup_gpu_memory():
    """释放 GPU 显存缓存，防止多页 PDF 识别时显存碎片累积导致 bad allocation"""
    if not _use_gpu:
        return
    # ONNX Runtime session 由 Python 引用计数管理，gc.collect() 会回收无引用的
    # session 及其显存。不要尝试手动调用 session.run_options.free()——RunOptions
    # 没有公开的 free() 方法，调用会抛 AttributeError 被静默吞掉，实际无效。
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _select_engine(use_gpu, cpu_threads=None):
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
        cfg = {
            "device_type": "gpu",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "provider_options": [
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "DEFAULT",
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
    global _ocr, _recognizer, _det, _use_gpu, _shrink_ratio
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
    engine, engine_config = _select_engine(use_gpu, args.cpu_threads)

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
    """
    if not poly or ratio <= 0:
        return [[int(p[0]), int(p[1])] for p in poly]
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
    """
    if not rec_polys or not dt_polys or not texts:
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
    global _ocr, _recognizer
    if _ocr is None and _recognizer is None:
        return {"code": 901, "data": "引擎未初始化"}
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
                if rec_polys and _validate_rec_polys(rec_polys, dt_polys, texts):
                    polys = rec_polys
                else:
                    polys = dt_polys
            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 0.0
                if polys and i < len(polys):
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
        if "bad allocation" in msg.lower() or "out of memory" in msg.lower():
            return {"code": 902, "data": "GPU 显存不足（bad allocation）。请降低'识别批处理数'后重试。"}
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
