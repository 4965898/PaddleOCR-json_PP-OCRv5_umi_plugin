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
    gc.collect()
    try:
        import onnxruntime as ort
        for sess in ort.get_all_sessions():
            sess.run_options.free()
    except Exception:
        pass
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
        cfg = {
            "device_type": "gpu",
            "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "provider_options": [
                {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"},
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
    global _ocr, _recognizer, _det, _use_gpu
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
            ocr_args["text_detection_model_dir"] = det_onnx_dir
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


def run_ocr(cmd):
    global _ocr, _recognizer
    if _ocr is None and _recognizer is None:
        return {"code": 901, "data": "引擎未初始化"}
    try:
        if "image_path" in cmd:
            input_data = cmd["image_path"]
        elif "image_base64" in cmd:
            img_bytes = base64.b64decode(cmd["image_base64"])
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.write(img_bytes)
            tmp.close()
            input_data = tmp.name
        else:
            return {"code": 403, "data": "No valid tasks."}

        if _det:
            result = list(_ocr.predict(input_data))
        else:
            result = list(_recognizer.predict(input_data))

        if "image_base64" in cmd:
            try:
                os.unlink(tmp.name)
            except:
                pass

        # 解析结果，兼容 det（复数字段）与 rec-only（单数字段）两种输出
        data_list = []
        for page in result:
            if page is None:
                continue
            texts = _get(page, "rec_texts", None)
            scores = _get(page, "rec_scores", None)
            polys = _get(page, "rec_polys", None) or _get(page, "dt_polys", None)
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
                polys = polys or []
            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 0.0
                if polys and i < len(polys):
                    poly = polys[i]
                    box = [[int(p[0]), int(p[1])] for p in poly]
                else:
                    box = [[0, 0], [0, 0], [0, 0], [0, 0]]
                data_list.append({"box": box, "text": text, "score": score})

        if not data_list:
            return {"code": 101, "data": f'No text found in image. Path:"{input_data}"'}
        return {"code": 100, "data": data_list}
    except Exception as e:
        msg = str(e)
        if "bad allocation" in msg.lower() or "out of memory" in msg.lower():
            return {"code": 902, "data": f"GPU 显存不足（bad allocation）。请降低"识别批处理数"后重试。"}
        return {"code": 902, "data": f"OCR 异常: {e}"}
    finally:
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
