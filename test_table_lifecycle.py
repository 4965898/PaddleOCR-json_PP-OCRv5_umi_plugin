#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""表格管道生命周期 + DirectML 周期重建 —— mock 状态机测试。

不依赖真实 paddleocr/onnxruntime：mock 掉 _run_ocr_with_table / _cleanup_gpu_memory /
init_ocr / time.monotonic / _ocr.predict，只驱动 run_ocr 内的状态机决策。

覆盖 13 个场景：连续无表格销毁、冷却跳过、表格重置、穿插不销毁、空闲销毁、
引擎重建、连续表格不销毁、DML/CUDA 重建阈值等。
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "umi_plugin_v6"))
import ppocr_v6_server as S  # noqa: E402

# ---- 可控时钟 ----
_clock = {"t": 1000.0}
S.time = types.SimpleNamespace(monotonic=lambda: _clock["t"])

# ---- mock 重依赖 ----
S._cleanup_gpu_memory = lambda: None
S.init_ocr = lambda args: None


class FakeOcr:
    def predict(self, path):
        return [{"rec_texts": [], "rec_scores": [], "rec_polys": None, "dt_polys": None}]


# ---- mock _run_ocr_with_table：模拟管道创建/复用副作用，返回排队结果 ----
_table_outputs = []


def fake_run_ocr_with_table(input_data):
    S._table_pipe = object()                 # 模拟管道已存在/刚创建
    S._table_last_used = S.time.monotonic()  # 模拟刷新空闲计时
    if _table_outputs:
        return _table_outputs.pop(0)
    return None, None


S._run_ocr_with_table = fake_run_ocr_with_table

_BLOCK = [{"box": [[0, 0], [0, 0], [0, 0], [0, 0]], "text": "T", "score": 1.0, "is_table": True}]
_CMD = {"image_path": "fake.png"}


def reset_state():
    _clock["t"] = 1000.0
    S._ocr = FakeOcr()
    S._recognizer = None
    S._det = True
    S._table_pipe = None
    S._table_mode = True
    S._table_format = "html"
    S._table_no_table_streak = 0
    S._table_last_used = _clock["t"]
    S._table_destroy_cooldown_until = 0.0
    S._page_count = 0
    S._gpu_backend = None
    S._init_args = object()


def run():
    return S.run_ocr(_CMD)


passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print("  PASS  " + name)
    else:
        failed += 1
        print("  FAIL  " + name)


# ===== T1-T4：连续无表格销毁 → 冷却 → 跳过 → 过期重建 =====
reset_state()
S._table_mode = True
_table_outputs[:] = [(None, None)] * 5
for _ in range(5):
    run()
check("T1  连续5页无表格→销毁管道", S._table_pipe is None)
check("T2  连续无表格销毁后设30s冷却", S.time.monotonic() < S._table_destroy_cooldown_until)
_table_outputs[:] = [(None, None)]
run()
check("T3  冷却期跳过自动检测(管道不重建,streak不累加)",
      S._table_pipe is None and S._table_no_table_streak == 0)
_clock["t"] = 1031  # 过冷却
_table_outputs[:] = [(_BLOCK, [])]
run()
check("T4  冷却过期后自动重建检测", S._table_pipe is not None)

# ===== T5：表格页重置连续无表格计数 =====
reset_state()
S._table_mode = True
_table_outputs[:] = [(None, None), (None, None), (_BLOCK, [])]
run(); run(); run()
check("T5  表格页重置连续无表格计数", S._table_no_table_streak == 0 and S._table_pipe is not None)

# ===== T6：穿插表格/无表格不销毁 =====
reset_state()
S._table_mode = True
_table_outputs[:] = [(_BLOCK, []), (None, None), (_BLOCK, []), (None, None), (_BLOCK, [])]
for _ in range(5):
    run()
check("T6  穿插表格/无表格不销毁", S._table_pipe is not None and S._table_no_table_streak == 0)

# ===== T7-T8：关闭开关后空闲60s销毁（不设冷却）=====
reset_state()
S._table_mode = True
_table_outputs[:] = [(_BLOCK, [])]
run()  # 管道创建, last_used=1000
S._table_mode = False  # 用户关闭表格开关
_clock["t"] = 1061     # 61s 后
_table_outputs[:] = [(None, None)]
run()
check("T7  关闭开关后空闲60s销毁管道", S._table_pipe is None)
check("T8  空闲销毁不设冷却", not (S.time.monotonic() < S._table_destroy_cooldown_until))

# ===== T9：空闲未到60s不销毁 =====
reset_state()
S._table_mode = True
_table_outputs[:] = [(_BLOCK, [])]
run()
S._table_mode = False
_clock["t"] = 1050  # 50s < 60
_table_outputs[:] = [(None, None)]
run()
check("T9  空闲未到60s不销毁", S._table_pipe is not None)

# ===== T10-T11：引擎重建销毁表格管道（不设冷却）→ 下张图立即重建 =====
reset_state()
S._table_mode = True
_table_outputs[:] = [(_BLOCK, [])]
run()  # 管道创建
S._ocr = FakeOcr()
S._rebuild_ocr()
check("T10 引擎重建顺带销毁表格管道", S._table_pipe is None)
S._ocr = FakeOcr()  # 模拟重建后引擎就绪
S._table_mode = True
_clock["t"] = 1001
_table_outputs[:] = [(_BLOCK, [])]
run()
check("T11 引擎重建后下张图立即重建检测", S._table_pipe is not None)

# ===== T12：连续表格页不反复销毁（streak 保持 0）=====
reset_state()
S._table_mode = True
_table_outputs[:] = [(_BLOCK, []), (_BLOCK, []), (_BLOCK, [])]
for _ in range(3):
    run()
check("T12 连续表格页不反复销毁(streak=0)", S._table_pipe is not None and S._table_no_table_streak == 0)

# ===== T13：DML 每30页重建 / CUDA 每50页重建（30页不触发）=====
_count = [0]
_orig_rebuild = S._rebuild_ocr


def counting_rebuild():
    _count[0] += 1
    S._page_count = 0


S._rebuild_ocr = counting_rebuild
reset_state()
S._gpu_backend = "directml"
S._table_mode = False
_count[0] = 0
for _ in range(30):
    run()
dml_count = _count[0]
reset_state()
S._gpu_backend = "cuda"
S._table_mode = False
_count[0] = 0
for _ in range(30):
    run()
cuda_count = _count[0]
S._rebuild_ocr = _orig_rebuild
check("T13 DML每30页重建/CUDA 30页不重建", dml_count == 1 and cuda_count == 0)

# ===== 汇总 =====
print("\n=== %d/%d passed ===" % (passed, passed + failed))
sys.exit(1 if failed else 0)
