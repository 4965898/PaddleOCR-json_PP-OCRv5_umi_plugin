# -*- coding: utf-8 -*-
"""issue #13 回归测试：用 Umi-OCR v2.1.5 的真实 tbpu 源码验证插件输出的兼容性。

结论目标：
A. 复现原始崩溃：空文本块进入「多栏-自然段」解析器 → StatisticsError（证明测试有效）
B~G. 模拟修复后插件的所有输出形态（垃圾字符/退化框/表格大块/海量块等），均不应抛异常
"""
import random
import statistics
import sys
import os

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg_dir = os.path.join(_repo_root, "_tbpu_v215")
for p in (_repo_root, _pkg_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from _tbpu_v215.parser_multi_para import MultiPara  # noqa: E402

_plugin_dir = os.path.join(_repo_root, "umi_plugin_v6")
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)

from ppocr_v6_server import _ensure_valid_box  # noqa: E402

DROP_SCORE = 0.5  # 与 ppocr_v6_server.py 的默认 --drop_score 保持一致


def plugin_postfix_assemble(items):
    """按修复后 ppocr_v6_server.run_ocr 的组装逻辑模拟输出。

    items: [(box, text, score), ...]
    """
    data_list = []
    for box, text, score in items:
        if text is None or not str(text).strip():
            continue
        if score < DROP_SCORE:
            continue
        box = _ensure_valid_box(box)
        if box is None:
            continue
        data_list.append({"box": box, "text": text, "score": score})
    return data_list


def make_tb(box, text, score=0.9, **extra):
    tb = {"box": box, "text": text, "score": score}
    tb.update(extra)
    return tb


def run_parser_expect_crash(tbs, label):
    """宿主端已知缺陷的记录性测试：这些输入会让 v2.1.5 GapTree 崩溃（插件必须拦截）。"""
    try:
        MultiPara().run([dict(tb) for tb in tbs])
        print(f"[FAIL] {label}: 预期崩溃未发生（Umi-OCR 可能已修复该缺陷，可移除此记录）")
        return False
    except Exception as e:
        print(f"[PASS] {label}: 确认宿主崩溃（{type(e).__name__}），插件必须拦截此类输入")
        return True


def run_parser(tbs, label):
    try:
        MultiPara().run([dict(tb) for tb in tbs])
        print(f"[PASS] {label}")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: {type(e).__name__}: {e}")
        return False


def test_a_original_bug():
    """原始 bug 复现：空文本块被 linePreprocessing 过滤后 median([]) 崩溃。"""
    tbs = [make_tb([[10, 10], [50, 10], [50, 30], [10, 30]], "")]
    try:
        MultiPara().run(tbs)
        print("[FAIL] A: 未复现原始 StatisticsError，测试环境无效！")
        return False
    except statistics.StatisticsError:
        print("[PASS] A: 复现原始 StatisticsError（issue #13 原始崩溃链）")
        return True


def test_b_all_filtered_empty():
    """全部块 text 为空白字符串（修复后插件会过滤，不会输出）。防御性确认崩溃条件。"""
    tbs = [make_tb([[10, 10], [50, 10], [50, 30], [10, 30]], "  ")]
    try:
        MultiPara().run(tbs)
        print("[PASS] B: 纯空白文本块可正常通过解析（插件已过滤，实际不会输出）")
        return True
    except statistics.StatisticsError:
        print("[FAIL] B: 纯空白文本块崩溃（说明插件 strip 过滤是必要的）")
        return False


def test_c_garbage_noise_blocks():
    """small 模型在空白页可能输出的垃圾非空文本 + 随机小框（含退化框）。"""
    random.seed(42)
    garbage = ["|", "1", "。", ",", "L", "一", "0.", "l)", "Ⅱ", "?", ":", "\""]
    tbs = []
    for _ in range(80):
        x = random.randint(0, 2000)
        y = random.randint(0, 2800)
        w = random.randint(0, 60)
        h = random.randint(0, 30)
        box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        tbs.append(make_tb(box, random.choice(garbage), random.uniform(0.05, 0.6)))
    return run_parser(tbs, "C: 80个低分垃圾块（随机退化小框）")


def test_d_table_block_mixed():
    """表格大块（HTML 长文本，覆盖大区域）与普通文本块混排。"""
    html = "<table><tbody>" + "".join(
        f"<tr><td>单元格{i}a</td><td>单元格{i}b</td></tr>" for i in range(50)
    ) + "</tbody></table>"
    tbs = [
        make_tb([[100, 100], [1800, 100], [1800, 1500], [100, 1500]], html, 1.0, is_table=True),
        make_tb([[150, 200], [400, 200], [400, 260], [150, 260]], "表格上方文字"),
        make_tb([[150, 1600], [900, 1600], [900, 1660], [150, 1660]], "表格下方文字，共一行。"),
        make_tb([[950, 1600], [1800, 1600], [1800, 1660], [950, 1660]], "右侧并列内容"),
    ]
    return run_parser(tbs, "D: 表格HTML大块与文本混排")


def test_e_many_blocks():
    """500 个正常文本块（无限制边长大页面的量级）。"""
    random.seed(7)
    tbs = []
    for i in range(500):
        x = random.randint(50, 3500)
        y = random.randint(50, 4500)
        w = random.randint(50, 400)
        h = random.randint(20, 50)
        tbs.append(make_tb([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], f"第{i}行测试文本"))
    return run_parser(tbs, "E: 500个正常文本块")


def test_f_zero_area_boxes():
    """polys 缺失时的全零退化兜底框 [[0,0]]x4 —— 宿主端 GapTree 已知崩溃条件。"""
    tbs = [make_tb([[0, 0], [0, 0], [0, 0], [0, 0]], "零面积框A"),
           make_tb([[0, 0], [0, 0], [0, 0], [0, 0]], "零面积框B")]
    return run_parser_expect_crash(tbs, "F: 全零退化兜底框x2")


def test_f2_single_zero_box():
    """单个全零退化框 —— 宿主端 GapTree 已知崩溃条件。"""
    return run_parser_expect_crash([make_tb([[0, 0], [0, 0], [0, 0], [0, 0]], "单零框")],
                                   "F2: 单个全零框")


def test_f3_zero_width_boxes():
    """整数化后零宽框（det 噪点框收缩取整的典型产物，x0==x2）。"""
    tbs = [make_tb([[100, 200], [100, 200], [100, 203], [100, 203]], "|"),
           make_tb([[100, 400], [100, 400], [100, 402], [100, 402]], "。"),
           make_tb([[300, 500], [500, 500], [500, 530], [300, 530]], "正常文本行")]
    return run_parser(tbs, "F3: 零宽框与正常框混排")


def test_f4_single_zero_width_alone():
    """单个零宽框独占页面（近空白页仅检出 1 个噪点框的场景）。"""
    return run_parser([make_tb([[100, 200], [100, 200], [100, 203], [100, 203]], "|")],
                      "F4: 单个零宽框独占页面")


def test_g_single_block_and_whitespace():
    """单块 + 纯空白文本（真值，会进入解析器）。"""
    ok1 = run_parser([make_tb([[10, 10], [50, 10], [50, 30], [10, 30]], "单")], "G1: 单个文本块")
    ok2 = run_parser([make_tb([[10, 10], [50, 10], [50, 30], [10, 30]], " "),
                      make_tb([[10, 40], [50, 40], [50, 60], [10, 60]], "x")], "G2: 空白+正常块")
    return ok1 and ok2


def test_h_ensure_valid_box_unit():
    """_ensure_valid_box 单元验证。"""
    ok = True
    # 全零兜底框 → 修复为合法框
    r = _ensure_valid_box([[0, 0], [0, 0], [0, 0], [0, 0]])
    ok &= r is not None and len(r) == 4
    # 塌缩框（收缩+取整产物）→ 修复为合法框
    r = _ensure_valid_box([[100, 200], [100, 200], [100, 203], [100, 203]])
    ok &= r is not None and len(r) == 4
    # 正常框 → 原样保留
    normal = [[10, 10], [50, 10], [50, 30], [10, 30]]
    ok &= _ensure_valid_box(normal) == normal
    # 结构非法 → None
    ok &= _ensure_valid_box([]) is None
    ok &= _ensure_valid_box([[1, 2], [3, 4]]) is None
    print(f"[{'PASS' if ok else 'FAIL'}] H: _ensure_valid_box 单元验证")
    return bool(ok)


def test_i_postfix_output_never_crashes():
    """端到端：修复后插件组装逻辑的输出（含噪点/退化/表格场景）全部可安全通过解析器。"""
    random.seed(99)
    ok = True
    # 场景1：近空白页（含空文本、低分垃圾、塌缩框——修复后分别被过滤/修复）
    items = [
        ([[100, 200], [100, 200], [100, 203], [100, 203]], "|", 0.2),
        ([[500, 300], [500, 300], [500, 302], [500, 302]], "。", 0.15),
        ([[0, 0], [0, 0], [0, 0], [0, 0]], "1", 0.1),
        ([[0, 0], [0, 0], [0, 0], [0, 0]], "", 0.9),
        ([[2000, 1500], [2200, 1500], [2200, 1540], [2000, 1540]], "正文标题", 0.96),
    ]
    tbs = plugin_postfix_assemble(items)
    tbs = [make_tb(tb["box"], tb["text"], tb["score"]) for tb in tbs]
    ok &= run_parser(tbs, "I1: 近空白页修复后输出（若空列表则上游 code 101 跳过解析）")
    # 场景2：大量噪声 + 全部走修复路径
    items = []
    for _ in range(200):
        x, y = random.randint(0, 3000), random.randint(0, 4000)
        w = random.choice([0, 0.3, 0.8, 1, 5, 50, 300])
        h = random.choice([0, 0.2, 0.9, 1, 3, 30])
        box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        items.append((box, random.choice(["|", "。", "字", "text"]), random.uniform(0.05, 0.99)))
    tbs = plugin_postfix_assemble(items)
    tbs = [make_tb(tb["box"], tb["text"], tb["score"]) for tb in tbs]
    ok &= run_parser(tbs, f"I2: 200个随机噪声块修复后输出（{len(tbs)}块通过过滤）")
    return ok


if __name__ == "__main__":
    results = [
        test_a_original_bug(),
        test_b_all_filtered_empty(),
        test_c_garbage_noise_blocks(),
        test_d_table_block_mixed(),
        test_e_many_blocks(),
        test_f_zero_area_boxes(),
        test_f2_single_zero_box(),
        test_f3_zero_width_boxes(),
        test_f4_single_zero_width_alone(),
        test_g_single_block_and_whitespace(),
        test_h_ensure_valid_box_unit(),
        test_i_postfix_output_never_crashes(),
    ]
    print()
    print(f"结果: {sum(results)}/{len(results)} 通过")
    sys.exit(0 if all(results) else 1)
