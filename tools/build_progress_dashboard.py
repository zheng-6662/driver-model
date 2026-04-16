# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
from collections import Counter
import html
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROGRESS = REPORTS / "progress"
DAILY = PROGRESS / "daily"

HUB = REPORTS / "project_progress_hub.md"
DECISIONS = PROGRESS / "decision_log.md"
EXPERIMENTS = PROGRESS / "experiment_registry.md"
GLOSSARY = PROGRESS / "glossary.md"

OUT = {
    "dashboard": REPORTS / "project_progress_dashboard.html",
    "decisions": REPORTS / "project_progress_decisions.html",
    "experiments": REPORTS / "project_progress_experiments.html",
    "daily": REPORTS / "project_progress_daily.html",
    "glossary": REPORTS / "project_progress_glossary.html",
}

SCAN_ROOTS = [
    ROOT / "tmp",
    ROOT / "datasetprocess" / "多模态数据" / "程序运行结果",
]

TOK = re.compile(r"`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)")
TIME_PAT = re.compile(r"_(\d{8})_(\d{6})$")
NEG_HINTS = ("阻塞", "问题", "恶化", "不支持", "不满足", "不能", "未", "缺")

SCRIPT_BASE = r"""
function bind(cfg){const s=document.getElementById(cfg.s),f=document.getElementById(cfg.f),ts=[...document.querySelectorAll(cfg.q)];let a='all';function ap(){const term=s?(s.value||'').trim().toLowerCase():'';ts.forEach(n=>{const tags=(n.dataset.filter||'').split(/\s+/).filter(Boolean);const ok1=a==='all'||tags.includes(a);const ok2=!term||(n.dataset.search||'').indexOf(term)!==-1;n.hidden=!(ok1&&ok2);});}if(f)f.addEventListener('click',e=>{const b=e.target.closest('button[data-filter]');if(!b)return;a=b.dataset.filter;f.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));ap();});if(s)s.addEventListener('input',ap);ap();}const ob=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)e.target.classList.add('on')}),{threshold:.12});document.querySelectorAll('.fade').forEach(n=>ob.observe(n));
"""


def read(p): return p.read_text(encoding="utf-8")


def tree(text):
    r = {"level": 0, "title": "", "lines": [], "children": []}
    stack = [r]
    for raw in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", raw)
        if m:
            lv = len(m.group(1))
            n = {"level": lv, "title": m.group(2).strip(), "lines": [], "children": []}
            while stack and stack[-1]["level"] >= lv:
                stack.pop()
            stack[-1]["children"].append(n)
            stack.append(n)
        else:
            stack[-1]["lines"].append(raw.rstrip())
    return r


def walk(n):
    for c in n.get("children", []):
        yield c
        for x in walk(c):
            yield x


def find(n, title):
    for c in walk(n):
        if c["title"] == title:
            return c
    return {"title": title, "lines": [], "children": []}


def list_items(lines, ordered=False):
    pat = r"^\s*\d+\.\s+(.*)$" if ordered else r"^\s*[-*]\s+(.*)$"
    return [m.group(1).strip() for line in lines for m in [re.match(pat, line)] if m]


def table_rows(lines):
    block = []
    for line in lines:
        if "|" in line:
            block.append(line.rstrip())
        elif block:
            break
    if len(block) < 2:
        return [], []
    heads = [c.strip() for c in block[0].strip().strip("|").split("|")]
    rows = []
    for line in block[2:]:
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        cells += [""] * (len(heads) - len(cells))
        rows.append({h: cells[i] for i, h in enumerate(heads)})
    return heads, rows


def rel(href, src):
    if not href or href.startswith("#") or re.match(r"^[a-zA-Z]+:", href):
        return href
    return Path(os.path.relpath(str((src.parent / href).resolve()), str(REPORTS))).as_posix()


def href_from_path(path):
    return Path(os.path.relpath(str(path.resolve()), str(REPORTS))).as_posix()


def inline(text, src):
    text = text or ""
    out, cur = [], 0
    for m in TOK.finditer(text):
        out.append(html.escape(text[cur:m.start()]))
        if m.group(1) is not None:
            out.append(f"<code>{html.escape(m.group(1))}</code>")
        else:
            out.append(f'<a href="{html.escape(rel(m.group(3), src), quote=True)}">{html.escape(m.group(2))}</a>')
        cur = m.end()
    out.append(html.escape(text[cur:]))
    return "".join(out)


def pick(row, *names):
    for n in names:
        if row.get(n):
            return row[n]
    return ""


def tag(title):
    m = re.match(r"^\[([^\]]+)\]\s*(.*)$", title)
    return (m.group(1), m.group(2).strip()) if m else ("note", title)


def split_kv(text):
    s = re.sub(r"^\s*[-*]\s+", "", text).strip()
    m = re.match(r"^([^:：]{1,24})[:：]\s*(.*)$", s)
    return (m.group(1).strip(), m.group(2).strip()) if m else ("", s)


def sclass(label):
    k = (label or "").strip().lower()
    if k in ("active", "proceed"): return "ok"
    if k in ("needs control", "keep, not replace", "proceed with label analysis"): return "watch"
    if k == "no-go": return "stop"
    return "neutral"


def vfilter(label):
    k = (label or "").strip().lower()
    if k in ("proceed", "proceed with label analysis"): return "proceed"
    if k == "keep, not replace": return "keep"
    if k == "needs control": return "watch"
    if k == "no-go": return "stop"
    if k == "archive": return "archive"
    if k == "active": return "active"
    return "other"


def fmt(v, d=3):
    if v is None or v == "":
        return "—"
    try:
        x = float(v)
    except Exception:
        return str(v)
    if abs(x) >= 100: return f"{x:.1f}"
    if abs(x) >= 10: return f"{x:.2f}"
    return f"{x:.{d}f}"


def nget(data, *keys):
    cur = data
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def parse_ts(name, path):
    m = TIME_PAT.search(name)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def family(prefix):
    u = (prefix or "").upper()
    if "MATCHED" in u and "EVENT_CONDITIONED" in u: return "matched", "matched 公平对照"
    if "EVENT_CONDITIONED" in u: return "event-conditioned", "event-conditioned"
    if "PROTOCOL_SAFE" in u: return "protocol-safe", "protocol-safe"
    if "INTERACTION_MULTIHYP" in u: return "multihyp", "interaction multihyp"
    if u.startswith("EXP2_"): return "legacy-exp2", "EXP2 历史实验"
    return "other", "其他"


def nickname(run):
    if run.get("family_key") == "protocol-safe":
        parts = ["protocol-safe"]
        if run.get("teacher_state_mode"):
            parts.append(str(run["teacher_state_mode"]))
        if run.get("smoke_test"):
            parts.append("smoke")
        return " · ".join(parts)
    parts = []
    mode = run.get("conditioning_mode")
    if mode == "baseline": parts.append("baseline 对照")
    elif mode == "structured_v2": parts.append("structured_v2")
    tf = run.get("teacher_forcing_ratio")
    if tf is not None:
        try:
            tf = float(tf)
            parts.append(f"TF{int(tf)}" if abs(tf-round(tf)) < 1e-9 else f"TF{tf:.2f}")
        except Exception:
            parts.append(f"TF {tf}")
    if run.get("event_residual_scale") == 0: parts.append("noresid")
    gt = run.get("gate_temperature")
    if gt not in (None, "", 0.04): parts.append(f"gate {fmt(gt,2)}")
    if run.get("smoke_test"): parts.append("smoke")
    return " · ".join(parts) if parts else (run.get("run_prefix") or run.get("run_name") or "run")


def breakdown(run):
    if run.get("family_key") == "protocol-safe":
        bits = ["protocol-safe = 按固定被试划分执行的主线验证 run"]
        if run.get("teacher_state_mode"):
            bits.append(f"{run['teacher_state_mode']} = 当前 teacher-state 表示方式")
        if run.get("split_policy"):
            bits.append(f"split = {run['split_policy']}")
        if run.get("smoke_test"):
            bits.append("smoke = 小规模快速验证")
        return "；".join(bits)
    p = (run.get("run_prefix") or "").upper()
    bits = []
    if "MATCHED" in p: bits.append("matched = 尽量公平对照")
    if "BASELINE" in p: bits.append("baseline = 基线对照")
    if "STRUCTV2" in p: bits.append("structured_v2 = 第二版结构化条件注入")
    if "TF0" in p: bits.append("TF0 = 不再强喂真实后续")
    if "NORESID" in p: bits.append("noresid = residual 支路关闭")
    if "GATE002" in p: bits.append("gate002 = gate temperature 0.02")
    return "；".join(bits) if bits else "自动发现目录名，当前没有更细的命名拆解。"


def plain(run):
    out = []
    if run["family_key"] == "protocol-safe": out.append("这是 protocol-safe 主线 run，用来按固定被试划分验证当前版本。")
    if run.get("teacher_state_mode") == "pca_latent": out.append("本轮使用 pca_latent teacher-state 表示。")
    if run.get("split_policy"): out.append(f"数据划分口径为 {run['split_policy']}。")
    if run["family_key"] == "matched": out.append("这是自动发现的 matched 对照 run，适合直接拿来做公平比较。")
    if run.get("conditioning_mode") == "baseline": out.append("它代表基础对照线。")
    if run.get("conditioning_mode") == "structured_v2": out.append("它代表结构化条件注入的第二版方案。")
    if run.get("teacher_forcing_ratio") == 0: out.append("TF0 主要是在看自由滚动预测会不会更稳。")
    if run.get("event_residual_scale") == 0: out.append("noresid 这支是在检查 residual 分支是不是问题来源。")
    if run.get("gate_temperature") not in (None, "", 0.04): out.append(f"这次还顺手调整了 gate temperature 到 {fmt(run.get('gate_temperature'),2)}。")
    return "".join(out) if out else ("这是 rich summary，可直接自动抽指标。" if run["schema"] == "rich" else "这是 legacy summary，当前更适合作为目录定位。")


def short_label(run):
    if run.get("family_key") == "protocol-safe":
        if run.get("teacher_state_mode") == "pca_latent":
            return "pca protocol"
        return "protocol"
    p = (run.get("run_prefix") or "").upper()
    if "MATCHED_BASELINE_TF0" in p: return "base TF0"
    if "MATCHED_BASELINE" in p: return "base"
    if "NORESID" in p: return "noresid"
    if "GATE002" in p: return "gate 0.02"
    if "STRUCTV2_TF0" in p: return "struct TF0"
    if "STRUCTV2" in p: return "struct v2"
    return run.get("nickname") or run.get("run_name") or "run"


def parse_run(summary_path):
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    run_dir = Path(data.get("run_root") or data.get("output_root") or summary_path.parent)
    config = data.get("config") or data.get("experiment_config") or {}
    prefix = config.get("run_prefix") or data.get("run_prefix") or run_dir.name
    ts = parse_ts(run_dir.name, summary_path)
    test = nget(data, "selection_compare", "active", "test") or data.get("final_test_metrics") or {}
    sel = test.get("selection_summary") or {}
    fam_key, fam_label = family(prefix)
    run = {
        "run_name": run_dir.name,
        "run_prefix": prefix,
        "summary_href": href_from_path(summary_path),
        "summary_label": summary_path.name,
        "dir_href": href_from_path(run_dir),
        "timestamp": ts,
        "timestamp_text": ts.strftime("%Y-%m-%d %H:%M"),
        "date": ts.strftime("%Y-%m-%d"),
        "family_key": fam_key,
        "family_label": fam_label,
        "schema": "rich" if test or data.get("config") else "legacy",
        "conditioning_mode": config.get("conditioning_mode"),
        "teacher_forcing_ratio": config.get("teacher_forcing_ratio"),
        "event_residual_scale": config.get("event_residual_scale"),
        "gate_temperature": config.get("gate_temperature"),
        "smoke_test": bool(data.get("smoke_test")),
        "teacher_state_mode": config.get("teacher_state_mode") or config.get("TEACHER_STATE_MODE"),
        "split_policy": config.get("split_policy_applied") or config.get("split_policy_expected"),
        "steer_rmse": test.get("steer_rmse"),
        "selection_score": sel.get("selection_score"),
        "boundary_shift_abs_err": sel.get("boundary_shift_abs_err"),
        "peak_time_abs_err_s": sel.get("peak_time_abs_err_s"),
        "manual_hint": "",
    }
    run["nickname"] = nickname(run)
    run["short_label"] = short_label(run)
    run["naming"] = breakdown(run)
    run["plain"] = plain(run)
    run["preview_images"] = collect_preview_images(run_dir)
    run["filters"] = " ".join(x for x in [run["schema"], run["family_key"], "tf0" if run.get("teacher_forcing_ratio") == 0 else ""] if x)
    run["search"] = " ".join([run["run_name"], run["run_prefix"], run["nickname"], run["naming"], run["plain"]]).lower()
    return run


def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_training_summary(summary_path):
    run_dir = summary_path.parent
    training = load_json(summary_path)
    config = load_json(run_dir / "run_config.json")
    test_basic = load_json(run_dir / "test_metrics_basic.json")
    if not training or not config:
        return None

    prefix = config.get("MODEL_VER") or config.get("protocol_version") or run_dir.name
    ts = parse_ts(run_dir.name, summary_path)
    fam_key, fam_label = family(f"{prefix} {run_dir.name}")
    run = {
        "run_name": run_dir.name,
        "run_prefix": prefix,
        "summary_href": href_from_path(summary_path),
        "summary_label": summary_path.name,
        "dir_href": href_from_path(run_dir),
        "timestamp": ts,
        "timestamp_text": ts.strftime("%Y-%m-%d %H:%M"),
        "date": ts.strftime("%Y-%m-%d"),
        "family_key": fam_key,
        "family_label": fam_label,
        "schema": "rich",
        "conditioning_mode": None,
        "teacher_forcing_ratio": None,
        "event_residual_scale": None,
        "gate_temperature": None,
        "smoke_test": bool(config.get("smoke_mode")),
        "teacher_state_mode": config.get("TEACHER_STATE_MODE") or config.get("teacher_state_mode"),
        "split_policy": config.get("split_policy_applied") or config.get("split_policy_expected"),
        "steer_rmse": test_basic.get("rmse_steer") or nget(training, "test_metrics_basic", "rmse_steer"),
        "selection_score": None,
        "boundary_shift_abs_err": None,
        "peak_time_abs_err_s": None,
        "manual_hint": "",
    }
    run["nickname"] = nickname(run)
    run["short_label"] = short_label(run)
    run["naming"] = breakdown(run)
    run["plain"] = plain(run)
    run["preview_images"] = collect_preview_images(run_dir)
    run["filters"] = " ".join(x for x in [run["schema"], run["family_key"], "smoke" if run.get("smoke_test") else ""] if x)
    run["search"] = " ".join([run["run_name"], run["run_prefix"], run["nickname"], run["naming"], run["plain"]]).lower()
    return run


def parse_metrics_bundle(metrics_path):
    run_dir = metrics_path.parent.parent if metrics_path.parent.name == "figures" else metrics_path.parent
    config = load_json(run_dir / "run_config.json")
    metrics = load_json(metrics_path)
    peak = load_json(run_dir / "figures" / "test_metrics_peak.json")
    tail = load_json(run_dir / "figures" / "test_metrics_tail.json")
    if not config or not metrics:
        return None

    prefix = config.get("MODEL_VER") or config.get("protocol_version") or run_dir.name
    ts = parse_ts(run_dir.name, metrics_path)
    fam_key, fam_label = family(f"{prefix} {run_dir.name} {run_dir.parent.name}")
    run = {
        "run_name": run_dir.name,
        "run_prefix": prefix,
        "summary_href": href_from_path(metrics_path),
        "summary_label": metrics_path.name,
        "dir_href": href_from_path(run_dir),
        "timestamp": ts,
        "timestamp_text": ts.strftime("%Y-%m-%d %H:%M"),
        "date": ts.strftime("%Y-%m-%d"),
        "family_key": fam_key,
        "family_label": fam_label,
        "schema": "rich",
        "conditioning_mode": None,
        "teacher_forcing_ratio": None,
        "event_residual_scale": None,
        "gate_temperature": None,
        "smoke_test": bool(config.get("smoke_mode")),
        "teacher_state_mode": config.get("TEACHER_STATE_MODE") or config.get("teacher_state_mode"),
        "split_policy": config.get("split_policy_applied") or config.get("split_policy_expected"),
        "steer_rmse": metrics.get("rmse_steer"),
        "selection_score": None,
        "boundary_shift_abs_err": tail.get("tail_slope_mae"),
        "peak_time_abs_err_s": peak.get("peak_time_mae_sec"),
        "manual_hint": "",
    }
    run["nickname"] = nickname(run)
    run["short_label"] = short_label(run)
    run["naming"] = breakdown(run)
    run["plain"] = plain(run)
    run["preview_images"] = collect_preview_images(run_dir)
    run["filters"] = " ".join(x for x in [run["schema"], run["family_key"], "smoke" if run.get("smoke_test") else ""] if x)
    run["search"] = " ".join([run["run_name"], run["run_prefix"], run["nickname"], run["naming"], run["plain"]]).lower()
    return run


def preview_caption(name):
    low = name.lower()
    if "pred_vs_gt" in low and "steer_only" in low:
        return "预测 vs 真值 · steer 细看"
    if "pred_vs_gt" in low:
        return "预测 vs 真值"
    if "trajectory_focus_comparison" in low:
        return "轨迹聚焦对比"
    if "best_model_test_steer_overview" in low:
        return "全样本总览"
    if "best_model_test_steer_detail" in low:
        return "局部细节"
    if "best_model_test_steer_all_samples" in low:
        return "全样本细节"
    if "pairwise_delta" in low:
        return "差值对比"
    return "预测图"


def collect_preview_images(run_dir, limit=3):
    img_exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    files, seen = [], set()
    for folder in [run_dir / "figures", run_dir]:
        if not folder.exists():
            continue
        for pattern in img_exts:
            for path in sorted(folder.glob(pattern)):
                key = str(path.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                files.append(path)

    if not files:
        return []

    preferred = (
        "pred_vs_gt",
        "trajectory_focus_comparison",
        "best_model_test_steer_overview",
        "best_model_test_steer_detail",
        "best_model_test_steer_all_samples",
        "pairwise_delta",
    )
    avoid = ("loss_curve", "state_vs_peak", "test_metric", "val_metric", "teacher_base_missing", "split_")

    picked, used = [], set()

    def add(path):
        key = str(path.resolve()).lower()
        if key in used:
            return
        used.add(key)
        picked.append({
            "href": href_from_path(path),
            "name": path.name,
            "caption": preview_caption(path.name),
        })

    for hint in preferred:
        for path in files:
            if hint in path.name.lower():
                add(path)
                if len(picked) >= limit:
                    return picked

    for path in files:
        name = path.name.lower()
        if any(bad in name for bad in avoid):
            continue
        add(path)
        if len(picked) >= limit:
            return picked

    return picked


def discover_runs():
    runs, seen_paths, seen_dirs = [], set(), set()
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for summary_path in sorted(root.rglob("run_summary.json")):
            key = str(summary_path.resolve()).lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            run = parse_run(summary_path)
            if run:
                seen_dirs.add(str(summary_path.parent.resolve()).lower())
                runs.append(run)
        for summary_path in sorted(root.rglob("training_summary.json")):
            dir_key = str(summary_path.parent.resolve()).lower()
            if dir_key in seen_dirs:
                continue
            key = str(summary_path.resolve()).lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            run = parse_training_summary(summary_path)
            if run:
                seen_dirs.add(dir_key)
                runs.append(run)
        for metrics_path in sorted(root.rglob("test_metrics.json")):
            if metrics_path.parent.name != "figures":
                continue
            dir_key = str(metrics_path.parent.parent.resolve()).lower()
            if dir_key in seen_dirs:
                continue
            key = str(metrics_path.resolve()).lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            run = parse_metrics_bundle(metrics_path)
            if run:
                seen_dirs.add(dir_key)
                runs.append(run)
    runs.sort(key=lambda x: x["timestamp"], reverse=True)
    return runs


H = tree(read(HUB)); D = tree(read(DECISIONS)); E = tree(read(EXPERIMENTS)); G = tree(read(GLOSSARY))
_, drows = table_rows(find(D, "决策表")["lines"])
_, erows = table_rows(find(E, "实验表")["lines"])
_, dprev = table_rows(find(H, "最近关键判断")["lines"])
_, eprev = table_rows(find(H, "最近实验快照")["lines"])

ctx = {
    "status": [dict(zip(("label", "value"), split_kv(x))) | {"raw": x} for x in list_items(find(H, "当前状态")["lines"])],
    "plain": list_items(find(H, "30 秒白话版")["lines"]),
    "prio": list_items(find(H, "当前优先级")["lines"], True),
    "paths": list_items(find(H, "建议查阅路径")["lines"]),
    "decisions": [{"date": r.get("日期", ""), "title": r.get("决策", ""), "plain": pick(r, "白话解释", "白话"), "reason": r.get("触发原因", ""), "impact": r.get("影响范围", ""), "link": r.get("证据 / 入口", ""), "state": r.get("状态", ""), "filters": vfilter(r.get("状态", "")), "search": " ".join(r.values()).lower()} for r in drows],
    "experiments": [{"date": r.get("日期", ""), "name": r.get("实验 / 分析", ""), "naming": pick(r, "命名拆解"), "plain": pick(r, "白话解释", "白话"), "compare": r.get("可比性", ""), "change": r.get("变更", ""), "result": r.get("关键结果", ""), "verdict": r.get("判定", ""), "detail": r.get("详情", ""), "filters": " ".join(x for x in [vfilter(r.get("判定", "")), r.get("可比性", "").strip().lower().replace(" ", "-")] if x), "search": " ".join(r.values()).lower()} for r in erows],
    "dprev": dprev,
    "eprev": eprev,
    "glossary": [],
    "days": [],
    "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
}

for sec in walk(G):
    if sec["level"] != 2:
        continue
    heads, rows = table_rows(sec["lines"])
    ctx["glossary"].append({"title": sec["title"], "heads": heads, "rows": rows, "bullets": list_items(sec["lines"]) + list_items(sec["lines"], True)})

for p in sorted(DAILY.glob("*.md"), reverse=True):
    T = tree(read(p)); hi = list_items(find(T, "今日重点")["lines"]); entries = []
    for n in walk(T):
        if n["level"] == 3:
            tg, tt = tag(n["title"]); entries.append({"tag": tg, "title": tt, "lines": [x for x in n["lines"] if x.strip()], "src": p})
    ctx["days"].append({"date": p.stem, "path": p.relative_to(REPORTS).as_posix(), "high": hi, "entries": entries, "src": p})

auto_runs = discover_runs()
for run in auto_runs:
    rnorm = re.sub(r"[^a-z0-9]+", "", (run["run_name"] + run["run_prefix"]).lower())
    for row in ctx["experiments"]:
        t = re.sub(r"[^a-z0-9]+", "", row["name"].lower())
        if t and t in rnorm:
            run["manual_hint"] = row["plain"] or row["result"]
            break

stats = {
    "latest": ctx["days"][0]["date"] if ctx["days"] else "n/a",
    "days": len(ctx["days"]),
    "decisions": sum(1 for x in ctx["decisions"] if x["state"].lower() == "active"),
    "experiments": len(ctx["experiments"]),
    "auto": len(auto_runs),
    "rich": sum(1 for x in auto_runs if x["schema"] == "rich"),
}


def nav(active):
    items = [("dashboard", "总览", "project_progress_dashboard.html"), ("decisions", "决策", "project_progress_decisions.html"), ("experiments", "实验", "project_progress_experiments.html"), ("daily", "日志", "project_progress_daily.html"), ("glossary", "词典", "project_progress_glossary.html")]
    return "".join(f'<a href="{h}" class="{"active" if k == active else ""}">{t}</a>' for k, t, h in items)


def ul(items, src, ordered=False):
    tagname = "ol" if ordered else "ul"
    return f"<{tagname} class=\"bullets\">" + "".join(f"<li>{inline(x, src)}</li>" for x in items) + f"</{tagname}>"


def badge(text, cls="tag"):
    return f'<span class="{cls}">{html.escape(text)}</span>'


def card(title, body, extra=""):
    return f"<div class='item'>{extra}<div class='title'>{html.escape(title)}</div><p>{body}</p></div>"


def section(title, sub, body, eyebrow=""):
    e = f"<p class='title'>{html.escape(eyebrow)}</p>" if eyebrow else ""
    return f"<section class='panel fade'><div class='head'><div>{e}<h2>{html.escape(title)}</h2><p class='sub'>{html.escape(sub)}</p></div></div>{body}</section>"


def shell(active, title, kicker, lead, actions, side, body, script=""):
    # 全新的工作台左右布局，不再使用巨型 Hero
    return f"""<!DOCTYPE html><html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="project_progress_site.css">
</head>
<body>
<div class="top">
    <div class="topin">
        <div class="brand"><strong>Progress</strong><span>项目进度工作台</span></div>
        <nav class="nav">{nav(active)}</nav>
    </div>
</div>
<div class="layoutWrap">
    <main class="layoutMain">
        <header class="pageHeader fade">
            <div class="kicker">{html.escape(kicker)}</div>
            <h1>{html.escape(title)}</h1>
            <p class="lead">{html.escape(lead)}</p>
            <div class="actions">{actions}</div>
            <div class="stats">
                <div class="stat"><span>最新日志</span><strong>{stats['latest']}</strong></div>
                <div class="stat"><span>扫描 run</span><strong>{stats['auto']}</strong></div>
                <div class="stat"><span>Rich 摘要</span><strong>{stats['rich']}</strong></div>
                <div class="stat"><span>Active 决策</span><strong>{stats['decisions']}</strong></div>
                <div class="stat"><span>人工登记</span><strong>{stats['experiments']}</strong></div>
            </div>
        </header>
        {body}
        <p class="foot">生成时间：{html.escape(ctx['generated'])} · <code>py -3 tools/build_progress_dashboard.py</code></p>
    </main>
    <aside class="layoutSide fade">
        {side}
    </aside>
</div>
<script>{SCRIPT_BASE}{script}</script>
</body>
</html>"""

def side_block(title, content):
    return f"<div class='sideBlock'><div class='title'>{html.escape(title)}</div>{content}</div>"

def lane_item(title, text, src_label, href, tone="note", date=""):
    search = " ".join([title, text, src_label, date]).lower()
    return f"<article class='laneItem {tone}' data-search='{html.escape(search, quote=True)}'><div class='laneMeta'>{badge(date or '当前', 'pill')} {badge(src_label, 'tag')}</div><h3>{inline(title, HUB)}</h3><p>{inline(text, HUB)}</p><div class='laneLink'><a href='{html.escape(href, quote=True)}'>查看原文 &rarr;</a></div></article>"


def lanes():
    blocked, active, archived = [], [], []
    for item in ctx["status"]:
        txt = item["value"] or item["raw"]
        if item["label"] in ("当前可信判断", "当前模型问题") or any(k in txt for k in NEG_HINTS):
            blocked.append(lane_item(item["label"] or "当前状态", txt, "项目中枢", href_from_path(HUB), "warn", stats["latest"]))
    for row in ctx["decisions"]:
        if row["state"].lower() == "active":
            active.append(lane_item(row["title"], row["plain"] or row["impact"], "决策日志", href_from_path(DECISIONS), "ok", row["date"]))
    for row in ctx["experiments"]:
        vf = vfilter(row["verdict"])
        if vf == "watch":
            blocked.append(lane_item(row["name"], row["plain"] or row["result"], "实验登记表", href_from_path(EXPERIMENTS), "warn", row["date"]))
        elif vf in ("proceed", "keep"):
            active.append(lane_item(row["name"], row["plain"] or row["result"], "实验登记表", href_from_path(EXPERIMENTS), "ok" if vf == "proceed" else "watch", row["date"]))
        elif vf in ("stop", "archive"):
            archived.append(lane_item(row["name"], row["plain"] or row["result"], "实验登记表", href_from_path(EXPERIMENTS), "stop" if vf == "stop" else "neutral", row["date"]))
    def mk(title, sub, cls, items):
        empty = "<div class='note' style='padding: 12px; font-size: 0.85rem; color: var(--muted);'>当前没有条目。</div>"
        body = "".join(items) if items else empty
        return f"<div class='lane {cls}'><div class='laneHead'><div><h3>{html.escape(title)}</h3><p class='muted'>{html.escape(sub)}</p></div>{badge(str(len(items)), 'pill')}</div><div class='laneStack'>{body}</div></div>"
    return f"<div class='laneBoard'>{mk('当前阻塞 / 需确认', '需要谨慎对待的观察或未达标指标。', 'laneWarn', blocked)}{mk('正在推进 / 生效中', '确认值得继续推的方向和 Active 决策。', 'laneOk', active)}{mk('已归档 / 暂停', '暂时停止或已完成归档的路线。', 'laneStop', archived)}</div>"


def chart_svg(points, color, higher=False):
    # 扩大 viewBox 宽度，避免文本拥挤；使用居中水平文本
    vals = [p["value"] for p in points if p["value"] is not None]
    if len(vals) < 2:
        return "<svg class='chartSvg' viewBox='0 0 720 240'><text x='24' y='40' fill='#64748b' font-size='14'>数据点不足</text></svg>"
    w, h, pl, pr, pt, pb = 720, 240, 60, 40, 20, 40
    low, high = min(vals), max(vals)
    if abs(high - low) < 1e-9:
        low -= .5; high += .5
    pts = []
    for i, p in enumerate(points):
        if p["value"] is None: continue
        x = pl + i * ((w - pl - pr) / max(len(points) - 1, 1))
        y = pt + (h - pt - pb) - ((p["value"] - low) / (high - low)) * (h - pt - pb)
        pts.append((x, y, p))
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y, _ in pts)
    area = path + f" L {pts[-1][0]:.1f} {h-pb:.1f} L {pts[0][0]:.1f} {h-pb:.1f} Z"
    best = max(vals) if higher else min(vals)
    grid = []
    for i in range(4):
        y = pt + (h - pt - pb) - i / 3 * (h - pt - pb)
        v = low + (high - low) * i / 3
        grid.append(f"<line x1='{pl}' y1='{y:.1f}' x2='{w-pr}' y2='{y:.1f}' stroke='#e2e8f0' stroke-dasharray='4 4'/>")
        grid.append(f"<text x='{pl-12}' y='{y+4:.1f}' text-anchor='end' font-size='11' fill='#94a3b8'>{fmt(v)}</text>")
    marks, labels = [], []
    for x, y, p in pts:
        r = 6 if abs(p['value'] - best) < 1e-9 else 4
        sw = 2 if abs(p['value'] - best) < 1e-9 else 1.5
        marks.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{color}' stroke='#ffffff' stroke-width='{sw}'/>")
        # 水平居中标签，截断过长文字
        lbl = p['label'] if len(p['label']) < 12 else p['label'][:10] + ".."
        labels.append(f"<text x='{x:.1f}' y='{h-12}' text-anchor='middle' font-size='10' fill='#64748b'>{html.escape(lbl)}</text>")
    return f"<svg class='chartSvg' viewBox='0 0 {w} {h}'><defs><linearGradient id='g{color[1:]}' x1='0' x2='0' y1='0' y2='1'><stop offset='0%' stop-color='{color}' stop-opacity='.15'/><stop offset='100%' stop-color='{color}' stop-opacity='.0'/></linearGradient></defs>{''.join(grid)}<path d='{area}' fill='url(#g{color[1:]})'/><path d='{path}' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/>{''.join(marks)}{''.join(labels)}</svg>"


def chart_card(title, sub, points, color, higher=False):
    valid = [p for p in points if p["value"] is not None]
    best = max(valid, key=lambda x: x["value"]) if valid and higher else (min(valid, key=lambda x: x["value"]) if valid else None)
    legend = "".join(f"<div class='legendItem'><span>{html.escape(p['label'])}</span><strong>{fmt(p['value'])}</strong></div>" for p in valid)
    note = f"当前最优：{best['label']} ({fmt(best['value'])})" if best else "暂无数据"
    return f"<article class='chartCard'><div class='chartHead'><div><h3>{html.escape(title)}</h3><p>{html.escape(sub)}</p></div>{badge('越大越好' if higher else '越小越好', 'tag')}</div>{chart_svg(points, color, higher)}<div class='chartNote'>{badge(note, 'pill')}</div><div class='legendGrid'>{legend}</div></article>"


def matched_runs():
    return [x for x in sorted(auto_runs, key=lambda y: y["timestamp"]) if x["family_key"] == "matched" and x["schema"] == "rich"]


def trend_section():
    runs = matched_runs()
    mk = lambda key: [{"label": r["short_label"], "value": r.get(key)} for r in runs]
    return f"<div class='chartGrid'>{chart_card('steer_rmse', '整体转向误差，先看大盘是否收口。', mk('steer_rmse'), '#0284c7')}{chart_card('selection_score', '项目综合分，快速扫面用。', mk('selection_score'), '#ea580c', True)}{chart_card('boundary_shift_abs_err', '关键边界偏移，当前核心阻塞指标。', mk('boundary_shift_abs_err'), '#ca8a04')}{chart_card('peak_time_abs_err_s', '峰值时间偏差，校验时序是否走对。', mk('peak_time_abs_err_s'), '#16a34a')}</div>"


def matched_table():
    runs = matched_runs()
    if not runs:
        return "<div class='note'><p>当前没有可比较的 matched rich run。</p></div>"
    rows = "".join(f"<tr><td><div class='tableTitle'>{html.escape(r['nickname'])}</div><div class='muted' style='font-size:0.75rem'><code>{html.escape(r['run_name'])}</code></div></td><td>{html.escape(r['timestamp_text'])}</td><td>{html.escape(r['plain'])}</td><td>{fmt(r['steer_rmse'])}</td><td>{fmt(r['selection_score'])}</td><td>{fmt(r['boundary_shift_abs_err'])}</td><td>{fmt(r['peak_time_abs_err_s'])}</td><td><a href='{html.escape(r['summary_href'], quote=True)}'>详情</a></td></tr>" for r in runs)
    return f"<div class='tableWrap'><table class='dataTable'><thead><tr><th>Run 缩写与目录</th><th>时间</th><th>白话解释</th><th>Steer RMSE</th><th>Score</th><th>Boundary</th><th>Peak Time</th><th>入口</th></tr></thead><tbody>{rows}</tbody></table></div>"


def preview(rows, sentence, link):
    return "".join(f"<div style='margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px dashed var(--line);'><p style='margin:0 0 4px; font-size:0.9rem;'>{inline(r.get(sentence,''), HUB)}</p><div class='muted' style='font-size:0.8rem;'>{inline(r.get(link,''), HUB)}</div></div>" for r in rows)


def glossary_terms(names):
    wanted = {x.replace("`", "") for x in names}
    cards = []
    for sec in ctx["glossary"]:
        if not sec["rows"] or not sec["heads"]:
            continue
        key = sec["heads"][0]
        for row in sec["rows"]:
            raw = row.get(key, "")
            if raw.replace("`", "") not in wanted:
                continue
            metas = "".join(f"<div class='metaBlock'><strong>{html.escape(h)}</strong><p>{inline(row.get(h,''), GLOSSARY)}</p></div>" for h in sec["heads"][1:] if row.get(h))
            cards.append(f"<article class='term'><h3>{inline(raw, GLOSSARY)}</h3>{metas}</article>")
    return "<div class='termGrid'>" + ("".join(cards) if cards else "<div class='note'><p>暂无相关术语卡片。</p></div>") + "</div>"


def preview_block(run, limit=3):
    imgs = run.get("preview_images") or []
    if not imgs:
        return ""
    cover = imgs[0]
    thumbs = imgs[1:limit]
    rail = "".join(
        f"<a class='previewThumb' href='{html.escape(img['href'], quote=True)}' title='{html.escape(img['name'])}'><img loading='lazy' src='{html.escape(img['href'], quote=True)}' alt='{html.escape(img['caption'])}'></a>"
        for img in thumbs
    )
    rail_html = f"<div class='previewRail'>{rail}</div>" if rail else ""
    count = badge(f"{len(imgs)} 张预测图", "tag")
    return (
        f"<div class='previewBlock'>"
        f"<div class='previewHead'><strong>预测图预览</strong>{count}</div>"
        f"<a class='previewHero' href='{html.escape(cover['href'], quote=True)}' title='{html.escape(cover['name'])}'>"
        f"<img loading='lazy' src='{html.escape(cover['href'], quote=True)}' alt='{html.escape(cover['caption'])}'>"
        f"</a>"
        f"<div class='previewCaption'>{html.escape(cover['caption'])}</div>"
        f"{rail_html}"
        f"</div>"
    )


def prediction_wall(limit_runs=6, per_run=2):
    runs = [r for r in auto_runs if r.get("preview_images")]
    if not runs:
        return "<div class='note'><p>当前还没有可自动提取的预测图目录。</p></div>"
    cards = []
    for run in runs[:limit_runs]:
        hero = run["preview_images"][0]
        thumbs = run["preview_images"][1:per_run]
        thumb_html = "".join(
            f"<a class='wallThumb' href='{html.escape(img['href'], quote=True)}' title='{html.escape(img['name'])}'><img loading='lazy' src='{html.escape(img['href'], quote=True)}' alt='{html.escape(img['caption'])}'></a>"
            for img in thumbs
        )
        thumb_row = f"<div class='wallThumbRow'>{thumb_html}</div>" if thumb_html else ""
        metric_line = " · ".join(
            x for x in [
                f"steer {fmt(run.get('steer_rmse'))}",
                f"score {fmt(run.get('selection_score'))}" if run.get("selection_score") is not None else "",
                f"boundary {fmt(run.get('boundary_shift_abs_err'))}" if run.get("boundary_shift_abs_err") is not None else "",
            ] if x
        )
        cards.append(
            f"<article class='previewRun'>"
            f"<div class='previewRunMeta'>{badge(run['timestamp_text'], 'pill')} {badge(run['family_label'], 'tag')}</div>"
            f"<h3>{html.escape(run['nickname'])}</h3>"
            f"<p class='muted'>{html.escape(run['plain'])}</p>"
            f"<a class='previewPoster' href='{html.escape(hero['href'], quote=True)}' title='{html.escape(hero['name'])}'>"
            f"<img loading='lazy' src='{html.escape(hero['href'], quote=True)}' alt='{html.escape(hero['caption'])}'>"
            f"</a>"
            f"<div class='previewPosterCaption'>{html.escape(hero['caption'])}</div>"
            f"{thumb_row}"
            f"<div class='previewRunFoot'><span>{html.escape(metric_line or '当前以图像直观比对为主')}</span><a href='{html.escape(run['dir_href'], quote=True)}'>打开实验目录</a></div>"
            f"</article>"
        )
    return "<div class='previewWall'>" + "".join(cards) + "</div>"


def run_card(r):
    # 显著强调人工登记表的关联补充
    hint = f"<div class='linkedManual'><div class='lmHead'>📌 关联人工登记表</div><p>{html.escape(r['manual_hint'])}</p></div>" if r.get("manual_hint") else ""
    return f"<article class='runCard' data-filter='{html.escape(r['filters'], quote=True)}' data-search='{html.escape(r['search'], quote=True)}'><div class='runMeta'>{badge(r['timestamp_text'], 'pill')} {badge(r['schema'], 'tag')} {badge(r['family_label'], 'tag')}</div><h3>{html.escape(r['nickname'])}</h3><div class='muted' style='font-size:0.75rem'><code>{html.escape(r['run_name'])}</code></div>{preview_block(r, 3)}<div class='callout'><strong>命名拆解</strong><p>{html.escape(r['naming'])}</p></div><div class='detail'><strong>白话解释</strong><p>{html.escape(r['plain'])}</p></div>{hint}<div class='metricMini'><div><span>steer_rmse</span><strong>{fmt(r['steer_rmse'])}</strong></div><div><span>selection_score</span><strong>{fmt(r['selection_score'])}</strong></div><div><span>boundary_shift</span><strong>{fmt(r['boundary_shift_abs_err'])}</strong></div><div><span>peak_time</span><strong>{fmt(r['peak_time_abs_err_s'])}</strong></div></div><div class='detail' style='margin-top:auto'><strong>入口</strong><p><a href='{html.escape(r['summary_href'], quote=True)}'>{html.escape(r.get('summary_label', 'summary'))}</a> · <a href='{html.escape(r['dir_href'], quote=True)}'>实验目录</a></p></div></article>"


def run_index():
    rows = "".join(f"<tr><td><div class='tableTitle'>{html.escape(r['nickname'])}</div><div class='muted' style='font-size:0.75rem'><code>{html.escape(r['run_name'])}</code></div></td><td>{html.escape(r['timestamp_text'])}</td><td>{html.escape(r['family_label'])}</td><td>{html.escape(r['schema'])}</td><td>{html.escape(r['plain'])}</td><td>steer {fmt(r['steer_rmse'])}<br>select {fmt(r['selection_score'])}<br>boundary {fmt(r['boundary_shift_abs_err'])}</td><td><a href='{html.escape(r['summary_href'], quote=True)}'>summary</a></td></tr>" for r in auto_runs)
    return f"<details class='fold'><summary>展开全部自动扫描 Run（{len(auto_runs)} 条）</summary><div class='foldBody'><div class='tableWrap'><table class='dataTable'><thead><tr><th>Run</th><th>时间</th><th>家族</th><th>Schema</th><th>白话解释</th><th>关键指标</th><th>入口</th></tr></thead><tbody>{rows}</tbody></table></div></div></details>"


def decision_feed():
    return "".join(f"<article class='article' data-filter='{html.escape(x['filters'], quote=True)}' data-search='{html.escape(x['search'], quote=True)}'><div class='meta'><span class='pill'>{html.escape(x['date'])}</span><span class='state {sclass(x['state'])}'>{html.escape(x['state'])}</span></div><div class='content'><h2>{inline(x['title'], DECISIONS)}</h2><div class='callout'><strong>白话解释</strong><p>{inline(x['plain'], DECISIONS)}</p></div><div class='detail'><strong>触发原因</strong><p>{inline(x['reason'], DECISIONS)}</p></div><div class='detail'><strong>影响范围</strong><p>{inline(x['impact'], DECISIONS)}</p></div><div class='detail'><strong>证据入口</strong><p>{inline(x['link'], DECISIONS)}</p></div></div></article>" for x in ctx["decisions"])


def experiment_feed():
    return "".join(f"<article class='article' data-filter='{html.escape(x['filters'], quote=True)}' data-search='{html.escape(x['search'], quote=True)}'><div class='meta'><span class='pill'>{html.escape(x['date'])}</span><span class='tag'>{inline(x['compare'], EXPERIMENTS)}</span><span class='state {sclass(x['verdict'])}'>{html.escape(x['verdict'])}</span></div><div class='content'><h2>{inline(x['name'], EXPERIMENTS)}</h2><div class='callout'><strong>命名拆解</strong><p>{inline(x['naming'], EXPERIMENTS)}</p></div><div class='callout'><strong>白话解释</strong><p>{inline(x['plain'], EXPERIMENTS)}</p></div><div class='detail'><strong>变更</strong><p>{inline(x['change'], EXPERIMENTS)}</p></div><div class='detail'><strong>关键结果</strong><p>{inline(x['result'], EXPERIMENTS)}</p></div><div class='detail'><strong>详情</strong><p>{inline(x['detail'], EXPERIMENTS)}</p></div></div></article>" for x in ctx["experiments"])


def linebox(line, src):
    ind = min((len(line)-len(line.lstrip(" ")))//2, 3)
    s = line.strip()
    m1 = re.match(r"^[-*]\s+(.*)$", s); m2 = re.match(r"^(\d+\.)\s+(.*)$", s)
    text = m1.group(1) if m1 else (m2.group(2) if m2 else s)
    k, v = split_kv(text)
    if k:
        return f"<div class='kv indent{ind}'><div class='k'>{html.escape(k)}</div><div>{inline(v, src)}</div></div>"
    return f"<div class='txt indent{ind}'><div class='k'>记录</div><div>{inline(text, src)}</div></div>"


def day_feed():
    days = []
    for d in ctx["days"]:
        entries = "".join(f"<details class='entry' data-filter='{html.escape(e['tag'], quote=True)}' data-search='{html.escape(' '.join([d['date'], e['tag'], e['title']] + e['lines']).lower(), quote=True)}'><summary><span class='tag'>{html.escape(e['tag'])}</span><strong>{inline(e['title'], d['src'])}</strong></summary><div class='entryBody'>{''.join(linebox(x, d['src']) for x in e['lines'])}</div></details>" for e in d["entries"])
        days.append(f"<section class='day fade'><div class='dayHead'><div class='row'><span class='pill'>{html.escape(d['date'])}</span><a class='muted' href='{html.escape(d['path'], quote=True)}'>查看原始日志</a></div></div>{ul(d['high'], d['src']) if d['high'] else ''}<div class='stack'>{entries}</div></section>")
    return "<div class='timeline'>" + "".join(days) + "</div>"


def glossary_sections():
    out = []
    for sec in ctx["glossary"]:
        cards = []
        if sec["rows"]:
            first = sec["heads"][0]
            for row in sec["rows"]:
                metas = "".join(f"<div class='metaBlock'><strong>{html.escape(h)}</strong><p>{inline(row.get(h,''), GLOSSARY)}</p></div>" for h in sec["heads"][1:] if row.get(h))
                cards.append(f"<article class='term' data-filter='{html.escape(sec['title'], quote=True)}' data-search='{html.escape(' '.join(row.values()).lower(), quote=True)}'><h3>{inline(row.get(first,''), GLOSSARY)}</h3>{metas}</article>")
        extra = ul(sec["bullets"], GLOSSARY) if sec["bullets"] else ""
        out.append(f"<section class='section fade'><div class='head'><div><p class='title'>Category</p><h2>{html.escape(sec['title'])}</h2></div></div>{extra}<div class='termGrid'>{''.join(cards)}</div></section>")
    return "".join(out)


families = Counter(r["family_label"] for r in auto_runs)
family_chips = "".join(badge(f"{k} {v}", "tag") for k, v in families.most_common(5))
auto_note = "自动同步已接入 run_summary.json / training_summary.json；像 protocol-safe 这类不产出 run_summary 的 run，现在也能进入自动索引。" if any(r["family_key"] == "protocol-safe" for r in auto_runs) else "自动同步当前主要覆盖 run_summary.json；未统一产出摘要文件的探索性实验，仍需人工通过「实验表」追溯。"

# Sidebars as grouped side blocks
main_side = side_block("30 秒白话版", ul(ctx['plain'], HUB)) + side_block("当前优先级", ul(ctx['prio'], HUB, True)) + side_block("自动同步大盘", f"<div class='stack'><div class='muted' style='font-size:0.85rem'>{auto_note}</div><div>{family_chips}</div></div>")

# Main Content
main_body = section("当前状态总览", "将中枢页核心提取为卡片，先抓结论，再行下钻。", "<div class='statusGrid'>" + "".join(f"<div class='item'><div class='title'>{html.escape(x['label'] or '状态')}</div><p>{inline(x['value'] or x['raw'], HUB)}</p></div>" for x in ctx["status"]) + "</div>", "Current State") \
          + section("项目泳道看板", "拉平“当前阻塞 / 正在推进 / 已归档”信息源，清晰界定排查重点。", lanes(), "Swimlanes") \
          + section("自动跟踪指标趋势", "自动扫描公平对照（Matched）组的关键实验，追踪水位变化。", trend_section(), "Metric Trend") \
          + section("Matched 对照明细", "这里只展示 matched 公平对照 run，所以时间可能不会覆盖当天所有新 run。", matched_table(), "Compare Table") \
          + section("最新自动扫描 Run", "这里看全量最新 run，不只 matched；当天新的 protocol-safe / 主线 smoke 也会出现在这里。", "<div class='runGrid'>" + "".join(run_card(r) for r in auto_runs[:8]) + "</div>", "Latest Runs") \
          + section("预测效果速览", "先看图像上像不像，再回头对照指标。这里自动抓最近 run 的预测图 / 轨迹图。", prediction_wall(4, 2), "Prediction Wall") \
          + section("近期研判快照", "无需返回原始 Markdown，即刻获悉方向变更与快速结论。", f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px;'><div class='sideBlock'><div class='title'>最近关键判断</div>{preview(ctx['dprev'], '一句话结论', '入口')}</div><div class='sideBlock'><div class='title'>最近实验快照</div>{preview(ctx['eprev'], '一句话结果', '入口')}</div></div>", "Snapshots")

dec_side = side_block("如何阅读本页", "<div class='muted' style='font-size:0.85rem'>优先看白话解释，随后查看触发原因。如果只关注“现在为什么这么做”，这比查阅历史流水账快得多。</div>")
dec_body = section("关键决策时间线", "支持按决策名称、白话说明、触发原因进行联合检索。", "<div class='tools'><label class='search'><input id='dsearch' type='search' placeholder='检索决策、白话解释或触发原因...'></label><div class='chips' id='dfilters'><button class='chipbtn active' data-filter='all'>全 部</button><button class='chipbtn' data-filter='active'>Active (生效中)</button></div></div>" + decision_feed(), "Decision Feed")

recent_runs = "".join(run_card(r) for r in auto_runs[:18])
exp_side = side_block("关于自动同步", "<div class='muted' style='font-size:0.85rem'>系统扫描本地输出目录的 run_summary.json / training_summary.json，并顺手抓取 figures 里的预测图。这免去了“抄数字 + 翻目录找图”的工作；配合人工登记表的“白话判定”，可达到最佳追溯效果。</div>")
exp_body = section("指标防迷失指南", "读表前，先对齐以下高频评价维度：", glossary_terms(['RMSE', 'selection_score', 'boundary_shift_abs_err', 'peak_time_abs_err_s']), "Metric Guide") \
         + section("公平对照 (Matched) 趋势", "用于衡量结构改动是真实改善了瓶颈，还是仅仅转移了问题区。", trend_section(), "Trend View") \
         + section("预测图速览墙", "专门用来横向看不同版本的预测效果。这里优先挑 pred_vs_gt / 轨迹对比图，而不是训练损失图。", prediction_wall(8, 2), "Prediction Wall") \
         + section("最新自动扫描 Run", "展示最近 18 组扫描结果（若有匹配的人工登记判定，会在卡片内加红高亮）。", "<div class='tools'><label class='search'><input id='asearch' type='search' placeholder='检索 run 名、命名拆解或解释...'></label><div class='chips' id='afilters'><button class='chipbtn active' data-filter='all'>全 部</button><button class='chipbtn' data-filter='rich'>Rich 数据</button><button class='chipbtn' data-filter='legacy'>Legacy 数据</button><button class='chipbtn' data-filter='matched'>Matched 对照</button><button class='chipbtn' data-filter='event-conditioned'>Event-Conditioned</button><button class='chipbtn' data-filter='protocol-safe'>Protocol Safe</button></div></div><div class='runGrid'>" + recent_runs + "</div>", "Auto Runs") \
         + section("完整索引表", "折叠保留所有扫描数据，供极限排查时定位。", run_index(), "Full Index") \
         + section("人工实验登记表", "补充自动图表缺失的：实验动机、命名拆解和最终判定。", "<div class='tools'><label class='search'><input id='esearch' type='search' placeholder='检索实验名、变动或结果...'></label><div class='chips' id='efilters'><button class='chipbtn active' data-filter='all'>全 部</button><button class='chipbtn' data-filter='proceed'>Proceed (通过)</button><button class='chipbtn' data-filter='watch'>Watch (需观察)</button><button class='chipbtn' data-filter='stop'>Stop (终止)</button><button class='chipbtn' data-filter='keep'>Keep (保留比对)</button></div></div>" + experiment_feed(), "Manual Registry")

day_side = side_block("如何阅读本页", "<div class='muted' style='font-size:0.85rem'>优先看每日 Highlights。如果你只需了解“今日为什么重要”，读这里；如果想确认“某条旧判断现在是否生效”，请移步至决策中心。</div>")
day_body = section("按日回溯推进轴", "按标签分类筛查，避免在 Markdown 中全文搜索的效率衰减。", "<div class='tools'><label class='search'><input id='tsearch' type='search' placeholder='检索日期、标题或核心记录...'></label><div class='chips' id='tfilters'><button class='chipbtn active' data-filter='all'>全 部</button><button class='chipbtn' data-filter='workflow'>Workflow</button><button class='chipbtn' data-filter='model'>Model</button><button class='chipbtn' data-filter='data'>Data</button><button class='chipbtn' data-filter='infra'>Infra</button><button class='chipbtn' data-filter='note'>Note</button></div></div>" + day_feed(), "Daily Timeline")

g_side = side_block("使用建议", "<div class='muted' style='font-size:0.85rem'>遇到未知的长后缀，先来这里定位原义，再回去看结论。如发现未录入的反复出现的强行话，建议及时补录 Markdown 字典。</div>")
g_body = section("术语分级速查表", "收拢英文命名规则、模型评估维度与测试协议词。", "<div class='tools'><label class='search'><input id='gsearch' type='search' placeholder='检索缩写、后缀、维度名...'></label><div class='chips' id='gfilters'><button class='chipbtn active' data-filter='all'>全 部</button><button class='chipbtn' data-filter='命名规则速查'>命名与后缀</button><button class='chipbtn' data-filter='模型与路线术语'>模型术语</button><button class='chipbtn' data-filter='指标与评估术语'>评价维度</button><button class='chipbtn' data-filter='协议与数据安全术语'>实验协议</button></div></div>" + glossary_sections(), "Glossary Core")

# Generate
OUT["dashboard"].write_text(shell("dashboard", "项目大盘工作台", "Project Overview", "梳理当前阻塞点与可用结论。先看泳道与判定卡片，后看指标趋势与基线偏移。", "<a class='btn primary' href='project_progress_experiments.html'>查看实验大盘 &rarr;</a><a class='btn' href='project_progress_decisions.html'>溯源决策项</a>", main_side, main_body), encoding="utf-8")
OUT["decisions"].write_text(shell("decisions", "关键决策池", "Decision Center", "收拢真正改变方向与边界的判断。提供白话翻译、触发缘由与影响面。", "<a class='btn primary' href='project_progress_dashboard.html'>&larr; 回到大盘</a><a class='btn' href='progress/decision_log.md'>查看源 Markdown</a>", dec_side, dec_body, "bind({s:'dsearch',f:'dfilters',q:'.article'});"), encoding="utf-8")
OUT["experiments"].write_text(shell("experiments", "实验控制中心", "Experiment Center", "人工评估逻辑与自动捕获的数据双线并轨，解决「如何改的」和「涨点多少」两大数据缝隙。", "<a class='btn primary' href='project_progress_dashboard.html'>&larr; 回到大盘</a><a class='btn' href='project_progress_glossary.html'>查阅字典</a>", exp_side, exp_body, "bind({s:'asearch',f:'afilters',q:'.runCard'});bind({s:'esearch',f:'efilters',q:'.article'});"), encoding="utf-8")
OUT["daily"].write_text(shell("daily", "工作流水日志", "Daily Timeline", "剥离无效噪音的逐日演进切片。利用标签组合快速还原旧版故障。", "<a class='btn primary' href='project_progress_dashboard.html'>&larr; 回到大盘</a><a class='btn' href='progress/daily/2026-04-14.md'>最新日志源文件</a>", day_side, day_body, "bind({s:'tsearch',f:'tfilters',q:'.entry'});"), encoding="utf-8")
OUT["glossary"].write_text(shell("glossary", "行话与术语库", "Glossary Map", "统一拉齐内部认知的释义表，降低缩写与后缀带来的阅读疲劳。", "<a class='btn primary' href='project_progress_dashboard.html'>&larr; 回到大盘</a><a class='btn' href='project_progress_experiments.html'>前往实验中心</a>", g_side, g_body, "bind({s:'gsearch',f:'gfilters',q:'.term'});"), encoding="utf-8")

print("Wrote progress site:")
for k in ["dashboard", "decisions", "experiments", "daily", "glossary"]:
    print(" - {}".format(OUT[k]))
