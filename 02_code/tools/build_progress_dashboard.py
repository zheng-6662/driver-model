# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
from collections import Counter
import html
import json
import os
import re

def resolve_layout(script_path: Path):
    for candidate in script_path.parents:
        new_reports = candidate / "04_project_logs" / "reports"
        if new_reports.exists():
            return candidate, new_reports
        old_reports = candidate / "reports"
        if old_reports.exists():
            return candidate, old_reports
    fallback_root = script_path.parents[1] if len(script_path.parents) > 1 else script_path.parent
    return fallback_root, fallback_root / "reports"

ROOT, REPORTS = resolve_layout(Path(__file__).resolve())
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

def build_scan_roots():
    candidates = [
        ROOT / "03_results" / "tmp",
        ROOT / "03_results" / "output",
        ROOT / "03_results" / "artifacts",
        ROOT / "03_results" / "trajectory_plot_output",
        ROOT / "03_results" / "多模态数据" / "程序运行结果",
        ROOT / "tmp",
        ROOT / "datasetprocess" / "多模态数据" / "程序运行结果",
    ]
    roots, seen = [], set()
    for path in candidates:
        key = str(path).lower()
        if key in seen or not path.exists():
            continue
        seen.add(key)
        roots.append(path)
    return roots

SCAN_ROOTS = build_scan_roots()

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

def find_any(n, *titles):
    for title in titles:
        for c in walk(n):
            if c["title"] == title:
                return c
    return {"title": titles[0] if titles else "", "lines": [], "children": []}

def list_items(lines, ordered=False):
    pat = r"^\s*\d+\.\s+(.*)$" if ordered else r"^\s*[-*]\s+(.*)$"
    return [m.group(1).strip() for line in lines for m in [re.match(pat, line)] if m]

def split_md_row(line):
    text = line.strip().strip("|")
    cells, cur, in_code = [], [], False
    for ch in text:
        if ch == "`":
            in_code = not in_code
            cur.append(ch)
            continue
        if ch == "|" and not in_code:
            cells.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    cells.append("".join(cur).strip())
    return cells

def table_rows(lines):
    block = []
    for line in lines:
        if "|" in line:
            block.append(line.rstrip())
        elif block:
            break
    if len(block) < 2:
        return [], []
    heads = split_md_row(block[0])
    rows = []
    for line in block[2:]:
        if not line.strip().startswith("|"):
            continue
        cells = split_md_row(line)
        cells += [""] * (len(heads) - len(cells))
        rows.append({h: cells[i] for i, h in enumerate(heads)})
    return heads, rows

def href_from_path(path):
    return Path(os.path.relpath(str(path.resolve()), str(REPORTS))).as_posix()

def rel(href, src):
    if not href or href.startswith("#"):
        return href
    if re.match(r"^[A-Za-z]:[\\/]", href):
        path = Path(href)
        if path.exists():
            try:
                return href_from_path(path)
            except Exception:
                return path.as_uri()
        return href.replace("\\", "/")
    if re.match(r"^[a-zA-Z]+:", href):
        return href
    return Path(os.path.relpath(str((src.parent / href).resolve()), str(REPORTS))).as_posix()

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
    lowered = {str(k).strip().lower(): v for k, v in row.items()}
    for n in names:
        if row.get(n):
            return row[n]
        value = lowered.get(str(n).strip().lower())
        if value:
            return value
    return ""

def tag(title):
    m = re.match(r"^\[([^\]]+)\]\s*(.*)$", title)
    return (m.group(1), m.group(2).strip()) if m else ("note", title)

def split_kv(text):
    s = re.sub(r"^\s*[-*]\s+", "", text).strip()
    if re.match(r"^`?[A-Za-z]:[\\/]", s) or re.match(r"^`?[A-Za-z]:/", s):
        return ("", s)
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
    if mode == "baseline": parts.append("baseline")
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
            "caption": path.name,
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

def breakdown(run): return run.get("run_prefix", "")

def plain(run): return run.get("run_prefix", "")

def short_label(run):
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

# 执行扫描与解析
H = tree(read(HUB)); D = tree(read(DECISIONS)); E = tree(read(EXPERIMENTS)); G = tree(read(GLOSSARY))
_, drows = table_rows(find_any(D, "决策表", "Decision Table")["lines"])
_, erows = table_rows(find_any(E, "实验表", "Experiment Table")["lines"])

ctx = {
    "status": [dict(zip(("label", "value"), split_kv(x))) | {"raw": x} for x in list_items(find_any(H, "当前状态", "Current State")["lines"])],
    "plain": list_items(find_any(H, "30 秒白话版", "30-second Plain Version", "30 秒白话")["lines"]),
    "prio": list_items(find_any(H, "当前优先级", "Current Priority")["lines"], True),
    "paths": list_items(find_any(H, "建议查阅路径", "Recommended Reading Path")["lines"]),
    "decisions": [{"date": pick(r,"日期", "Date"), "title": pick(r,"决策", "Decision"), "plain": pick(r,"白话解释","白话", "Plain-language meaning", "Plain language meaning"), "reason": pick(r,"触发原因", "Reason"), "impact": pick(r,"影响范围", "Impact"), "link": pick(r,"证据 / 入口", "证据/入口", "Link", "Details"), "state": pick(r,"状态", "State"), "filters": vfilter(pick(r,"状态", "State")), "search": " ".join(str(v) for v in r.values()).lower()} for r in drows],
    "experiments": [{"date": pick(r,"日期", "Date"), "name": pick(r,"实验 / 分析", "实验/分析", "Experiment / analysis", "Experiment / Analysis"), "naming": pick(r,"命名拆解", "Name decode"), "plain": pick(r,"白话解释","白话", "Plain-language meaning", "Plain language meaning"), "compare": pick(r,"可比性", "Comparability"), "change": pick(r,"变更", "Change"), "result": pick(r,"关键结果", "Key result"), "verdict": pick(r,"判定", "Decision"), "detail": pick(r,"详情", "Details"), "filters": " ".join(x for x in [vfilter(pick(r,"判定", "Decision")), pick(r,"可比性", "Comparability").strip().lower().replace(" ","-")] if x), "search": " ".join(str(v) for v in r.values()).lower()} for r in erows],
    "glossary": [], "days": [], "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
}

for sec in walk(G):
    if sec["level"] != 2: continue
    heads, rows = table_rows(sec["lines"])
    ctx["glossary"].append({"title": sec["title"], "heads": heads, "rows": rows, "bullets": list_items(sec["lines"]) + list_items(sec["lines"], True)})

for p in sorted(DAILY.glob("*.md"), reverse=True):
    T = tree(read(p))
    hi = list_items(find_any(T, "今日重点", "Today Focus", "Current Takeaway")["lines"])
    entries = []
    source_nodes = [n for n in walk(T) if n["level"] == 2 and n["title"].startswith("[")] or [n for n in walk(T) if n["level"] == 3]
    for n in source_nodes:
        tg, tt = tag(n["title"]); lines = [x for x in n["lines"] if x.strip()]
        if lines: entries.append({"tag": tg, "title": tt, "lines": lines, "src": p})
    ctx["days"].append({"date": p.stem, "path": p.relative_to(REPORTS).as_posix(), "high": hi, "entries": entries, "src": p})

auto_runs = discover_runs()
for run in auto_runs:
    rnorm = re.sub(r"[^a-z0-9]+", "", (run["run_name"] + run["run_prefix"]).lower())
    for row in ctx["experiments"]:
        t = re.sub(r"[^a-z0-9]+", "", row["name"].lower())
        if t and t in rnorm: run["manual_hint"] = row["plain"] or row["result"]; break

stats = {
    "latest": ctx["days"][0]["date"] if ctx["days"] else "n/a", "days": len(ctx["days"]),
    "decisions": sum(1 for x in ctx["decisions"] if x["state"].lower() == "active"),
    "experiments": len(ctx["experiments"]), "auto": len(auto_runs), "rich": sum(1 for x in auto_runs if x["schema"] == "rich"),
}

# === 视觉 HTML 组装逻辑 ===

def nav(active):
    items = [("dashboard", "总览", "project_progress_dashboard.html"), ("decisions", "决策", "project_progress_decisions.html"), ("experiments", "实验", "project_progress_experiments.html"), ("daily", "日志", "project_progress_daily.html"), ("glossary", "词典", "project_progress_glossary.html")]
    return "".join(f'<a href="{h}" class="{"active" if k == active else ""}">{t}</a>' for k, t, h in items)

def badge(text, cls="tag"): return f'<span class="{cls}">{html.escape(text)}</span>'

def section(title, sub, body, eyebrow=""):
    return f"<section class='panel'><div class='head'><h2>{html.escape(title)}</h2><p class='sub'>{html.escape(sub)}</p></div>{body}</section>"

def shell(active, title, kicker, lead, actions, side, body, script=""):
    return f"""<!DOCTYPE html><html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="project_progress_site.css">
</head>
<body>
<div class="top"><div class="topin"><div class="brand">🚀 <strong>Progress</strong><span>项目控制台</span></div><nav class="nav">{nav(active)}</nav></div></div>
<div class="layoutWrap">
    <main class="layoutMain">
        <header class="pageHeader">
            <h1 style="color:var(--ink);">{html.escape(title)}</h1>
            <p class="lead">{html.escape(lead)}</p>
        </header>
        {body}
    </main>
    <aside class="layoutSide">{side}</aside>
</div>
<script>{SCRIPT_BASE}{script}</script>
</body></html>"""

def side_block(title, content): return f"<div class='sideBlock'><div class='title'>{html.escape(title)}</div>{content}</div>"

def build_status_grid():
    html_str = "<div class='statusGrid'>"
    for item in ctx["status"]:
        val = item["value"] or item["raw"]
        is_danger = any(k in val for k in NEG_HINTS) or item["label"] in ("当前可信判断", "当前模型问题")
        color = "red" if is_danger else "green"
        danger_class = " danger-bg" if is_danger else ""
        html_str += f"<div class='item{danger_class}'><div class='title'><span class='status-dot {color}'></span> {html.escape(item['label'] or '状态')}</div><p>{inline(val, HUB)}</p></div>"
    return html_str + "</div>"

def compact_run_list(runs, limit=8):
    html_str = "<div class='runGrid'>"
    for r in runs[:limit]:
        html_str += f"""
        <a href="{html.escape(r['dir_href'], quote=True)}" class='runCard' data-filter='{html.escape(r['filters'], quote=True)}' data-search='{html.escape(r['search'], quote=True)}'>
            <div class='runCard-info'>
                {badge(r['schema'], 'tag accent')}
                {badge(r['family_label'], 'tag')}
                <h3>{html.escape(r['nickname'])}</h3>
            </div>
            <div class='metricMini'>
                <div><span>Steer RMSE</span><strong>{fmt(r['steer_rmse'])}</strong></div>
                <div><span>Score</span><strong>{fmt(r['selection_score'])}</strong></div>
            </div>
        </a>"""
    return html_str + "</div>"

def visual_prediction_wall(limit_runs=6, per_run=2):
    runs = [r for r in auto_runs if r.get("preview_images")]
    if not runs: return "<div style='color:var(--muted)'>暂无自动提取的预测图。</div>"
    cards = []
    for run in runs[:limit_runs]:
        hero = run["preview_images"][0]
        thumbs = run["preview_images"][1:per_run+1]
        thumb_html = "".join(f"<a class='wallThumb' href='{html.escape(img['href'], quote=True)}'><img loading='lazy' src='{html.escape(img['href'], quote=True)}' alt='thumb'></a>" for img in thumbs)
        thumb_row = f"<div class='wallThumbRow'>{thumb_html}</div>" if thumb_html else ""
        cards.append(f"""
        <article class='previewRun'>
            <a class='previewPoster' href='{html.escape(hero['href'], quote=True)}'>
                <img loading='lazy' src='{html.escape(hero['href'], quote=True)}'>
                <div class='previewPosterCaption'>{html.escape(hero['caption'])}</div>
            </a>
            {thumb_row}
            <div class='previewBody'>
                <div class='previewRunMeta'>{badge(run['timestamp_text'], 'pill')} {badge(run['family_label'], 'tag')}</div>
                <h3>{html.escape(run['nickname'])}</h3>
            </div>
        </article>""")
    return "<div class='previewWall'>" + "".join(cards) + "</div>"

def chart_svg(points, color, higher=False):
    vals = [p["value"] for p in points if p["value"] is not None]
    if len(vals) < 2: return "<svg class='chartSvg' viewBox='0 0 720 240'><text x='24' y='40'>数据点不足</text></svg>"
    w, h, pl, pr, pt, pb = 720, 240, 60, 40, 20, 40
    low, high = min(vals), max(vals)
    if abs(high - low) < 1e-9: low -= .5; high += .5
    pts = [(pl + i * ((w - pl - pr) / max(len(points) - 1, 1)), pt + (h - pt - pb) - ((p["value"] - low) / (high - low)) * (h - pt - pb), p) for i, p in enumerate(points) if p["value"] is not None]
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y, _ in pts)
    grid = "".join(f"<line x1='{pl}' y1='{pt + (h - pt - pb) - i / 3 * (h - pt - pb):.1f}' x2='{w-pr}' y2='{pt + (h - pt - pb) - i / 3 * (h - pt - pb):.1f}' stroke-dasharray='4 4'/>" for i in range(4))
    marks = "".join(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}'/>" for x, y, p in pts)
    labels = "".join(f"<text x='{x:.1f}' y='{h-12}' text-anchor='middle' font-size='11'>{html.escape(p['label'] if len(p['label'])<12 else p['label'][:10]+'..')}</text>" for x, y, p in pts)
    return f"<svg class='chartSvg' viewBox='0 0 {w} {h}'>{grid}<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/>{marks}{labels}</svg>"

def trend_section():
    runs = [x for x in sorted(auto_runs, key=lambda y: y["timestamp"]) if x["family_key"] == "matched" and x["schema"] == "rich"]
    mk = lambda key: [{"label": r["short_label"], "value": r.get(key)} for r in runs]
    def ccard(title, sub, p, c): return f"<div class='chartCard'><h3>{title}</h3><p style='color:var(--muted);font-size:0.8rem;margin-bottom:12px;'>{sub}</p>{chart_svg(p, c)}</div>"
    return f"<div class='chartGrid'>{ccard('steer_rmse', '整体转向误差', mk('steer_rmse'), '#8b5cf6')}{ccard('selection_score', '项目综合分', mk('selection_score'), '#f59e0b')}</div>"

active_pct = int((stats['decisions'] / max(len(ctx["decisions"]), 1)) * 100)
main_side = side_block("研发态势指标 (基于真实数据)", f"""
<div style='margin-bottom:12px; font-size:0.85rem; color:var(--muted);'>最近日志: {stats['latest']}</div>
<div style='margin-bottom:16px;'>
    <div style='display:flex; justify-content:space-between; font-size:0.85rem;'><span>活跃决策比</span> <strong>{stats['decisions']}/{len(ctx['decisions'])}</strong></div>
    <div class='stat-bar-container'><div class='stat-bar-fill' style='width:{active_pct}%'></div></div>
</div>
<div style='margin-bottom:16px;'>
    <div style='display:flex; justify-content:space-between; font-size:0.85rem;'><span>Rich 摘要覆盖率</span> <strong>{stats['rich']}/{stats['auto']}</strong></div>
    <div class='stat-bar-container'><div class='stat-bar-fill' style='width:{int((stats['rich']/max(stats['auto'],1))*100)}%; background:var(--ok);'></div></div>
</div>
""") + side_block("建议查阅路径", "".join(f"<li style='font-size:0.85rem; color:var(--muted); margin-bottom:4px;'>{inline(x, HUB)}</li>" for x in ctx['paths']))

main_body = section("当前状态与判定", "由 project_progress_hub 解析，状态灯与高危底色自动映射。", build_status_grid()) \
          + section("最新大图预览墙", "从实验目录中自动提取最新的 figures 图表比对。", visual_prediction_wall(4, 2)) \
          + section("近期实验指标走势 (Matched)", "基于 Rich Summary 自动生成图表。", trend_section()) \
          + section("最新扫描 Run (高密度视图)", "仅罗列核心指标，点击进入实验目录。", compact_run_list(auto_runs, 10))

def decision_feed(): return "".join(f"<article class='article'><h3>{inline(x['title'], DECISIONS)}</h3><p style='color:var(--muted); font-size:0.9rem;'>{inline(x['plain'] or x['impact'], DECISIONS)}</p><div style='margin-top:8px;'>{badge(x['date'], 'pill')} {badge(x['state'], 'tag')}</div></article>" for x in ctx["decisions"])
dec_body = section("决策时间线", "支持文本检索", "<div class='tools'><div class='search'><input id='dsearch' type='search' placeholder='检索...'></div></div>" + decision_feed())

def experiment_feed(): return "".join(f"<article class='article'><h3>{inline(x['name'], EXPERIMENTS)}</h3><p style='color:var(--muted); font-size:0.9rem;'>{inline(x['plain'] or x['result'], EXPERIMENTS)}</p><div style='margin-top:8px;'>{badge(x['date'], 'pill')} {badge(x['verdict'], 'tag')}</div></article>" for x in ctx["experiments"])
exp_body = section("自动扫描完整大图墙", "扫描所有匹配的图像结果。", visual_prediction_wall(12, 2)) \
         + section("全量自动索引", "紧凑列表，收纳所有扫描结果。", compact_run_list(auto_runs, 50)) \
         + section("人工实验库", "登记表解析", experiment_feed())

def day_feed(): return "<div class='timeline'>" + "".join(f"<section class='day'><div style='margin-bottom:12px;'>{badge(d['date'], 'pill')}</div>" + "".join(f"<details class='entry'><summary>{badge(e['tag'], 'tag')} {inline(e['title'], d['src'])}</summary><div class='entryBody' style='color:var(--muted);'>{'<br>'.join(inline(x, d['src']) for x in e['lines'])}</div></details>" for e in d["entries"]) + "</section>" for d in ctx["days"]) + "</div>"
day_body = section("逐日推进回溯", "基于 md 解析", day_feed())

def glossary_feed(): return "".join(f"<div style='margin-bottom:20px;'><h3 style='color:var(--accent); margin:0 0 8px;'>{sec['title']}</h3><div style='display:flex; flex-wrap:wrap; gap:8px;'>{''.join(badge(row.get(sec['heads'][0],''), 'pill') for row in sec['rows'] if sec['rows'])}</div></div>" for sec in ctx["glossary"])
g_body = section("全量术语字典", "自动读取并构建标签库", glossary_feed())

OUT["dashboard"].write_text(shell("dashboard", "态势大盘", "Cockpit", "基于深色工作站视图强化核心数据。", "", main_side, main_body), encoding="utf-8")
OUT["decisions"].write_text(shell("decisions", "核心决策", "Decisions", "记录关键路口。", "", side_block("提示", "从 md 抽取"), dec_body), encoding="utf-8")
OUT["experiments"].write_text(shell("experiments", "实验控制中心", "Experiments", "自动图表与人工结合。", "", side_block("说明", "支持大图预览"), exp_body), encoding="utf-8")
OUT["daily"].write_text(shell("daily", "逐日日志", "Daily", "日更归档。", "", side_block("过滤", "支持检索"), day_body), encoding="utf-8")
OUT["glossary"].write_text(shell("glossary", "行话与术语库", "Glossary", "词典索引。", "", side_block("来源", "glossary.md"), g_body), encoding="utf-8")

print("Wrote visual site:")
for k in ["dashboard", "decisions", "experiments", "daily", "glossary"]: print(" - {}".format(OUT[k]))