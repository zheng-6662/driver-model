from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pyzotero import zotero as zotero_api


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "startup" / "academic_zotero_config.json"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"

ORGANIZED_ROOT = "论文整理_驾驶员反应建模"


@dataclass(frozen=True)
class CategorySpec:
    path: tuple[str, ...]
    description: str


CATEGORY_SPECS: dict[str, CategorySpec] = {
    "core_review": CategorySpec(
        path=(ORGANIZED_ROOT, "01_核心主线", "综述与研究定位"),
        description="综述、问题定义、研究框架类文献",
    ),
    "core_prediction": CategorySpec(
        path=(ORGANIZED_ROOT, "01_核心主线", "驾驶员行为建模与短时预测"),
        description="驾驶员行为建模、反应预测、意图预测、操纵预测",
    ),
    "core_extreme": CategorySpec(
        path=(ORGANIZED_ROOT, "01_核心主线", "极限工况_车辆稳定性_风险评估"),
        description="极限驾驶工况、失稳、风险评估、车辆稳定性",
    ),
    "core_shared_control": CategorySpec(
        path=(ORGANIZED_ROOT, "01_核心主线", "共享控制_接管_人机协同"),
        description="共享控制、人机共驾、接管、情景意识、人因交互",
    ),
    "core_multimodal": CategorySpec(
        path=(ORGANIZED_ROOT, "01_核心主线", "多模态_生理_脑电"),
        description="生理、脑电、多模态驾驶行为分析",
    ),
    "method_state": CategorySpec(
        path=(ORGANIZED_ROOT, "02_方法借鉴", "驾驶状态_分心_情绪识别"),
        description="驾驶状态、分心、疲劳、情绪识别等方法借鉴",
    ),
    "method_domain": CategorySpec(
        path=(ORGANIZED_ROOT, "02_方法借鉴", "机器视觉_域泛化_迁移学习"),
        description="机器视觉、域泛化、域对齐、元学习、迁移学习等",
    ),
    "method_signal": CategorySpec(
        path=(ORGANIZED_ROOT, "02_方法借鉴", "多模态学习_信号处理方法"),
        description="信号处理、多模态方法、特征工程等",
    ),
    "pending_auto": CategorySpec(
        path=(ORGANIZED_ROOT, "03_待处理", "自动导入待判定"),
        description="自动导入但暂时不适合直接归主线的文献",
    ),
    "pending_manual": CategorySpec(
        path=(ORGANIZED_ROOT, "03_待处理", "待人工复核"),
        description="标题缺失、元数据异常或规则不够确定的条目",
    ),
    "pending_cleanup": CategorySpec(
        path=(ORGANIZED_ROOT, "03_待处理", "异常条目_待清理"),
        description="明显异常或无效条目，建议后续人工删除或修复",
    ),
    "side_robotics": CategorySpec(
        path=(ORGANIZED_ROOT, "90_旁支归档", "机器人与课程作业"),
        description="机器人/UAV/课程作业类文献",
    ),
    "side_manufacturing": CategorySpec(
        path=(ORGANIZED_ROOT, "90_旁支归档", "智能制造"),
        description="智能制造及自动化相关文献",
    ),
    "side_evtol": CategorySpec(
        path=(ORGANIZED_ROOT, "90_旁支归档", "电动飞行器与电子电气"),
        description="电动飞行器、电推进、电气架构等",
    ),
    "side_other": CategorySpec(
        path=(ORGANIZED_ROOT, "90_旁支归档", "其他非当前论文"),
        description="与当前论文主线关系较弱的其他主题",
    ),
}

FOCUS_COLLECTION_SPECS: dict[str, tuple[str, ...]] = {
    "focus_review": (ORGANIZED_ROOT, "04_论文写作用核心集", "A_研究定位与综述"),
    "focus_prediction": (ORGANIZED_ROOT, "04_论文写作用核心集", "B_驾驶员行为建模与短时预测"),
    "focus_extreme": (ORGANIZED_ROOT, "04_论文写作用核心集", "C_极限工况_风险_稳定性"),
    "focus_shared": (ORGANIZED_ROOT, "04_论文写作用核心集", "D_共享控制_接管_人机协同"),
    "focus_multimodal": (ORGANIZED_ROOT, "04_论文写作用核心集", "E_多模态_生理_脑电"),
    "focus_state": (ORGANIZED_ROOT, "04_论文写作用核心集", "F_方法借鉴_状态识别"),
    "focus_domain": (ORGANIZED_ROOT, "04_论文写作用核心集", "G_方法借鉴_域泛化"),
}

FOCUS_BUCKET_CONFIG: dict[str, dict[str, Any]] = {
    "focus_review": {"source_categories": {"core_review"}, "limit": 8},
    "focus_prediction": {"source_categories": {"core_prediction"}, "limit": 10},
    "focus_extreme": {"source_categories": {"core_extreme"}, "limit": 8},
    "focus_shared": {"source_categories": {"core_shared_control"}, "limit": 8},
    "focus_multimodal": {"source_categories": {"core_multimodal"}, "limit": 10},
    "focus_state": {"source_categories": {"method_state"}, "limit": 6},
    "focus_domain": {"source_categories": {"method_domain"}, "limit": 6},
}

MANUAL_ITEM_OVERRIDES: dict[str, str] = {
    "7YAEPLG6": "pending_cleanup",
    "FFMETG2C": "pending_cleanup",
    "RDTI5ECH": "pending_cleanup",
    "9JG5SID5": "side_other",
}


DIRECT_COLLECTION_HINTS: list[tuple[str, str, int]] = [
    ("极限工况驾驶员反应建模", "core_prediction", 5),
    ("短时预测_行为识别_共享控制", "core_prediction", 5),
    ("极限驾驶工况", "core_extreme", 4),
    ("车辆判稳相关文献", "core_extreme", 5),
    ("驾驶安全风险", "core_extreme", 5),
    ("人因文献", "core_shared_control", 4),
    ("脑电信号处理", "core_multimodal", 5),
    ("驾驶行为建模 / 综述", "core_review", 5),
    ("驾驶行为建模", "core_prediction", 3),
    ("驾驶员驾驶状态判别", "method_state", 5),
    ("驾驶员分心状态检测", "method_state", 5),
    ("机器视觉", "method_domain", 4),
    ("域泛化方法", "method_domain", 5),
    ("域对齐", "method_domain", 5),
    ("元学习方法", "method_domain", 5),
    ("数据增强", "method_domain", 5),
    ("自动导入", "pending_auto", 4),
    ("ScholarAIO", "pending_auto", 4),
    ("机器人技术结课论文参考文献", "side_robotics", 5),
    ("智能制造系统", "side_manufacturing", 6),
    ("电动飞行器电子电气系统架构", "side_evtol", 6),
]


KEYWORD_HINTS: list[tuple[str, str, int]] = [
    ("survey", "core_review", 3),
    ("review", "core_review", 3),
    ("overview", "core_review", 3),
    ("综述", "core_review", 3),
    ("state of the art", "core_review", 3),
    ("comprehensive review", "core_review", 4),
    ("future directions", "core_review", 4),
    ("developments", "core_review", 2),
    ("human driver", "core_review", 2),
    ("driver behavior model", "core_prediction", 4),
    ("driver behavior", "core_prediction", 3),
    ("driving behavior", "core_prediction", 3),
    ("driver model", "core_prediction", 4),
    ("intention recognition", "core_prediction", 3),
    ("lane change intention", "core_prediction", 4),
    ("motion prediction", "core_prediction", 4),
    ("trajectory prediction", "core_prediction", 4),
    ("intent prediction", "core_prediction", 4),
    ("reaction", "core_prediction", 3),
    ("response", "core_prediction", 3),
    ("steering", "core_prediction", 3),
    ("path tracking", "core_prediction", 2),
    ("path following", "core_prediction", 2),
    ("active steering", "core_prediction", 2),
    ("maneuver", "core_prediction", 2),
    ("behavior prediction", "core_prediction", 4),
    ("极限工况", "core_extreme", 5),
    ("critical driving", "core_extreme", 4),
    ("critical situation", "core_extreme", 4),
    ("loss of control", "core_extreme", 5),
    ("risk assessment", "core_extreme", 4),
    ("risk", "core_extreme", 2),
    ("stability", "core_extreme", 4),
    ("yaw", "core_extreme", 2),
    ("tire force", "core_extreme", 3),
    ("run-off-road", "core_extreme", 4),
    ("crash", "core_extreme", 3),
    ("shared control", "core_shared_control", 5),
    ("takeover", "core_shared_control", 4),
    ("human-vehicle collaboration", "core_shared_control", 5),
    ("human vehicle collaboration", "core_shared_control", 5),
    ("automated vehicles", "core_shared_control", 3),
    ("conditional automation", "core_shared_control", 4),
    ("trust in emerging adas", "core_shared_control", 4),
    ("anticipatory driving", "core_shared_control", 3),
    ("handbook of human factors", "core_shared_control", 4),
    ("situation awareness", "core_shared_control", 3),
    ("human factors", "core_shared_control", 3),
    ("人机共驾", "core_shared_control", 5),
    ("共享控制", "core_shared_control", 5),
    ("接管", "core_shared_control", 4),
    ("情景意识", "core_shared_control", 3),
    ("eeg", "core_multimodal", 5),
    ("ecg", "core_multimodal", 4),
    ("emg", "core_multimodal", 4),
    ("eda", "core_multimodal", 4),
    ("physio", "core_multimodal", 4),
    ("physiological", "core_multimodal", 4),
    ("biosignal", "core_multimodal", 4),
    ("multimodal", "core_multimodal", 3),
    ("脑电", "core_multimodal", 5),
    ("生理", "core_multimodal", 4),
    ("分心", "method_state", 5),
    ("distract", "method_state", 5),
    ("fatigue", "method_state", 4),
    ("drows", "method_state", 4),
    ("emotion", "method_state", 4),
    ("anger", "method_state", 3),
    ("driving state", "method_state", 4),
    ("state detection", "method_state", 4),
    ("驾驶状态", "method_state", 5),
    ("情绪", "method_state", 4),
    ("疲劳", "method_state", 4),
    ("机器视觉", "method_domain", 5),
    ("computer vision", "method_domain", 5),
    ("vision", "method_domain", 2),
    ("domain generalization", "method_domain", 6),
    ("domain adaptation", "method_domain", 6),
    ("domain alignment", "method_domain", 6),
    ("transfer learning", "method_domain", 5),
    ("meta-learning", "method_domain", 5),
    ("metalearning", "method_domain", 5),
    ("data augmentation", "method_domain", 5),
    ("域泛化", "method_domain", 6),
    ("域对齐", "method_domain", 6),
    ("元学习", "method_domain", 5),
    ("数据增强", "method_domain", 5),
    ("signal processing", "method_signal", 4),
    ("wavelet", "method_signal", 3),
    ("feature extraction", "method_signal", 3),
    ("fusion", "method_signal", 2),
    ("representation", "method_signal", 2),
    ("信号处理", "method_signal", 4),
    ("特征提取", "method_signal", 4),
    ("uav", "side_robotics", 6),
    ("robot", "side_robotics", 5),
    ("robotics", "side_robotics", 6),
    ("课程", "side_robotics", 2),
    ("manufacturing", "side_manufacturing", 6),
    ("智能制造", "side_manufacturing", 7),
    ("自动化设备", "side_manufacturing", 5),
    ("privileged and sensitive information", "side_other", 7),
    ("labour market", "side_other", 7),
    ("collaborative filtering", "side_other", 6),
    ("evtol", "side_evtol", 7),
    ("aircraft", "side_evtol", 5),
    ("air mobility", "side_evtol", 4),
    ("electric propulsion", "side_evtol", 5),
    ("电动飞行器", "side_evtol", 7),
    ("飞机", "side_evtol", 4),
    ("电推进", "side_evtol", 5),
]

PAPER_RELEVANCE_TERMS = [
    "driver",
    "driving",
    "vehicle",
    "automated vehicle",
    "autonomous vehicle",
    "adas",
    "steering",
    "trajectory",
    "car-following",
    "car following",
    "takeover",
    "shared control",
    "human-vehicle",
    "human vehicle",
    "eeg",
    "ecg",
    "emg",
    "eda",
    "physiological",
    "biosignal",
    "brain-computer",
    "brain computer",
    "road",
    "lane",
    "crash",
    "traffic",
    "risk",
    "stability",
    "workload",
    "fatigue",
    "emotion",
    "distract",
    "driver state",
    "autonomous vehicle",
    "automated vehicle",
    "驾驶员",
    "驾驶",
    "车辆",
    "方向",
    "共享控制",
    "接管",
    "脑电",
    "生理",
    "分心",
    "疲劳",
    "情绪",
    "风险",
    "稳定性",
]

METHOD_RELEVANCE_TERMS = [
    "domain generalization",
    "domain adaptation",
    "domain alignment",
    "transfer learning",
    "meta-learning",
    "metalearning",
    "data augmentation",
    "computer vision",
    "representation",
    "invariant feature",
    "feature-critic",
    "causal matching",
    "distributional shifts",
    "域泛化",
    "域对齐",
    "元学习",
    "数据增强",
    "机器视觉",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_zotero_client(config_path: Path) -> zotero_api.Zotero:
    cfg = load_config(config_path)
    zotero_cfg = cfg["zotero"]
    return zotero_api.Zotero(
        str(zotero_cfg["library_id"]),
        str(zotero_cfg["library_type"]),
        api_key=str(zotero_cfg["api_key"]),
    )


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip().lower()


def iter_collection_paths(
    collection_keys: list[str],
    collections_by_key: dict[str, dict[str, Any]],
) -> list[str]:
    paths: list[str] = []
    for key in collection_keys:
        parts: list[str] = []
        current = collections_by_key.get(key)
        seen: set[str] = set()
        while current and current["key"] not in seen:
            seen.add(current["key"])
            parts.append(current["data"]["name"])
            parent = current["data"].get("parentCollection")
            current = collections_by_key.get(parent) if parent else None
        paths.append(" / ".join(reversed(parts)))
    return paths


def collection_path_tuple(
    collection_key: str,
    collections_by_key: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    parts: list[str] = []
    current = collections_by_key.get(collection_key)
    seen: set[str] = set()
    while current and current["key"] not in seen:
        seen.add(current["key"])
        parts.append(current["data"]["name"])
        parent = current["data"].get("parentCollection")
        current = collections_by_key.get(parent) if parent else None
    return tuple(reversed(parts))


def score_item(item: dict[str, Any], collection_paths: list[str]) -> tuple[str, dict[str, int], str]:
    data = item["data"]
    item_key = item["key"]
    if item_key in MANUAL_ITEM_OVERRIDES:
        category = MANUAL_ITEM_OVERRIDES[item_key]
        scores = {key: 0 for key in CATEGORY_SPECS}
        scores[category] = 100
        return category, scores, f"manual_override={category}"

    title = data.get("title") or ""
    abstract = data.get("abstractNote") or ""
    publication = data.get("publicationTitle") or ""
    extra = data.get("extra") or ""
    item_type = data.get("itemType") or ""
    tags = " ".join(tag.get("tag", "") for tag in (data.get("tags") or []))
    path_blob = " || ".join(collection_paths)
    corpus = normalize_text(" ".join([title, abstract, publication, extra, tags, path_blob]))
    scores = {key: 0 for key in CATEGORY_SPECS}

    title_norm = normalize_text(title)
    if not title_norm or item_type in {"computerProgram", "attachment", "annotation"}:
        scores["pending_manual"] += 20

    for needle, category, weight in DIRECT_COLLECTION_HINTS:
        if normalize_text(needle) in normalize_text(path_blob):
            scores[category] += weight

    for needle, category, weight in KEYWORD_HINTS:
        if needle in corpus:
            scores[category] += weight

    paper_relevant = any(term in corpus for term in PAPER_RELEVANCE_TERMS)
    method_relevant = any(term in corpus for term in METHOD_RELEVANCE_TERMS)

    if "codex-academic-import" in corpus:
        scores["pending_auto"] += 1

    if "savedrecs" in normalize_text(path_blob) and max(scores.values()) < 5:
        scores["side_other"] += 4

    if scores["core_multimodal"] > 0 and scores["method_signal"] > 0:
        scores["core_multimodal"] += 1

    if scores["core_review"] > 0 and (
        scores["core_prediction"] > 0
        or scores["core_extreme"] > 0
        or scores["core_shared_control"] > 0
    ):
        scores["core_review"] += 1

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_category, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    best_non_side = next(
        ((category, score) for category, score in ranked if not category.startswith("side_")),
        ("pending_manual", 0),
    )

    if best_score <= 0:
        best_category = "pending_manual"
    elif best_score < 4 and best_category not in {"side_robotics", "side_manufacturing", "side_evtol"}:
        best_category = "pending_manual"
    elif not paper_relevant and not method_relevant and not best_category.startswith("side_"):
        best_category = "pending_manual"
    elif best_category.startswith("side_") and paper_relevant and best_non_side[1] >= best_score - 1:
        best_category = best_non_side[0]
    elif best_category.startswith("core_") and not paper_relevant and method_relevant:
        best_category = "method_domain"
    elif best_score - second_score <= 1 and best_category == "core_review" and "review" in corpus:
        best_category = "core_review"

    reason = f"best={best_category}:{scores[best_category]} second={second_score}"
    return best_category, scores, reason


def focus_bonus(bucket: str, item: dict[str, Any]) -> int:
    data = item["data"]
    text = normalize_text(
        " ".join(
            [
                data.get("title") or "",
                data.get("abstractNote") or "",
                data.get("publicationTitle") or "",
                data.get("extra") or "",
            ]
        )
    )
    bonus = 0
    bonus_terms: dict[str, list[str]] = {
        "focus_review": ["review", "survey", "overview", "future directions", "综述"],
        "focus_prediction": ["prediction", "driver model", "driver behavior", "trajectory", "steering", "intention"],
        "focus_extreme": ["risk", "stability", "critical", "emergency", "loss of control", "crash"],
        "focus_shared": ["shared control", "takeover", "collaboration", "anticipatory", "conditional automation", "trust"],
        "focus_multimodal": ["multimodal", "physiological", "eeg", "ecg", "eda", "workload", "emotion", "stress"],
        "focus_state": ["driver state", "distract", "fatigue", "stress", "workload", "emotion"],
        "focus_domain": ["domain generalization", "domain adaptation", "transfer learning", "meta-learning", "data augmentation"],
    }
    for term in bonus_terms.get(bucket, []):
        if term in text:
            bonus += 2
    return bonus


def build_focus_assignments(
    item_list: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    item_lookup = {item["key"]: item for item in item_list}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in assignments:
        for bucket, cfg in FOCUS_BUCKET_CONFIG.items():
            if row["category"] in cfg["source_categories"]:
                item = item_lookup[row["item_key"]]
                priority = row["scores"].get(row["category"], 0) + focus_bonus(bucket, item)
                enriched = dict(row)
                enriched["focus_priority"] = priority
                grouped[bucket].append(enriched)

    selected: dict[str, list[dict[str, Any]]] = {}
    for bucket, rows in grouped.items():
        rows = sorted(
            rows,
            key=lambda row: (
                -int(row["focus_priority"]),
                -(1 if row["item_type"] in {"journalArticle", "conferencePaper", "thesis", "book"} else 0),
                row["title"] or "",
            ),
        )
        selected[bucket] = rows[: int(FOCUS_BUCKET_CONFIG[bucket]["limit"])]
    return selected


def ensure_collection_path(
    zot: zotero_api.Zotero,
    collections_by_key: dict[str, dict[str, Any]],
    path: tuple[str, ...],
) -> str:
    path_to_key: dict[tuple[str, ...], str] = {}
    for collection in collections_by_key.values():
        parts: list[str] = []
        current = collection
        seen: set[str] = set()
        while current and current["key"] not in seen:
            seen.add(current["key"])
            parts.append(current["data"]["name"])
            parent = current["data"].get("parentCollection")
            current = collections_by_key.get(parent) if parent else None
        path_to_key[tuple(reversed(parts))] = collection["key"]

    current_parent: str | None = None
    current_path: list[str] = []
    for segment in path:
        current_path.append(segment)
        current_tuple = tuple(current_path)
        if current_tuple in path_to_key:
            current_parent = path_to_key[current_tuple]
            continue
        payload = {"name": segment}
        if current_parent:
            payload["parentCollection"] = current_parent
        response = zot.create_collections([payload])
        created_key = response["success"]["0"]
        created = zot.collection(created_key)
        collections_by_key[created_key] = created
        path_to_key[current_tuple] = created_key
        current_parent = created_key
    if current_parent is None:
        raise RuntimeError(f"Failed to create collection path: {' / '.join(path)}")
    return current_parent


def sync_item_membership(
    zot: zotero_api.Zotero,
    item: dict[str, Any],
    desired_managed_keys: set[str],
    managed_collection_keys: set[str],
) -> bool:
    current_collections = list(item["data"].get("collections") or [])
    unmanaged = [key for key in current_collections if key not in managed_collection_keys]
    desired = unmanaged + [key for key in current_collections if key in desired_managed_keys]
    for key in desired_managed_keys:
        if key not in desired:
            desired.append(key)
    if desired == current_collections:
        return False
    item["data"]["collections"] = desired
    zot.update_item(item)
    return True


def build_report_markdown(
    *,
    applied: bool,
    created_paths: list[str],
    category_counter: Counter[str],
    assignments: list[dict[str, Any]],
    focus_assignments: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Zotero 论文导向整理报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 模式：{'apply' if applied else 'dry-run'}")
    lines.append(f"- 输出文件：`{output_path}`")
    lines.append("")
    lines.append("## 分类结果概览")
    lines.append("")
    for category, count in sorted(category_counter.items(), key=lambda item: (-item[1], item[0])):
        spec = CATEGORY_SPECS[category]
        lines.append(f"- `{' / '.join(spec.path)}`：{count} 篇")
    lines.append("")
    if created_paths:
        lines.append("## 新建 Collection 路径")
        lines.append("")
        for path in created_paths:
            lines.append(f"- `{path}`")
        lines.append("")
    review_items = [row for row in assignments if row["category"] == "pending_manual"][:40]
    if review_items:
        lines.append("## 待人工复核样本")
        lines.append("")
        for row in review_items:
            lines.append(
                f"- `{row['item_key']}` | {row['title'] or '<无标题>'} | 当前：`{'; '.join(row['current_paths']) or '<未归类>'}`"
            )
        lines.append("")
    cleanup_items = [row for row in assignments if row["category"] == "pending_cleanup"][:40]
    if cleanup_items:
        lines.append("## 异常条目待清理")
        lines.append("")
        for row in cleanup_items:
            lines.append(
                f"- `{row['item_key']}` | {row['title'] or '<无标题>'} | 当前：`{'; '.join(row['current_paths']) or '<未归类>'}`"
            )
        lines.append("")
    lines.append("## 论文写作用核心集")
    lines.append("")
    for bucket, path in FOCUS_COLLECTION_SPECS.items():
        selected = focus_assignments.get(bucket, [])
        lines.append(f"### {' / '.join(path)}")
        if not selected:
            lines.append("- 无")
            lines.append("")
            continue
        for row in selected:
            lines.append(f"- {row['title'] or '<无标题>'}")
        lines.append("")
    lines.append("## 分类样本")
    lines.append("")
    for category in sorted(CATEGORY_SPECS):
        lines.append(f"### {' / '.join(CATEGORY_SPECS[category].path)}")
        samples = [row for row in assignments if row["category"] == category][:10]
        if not samples:
            lines.append("- 无")
            lines.append("")
            continue
        for row in samples:
            lines.append(f"- {row['title'] or '<无标题>'}")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按论文主线重新整理 Zotero 文献结构")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Zotero 配置 JSON 路径",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="报告输出目录",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正写回 Zotero；默认只做 dry-run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    zot = make_zotero_client(args.config)
    collection_list = zot.everything(zot.collections())
    item_list = zot.everything(zot.top(itemType="-attachment || note || annotation"))
    collections_by_key = {collection["key"]: collection for collection in collection_list}

    assignments: list[dict[str, Any]] = []
    category_counter: Counter[str] = Counter()
    required_paths: set[tuple[str, ...]] = set()

    for item in item_list:
        current_paths = iter_collection_paths(item["data"].get("collections") or [], collections_by_key)
        category, scores, reason = score_item(item, current_paths)
        spec = CATEGORY_SPECS[category]
        required_paths.add(spec.path)
        category_counter[category] += 1
        assignments.append(
            {
                "item_key": item["key"],
                "title": item["data"].get("title", ""),
                "item_type": item["data"].get("itemType", ""),
                "category": category,
                "target_path": " / ".join(spec.path),
                "current_paths": current_paths,
                "reason": reason,
                "scores": scores,
            }
        )

    focus_assignments = build_focus_assignments(item_list, assignments)
    focus_required_paths = set(FOCUS_COLLECTION_SPECS.values())
    created_paths: list[str] = []
    path_key_map: dict[tuple[str, ...], str] = {}

    if args.apply:
        for path in sorted(required_paths | focus_required_paths):
            collection_key_before = {
                tuple(iter_collection_paths([key], collections_by_key)[0].split(" / ")): key
                for key in collections_by_key
            }
            target_key = ensure_collection_path(zot, collections_by_key, path)
            path_key_map[path] = target_key
            if path not in collection_key_before:
                created_paths.append(" / ".join(path))

        managed_collection_keys = {
            key
            for key in collections_by_key
            if collection_path_tuple(key, collections_by_key)[:1] == (ORGANIZED_ROOT,)
        }
        item_lookup = {item["key"]: item for item in item_list}
        updates = 0
        item_to_focus_keys: dict[str, set[str]] = defaultdict(set)
        for bucket, rows in focus_assignments.items():
            focus_key = path_key_map[FOCUS_COLLECTION_SPECS[bucket]]
            for row in rows:
                item_to_focus_keys[row["item_key"]].add(focus_key)
        for row in assignments:
            target_key = path_key_map[tuple(row["target_path"].split(" / "))]
            desired_keys = {target_key} | item_to_focus_keys.get(row["item_key"], set())
            changed = sync_item_membership(
                zot,
                item_lookup[row["item_key"]],
                desired_keys,
                managed_collection_keys,
            )
            if changed:
                updates += 1
        print(
            json.dumps(
                {"event": "apply_complete", "updated_items": updates, "focus_updates": sum(len(v) for v in item_to_focus_keys.values())},
                ensure_ascii=False,
            )
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json_path = args.report_dir / f"zotero_thesis_organize_report_{timestamp}.json"
    report_md_path = args.report_dir / f"zotero_thesis_organize_report_{timestamp}.md"

    report_payload = {
        "generated_at": timestamp,
        "mode": "apply" if args.apply else "dry-run",
        "total_items": len(item_list),
        "categories": dict(category_counter),
        "created_paths": created_paths,
        "assignments": assignments,
        "focus_assignments": focus_assignments,
    }
    with report_json_path.open("w", encoding="utf-8") as handle:
        json.dump(report_payload, handle, ensure_ascii=False, indent=2)

    report_md = build_report_markdown(
        applied=args.apply,
        created_paths=created_paths,
        category_counter=category_counter,
        assignments=assignments,
        focus_assignments=focus_assignments,
        output_path=report_json_path,
    )
    with report_md_path.open("w", encoding="utf-8") as handle:
        handle.write(report_md)

    print(
        json.dumps(
            {
                "event": "report_ready",
                "mode": "apply" if args.apply else "dry-run",
                "total_items": len(item_list),
                "json_report": str(report_json_path),
                "md_report": str(report_md_path),
                "categories": dict(category_counter),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
