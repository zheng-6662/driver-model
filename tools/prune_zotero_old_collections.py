from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pyzotero import zotero as zotero_api


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "startup" / "academic_zotero_config.json"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"
MANAGED_ROOT_NAME = "论文整理_驾驶员反应建模"


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


def is_under_managed_root(
    collection_key: str,
    collections_by_key: dict[str, dict[str, Any]],
) -> bool:
    path = collection_path_tuple(collection_key, collections_by_key)
    return bool(path) and path[0] == MANAGED_ROOT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="删除 Zotero 中旧的历史 collections，只保留新的论文整理树")
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
        help="真正删除旧 collections；默认仅 dry-run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    zot = make_zotero_client(args.config)
    collections = zot.everything(zot.collections())
    items = zot.everything(zot.top(itemType="-attachment || note || annotation"))
    collections_by_key = {collection["key"]: collection for collection in collections}

    old_collections = [
        collection
        for collection in collections
        if not is_under_managed_root(collection["key"], collections_by_key)
    ]
    new_collection_keys = {
        collection["key"]
        for collection in collections
        if is_under_managed_root(collection["key"], collections_by_key)
    }
    old_collection_keys = {collection["key"] for collection in old_collections}

    missing_items: list[dict[str, Any]] = []
    for item in items:
        collection_keys = item["data"].get("collections") or []
        if not any(key in old_collection_keys for key in collection_keys):
            continue
        if not any(key in new_collection_keys for key in collection_keys):
            missing_items.append(
                {
                    "key": item["key"],
                    "title": item["data"].get("title", ""),
                    "old_paths": [
                        " / ".join(collection_path_tuple(key, collections_by_key))
                        for key in collection_keys
                        if key in old_collection_keys
                    ],
                }
            )

    report = {
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "mode": "apply" if args.apply else "dry-run",
        "managed_root": MANAGED_ROOT_NAME,
        "old_collection_count": len(old_collections),
        "old_collections": [
            {
                "key": collection["key"],
                "path": " / ".join(collection_path_tuple(collection["key"], collections_by_key)),
                "numItems": collection["meta"].get("numItems", 0),
                "numCollections": collection["meta"].get("numCollections", 0),
            }
            for collection in sorted(
                old_collections,
                key=lambda collection: (
                    -len(collection_path_tuple(collection["key"], collections_by_key)),
                    " / ".join(collection_path_tuple(collection["key"], collections_by_key)),
                ),
            )
        ],
        "items_missing_new_membership": missing_items,
    }

    if missing_items:
        raise RuntimeError(
            f"发现 {len(missing_items)} 条文献仍只存在于旧 collections 中，已停止删除。"
        )

    deleted = 0
    if args.apply:
        for collection in report["old_collections"]:
            zot.delete_collection({"key": collection["key"], "version": collections_by_key[collection["key"]]["version"]})
            deleted += 1
        report["deleted_collection_count"] = deleted

    stamp = report["generated_at"]
    json_path = args.report_dir / f"zotero_old_collection_prune_{stamp}.json"
    md_path = args.report_dir / f"zotero_old_collection_prune_{stamp}.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    lines = [
        "# Zotero 旧分类清理报告",
        "",
        f"- 生成时间：{stamp}",
        f"- 模式：{report['mode']}",
        f"- 旧 collections 数量：{report['old_collection_count']}",
        f"- 缺失新归属的条目数：{len(missing_items)}",
        "",
        "## 待删除旧 Collections",
        "",
    ]
    for collection in report["old_collections"]:
        lines.append(
            f"- `{collection['path']}` | items={collection['numItems']} | subcollections={collection['numCollections']}"
        )
    if args.apply:
        lines.extend(
            [
                "",
                "## 删除结果",
                "",
                f"- 已删除旧 collections：{deleted}",
            ]
        )
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(
        json.dumps(
            {
                "event": "prune_ready" if not args.apply else "prune_complete",
                "mode": report["mode"],
                "old_collection_count": report["old_collection_count"],
                "deleted_collection_count": deleted,
                "json_report": str(json_path),
                "md_report": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
