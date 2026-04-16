from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import requests
from pyzotero import zotero as zotero_api


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "startup" / "academic_zotero_config.json"
DEFAULT_DOWNLOAD_ROOT = REPO_ROOT / "tmp" / "literature_downloads"
DEFAULT_TRANSLATION_SERVER_URL = "http://127.0.0.1:1969"
CROSSREF_WORKS_URL = "https://api.crossref.org/works/{doi}"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"
USER_AGENT = "CodexAcademicIngest/0.1 (+local workflow)"
PDF_SIGNATURE = b"%PDF"
MIN_PDF_BYTES = 10 * 1024
MAX_PDF_BYTES = 100 * 1024 * 1024
WEAK_TITLE_SIMILARITY = 0.92


class AcademicImportError(RuntimeError):
    pass


@dataclass
class AppConfig:
    zotero_library_type: str | None = None
    zotero_library_id: str | None = None
    zotero_api_key: str | None = None
    default_collection_path: str | None = None
    translation_server_url: str | None = None
    unpaywall_email: str | None = None
    openalex_mailto: str | None = None
    download_root: Path = DEFAULT_DOWNLOAD_ROOT


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(config_path: str | None) -> AppConfig:
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        raw = read_json(path)

    zotero_block = raw.get("zotero", {})
    defaults_block = raw.get("defaults", {})
    services_block = raw.get("services", {})

    def env(name: str, fallback: Any = None) -> Any:
        return os.environ.get(name, fallback)

    download_root = (
        env("ACADEMIC_DOWNLOAD_ROOT")
        or defaults_block.get("download_root")
        or str(DEFAULT_DOWNLOAD_ROOT)
    )

    return AppConfig(
        zotero_library_type=env(
            "ZOTERO_LIBRARY_TYPE", zotero_block.get("library_type")
        ),
        zotero_library_id=env("ZOTERO_LIBRARY_ID", zotero_block.get("library_id")),
        zotero_api_key=env("ZOTERO_API_KEY", zotero_block.get("api_key")),
        default_collection_path=env(
            "ZOTERO_COLLECTION_PATH", defaults_block.get("collection_path")
        ),
        translation_server_url=env(
            "TRANSLATION_SERVER_URL",
            services_block.get("translation_server_url", DEFAULT_TRANSLATION_SERVER_URL),
        ),
        unpaywall_email=env("UNPAYWALL_EMAIL", services_block.get("unpaywall_email")),
        openalex_mailto=env("OPENALEX_MAILTO", services_block.get("openalex_mailto")),
        download_root=Path(download_root),
    )


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def require_zotero_config(config: AppConfig) -> None:
    missing: list[str] = []
    if not config.zotero_library_type:
        missing.append("zotero.library_type / ZOTERO_LIBRARY_TYPE")
    if not config.zotero_library_id:
        missing.append("zotero.library_id / ZOTERO_LIBRARY_ID")
    if not config.zotero_api_key:
        missing.append("zotero.api_key / ZOTERO_API_KEY")
    if missing:
        raise AcademicImportError(
            "缺少 Zotero 配置: " + ", ".join(missing)
        )


def make_zotero_client(config: AppConfig) -> zotero_api.Zotero:
    require_zotero_config(config)
    return zotero_api.Zotero(
        str(config.zotero_library_id),
        str(config.zotero_library_type),
        api_key=str(config.zotero_api_key),
    )


def json_print(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def log_event(event: str, *, level: str = "INFO", **fields: Any) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
    }
    payload.update({key: value for key, value in fields.items() if value is not None})
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip()
    cleaned = cleaned.replace("https://doi.org/", "")
    cleaned = cleaned.replace("http://doi.org/", "")
    cleaned = cleaned.replace("doi:", "")
    return cleaned.strip().lower() or None


def normalize_title(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip().lower()
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "", cleaned)
    return cleaned or None


def normalize_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlparse(value.strip())
    if not parsed.scheme or not parsed.netloc:
        return value.strip().rstrip("/") or None
    path = parsed.path.rstrip("/")
    normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"
    if parsed.query:
        normalized = f"{normalized}?{parsed.query}"
    return normalized or None


def extract_year(value: str | int | None) -> str | None:
    if value is None:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    if match:
        return match.group(0)
    return None


def normalize_person_name(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip().lower()
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "", cleaned)
    return cleaned or None


def title_similarity(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, normalize_title(left) or "", normalize_title(right) or "").ratio()


def unique_nonempty(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def clean_collection_segments(path_spec: str | None) -> list[str]:
    if not path_spec:
        return []
    return [segment.strip() for segment in re.split(r"[\\/]+", path_spec) if segment.strip()]


def openalex_params(config: AppConfig, **extra: Any) -> dict[str, Any]:
    params = {key: value for key, value in extra.items() if value not in (None, "", [])}
    if config.openalex_mailto:
        params["mailto"] = config.openalex_mailto
    return params


def get_json(session: requests.Session, url: str, **kwargs: Any) -> Any:
    response = session.get(url, timeout=30, **kwargs)
    response.raise_for_status()
    return response.json()


def post_json(
    session: requests.Session,
    url: str,
    *,
    data: str | bytes | None = None,
    json_body: Any = None,
    content_type: str = "application/json",
) -> requests.Response:
    headers = {"Content-Type": content_type}
    return session.post(url, data=data, json=json_body, headers=headers, timeout=60)


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    if not inverted_index:
        return None
    words: list[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        for position in positions:
            words.append((position, token))
    if not words:
        return None
    ordered = [token for _, token in sorted(words, key=lambda item: item[0])]
    text = " ".join(ordered)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = text.replace(" )", ")").replace("( ", "(")
    return text.strip() or None


def strip_markup(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"<[^>]+>", " ", unescape(value))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def nested_get(data: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def pick_openalex_source(work: dict[str, Any]) -> str | None:
    source = (
        nested_get(work, "primary_location", "source", "display_name")
        or nested_get(work, "best_oa_location", "source", "display_name")
        or nested_get(work, "primary_location", "source", "host_organization_name")
    )
    if source:
        return str(source)
    return None


def first_list_value(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def crossref_date_to_string(message: dict[str, Any]) -> str | None:
    for field in ("published-print", "published-online", "issued", "created"):
        parts = nested_get(message, field, "date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list):
            values = [str(part) for part in parts[0] if part is not None]
            if values:
                return "-".join(values)
    return None


def pick_crossref_source(message: dict[str, Any]) -> str | None:
    return first_list_value(message.get("container-title")) or first_list_value(
        message.get("short-container-title")
    )


def collect_crossref_pdf_candidates(message: dict[str, Any]) -> list[str]:
    candidates = [
        nested_get(message, "resource", "primary", "URL"),
        message.get("URL"),
    ]
    for link in message.get("link") or []:
        if not isinstance(link, dict):
            continue
        content_type = (link.get("content-type") or link.get("content_type") or "").lower()
        intended = (link.get("intended-application") or "").lower()
        if "pdf" in content_type or "text-mining" in intended or "similarity-checking" in intended:
            candidates.append(link.get("URL"))
    return unique_nonempty(candidates)


def collect_openalex_pdf_candidates(work: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    open_access = work.get("open_access") or {}
    best_oa_location = work.get("best_oa_location") or {}
    primary_location = work.get("primary_location") or {}

    add(open_access.get("oa_url"))
    add(best_oa_location.get("pdf_url"))
    add(best_oa_location.get("landing_page_url"))
    add(primary_location.get("pdf_url"))
    add(primary_location.get("landing_page_url"))

    for location in work.get("locations") or []:
        if not isinstance(location, dict):
            continue
        add(location.get("pdf_url"))
        if location.get("is_oa"):
            add(location.get("landing_page_url"))

    doi = normalize_doi(nested_get(work, "ids", "doi"))
    if doi:
        add(f"https://doi.org/{doi}")

    return unique_nonempty(candidates)


def summarize_openalex_work(work: dict[str, Any]) -> dict[str, Any]:
    doi = normalize_doi(nested_get(work, "ids", "doi"))
    source = pick_openalex_source(work)
    authors = []
    for authorship in work.get("authorships") or []:
        name = nested_get(authorship, "author", "display_name")
        if name:
            authors.append(name)

    return {
        "title": work.get("display_name"),
        "year": work.get("publication_year"),
        "doi": doi,
        "type": work.get("type"),
        "source": source,
        "authors": authors,
        "is_oa": nested_get(work, "open_access", "is_oa"),
        "oa_status": nested_get(work, "open_access", "oa_status"),
        "landing_page_url": nested_get(work, "primary_location", "landing_page_url"),
        "pdf_candidates": collect_openalex_pdf_candidates(work),
        "openalex_id": work.get("id"),
        "cited_by_count": work.get("cited_by_count"),
    }


def search_openalex(
    session: requests.Session,
    config: AppConfig,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    params = openalex_params(config, search=query, **{"per-page": limit})
    payload = get_json(session, OPENALEX_WORKS_URL, params=params)
    results = payload.get("results") or []
    return [summarize_openalex_work(work) for work in results]


def fetch_openalex_by_doi(
    session: requests.Session,
    config: AppConfig,
    doi: str,
) -> dict[str, Any] | None:
    params = openalex_params(config, filter=f"doi:{normalize_doi(doi)}", **{"per-page": 1})
    payload = get_json(session, OPENALEX_WORKS_URL, params=params)
    results = payload.get("results") or []
    return results[0] if results else None


def fetch_crossref_by_doi(
    session: requests.Session,
    doi: str,
) -> dict[str, Any] | None:
    payload = get_json(session, CROSSREF_WORKS_URL.format(doi=doi))
    message = payload.get("message") or {}
    return message if isinstance(message, dict) and message else None


def fetch_unpaywall_record(
    session: requests.Session,
    config: AppConfig,
    doi: str | None,
) -> dict[str, Any] | None:
    if not doi or not config.unpaywall_email:
        return None
    url = UNPAYWALL_URL.format(doi=doi)
    payload = get_json(
        session,
        url,
        params={"email": config.unpaywall_email},
    )
    return payload if isinstance(payload, dict) else None


def item_type_for_openalex(work_type: str | None) -> str:
    mapping = {
        "article": "journalArticle",
        "journal-article": "journalArticle",
        "proceedings-article": "conferencePaper",
        "book-chapter": "bookSection",
        "book": "book",
        "dataset": "report",
        "dissertation": "thesis",
        "preprint": "preprint",
        "report": "report",
    }
    if work_type in mapping:
        return mapping[work_type]
    return "journalArticle"


def item_type_for_crossref(work_type: str | None) -> str:
    return item_type_for_openalex(work_type)


def append_extra_line(item: dict[str, Any], line: str) -> None:
    extra = item.get("extra") or ""
    lines = [entry.strip() for entry in str(extra).splitlines() if entry.strip()]
    if line not in lines:
        lines.append(line)
    if lines:
        item["extra"] = "\n".join(lines)


def first_creator_name(creators: list[dict[str, Any]] | None) -> str | None:
    if not creators:
        return None
    for creator in creators:
        if not isinstance(creator, dict):
            continue
        if creator.get("creatorType") not in (None, "", "author"):
            continue
        literal = creator.get("name")
        if isinstance(literal, str) and literal.strip():
            return literal.strip()
        family = creator.get("lastName")
        given = creator.get("firstName")
        if family and given:
            return f"{given} {family}".strip()
        if family:
            return str(family).strip()
        if given:
            return str(given).strip()
    return None


def make_openalex_zotero_item(
    zot: zotero_api.Zotero,
    work: dict[str, Any],
    collection_key: str | None,
) -> dict[str, Any]:
    item_type = item_type_for_openalex(work.get("type"))
    item = zot.item_template(item_type)
    item["title"] = work.get("display_name") or ""

    authors = []
    for authorship in work.get("authorships") or []:
        name = nested_get(authorship, "author", "display_name")
        if name:
            authors.append({"creatorType": "author", "name": name})
    if authors:
        item["creators"] = authors

    source = pick_openalex_source(work)
    if item_type in {"journalArticle", "conferencePaper", "preprint"} and source:
        if "publicationTitle" in item:
            item["publicationTitle"] = source
        elif "proceedingsTitle" in item:
            item["proceedingsTitle"] = source

    doi = normalize_doi(nested_get(work, "ids", "doi"))
    if doi and "DOI" in item:
        item["DOI"] = doi

    landing_page = (
        nested_get(work, "primary_location", "landing_page_url")
        or nested_get(work, "best_oa_location", "landing_page_url")
        or (f"https://doi.org/{doi}" if doi else "")
    )
    if landing_page and "url" in item:
        item["url"] = landing_page

    publication_date = work.get("publication_date") or work.get("publication_year")
    if publication_date and "date" in item:
        item["date"] = str(publication_date)

    if "abstractNote" in item:
        item["abstractNote"] = reconstruct_abstract(work.get("abstract_inverted_index")) or ""

    biblio = work.get("biblio") or {}
    if biblio.get("volume") and "volume" in item:
        item["volume"] = str(biblio["volume"])
    if biblio.get("issue") and "issue" in item:
        item["issue"] = str(biblio["issue"])

    first_page = biblio.get("first_page")
    last_page = biblio.get("last_page")
    if first_page and last_page and "pages" in item:
        item["pages"] = f"{first_page}-{last_page}"
    elif first_page and "pages" in item:
        item["pages"] = str(first_page)

    language = work.get("language")
    if language and "language" in item:
        item["language"] = language

    extra_lines = []
    if work.get("id"):
        extra_lines.append(f"OpenAlex: {work['id']}")
    if work.get("cited_by_count") is not None:
        extra_lines.append(f"Cited by: {work['cited_by_count']}")
    oa_status = nested_get(work, "open_access", "oa_status")
    if oa_status:
        extra_lines.append(f"OA status: {oa_status}")
    if extra_lines and "extra" in item:
        item["extra"] = "\n".join(extra_lines)
        append_extra_line(item, "Metadata source: OpenAlex")

    item["tags"] = item.get("tags") or []
    item["tags"].append({"tag": "codex-academic-import"})

    if collection_key:
        item["collections"] = [collection_key]

    return item


def make_crossref_zotero_item(
    zot: zotero_api.Zotero,
    message: dict[str, Any],
    collection_key: str | None,
) -> dict[str, Any]:
    item_type = item_type_for_crossref(message.get("type"))
    item = zot.item_template(item_type)
    item["title"] = first_list_value(message.get("title")) or ""

    creators = []
    for author in message.get("author") or []:
        if not isinstance(author, dict):
            continue
        family = (author.get("family") or "").strip()
        given = (author.get("given") or "").strip()
        literal = (author.get("name") or "").strip()
        if family or given:
            creators.append(
                {
                    "creatorType": "author",
                    "firstName": given,
                    "lastName": family or literal,
                }
            )
        elif literal:
            creators.append({"creatorType": "author", "name": literal})
    if creators:
        item["creators"] = creators

    source = pick_crossref_source(message)
    if item_type in {"journalArticle", "conferencePaper", "preprint"} and source:
        if "publicationTitle" in item:
            item["publicationTitle"] = source
        elif "proceedingsTitle" in item:
            item["proceedingsTitle"] = source

    doi = normalize_doi(message.get("DOI"))
    if doi and "DOI" in item:
        item["DOI"] = doi

    landing_page = (
        nested_get(message, "resource", "primary", "URL")
        or message.get("URL")
        or (f"https://doi.org/{doi}" if doi else "")
    )
    if landing_page and "url" in item:
        item["url"] = landing_page

    publication_date = crossref_date_to_string(message)
    if publication_date and "date" in item:
        item["date"] = publication_date

    if "abstractNote" in item:
        item["abstractNote"] = strip_markup(message.get("abstract")) or ""

    if message.get("volume") and "volume" in item:
        item["volume"] = str(message["volume"])
    if message.get("issue") and "issue" in item:
        item["issue"] = str(message["issue"])
    if message.get("page") and "pages" in item:
        item["pages"] = str(message["page"])

    language = message.get("language")
    if language and "language" in item:
        item["language"] = str(language)

    item["tags"] = item.get("tags") or []
    item["tags"].append({"tag": "codex-academic-import"})

    if "extra" in item:
        append_extra_line(item, "Metadata source: Crossref")
        if message.get("type"):
            append_extra_line(item, f"Crossref type: {message['type']}")

    if collection_key:
        item["collections"] = [collection_key]

    return item


def clean_translator_item_for_create(
    zot: zotero_api.Zotero,
    item: dict[str, Any],
    collection_key: str | None,
) -> dict[str, Any]:
    item_type = item.get("itemType") or "journalArticle"
    template = zot.item_template(item_type)
    cleaned = {}
    for key in template.keys():
        if key in item:
            cleaned[key] = item[key]

    cleaned["itemType"] = item_type
    cleaned["creators"] = item.get("creators") or []
    cleaned["tags"] = item.get("tags") or []
    cleaned["tags"].append({"tag": "codex-academic-import"})
    cleaned["relations"] = item.get("relations") or {}

    if collection_key:
        cleaned["collections"] = [collection_key]

    if "extra" in template:
        extra_lines = []
        if item.get("extra"):
            extra_lines.append(str(item["extra"]))
        if item.get("url"):
            extra_lines.append(f"Imported from: {item['url']}")
        if extra_lines:
            cleaned["extra"] = "\n".join(extra_lines)
            append_extra_line(cleaned, "Metadata source: translation-server")

    return cleaned


def merge_zotero_items(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for key, value in secondary.items():
        if key in {"creators", "tags", "collections"}:
            continue
        if key == "extra":
            continue
        if not merged.get(key) and value not in (None, "", [], {}):
            merged[key] = value

    primary_creators = primary.get("creators") or []
    secondary_creators = secondary.get("creators") or []
    merged["creators"] = primary_creators or secondary_creators

    tag_names: set[str] = set()
    merged_tags: list[dict[str, Any]] = []
    for tag in list(primary.get("tags") or []) + list(secondary.get("tags") or []):
        if not isinstance(tag, dict):
            continue
        name = str(tag.get("tag") or "").strip()
        if not name or name in tag_names:
            continue
        tag_names.add(name)
        merged_tags.append(tag)
    if merged_tags:
        merged["tags"] = merged_tags

    collections = primary.get("collections") or secondary.get("collections")
    if collections:
        merged["collections"] = collections

    for line in [entry.strip() for entry in str(secondary.get("extra") or "").splitlines() if entry.strip()]:
        append_extra_line(merged, line)

    return merged


def extract_attachment_urls_from_translator_item(item: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for attachment in item.get("attachments") or []:
        if not isinstance(attachment, dict):
            continue
        url = attachment.get("url")
        content_type = attachment.get("mimeType") or attachment.get("contentType")
        title = attachment.get("title") or ""
        if isinstance(url, str) and url.strip():
            if (content_type and "pdf" in content_type.lower()) or "pdf" in title.lower():
                urls.append(url.strip())
    return unique_nonempty(urls)


def collect_unpaywall_pdf_candidates(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    candidates = [
        nested_get(payload, "best_oa_location", "url_for_pdf"),
        nested_get(payload, "best_oa_location", "url"),
        payload.get("doi_url"),
    ]
    return unique_nonempty(candidates)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "_", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned[:180] or "document"


def decode_content_disposition_filename(response: requests.Response) -> str | None:
    content_disposition = response.headers.get("Content-Disposition", "")
    if not content_disposition:
        return None
    match = re.search(r"filename\*?=(?:UTF-8''|\"?)([^\";]+)", content_disposition, flags=re.IGNORECASE)
    if not match:
        return None
    return unquote(match.group(1).strip().strip('"'))


def looks_like_pdf(response: requests.Response, content: bytes) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    if "application/pdf" in content_type:
        return True
    return content.startswith(PDF_SIGNATURE)


def resolve_url(base_url: str, candidate: str) -> str:
    return urljoin(base_url, candidate)


def find_pdf_link_in_html(base_url: str, html: str) -> str | None:
    meta_match = re.search(
        r'citation_pdf_url"\s+content="([^"]+)"',
        html,
        flags=re.IGNORECASE,
    )
    if meta_match:
        return resolve_url(base_url, meta_match.group(1))

    href_match = re.search(
        r'href="([^"]+\.pdf(?:\?[^"]*)?)"',
        html,
        flags=re.IGNORECASE,
    )
    if href_match:
        return resolve_url(base_url, href_match.group(1))
    return None


def find_page_title_in_html(html: str) -> str | None:
    patterns = [
        r'<meta[^>]+name="citation_title"[^>]+content="([^"]+)"',
        r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"',
        r"<title>([^<]+)</title>",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            title = strip_markup(match.group(1))
            if title:
                return title
    return None


def title_related_pdf_heuristic(
    expected_title: str | None,
    doi: str | None,
    *observed_values: str | None,
) -> tuple[bool, str]:
    if not expected_title:
        return True, "no_expected_title"

    normalized_expected = normalize_title(expected_title)
    observed_blob = " ".join(value for value in observed_values if isinstance(value, str) and value.strip())
    normalized_observed = normalize_title(observed_blob)

    if normalized_expected and normalized_observed:
        if normalized_expected in normalized_observed or normalized_observed in normalized_expected:
            return True, "normalized_title_match"

    if doi:
        compact_doi = doi.replace("/", "").lower()
        if compact_doi and compact_doi in observed_blob.lower().replace("/", ""):
            return True, "doi_hint_match"

    expected_tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", expected_title)
        if len(token) >= 4
    ]
    if expected_tokens:
        matched_tokens = sum(1 for token in expected_tokens[:8] if token in observed_blob.lower())
        if matched_tokens >= min(2, len(expected_tokens)):
            return True, f"title_token_overlap:{matched_tokens}"

    return False, "title_heuristic_failed"


def validate_pdf_response(
    response: requests.Response,
    content: bytes,
    *,
    expected_title: str | None,
    doi: str | None,
    observed_title: str | None = None,
) -> tuple[bool, str]:
    if not looks_like_pdf(response, content):
        return False, "not_pdf_signature_or_content_type"
    if len(content) < MIN_PDF_BYTES:
        return False, f"pdf_too_small:{len(content)}"
    if len(content) > MAX_PDF_BYTES:
        return False, f"pdf_too_large:{len(content)}"

    filename = decode_content_disposition_filename(response)
    ok, reason = title_related_pdf_heuristic(
        expected_title,
        doi,
        observed_title,
        filename,
        response.url,
    )
    if not ok:
        return False, reason
    return True, reason


def try_download_pdf(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    expected_title: str | None,
    doi: str | None,
) -> tuple[Path | None, str]:
    try:
        response = session.get(url, timeout=60, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        return None, f"http_error:{exc.__class__.__name__}"

    content = response.content
    valid, reason = validate_pdf_response(
        response,
        content,
        expected_title=expected_title,
        doi=doi,
    )
    if valid:
        destination.write_bytes(content)
        return destination, reason

    pdf_link = find_pdf_link_in_html(response.url, response.text)
    if not pdf_link or pdf_link == url:
        return None, reason

    try:
        nested = session.get(pdf_link, timeout=60, allow_redirects=True)
        nested.raise_for_status()
    except requests.RequestException as exc:
        return None, f"nested_http_error:{exc.__class__.__name__}"

    nested_content = nested.content
    observed_title = find_page_title_in_html(response.text)
    valid, nested_reason = validate_pdf_response(
        nested,
        nested_content,
        expected_title=expected_title,
        doi=doi,
        observed_title=observed_title,
    )
    if not valid:
        return None, nested_reason

    destination.write_bytes(nested_content)
    return destination, nested_reason


def download_pdf_from_candidates(
    session: requests.Session,
    title: str,
    doi: str | None,
    year: str | int | None,
    download_root: Path,
    candidates: list[str],
) -> tuple[Path | None, str | None]:
    if not candidates:
        return None, None

    ensure_directory(download_root)
    filename = sanitize_filename(f"{year or '0000'}_{title}.pdf")
    target = download_root / filename

    seen: set[str] = set()
    for url in candidates:
        if not url or url in seen:
            continue
        seen.add(url)
        log_event("pdf_candidate_try", url=url, title=title, doi=doi)
        downloaded, reason = try_download_pdf(
            session,
            url,
            target,
            expected_title=title,
            doi=doi,
        )
        if downloaded:
            log_event("pdf_candidate_selected", url=url, path=str(downloaded), reason=reason)
            return downloaded, url
        log_event("pdf_candidate_rejected", level="WARNING", url=url, reason=reason)
    return None, None


def ensure_collection_path(
    zot: zotero_api.Zotero,
    collection_path: str | None,
) -> str | None:
    segments = clean_collection_segments(collection_path)
    if not segments:
        return None

    all_collections = zot.everything(zot.all_collections())
    current_parent = ""

    for segment in segments:
        matched_key = None
        for collection in all_collections:
            data = collection.get("data") or {}
            if data.get("name") == segment and (data.get("parentCollection") or "") == current_parent:
                matched_key = data.get("key")
                break

        if matched_key is None:
            payload = [{"name": segment, "parentCollection": current_parent}]
            response = zot.create_collections(payload)
            matched_key = parse_write_response_key(response)
            all_collections = zot.everything(zot.all_collections())

        if not matched_key:
            raise AcademicImportError(f"无法创建或定位 Zotero 分类: {segment}")
        current_parent = matched_key

    return current_parent or None


def parse_write_response_key(response: dict[str, Any]) -> str | None:
    success = response.get("success")
    if isinstance(success, dict):
        for value in success.values():
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict):
                if value.get("key"):
                    return value["key"]
                links = value.get("links") or {}
                self_link = links.get("self") or {}
                href = self_link.get("href")
                if isinstance(href, str) and href.strip():
                    return href.rstrip("/").split("/")[-1]
    return None


def evaluate_existing_item_match(
    data: dict[str, Any],
    *,
    doi: str | None,
    title: str | None,
    url: str | None,
    first_author: str | None,
    year: str | None,
) -> tuple[bool, str | None]:
    existing_doi = normalize_doi(data.get("DOI"))
    existing_url = normalize_url(data.get("url"))
    existing_title = data.get("title")
    existing_first_author = first_creator_name(data.get("creators") or [])
    existing_year = extract_year(data.get("date"))

    if doi and existing_doi and existing_doi == normalize_doi(doi):
        return True, "doi"
    if url and existing_url and existing_url == normalize_url(url):
        return True, "url"
    if title and normalize_title(existing_title) == normalize_title(title):
        return True, "exact_title"

    similarity = title_similarity(existing_title, title)
    if (
        title
        and first_author
        and year
        and similarity >= WEAK_TITLE_SIMILARITY
        and normalize_person_name(existing_first_author) == normalize_person_name(first_author)
        and existing_year == year
    ):
        return True, f"weak_title_author_year:{similarity:.3f}"

    return False, None


def find_existing_zotero_item(
    zot: zotero_api.Zotero,
    doi: str | None,
    title: str | None,
    *,
    url: str | None = None,
    creators: list[dict[str, Any]] | None = None,
    date: str | int | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    normalized_title = normalize_title(title)
    first_author = first_creator_name(creators)
    year = extract_year(date)
    query_terms = unique_nonempty([doi, title, first_author, year])
    inspected: set[str] = set()

    for term in query_terms:
        try:
            results = zot.everything(zot.top(q=term, limit=50))
        except Exception as exc:
            log_event("dedupe_query_failed", level="WARNING", term=term, error=str(exc))
            continue
        for result in results:
            data = result.get("data") or {}
            if data.get("itemType") == "attachment":
                continue
            item_key = result.get("key") or data.get("key") or json.dumps(data, ensure_ascii=False, sort_keys=True)
            if item_key in inspected:
                continue
            inspected.add(item_key)
            matched, reason = evaluate_existing_item_match(
                data,
                doi=doi,
                title=title,
                url=url,
                first_author=first_author,
                year=year,
            )
            if matched:
                return result, reason

    if normalized_title:
        try:
            results = zot.everything(zot.top(limit=100))
        except Exception as exc:
            log_event("dedupe_fallback_failed", level="WARNING", error=str(exc))
            return None, None
        for result in results:
            data = result.get("data") or {}
            if data.get("itemType") == "attachment":
                continue
            matched, reason = evaluate_existing_item_match(
                data,
                doi=doi,
                title=title,
                url=url,
                first_author=first_author,
                year=year,
            )
            if matched:
                return result, reason

    return None, None


def create_item_and_attachment(
    zot: zotero_api.Zotero,
    item: dict[str, Any],
    pdf_path: Path | None,
) -> dict[str, Any]:
    response = zot.create_items([item])
    item_key = parse_write_response_key(response)
    if not item_key:
        raise AcademicImportError("Zotero 条目创建失败，未返回 key。")

    uploaded = None
    if pdf_path:
        uploaded = zot.attachment_simple([str(pdf_path)], parentid=item_key)

    return {"item_key": item_key, "attachment_result": uploaded, "create_result": response}


def translation_server_request(
    session: requests.Session,
    base_url: str,
    endpoint: str,
    payload: str | bytes,
    content_type: str = "text/plain",
) -> requests.Response:
    return post_json(
        session,
        f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}",
        data=payload,
        content_type=content_type,
    )


def translation_server_url_import(
    session: requests.Session,
    base_url: str,
    url: str,
    pick: int | None,
) -> list[dict[str, Any]]:
    response = translation_server_request(session, base_url, "/web", url.encode("utf-8"))

    if response.status_code == 300:
        payload = response.json()
        items = payload.get("items") or {}
        if not isinstance(items, dict) or not items:
            raise AcademicImportError("translation-server 返回了多选结果，但没有候选条目。")
        choices = list(items.items())
        if pick is None:
            raise AcademicImportError(
                "URL 对应多个候选结果，请追加 --pick N。候选如下:\n"
                + "\n".join(
                    f"{index + 1}. {value.get('title', key)}"
                    for index, (key, value) in enumerate(choices)
                )
            )
        if pick < 1 or pick > len(choices):
            raise AcademicImportError(f"--pick 超出范围，可选范围是 1 到 {len(choices)}。")

        selected_key = choices[pick - 1][0]
        payload["items"] = {selected_key: items[selected_key]}
        retry = post_json(
            session,
            f"{base_url.rstrip('/')}/web",
            json_body=payload,
            content_type="application/json",
        )
        retry.raise_for_status()
        return retry.json()

    response.raise_for_status()
    return response.json()


def translation_server_identifier_import(
    session: requests.Session,
    base_url: str,
    identifier: str,
) -> list[dict[str, Any]]:
    response = translation_server_request(
        session,
        base_url,
        "/search",
        identifier.encode("utf-8"),
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise AcademicImportError("translation-server/search 返回了未知格式。")


def translation_server_best_effort_url_import(
    session: requests.Session,
    base_url: str,
    url: str,
) -> list[dict[str, Any]]:
    response = translation_server_request(session, base_url, "/web", url.encode("utf-8"))
    if response.status_code == 300:
        payload = response.json()
        items = payload.get("items") or {}
        if isinstance(items, dict) and items:
            log_event(
                "translation_server_url_ambiguous",
                level="WARNING",
                url=url,
                candidate_count=len(items),
            )
        return []
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


def fetch_optional_source(name: str, loader: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        payload = loader(*args, **kwargs)
    except requests.HTTPError as exc:
        response = exc.response
        status_code = response.status_code if response is not None else None
        if status_code == 404:
            log_event("metadata_source_missing", source=name, status_code=status_code)
            return None
        log_event(
            "metadata_source_failed",
            level="WARNING",
            source=name,
            status_code=status_code,
            error=str(exc),
        )
        return None
    except requests.RequestException as exc:
        log_event("metadata_source_failed", level="WARNING", source=name, error=str(exc))
        return None

    if payload:
        log_event("metadata_source_resolved", source=name)
    else:
        log_event("metadata_source_missing", source=name)
    return payload


def pick_doi_landing_page(
    doi: str,
    *,
    crossref_message: dict[str, Any] | None,
    openalex_work: dict[str, Any] | None,
    unpaywall_record: dict[str, Any] | None,
) -> str:
    return (
        nested_get(crossref_message, "resource", "primary", "URL")
        or (crossref_message or {}).get("URL")
        or nested_get(unpaywall_record, "best_oa_location", "url")
        or (unpaywall_record or {}).get("doi_url")
        or nested_get(openalex_work, "primary_location", "landing_page_url")
        or nested_get(openalex_work, "best_oa_location", "landing_page_url")
        or f"https://doi.org/{doi}"
    )


def build_doi_item_from_sources(
    zot: zotero_api.Zotero,
    doi: str,
    collection_key: str | None,
    *,
    crossref_message: dict[str, Any] | None,
    openalex_work: dict[str, Any] | None,
    translator_item: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    item: dict[str, Any] | None = None
    metadata_sources: list[str] = []

    if crossref_message:
        item = make_crossref_zotero_item(zot, crossref_message, collection_key)
        metadata_sources.append("crossref")

    if openalex_work:
        openalex_item = make_openalex_zotero_item(zot, openalex_work, collection_key)
        item = merge_zotero_items(item, openalex_item) if item else openalex_item
        metadata_sources.append("openalex")

    if translator_item:
        cleaned = clean_translator_item_for_create(zot, translator_item, collection_key)
        item = merge_zotero_items(item, cleaned) if item else cleaned
        metadata_sources.append(
            "translation-server-enhancement" if metadata_sources else "translation-server"
        )

    if item and "extra" in item:
        append_extra_line(item, f"Resolved DOI: {doi}")

    return item, metadata_sources


def build_pdf_candidate_list(
    *,
    crossref_message: dict[str, Any] | None = None,
    openalex_work: dict[str, Any] | None,
    unpaywall_record: dict[str, Any] | None = None,
    translator_item: dict[str, Any] | None,
    prefer_translator: bool = False,
) -> list[str]:
    structured_candidates: list[str] = []
    translator_candidates: list[str] = []

    if crossref_message:
        structured_candidates.extend(collect_crossref_pdf_candidates(crossref_message))
    if openalex_work:
        structured_candidates.extend(collect_openalex_pdf_candidates(openalex_work))
    structured_candidates.extend(collect_unpaywall_pdf_candidates(unpaywall_record))

    if translator_item:
        translator_candidates.extend(extract_attachment_urls_from_translator_item(translator_item))
        source_url = translator_item.get("url")
        if isinstance(source_url, str) and source_url.strip():
            translator_candidates.append(source_url.strip())

    ordered = (
        translator_candidates + structured_candidates
        if prefer_translator
        else structured_candidates + translator_candidates
    )
    return unique_nonempty(ordered)


def command_search(args: argparse.Namespace, config: AppConfig, session: requests.Session) -> int:
    results = search_openalex(session, config, args.query, args.limit)
    json_print({"query": args.query, "count": len(results), "results": results})
    return 0


def command_import_query(args: argparse.Namespace, config: AppConfig, session: requests.Session) -> int:
    log_event("import_query_start", query=args.query, pick=args.pick, download_pdf=bool(args.download_pdf))
    zot = make_zotero_client(config)
    params = openalex_params(config, search=args.query, **{"per-page": args.limit})
    full_payload = get_json(session, OPENALEX_WORKS_URL, params=params)
    full_results = full_payload.get("results") or []
    if not full_results:
        raise AcademicImportError("OpenAlex 没有返回匹配结果。")
    if args.pick < 1 or args.pick > len(full_results):
        raise AcademicImportError(f"--pick 超出范围，可选范围是 1 到 {len(full_results)}。")

    work = full_results[args.pick - 1]
    summary = summarize_openalex_work(work)
    item = make_openalex_zotero_item(zot, work, None)

    existing, dedupe_reason = find_existing_zotero_item(
        zot,
        summary.get("doi"),
        summary.get("title"),
        url=item.get("url"),
        creators=item.get("creators"),
        date=item.get("date"),
    )
    if existing:
        log_event("dedupe_match", reason=dedupe_reason, title=summary.get("title"), doi=summary.get("doi"))
        json_print(
            {
                "status": "exists",
                "item_key": existing.get("key") or nested_get(existing, "data", "key"),
                "title": nested_get(existing, "data", "title"),
                "doi": nested_get(existing, "data", "DOI"),
                "dedupe_reason": dedupe_reason,
            }
        )
        return 0

    collection_path = args.collection or config.default_collection_path
    if collection_path and not args.dry_run:
        collection_key = ensure_collection_path(zot, collection_path)
        item["collections"] = [collection_key]

    pdf_path = None
    pdf_source = None
    if args.download_pdf:
        unpaywall_record = fetch_optional_source(
            "unpaywall",
            fetch_unpaywall_record,
            session,
            config,
            summary.get("doi"),
        )
        candidates = build_pdf_candidate_list(
            openalex_work=work,
            unpaywall_record=unpaywall_record,
            translator_item=None,
        )
        pdf_path, pdf_source = download_pdf_from_candidates(
            session,
            summary["title"],
            summary.get("doi"),
            summary["year"],
            config.download_root,
            candidates,
        )

    if args.dry_run:
        json_print(
            {
                "status": "dry_run",
                "selected": summary,
                "collection": collection_path,
                "item": item,
                "pdf_path": str(pdf_path) if pdf_path else None,
                "pdf_source": pdf_source,
            }
        )
        return 0

    created = create_item_and_attachment(zot, item, pdf_path)
    log_event(
        "import_query_completed",
        query=args.query,
        item_key=created["item_key"],
        doi=summary.get("doi"),
    )
    json_print(
        {
            "status": "imported",
            "selected": summary,
            "collection": collection_path,
            "item_key": created["item_key"],
            "pdf_path": str(pdf_path) if pdf_path else None,
            "pdf_source": pdf_source,
        }
    )
    return 0


def command_import_doi(args: argparse.Namespace, config: AppConfig, session: requests.Session) -> int:
    log_event("import_doi_start", doi=args.doi, download_pdf=bool(args.download_pdf))
    zot = make_zotero_client(config)
    doi = normalize_doi(args.doi)
    if not doi:
        raise AcademicImportError("DOI 不能为空。")

    collection_path = args.collection or config.default_collection_path
    crossref_message = fetch_optional_source("crossref", fetch_crossref_by_doi, session, doi)
    openalex_work = fetch_optional_source("openalex", fetch_openalex_by_doi, session, config, doi)
    unpaywall_record = fetch_optional_source(
        "unpaywall",
        fetch_unpaywall_record,
        session,
        config,
        doi,
    )

    landing_page_url = pick_doi_landing_page(
        doi,
        crossref_message=crossref_message,
        openalex_work=openalex_work,
        unpaywall_record=unpaywall_record,
    )

    selected_item = None
    if config.translation_server_url and landing_page_url:
        try:
            translator_items = translation_server_best_effort_url_import(
                session,
                config.translation_server_url,
                landing_page_url,
            )
            if translator_items:
                selected_item = translator_items[0]
                log_event("translation_server_enhancement_resolved", doi=doi, url=landing_page_url)
            else:
                log_event("translation_server_enhancement_missing", doi=doi, url=landing_page_url)
        except requests.RequestException as exc:
            log_event(
                "translation_server_enhancement_failed",
                level="WARNING",
                doi=doi,
                url=landing_page_url,
                error=str(exc),
            )

    item, metadata_sources = build_doi_item_from_sources(
        zot,
        doi,
        None,
        crossref_message=crossref_message,
        openalex_work=openalex_work,
        translator_item=selected_item,
    )
    if not item and config.translation_server_url:
        try:
            translator_items = translation_server_identifier_import(
                session,
                config.translation_server_url,
                doi,
            )
        except requests.RequestException as exc:
            translator_items = []
            log_event(
                "translation_server_identifier_failed",
                level="WARNING",
                doi=doi,
                error=str(exc),
            )
        if translator_items:
            selected_item = translator_items[0]
            item, metadata_sources = build_doi_item_from_sources(
                zot,
                doi,
                None,
                crossref_message=None,
                openalex_work=None,
                translator_item=selected_item,
            )

    if not item:
        raise AcademicImportError(
            "未能从 Crossref、OpenAlex、Unpaywall 及可选的 translation-server 回退路径获取该 DOI 的可用元数据。"
        )

    title = item.get("title") or doi
    year = extract_year(item.get("date")) or extract_year(
        crossref_date_to_string(crossref_message) if crossref_message else None
    )
    if not year and openalex_work:
        year = str(openalex_work.get("publication_year") or "")

    existing, dedupe_reason = find_existing_zotero_item(
        zot,
        doi,
        item.get("title"),
        url=item.get("url"),
        creators=item.get("creators"),
        date=item.get("date") or year,
    )
    if existing:
        log_event("dedupe_match", reason=dedupe_reason, title=item.get("title"), doi=doi)
        json_print(
            {
                "status": "exists",
                "item_key": existing.get("key") or nested_get(existing, "data", "key"),
                "title": nested_get(existing, "data", "title"),
                "doi": nested_get(existing, "data", "DOI"),
                "dedupe_reason": dedupe_reason,
                "metadata_sources": metadata_sources,
            }
        )
        return 0

    if collection_path and not args.dry_run:
        collection_key = ensure_collection_path(zot, collection_path)
        item["collections"] = [collection_key]

    pdf_path = None
    pdf_source = None
    if args.download_pdf:
        candidates = build_pdf_candidate_list(
            crossref_message=crossref_message,
            openalex_work=openalex_work,
            unpaywall_record=unpaywall_record,
            translator_item=selected_item,
        )
        pdf_path, pdf_source = download_pdf_from_candidates(
            session,
            str(title),
            doi,
            year,
            config.download_root,
            candidates,
        )

    if args.dry_run:
        json_print(
            {
                "status": "dry_run",
                "doi": doi,
                "collection": collection_path,
                "item": item,
                "pdf_path": str(pdf_path) if pdf_path else None,
                "pdf_source": pdf_source,
                "metadata_sources": metadata_sources,
            }
        )
        return 0

    created = create_item_and_attachment(zot, item, pdf_path)
    log_event("import_doi_completed", doi=doi, item_key=created["item_key"], metadata_sources=metadata_sources)
    json_print(
        {
            "status": "imported",
            "doi": doi,
            "collection": collection_path,
            "item_key": created["item_key"],
            "pdf_path": str(pdf_path) if pdf_path else None,
            "pdf_source": pdf_source,
            "metadata_sources": metadata_sources,
        }
    )
    return 0


def command_import_url(args: argparse.Namespace, config: AppConfig, session: requests.Session) -> int:
    log_event("import_url_start", url=args.url, download_pdf=bool(args.download_pdf))
    if not config.translation_server_url:
        raise AcademicImportError(
            "未配置 translation-server。请先运行 startup/start_zotero_translation_server.ps1，或在配置里设置 services.translation_server_url。"
        )

    zot = make_zotero_client(config)
    collection_path = args.collection or config.default_collection_path

    items = translation_server_url_import(
        session,
        config.translation_server_url,
        args.url,
        args.pick,
    )
    if not items:
        raise AcademicImportError("translation-server 没有返回可导入条目。")

    item_from_translator = items[0]
    cleaned_item = clean_translator_item_for_create(zot, item_from_translator, None)

    crossref_message = None
    openalex_work = None
    unpaywall_record = None
    metadata_sources = ["translation-server"]
    translator_doi = normalize_doi(cleaned_item.get("DOI"))
    if translator_doi:
        crossref_message = fetch_optional_source(
            "crossref",
            fetch_crossref_by_doi,
            session,
            translator_doi,
        )
        openalex_work = fetch_optional_source(
            "openalex",
            fetch_openalex_by_doi,
            session,
            config,
            translator_doi,
        )
        unpaywall_record = fetch_optional_source(
            "unpaywall",
            fetch_unpaywall_record,
            session,
            config,
            translator_doi,
        )
        merged_item, metadata_sources = build_doi_item_from_sources(
            zot,
            translator_doi,
            None,
            crossref_message=crossref_message,
            openalex_work=openalex_work,
            translator_item=item_from_translator,
        )
        if merged_item:
            cleaned_item = merged_item

    existing, dedupe_reason = find_existing_zotero_item(
        zot,
        normalize_doi(cleaned_item.get("DOI")),
        cleaned_item.get("title"),
        url=cleaned_item.get("url"),
        creators=cleaned_item.get("creators"),
        date=cleaned_item.get("date"),
    )
    if existing:
        log_event(
            "dedupe_match",
            reason=dedupe_reason,
            title=cleaned_item.get("title"),
            doi=normalize_doi(cleaned_item.get("DOI")),
        )
        json_print(
            {
                "status": "exists",
                "item_key": existing.get("key") or nested_get(existing, "data", "key"),
                "title": nested_get(existing, "data", "title"),
                "doi": nested_get(existing, "data", "DOI"),
                "dedupe_reason": dedupe_reason,
            }
        )
        return 0

    if collection_path and not args.dry_run:
        collection_key = ensure_collection_path(zot, collection_path)
        cleaned_item["collections"] = [collection_key]

    pdf_path = None
    pdf_source = None
    if args.download_pdf:
        candidates = build_pdf_candidate_list(
            crossref_message=crossref_message,
            openalex_work=openalex_work,
            unpaywall_record=unpaywall_record,
            translator_item=item_from_translator,
            prefer_translator=True,
        )
        pdf_path, pdf_source = download_pdf_from_candidates(
            session,
            cleaned_item.get("title") or "document",
            normalize_doi(cleaned_item.get("DOI")),
            cleaned_item.get("date") or "",
            config.download_root,
            candidates,
        )

    if args.dry_run:
        json_print(
            {
                "status": "dry_run",
                "url": args.url,
                "collection": collection_path,
                "item": cleaned_item,
                "pdf_path": str(pdf_path) if pdf_path else None,
                "pdf_source": pdf_source,
                "metadata_sources": metadata_sources,
            }
        )
        return 0

    created = create_item_and_attachment(zot, cleaned_item, pdf_path)
    log_event("import_url_completed", url=args.url, item_key=created["item_key"], metadata_sources=metadata_sources)
    json_print(
        {
            "status": "imported",
            "url": args.url,
            "collection": collection_path,
            "item_key": created["item_key"],
            "pdf_path": str(pdf_path) if pdf_path else None,
            "pdf_source": pdf_source,
            "metadata_sources": metadata_sources,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search academic literature, retrieve legal/open PDFs where possible, and import into Zotero.",
    )
    parser.add_argument(
        "--config",
        help=f"Path to config JSON. Defaults to {DEFAULT_CONFIG_PATH}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Search OpenAlex by keyword.")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    import_query_parser = subparsers.add_parser(
        "import-query",
        help="Search OpenAlex, pick one result, and import it into Zotero.",
    )
    import_query_parser.add_argument("query", help="Search query")
    import_query_parser.add_argument("--limit", type=int, default=5, help="Number of search candidates")
    import_query_parser.add_argument("--pick", type=int, default=1, help="1-based result index to import")
    import_query_parser.add_argument("--collection", help="Zotero collection path like A/B/C")
    import_query_parser.add_argument(
        "--download-pdf",
        action="store_true",
        help="Try to download an open/legal PDF and upload it as an attachment.",
    )
    import_query_parser.add_argument("--dry-run", action="store_true", help="Preview only")

    import_doi_parser = subparsers.add_parser(
        "import-doi",
        help="Import one paper by DOI into Zotero.",
    )
    import_doi_parser.add_argument("doi", help="DOI")
    import_doi_parser.add_argument("--collection", help="Zotero collection path like A/B/C")
    import_doi_parser.add_argument(
        "--download-pdf",
        action="store_true",
        help="Try to download an open/legal PDF and upload it as an attachment.",
    )
    import_doi_parser.add_argument("--dry-run", action="store_true", help="Preview only")

    import_url_parser = subparsers.add_parser(
        "import-url",
        help="Use Zotero translation-server to translate a page URL and import the result into Zotero.",
    )
    import_url_parser.add_argument("url", help="Article page URL or supported result page URL")
    import_url_parser.add_argument("--pick", type=int, help="1-based result index when translation-server returns multiple choices")
    import_url_parser.add_argument("--collection", help="Zotero collection path like A/B/C")
    import_url_parser.add_argument(
        "--download-pdf",
        action="store_true",
        help="Try to download an open/legal PDF and upload it as an attachment.",
    )
    import_url_parser.add_argument("--dry-run", action="store_true", help="Preview only")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    session = make_session()

    try:
        if args.command == "search":
            return command_search(args, config, session)
        if args.command == "import-query":
            return command_import_query(args, config, session)
        if args.command == "import-doi":
            return command_import_doi(args, config, session)
        if args.command == "import-url":
            return command_import_url(args, config, session)
        raise AcademicImportError(f"未知命令: {args.command}")
    except AcademicImportError as exc:
        log_event("command_failed", level="ERROR", command=args.command, error=str(exc))
        print(str(exc), file=sys.stderr)
        return 2
    except requests.HTTPError as exc:
        response = exc.response
        if response is not None:
            detail = response.text[:500]
            log_event(
                "http_failed",
                level="ERROR",
                command=args.command,
                status_code=response.status_code,
                reason=response.reason,
            )
            print(
                f"HTTP 请求失败: {response.status_code} {response.reason}\n{detail}",
                file=sys.stderr,
            )
        else:
            log_event("http_failed", level="ERROR", command=args.command, error=str(exc))
            print(f"HTTP 请求失败: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        log_event("command_failed", level="ERROR", command=args.command, error=str(exc))
        print(f"未处理错误: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
