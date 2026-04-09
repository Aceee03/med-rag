from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import trafilatura

from convert_pdfs_to_markdown import convert_pdf, page_chunks_to_markdown


DEFAULT_REGISTRY = Path("data/source_registry.json")
DEFAULT_OUTPUT_DIR = Path("data/clinical_markdown")
DEFAULT_MANIFEST_PATH = Path("data/clinical_source_manifest.json")
DEFAULT_CACHE_DIR = Path("data/source_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest official clinical guideline sources into clean Markdown. "
            "Uses trafilatura for HTML and pymupdf4llm for PDFs."
        )
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-authority", action="append", default=[])
    parser.add_argument("--only-condition", action="append", default=[])
    return parser.parse_args()


def slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")
    return text or "source"


def load_registry(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        sources = raw.get("sources", [])
    elif isinstance(raw, list):
        sources = raw
    else:
        raise TypeError(f"Unsupported registry format in {path}")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"No sources found in {path}")
    return sources


def filter_sources(sources: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    filtered = sources
    if args.only_authority:
        allowed = {value.lower() for value in args.only_authority}
        filtered = [item for item in filtered if str(item.get("authority", "")).lower() in allowed]
    if args.only_condition:
        allowed = {value.lower() for value in args.only_condition}
        filtered = [item for item in filtered if str(item.get("condition", "")).lower() in allowed]
    if args.limit is not None:
        filtered = filtered[: max(0, args.limit)]
    return filtered


def _requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 Clinical-GraphRAG-Ingest/1.0"})
    return session


def _download_binary(session: requests.Session, url: str, destination: Path) -> tuple[Path, str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination, response.url, response.headers.get("content-type", "")


def _html_to_markdown(session: requests.Session, url: str) -> tuple[str, str, str]:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        response = session.get(url, timeout=60, allow_redirects=True)
        response.raise_for_status()
        downloaded = response.text
        resolved_url = response.url
        content_type = response.headers.get("content-type", "")
    else:
        response = session.get(url, timeout=60, allow_redirects=True)
        response.raise_for_status()
        resolved_url = response.url
        content_type = response.headers.get("content-type", "")

    markdown = trafilatura.extract(
        downloaded,
        output_format="markdown",
        include_links=True,
        include_formatting=True,
        favor_precision=True,
        url=resolved_url,
    )
    if not markdown:
        raise RuntimeError(f"Trafilatura could not extract markdown from {url}")
    return markdown.strip() + "\n", resolved_url, content_type


def _pdf_to_markdown(pdf_path: Path) -> tuple[str, str]:
    page_chunks, conversion_mode, _, _ = convert_pdf(pdf_path, layout_enabled=True)
    return page_chunks_to_markdown(page_chunks), f"pymupdf4llm:{conversion_mode}"


def output_path_for(entry: dict[str, Any], output_dir: Path) -> Path:
    stem = "__".join(
        [
            slugify(str(entry.get("authority", ""))),
            slugify(str(entry.get("condition", ""))),
            slugify(str(entry.get("id") or entry.get("title") or entry.get("source_title") or "source")),
        ]
    )
    return output_dir / f"{stem}.md"


def ingest_one(
    entry: dict[str, Any],
    *,
    session: requests.Session,
    output_dir: Path,
    cache_dir: Path,
    force: bool,
) -> dict[str, Any]:
    source_url = str(entry["url"])
    source_format = str(entry.get("format", "html")).lower()
    output_path = output_path_for(entry, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        return {
            **entry,
            "status": "skipped_existing",
            "output_markdown": str(output_path),
        }

    parser_used = ""
    downloaded_file = ""
    resolved_url = source_url
    content_type = ""

    if source_format == "pdf":
        cache_name = output_path.with_suffix(".pdf").name
        local_pdf, resolved_url, content_type = _download_binary(
            session,
            source_url,
            cache_dir / "pdf" / cache_name,
        )
        downloaded_file = str(local_pdf)
        markdown, parser_used = _pdf_to_markdown(local_pdf)
    elif source_format == "html":
        markdown, resolved_url, content_type = _html_to_markdown(session, source_url)
        parser_used = "trafilatura"
    else:
        raise ValueError(f"Unsupported source format: {source_format}")

    output_path.write_text(markdown, encoding="utf-8")
    return {
        **entry,
        "status": "ingested",
        "output_markdown": str(output_path),
        "downloaded_file": downloaded_file,
        "resolved_url": resolved_url,
        "content_type": content_type,
        "markdown_char_count": len(markdown),
        "parser_used": parser_used,
    }


def main() -> int:
    args = parse_args()
    registry = filter_sources(load_registry(args.registry), args)
    if not registry:
        raise ValueError("No sources matched the provided filters.")

    session = _requests_session()
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest_path.resolve()
    cache_dir = args.cache_dir.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    print(f"Ingesting {len(registry)} clinical sources...")
    for index, entry in enumerate(registry, start=1):
        title = entry.get("source_title") or entry.get("title") or entry.get("id") or "source"
        print(f"[{index}/{len(registry)}] {title} ({entry.get('format', 'html')})", flush=True)
        result = ingest_one(
            entry,
            session=session,
            output_dir=output_dir,
            cache_dir=cache_dir,
            force=args.force,
        )
        documents.append(result)
        print(
            f"  -> {result['status']} | parser={result.get('parser_used', 'n/a')} | "
            f"output={result['output_markdown']}",
            flush=True,
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "ingest_clinical_sources.py",
        "registry": str(args.registry.resolve()),
        "output_dir": str(output_dir),
        "documents": documents,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
