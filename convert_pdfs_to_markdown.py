from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymupdf4llm


DEFAULT_INPUTS = ("data/DSM.pdf", "data/brochures")
DEFAULT_OUTPUT_DIR = Path("data/markdown")
DEFAULT_MANIFEST_PATH = Path("data/markdown_manifest.json")


def apply_layout_dtype_patch() -> None:
    """Patch pymupdf4llm's layout runtime so ONNX gets int64 edge indices."""

    from pymupdf.layout.onnx import BoxRFDGNN as boxrf_module

    if getattr(boxrf_module, "_codex_dtype_patch_applied", False):
        return

    original_get_nn_input = boxrf_module.get_nn_input_from_datadict

    def patched_get_nn_input(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        import numpy as np

        (
            x,
            edge_index,
            edge_attr,
            nn_index,
            nn_attr,
            rf_feature,
            text_feature,
            image_feature,
            image_data,
        ) = original_get_nn_input(*args, **kwargs)

        if edge_index is not None:
            edge_index = np.asarray(edge_index, dtype=np.int64)
        if nn_index is not None:
            nn_index = np.asarray(nn_index, dtype=np.int64)

        return (
            x,
            edge_index,
            edge_attr,
            nn_index,
            nn_attr,
            rf_feature,
            text_feature,
            image_feature,
            image_data,
        )

    boxrf_module.get_nn_input_from_datadict = patched_get_nn_input
    boxrf_module._codex_dtype_patch_applied = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert brochure / guideline PDFs into Markdown using pymupdf4llm. "
            "Outputs clean .md files plus a manifest with source metadata for citations."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "PDF files or directories to scan recursively. "
            f"Defaults to: {', '.join(DEFAULT_INPUTS)}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where converted Markdown files will be written.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path for the JSON manifest that stores PDF-to-Markdown provenance.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert PDFs even when the destination Markdown file already exists.",
    )
    parser.add_argument(
        "--disable-layout",
        action="store_true",
        help=(
            "Skip pymupdf4llm's newer layout engine and use the legacy extractor "
            "from the start."
        ),
    )
    return parser.parse_args()


def discover_input_roots(raw_inputs: list[str]) -> list[Path]:
    if raw_inputs:
        roots = [Path(item).resolve() for item in raw_inputs]
    else:
        roots = [Path(item).resolve() for item in DEFAULT_INPUTS if Path(item).exists()]

    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        raise FileNotFoundError(
            "No valid input PDFs or directories were found. "
            f"Checked: {', '.join(raw_inputs or DEFAULT_INPUTS)}"
        )
    return existing_roots


def discover_pdfs(input_roots: list[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for root in input_roots:
        if root.is_file():
            if root.suffix.lower() == ".pdf":
                discovered.add(root)
            continue

        for pdf_path in root.rglob("*.pdf"):
            if pdf_path.is_file():
                discovered.add(pdf_path.resolve())

    return sorted(discovered)


def sanitize_segment(segment: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in segment)
    cleaned = cleaned.strip("-_")
    return cleaned or "document"


def get_relative_stem(pdf_path: Path, input_roots: list[Path]) -> Path:
    for root in sorted(input_roots, key=lambda item: len(str(item)), reverse=True):
        if root.is_file() and pdf_path == root:
            return Path(root.stem)
        if root.is_dir():
            try:
                return pdf_path.relative_to(root).with_suffix("")
            except ValueError:
                continue
    return Path(pdf_path.stem)


def build_output_path(
    pdf_path: Path,
    input_roots: list[Path],
    output_dir: Path,
    used_paths: dict[Path, Path],
) -> Path:
    relative_stem = get_relative_stem(pdf_path, input_roots)
    name_parts = [sanitize_segment(part) for part in relative_stem.parts if part not in {".", ""}]
    base_name = "__".join(name_parts) or sanitize_segment(pdf_path.stem)
    output_path = output_dir / f"{base_name}.md"

    previous_source = used_paths.get(output_path)
    if previous_source and previous_source != pdf_path:
        digest = hashlib.sha1(str(relative_stem).encode("utf-8")).hexdigest()[:8]
        output_path = output_dir / f"{base_name}--{digest}.md"

    used_paths[output_path] = pdf_path
    return output_path


def load_previous_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    documents = manifest_data.get("documents", [])
    previous_entries: dict[str, dict[str, Any]] = {}
    for entry in documents:
        source_pdf = entry.get("source_pdf")
        if isinstance(source_pdf, str):
            previous_entries[source_pdf] = entry
    return previous_entries


def page_chunks_to_markdown(page_chunks: list[dict[str, Any]]) -> str:
    page_texts = []
    for chunk in page_chunks:
        text = (chunk.get("text") or "").strip()
        if text:
            page_texts.append(text)
    return "\n\n".join(page_texts).strip() + "\n"


def page_chunks_to_manifest_pages(page_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for chunk in page_chunks:
        metadata = dict(chunk.get("metadata") or {})
        pages.append(
            {
                "page": metadata.get("page") or metadata.get("page_number"),
                "metadata": metadata,
                "toc_items": chunk.get("toc_items") or [],
                "markdown": chunk.get("text") or "",
            }
        )
    return pages


def doc_metadata_from_page_chunks(page_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    if not page_chunks:
        return {}

    metadata = dict(page_chunks[0].get("metadata") or {})
    metadata.pop("page", None)
    metadata.pop("page_number", None)
    return metadata


def convert_with_mode(pdf_path: Path, use_layout: bool) -> list[dict[str, Any]]:
    if use_layout:
        apply_layout_dtype_patch()
    pymupdf4llm.use_layout(use_layout)
    converted = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    if not isinstance(converted, list):
        raise TypeError(
            f"Expected page_chunks output to be a list, got {type(converted).__name__}."
        )
    return converted


def convert_pdf(
    pdf_path: Path,
    layout_enabled: bool,
) -> tuple[list[dict[str, Any]], str, str | None, bool]:
    if layout_enabled:
        try:
            return convert_with_mode(pdf_path, use_layout=True), "layout", None, True
        except Exception as exc:
            fallback_chunks = convert_with_mode(pdf_path, use_layout=False)
            return fallback_chunks, "legacy-fallback", f"{type(exc).__name__}: {exc}", False

    return convert_with_mode(pdf_path, use_layout=False), "legacy", None, False


def build_manifest_entry(
    pdf_path: Path,
    output_path: Path,
    page_chunks: list[dict[str, Any]],
    conversion_mode: str,
    layout_error: str | None,
    status: str,
) -> dict[str, Any]:
    markdown_text = page_chunks_to_markdown(page_chunks)
    return {
        "status": status,
        "source_pdf": str(pdf_path),
        "source_file_name": pdf_path.name,
        "source_size_bytes": pdf_path.stat().st_size,
        "source_last_modified": datetime.fromtimestamp(
            pdf_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "output_markdown": str(output_path),
        "conversion_mode": conversion_mode,
        "layout_error": layout_error,
        "document_metadata": doc_metadata_from_page_chunks(page_chunks),
        "markdown_char_count": len(markdown_text),
        "pages": page_chunks_to_manifest_pages(page_chunks),
    }


def main() -> int:
    args = parse_args()
    input_roots = discover_input_roots(args.inputs)
    pdf_paths = discover_pdfs(input_roots)

    if not pdf_paths:
        print("No PDFs found in the provided inputs.")
        return 1

    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pdf_paths)} PDF(s) to process.", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Manifest path: {manifest_path}", flush=True)

    previous_entries = load_previous_manifest(manifest_path)
    used_paths: dict[Path, Path] = {}
    manifest_documents: list[dict[str, Any]] = []

    layout_enabled = not args.disable_layout
    layout_disabled_reason: str | None = None
    converted_count = 0
    skipped_count = 0

    for index, pdf_path in enumerate(pdf_paths, start=1):
        output_path = build_output_path(pdf_path, input_roots, output_dir, used_paths)
        print(
            f"[{index}/{len(pdf_paths)}] Preparing {pdf_path.name}...",
            flush=True,
        )

        if output_path.exists() and not args.force:
            previous_entry = previous_entries.get(str(pdf_path))
            if previous_entry:
                entry = dict(previous_entry)
                entry["status"] = "skipped_existing"
            else:
                entry = {
                    "status": "skipped_existing",
                    "source_pdf": str(pdf_path),
                    "output_markdown": str(output_path),
                }
            manifest_documents.append(entry)
            skipped_count += 1
            print(f"Skipped existing: {pdf_path} -> {output_path}", flush=True)
            continue

        print(
            f"[{index}/{len(pdf_paths)}] Converting {pdf_path}...",
            flush=True,
        )
        page_chunks, conversion_mode, layout_error, layout_succeeded = convert_pdf(
            pdf_path, layout_enabled=layout_enabled
        )

        if layout_enabled and not layout_succeeded:
            layout_enabled = False
            layout_disabled_reason = layout_error

        markdown_text = page_chunks_to_markdown(page_chunks)
        output_path.write_text(markdown_text, encoding="utf-8")

        entry = build_manifest_entry(
            pdf_path=pdf_path,
            output_path=output_path,
            page_chunks=page_chunks,
            conversion_mode=conversion_mode,
            layout_error=layout_error,
            status="converted",
        )
        manifest_documents.append(entry)
        converted_count += 1
        print(
            f"Converted ({conversion_mode}): {pdf_path} -> {output_path}",
            flush=True,
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "convert_pdfs_to_markdown.py",
        "input_roots": [str(path) for path in input_roots],
        "output_dir": str(output_dir),
        "documents": manifest_documents,
    }
    if layout_disabled_reason:
        manifest["layout_fallback_reason"] = layout_disabled_reason

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(flush=True)
    print(
        f"Finished conversion: {converted_count} converted, {skipped_count} skipped. "
        f"Manifest written to {manifest_path}",
        flush=True,
    )
    if layout_disabled_reason:
        print(
            "Layout mode was disabled after the first failure and the remaining PDFs "
            "used the legacy extractor.",
            flush=True,
        )
        print(f"Layout failure: {layout_disabled_reason}", flush=True)
    if args.output_dir == DEFAULT_OUTPUT_DIR and any(
        path.name.startswith("output") and path.suffix.lower() == ".md"
        for path in output_dir.iterdir()
        if path.is_file()
    ):
        print(
            "Note: existing cloud-exported Markdown files are still present in "
            f"{output_dir}. Remove them when you are ready to avoid duplicate ingestion.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
