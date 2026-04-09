from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser


DEFAULT_INPUT_PATH = Path("data/markdown/DSM.md")
DEFAULT_OUTPUT_PATH = Path("data/markdown_filtered/DSM.filtered.md")
DEFAULT_REPORT_PATH = Path("checkpoints/dsm_filter_report.json")

HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
HTML_PATTERN = re.compile(r"<[^>]+>")
STYLE_PATTERN = re.compile(r"[*_`~]+")
SPACE_PATTERN = re.compile(r"\s+")
ROMAN_NUMERAL_PATTERN = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
PAGE_MARKER_PATTERN = re.compile(r"^(?:page\s+)?(?:\d+|[ivxlcdm]+|[a-z]?\d+)$", re.IGNORECASE)
LIST_LINE_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+\.|[A-Z]\.|[a-z]\)|\([a-z]\))\s+")
ICD_CODE_PATTERN = re.compile(r"\b[A-TV-Z][0-9][0-9AB](?:\.[A-Z0-9]{1,4})?\b")
PICTURE_PLACEHOLDER_PATTERN = re.compile(
    r"^\*\*(?:==> picture .* omitted <==|----- (?:Start|End) of picture text -----)\*\*(?:<br>)?$",
    re.IGNORECASE,
)

SECTION_II_MARKERS = (
    "section ii",
    "diagnostic criteria and codes",
)
SECTION_III_MARKERS = (
    "section iii",
    "emerging measures and models",
)

DISCARD_HEADER_PHRASES = {
    "acknowledgment",
    "acknowledgements",
    "advisors",
    "alphabetical listing",
    "american psychiatric association",
    "appendix",
    "assembly",
    "board of trustees",
    "british library cataloguing in publication data",
    "cautionary statement",
    "chairs",
    "classification",
    "contributors",
    "contents",
    "copyright",
    "course specifiers and glossary",
    "cross cutting review group",
    "cross cutting review groups",
    "dsm 5 basics",
    "dsm 5 classification",
    "dsm 5 research group",
    "dsm 5 task force",
    "dsm 5 tr chairs",
    "dsm steering committee",
    "editorial and coding consultants",
    "ethnoracial equity and inclusion work group",
    "foreword",
    "glossary",
    "highlights of changes",
    "index",
    "introduction",
    "library of congress cataloging in publication data",
    "numerical listing",
    "officers",
    "office of the medical director",
    "preface",
    "review committees",
    "review groups",
    "reviewers",
    "section editor",
    "staff",
    "study groups",
    "suggested readings",
    "task force",
    "use of the manual",
    "work group",
    "work groups",
}

CLINICAL_SIGNALS = {
    "associated features",
    "associated with",
    "behavior",
    "behavioral",
    "clinical",
    "comorbid",
    "comorbidity",
    "consequences",
    "criterion",
    "criteria",
    "development and course",
    "development",
    "diagnosis",
    "diagnostic",
    "differential diagnosis",
    "disorder",
    "duration",
    "episodes",
    "features",
    "functional consequences",
    "functional impairment",
    "impairment",
    "mental disorder",
    "onset",
    "pattern",
    "prevalence",
    "presentation",
    "prognosis",
    "psychotherapy",
    "risk and prognostic",
    "risk factor",
    "severity",
    "specifier",
    "specifiers",
    "subtype",
    "suicidal",
    "suicide",
    "symptom",
    "symptoms",
    "treatment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter oversized DSM Markdown before graph construction by removing "
            "administrative sections and rejecting low-value chunks."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the DSM Markdown file produced by pymupdf4llm.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the section-filtered Markdown.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write the JSON audit report.",
    )
    parser.add_argument(
        "--keep-all-sections",
        action="store_true",
        help="Do not restrict filtering to Section II of the DSM.",
    )
    return parser.parse_args()


def strip_markdown_formatting(text: str) -> str:
    cleaned = LINK_PATTERN.sub(r"\1", text)
    cleaned = HTML_PATTERN.sub(" ", cleaned)
    cleaned = STYLE_PATTERN.sub("", cleaned)
    cleaned = cleaned.replace("™", " ").replace("®", " ")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    return SPACE_PATTERN.sub(" ", cleaned).strip()


def normalize_header(text: str) -> str:
    cleaned = strip_markdown_formatting(text).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return SPACE_PATTERN.sub(" ", cleaned).strip()


def split_by_headers(markdown_text: str) -> list[dict[str, Any]]:
    matches = list(HEADER_PATTERN.finditer(markdown_text))
    sections: list[dict[str, Any]] = []

    if not matches:
        return [
            {
                "index": 0,
                "header": "__document__",
                "normalized_header": "document",
                "level": 0,
                "content": markdown_text.strip(),
                "char_count": len(markdown_text.strip()),
            }
        ]

    preamble = markdown_text[: matches[0].start()].strip()
    if preamble:
        sections.append(
            {
                "index": 0,
                "header": "__preamble__",
                "normalized_header": "preamble",
                "level": 0,
                "content": preamble,
                "char_count": len(preamble),
            }
        )

    for match_index, match in enumerate(matches):
        header = match.group(2).strip()
        start = match.end()
        end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(markdown_text)
        content = markdown_text[start:end].strip()
        sections.append(
            {
                "index": len(sections),
                "header": header,
                "normalized_header": normalize_header(header),
                "level": len(match.group(1)),
                "content": content,
                "char_count": len(content),
            }
        )

    return sections


def is_discardable_header(header_text: str) -> tuple[bool, str | None]:
    normalized = normalize_header(header_text)

    if not normalized:
        return True, "empty_header"
    if normalized in {"preamble", "document"}:
        return True, "non_section_content"
    if PAGE_MARKER_PATTERN.fullmatch(normalized) or ROMAN_NUMERAL_PATTERN.fullmatch(normalized):
        return True, "page_marker_header"
    if normalized in {
        "diagnostic and statistical manual of mental disorders",
        "dsm 5 tr",
        "dsm 5",
    }:
        return True, "title_header"
    for phrase in DISCARD_HEADER_PHRASES:
        if phrase in normalized:
            return True, f"discard_header:{phrase}"
    return False, None


def find_section_window(sections: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    section_ii_index: int | None = None
    section_iii_index: int | None = None

    for index, section in enumerate(sections):
        normalized = section["normalized_header"]
        if section_ii_index is None and normalized == "section ii":
            section_ii_index = index
            continue

        if section_ii_index is not None and normalized == "section iii":
            section_iii_index = index
            break

    if section_ii_index is None:
        for index, section in enumerate(sections):
            if section["normalized_header"] == "section ii diagnostic criteria and codes":
                section_ii_index = index
                break

    if section_ii_index is not None and section_iii_index is None:
        for index in range(section_ii_index + 1, len(sections)):
            normalized = sections[index]["normalized_header"]
            if any(marker in normalized for marker in SECTION_III_MARKERS):
                section_iii_index = index
                break

    return section_ii_index, section_iii_index


def apply_section_filters(
    sections: list[dict[str, Any]],
    *,
    restrict_to_section_ii: bool = True,
) -> list[dict[str, Any]]:
    filtered_sections: list[dict[str, Any]] = []
    section_ii_index, section_iii_index = find_section_window(sections)

    for index, section in enumerate(sections):
        keep = True
        reason: str | None = None

        if restrict_to_section_ii and section_ii_index is not None:
            if index < section_ii_index:
                keep = False
                reason = "outside_section_ii_before_start"
            elif section_iii_index is not None and index >= section_iii_index:
                keep = False
                reason = "outside_section_ii_after_end"

        if keep:
            discard_header, discard_reason = is_discardable_header(section["header"])
            if discard_header:
                keep = False
                reason = discard_reason

        section_copy = dict(section)
        section_copy["keep"] = keep
        section_copy["discard_reason"] = reason
        filtered_sections.append(section_copy)

    return filtered_sections


def rebuild_markdown(sections: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for section in sections:
        if not section.get("keep"):
            continue
        if section["level"] > 0:
            parts.append(f"{'#' * section['level']} {section['header']}")
        if section["content"]:
            parts.append(section["content"])
    rebuilt = "\n\n".join(part for part in parts if part.strip()).strip()
    return rebuilt + "\n" if rebuilt else ""


def clean_chunk_text(text: str) -> str:
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        normalized_line = normalize_header(line)
        if not line:
            cleaned_lines.append("")
            continue
        if PICTURE_PLACEHOLDER_PATTERN.match(line):
            continue
        if PAGE_MARKER_PATTERN.fullmatch(normalized_line) or ROMAN_NUMERAL_PATTERN.fullmatch(normalized_line):
            continue
        cleaned_lines.append(raw_line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def chunk_quality_score(text: str, header_path: str = "") -> dict[str, Any]:
    cleaned_text = clean_chunk_text(text)
    combined_text = f"{header_path}\n{cleaned_text}".lower()
    lines = [line.strip() for line in cleaned_text.strip().splitlines() if line.strip()]
    list_lines = sum(1 for line in lines if LIST_LINE_PATTERN.match(line))
    table_lines = sum(1 for line in lines if "|" in line)
    alpha_chars = sum(1 for char in cleaned_text if char.isalpha())
    non_space_chars = sum(1 for char in cleaned_text if not char.isspace())
    clinical_hits = sorted(signal for signal in CLINICAL_SIGNALS if signal in combined_text)

    return {
        "char_count": len(cleaned_text),
        "word_count": len(cleaned_text.split()),
        "line_count": len(lines),
        "list_ratio": list_lines / max(len(lines), 1),
        "table_ratio": table_lines / max(len(lines), 1),
        "alpha_ratio": alpha_chars / max(non_space_chars, 1),
        "sentence_count": len(re.findall(r"[.!?]+", cleaned_text)),
        "icd_code_count": len(ICD_CODE_PATTERN.findall(cleaned_text)),
        "clinical_hits": clinical_hits,
        "has_clinical_signal": bool(clinical_hits),
        "cleaned_text": cleaned_text,
    }


def should_keep_chunk(text: str, header_path: str = "") -> tuple[bool, str, dict[str, Any]]:
    metrics = chunk_quality_score(text, header_path=header_path)

    if metrics["char_count"] < 150:
        return False, f"too_short:{metrics['char_count']}", metrics

    if metrics["table_ratio"] > 0.30 and not metrics["has_clinical_signal"]:
        return False, f"table_heavy:{metrics['table_ratio']:.2f}", metrics

    if metrics["list_ratio"] > 0.70 and metrics["char_count"] < 600 and not metrics["has_clinical_signal"]:
        return False, f"list_heavy:{metrics['list_ratio']:.2f}", metrics

    if metrics["icd_code_count"] >= 3 and metrics["char_count"] < 1000 and not metrics["has_clinical_signal"]:
        return False, f"code_heavy:{metrics['icd_code_count']}", metrics

    if metrics["alpha_ratio"] < 0.50:
        return False, f"low_alpha_ratio:{metrics['alpha_ratio']:.2f}", metrics

    if not metrics["has_clinical_signal"]:
        return False, "no_clinical_signal", metrics

    return True, "ok", metrics


def filter_nodes(nodes: list[Any]) -> tuple[list[Any], list[dict[str, Any]], Counter[str]]:
    kept_nodes: list[Any] = []
    rejected_chunks: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()

    for index, node in enumerate(nodes):
        metadata = dict(getattr(node, "metadata", {}) or {})
        header_path = metadata.get("header_path", "") or metadata.get("Header 1", "")
        keep, reason, metrics = should_keep_chunk(node.text, header_path=str(header_path))
        if keep:
            kept_nodes.append(node)
            continue

        rejection_reasons[reason] += 1
        cleaned_preview = metrics["cleaned_text"][:200].replace("\n", " ").strip()
        report_metrics = {key: value for key, value in metrics.items() if key != "cleaned_text"}
        rejected_chunks.append(
            {
                "index": index,
                "header_path": str(header_path),
                "reason": reason,
                "metrics": report_metrics,
                "preview": cleaned_preview or node.text[:200].replace("\n", " ").strip(),
            }
        )

    return kept_nodes, rejected_chunks, rejection_reasons


def build_final_markdown_from_nodes(kept_nodes: list[Any]) -> str:
    parts = []
    for node in kept_nodes:
        cleaned_text = clean_chunk_text(node.text)
        if cleaned_text:
            parts.append(cleaned_text)
    final_markdown = "\n\n".join(parts).strip()
    return final_markdown + "\n" if final_markdown else ""


def summarize_sections(sections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    discarded_sections = []
    discard_reasons: Counter[str] = Counter()

    for section in sections:
        if section.get("keep"):
            continue
        reason = section.get("discard_reason") or "discarded"
        discard_reasons[reason] += 1
        discarded_sections.append(
            {
                "header": section["header"],
                "normalized_header": section["normalized_header"],
                "level": section["level"],
                "char_count": section["char_count"],
                "reason": reason,
                "preview": section["content"][:200].replace("\n", " ").strip(),
            }
        )

    return discarded_sections, discard_reasons


def filter_markdown_text(
    markdown_text: str,
    *,
    source_name: str = "DSM-5-TR",
    restrict_to_section_ii: bool = True,
) -> dict[str, Any]:
    sections = split_by_headers(markdown_text)
    filtered_sections = apply_section_filters(
        sections,
        restrict_to_section_ii=restrict_to_section_ii,
    )
    kept_sections = [section for section in filtered_sections if section.get("keep")]
    section_filtered_markdown = rebuild_markdown(filtered_sections)

    parser = MarkdownNodeParser()
    document = Document(text=section_filtered_markdown, metadata={"source": source_name})
    nodes = parser.get_nodes_from_documents([document])
    kept_nodes, rejected_chunks, rejection_reasons = filter_nodes(nodes)
    final_markdown = build_final_markdown_from_nodes(kept_nodes)

    discarded_sections, discard_reasons = summarize_sections(filtered_sections)

    return {
        "sections": filtered_sections,
        "kept_sections": kept_sections,
        "discarded_sections": discarded_sections,
        "section_discard_reasons": dict(discard_reasons),
        "section_filtered_markdown": section_filtered_markdown,
        "final_markdown": final_markdown,
        "nodes_before_chunk_filter": len(nodes),
        "kept_nodes": kept_nodes,
        "rejected_chunks": rejected_chunks,
        "chunk_rejection_reasons": dict(rejection_reasons),
    }


def build_report(
    *,
    input_path: Path,
    output_path: Path,
    results: dict[str, Any],
    original_markdown: str,
) -> dict[str, Any]:
    section_filtered_markdown = results["section_filtered_markdown"]
    kept_nodes = results["kept_nodes"]

    return {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "original_char_count": len(original_markdown),
        "section_filtered_char_count": len(section_filtered_markdown),
        "final_chunk_char_count": len(results["final_markdown"]),
        "total_sections": len(results["sections"]),
        "kept_sections": len(results["kept_sections"]),
        "discarded_sections": len(results["discarded_sections"]),
        "nodes_before_chunk_filter": results["nodes_before_chunk_filter"],
        "kept_nodes": len(kept_nodes),
        "rejected_nodes": len(results["rejected_chunks"]),
        "section_discard_reasons": results["section_discard_reasons"],
        "chunk_rejection_reasons": results["chunk_rejection_reasons"],
        "discarded_sections_detail": results["discarded_sections"],
        "rejected_chunks_detail": results["rejected_chunks"],
    }


def write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_filter(
    input_path: Path,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    restrict_to_section_ii: bool = True,
) -> dict[str, Any]:
    original_markdown = input_path.read_text(encoding="utf-8")
    results = filter_markdown_text(
        original_markdown,
        source_name=input_path.stem,
        restrict_to_section_ii=restrict_to_section_ii,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(results["final_markdown"], encoding="utf-8")

    report = build_report(
        input_path=input_path,
        output_path=output_path,
        results=results,
        original_markdown=original_markdown,
    )
    write_report(report_path, report)
    return report


def main() -> int:
    args = parse_args()
    input_path = args.input_path.resolve()
    output_path = args.output_path.resolve()
    report_path = args.report_path.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input Markdown file not found: {input_path}")

    print(f"Reading: {input_path}", flush=True)
    report = run_filter(
        input_path,
        output_path=output_path,
        report_path=report_path,
        restrict_to_section_ii=not args.keep_all_sections,
    )

    print(f"Filtered Markdown written to: {output_path}", flush=True)
    print(f"Audit report written to: {report_path}", flush=True)
    print(
        "Summary: "
        f"{report['original_char_count']:,} -> "
        f"{report['section_filtered_char_count']:,} chars after section filtering, "
        f"{report['final_chunk_char_count']:,} chars across "
        f"{report['kept_nodes']} kept chunks.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
