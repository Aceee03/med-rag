from __future__ import annotations

import copy
import hashlib
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser

from progress_utils import ProgressPrinter


CHECKPOINT_VERSION = 1
ENRICHMENT_PROMPT_VERSION = "v3"


def _ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _pickle_dump(path: Path, obj: Any) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def _pickle_load(path: Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_fingerprint(manifest_path: str | Path | None) -> str | None:
    if not manifest_path:
        return None
    path = Path(manifest_path)
    if not path.exists():
        return None
    payload = {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_source_manifest_map(manifest_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not manifest_path:
        return {}
    path = Path(manifest_path)
    if not path.exists():
        return {}

    raw = _json_load(path)
    documents = raw.get("documents", []) if isinstance(raw, dict) else []
    mapping: dict[str, dict[str, Any]] = {}
    for entry in documents:
        output_markdown = entry.get("output_markdown")
        if not output_markdown:
            continue
        output_path = Path(str(output_markdown))
        metadata = {
            "source_title": entry.get("source_title") or entry.get("title") or output_path.stem,
            "source_url": entry.get("source_url") or entry.get("url") or "",
            "authority": entry.get("authority") or "",
            "condition": entry.get("condition") or "",
            "source_format": entry.get("source_format") or entry.get("format") or "",
        }
        mapping[str(output_path.resolve()).lower()] = metadata
        mapping[output_path.name.lower()] = metadata
    return mapping


def _attach_source_metadata(items: list[Any], source_manifest_map: dict[str, dict[str, Any]]) -> None:
    if not source_manifest_map:
        return
    for item in items:
        metadata = getattr(item, "metadata", None)
        if metadata is None:
            continue
        file_path = metadata.get("file_path")
        file_name = metadata.get("file_name")
        candidates = []
        if file_path:
            candidates.append(str(Path(str(file_path)).resolve()).lower())
        if file_name:
            candidates.append(Path(str(file_name)).name.lower())
        if file_path:
            candidates.append(Path(str(file_path)).name.lower())
        for candidate in candidates:
            matched = source_manifest_map.get(candidate)
            if matched:
                metadata.update(matched)
                break


def _build_source_state(input_dir: str | Path) -> dict[str, Any]:
    root = Path(input_dir).resolve()
    markdown_files = sorted(path for path in root.rglob("*.md") if path.is_file())
    files = [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in markdown_files
    ]
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "input_dir": str(root),
        "file_count": len(files),
        "files": files,
        "fingerprint": fingerprint,
    }


def _build_nodes_hash(nodes: list[Any]) -> str:
    digester = hashlib.sha256()
    for node in nodes:
        text = getattr(node, "text", "") or ""
        metadata = getattr(node, "metadata", {}) or {}
        original_text = metadata.get("original_text", text)
        payload = {
            "node_id": getattr(node, "node_id", None),
            "file_name": metadata.get("file_name"),
            "file_path": metadata.get("file_path"),
            "header_path": metadata.get("header_path"),
            "text_sha": hashlib.sha256(str(original_text).encode("utf-8")).hexdigest(),
        }
        digester.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digester.hexdigest()


def load_or_build_markdown_nodes(
    input_dir: str = "./data/markdown_filtered",
    checkpoint_dir: str = "./checkpoints/pipeline",
    source_manifest_path: str | None = None,
    force_rebuild: bool = False,
) -> tuple[list[Any], list[Any]]:
    checkpoint_root = _ensure_dir(checkpoint_dir)
    documents_path = checkpoint_root / "documents.pkl"
    nodes_path = checkpoint_root / "nodes.pkl"
    meta_path = checkpoint_root / "nodes_meta.json"

    source_state = _build_source_state(input_dir)
    current_meta = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "builder": "load_or_build_markdown_nodes",
        "parser": "MarkdownNodeParser",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_state": source_state,
        "source_manifest_fingerprint": _manifest_fingerprint(source_manifest_path),
    }

    if (
        not force_rebuild
        and documents_path.exists()
        and nodes_path.exists()
        and meta_path.exists()
    ):
        saved_meta = _json_load(meta_path)
        if (
            saved_meta.get("checkpoint_version") == CHECKPOINT_VERSION
            and saved_meta.get("parser") == "MarkdownNodeParser"
            and saved_meta.get("source_state", {}).get("fingerprint")
            == source_state["fingerprint"]
            and saved_meta.get("source_manifest_fingerprint")
            == current_meta["source_manifest_fingerprint"]
        ):
            documents = _pickle_load(documents_path)
            nodes = _pickle_load(nodes_path)
            print(
                f"Loaded cached documents/nodes from {checkpoint_root} "
                f"({len(nodes)} nodes from {source_state['file_count']} files)."
            )
            return documents, nodes

    print(
        f"Building documents/nodes from {source_state['file_count']} markdown files "
        f"in {source_state['input_dir']}..."
    )
    reader = SimpleDirectoryReader(input_dir=str(Path(input_dir)))
    documents = reader.load_data()
    source_manifest_map = _load_source_manifest_map(source_manifest_path)
    _attach_source_metadata(documents, source_manifest_map)
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    _attach_source_metadata(nodes, source_manifest_map)

    _pickle_dump(documents_path, documents)
    _pickle_dump(nodes_path, nodes)
    _json_dump(meta_path, current_meta)

    print(
        f"Saved documents/nodes checkpoint to {checkpoint_root} "
        f"({len(nodes)} nodes)."
    )
    return documents, nodes


def _clean_doc_name(raw_value: Any) -> str:
    if not raw_value:
        return "Unknown Document"
    name = Path(str(raw_value)).name
    name = re.sub(r"\.filtered(?=\.md$)", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.(md|markdown|txt)$", "", name, flags=re.IGNORECASE)
    return name.replace("_", " ").strip() or "Unknown Document"


def _extract_section_title(node_text: str) -> str:
    for line in node_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("[CONTEXT:"):
            continue
        if stripped.startswith("#"):
            title = re.sub(r"^#+\s*", "", stripped)
        else:
            title = stripped
        title = re.sub(r"[*`_]+", "", title).strip(" -:/")
        if title:
            return title[:120]
    return "General Section"


def _strip_existing_context(node_text: str) -> str:
    if not node_text.startswith("[CONTEXT:"):
        return node_text

    lines = node_text.splitlines()
    keep_from = 0
    for idx, line in enumerate(lines):
        if idx == 0 and line.startswith("[CONTEXT:"):
            continue
        if keep_from == 0 and line.strip() == "":
            continue
        keep_from = idx
        break
    return "\n".join(lines[keep_from:]).lstrip()


def _fallback_context(source_doc: str, section_title: str) -> str:
    if section_title and section_title != "General Section":
        return (
            f"This passage is from {source_doc} and focuses on "
            f"{section_title} in a mental health reference."
        )
    return f"This passage is from {source_doc} and provides mental health reference context."


def _build_enrichment_prompt(source_doc: str, section_title: str, excerpt: str) -> str:
    return f"""You are labeling chunks for mental-health retrieval and graph extraction.
Document: {source_doc}
Section: {section_title}

Write exactly one sentence, under 25 words, that states the disorder/topic and the subtopic of this chunk.
Use the excerpt faithfully. If the disorder is unclear, describe the clinical domain instead of guessing.
Do not mention file names, markdown, or say 'this text', 'this chunk', or 'this passage'.

Excerpt:
{excerpt}

Sentence:"""


def _get_model_name(llm_client: Any) -> str:
    metadata = getattr(llm_client, "metadata", None)
    if metadata is not None:
        model_name = getattr(metadata, "model_name", None)
        if model_name:
            return str(model_name)
    for attr in ("model", "model_name"):
        value = getattr(llm_client, attr, None)
        if value:
            return str(value)
    return "unknown-model"


def _load_enrichment_checkpoint(
    checkpoint_dir: Path,
    nodes_hash: str,
    model_name: str,
    prompt_version: str,
    node_count: int,
) -> list[Any] | None:
    state_path = checkpoint_dir / "enriched_nodes.pkl"
    meta_path = checkpoint_dir / "enriched_nodes_meta.json"
    if not state_path.exists() or not meta_path.exists():
        return None

    meta = _json_load(meta_path)
    if meta.get("checkpoint_version") != CHECKPOINT_VERSION:
        return None
    if meta.get("nodes_hash") != nodes_hash:
        return None
    if meta.get("model_name") != model_name:
        return None
    if meta.get("prompt_version") != prompt_version:
        return None
    if meta.get("node_count") != node_count:
        return None

    enriched_nodes = _pickle_load(state_path)
    if not isinstance(enriched_nodes, list):
        return None
    if len(enriched_nodes) > node_count:
        return None
    return enriched_nodes


def load_enriched_nodes_checkpoint(checkpoint_dir: str = "./checkpoints/pipeline") -> list[Any]:
    checkpoint_root = Path(checkpoint_dir)
    state_path = checkpoint_root / "enriched_nodes.pkl"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing enrichment checkpoint: {state_path}")
    enriched_nodes = _pickle_load(state_path)
    if not isinstance(enriched_nodes, list):
        raise TypeError(f"Unexpected enrichment checkpoint contents in {state_path}")
    return enriched_nodes


def _save_enrichment_checkpoint(
    checkpoint_dir: Path,
    enriched_nodes: list[Any],
    nodes_hash: str,
    model_name: str,
    prompt_version: str,
    node_count: int,
) -> None:
    state_path = checkpoint_dir / "enriched_nodes.pkl"
    meta_path = checkpoint_dir / "enriched_nodes_meta.json"
    _pickle_dump(state_path, enriched_nodes)
    _json_dump(
        meta_path,
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "node_count": node_count,
            "saved_count": len(enriched_nodes),
            "nodes_hash": nodes_hash,
            "model_name": model_name,
            "prompt_version": prompt_version,
        },
    )


def enrich_nodes_with_context(
    nodes: list[Any],
    llm_client: Any,
    preview_chars: int = 700,
    progress_every: int = 25,
    checkpoint_dir: str = "./checkpoints/pipeline",
    resume: bool = True,
    force_rebuild: bool = False,
    save_every: int = 25,
    prompt_version: str = ENRICHMENT_PROMPT_VERSION,
) -> list[Any]:
    checkpoint_root = _ensure_dir(checkpoint_dir)
    model_name = _get_model_name(llm_client)
    progress_every = max(1, int(progress_every))
    save_every = max(1, int(save_every))
    node_count = len(nodes)
    nodes_hash = _build_nodes_hash(nodes)

    enriched_nodes: list[Any] = []
    start_idx = 0
    if resume and not force_rebuild:
        cached = _load_enrichment_checkpoint(
            checkpoint_root,
            nodes_hash=nodes_hash,
            model_name=model_name,
            prompt_version=prompt_version,
            node_count=node_count,
        )
        if cached is not None:
            enriched_nodes = cached
            start_idx = len(cached)
            if start_idx >= node_count:
                print(
                    f"Loaded completed enrichment checkpoint from {checkpoint_root} "
                    f"({start_idx}/{node_count} nodes, model={model_name})."
                )
                return cached
            print(
                f"Resuming enrichment from node {start_idx + 1}/{node_count} "
                f"using checkpoint in {checkpoint_root}."
            )

    prompt_cache: dict[tuple[str, str, str], str] = {}
    if start_idx == 0:
        print(f"Starting enrichment for {node_count} nodes...")
    else:
        print(f"Continuing enrichment with {start_idx} nodes already saved.")

    progress = ProgressPrinter(
        label="ENRICH",
        total=node_count,
        every=progress_every,
        start_count=start_idx,
    )
    if start_idx > 0:
        progress.update(start_idx, extra="resume point", force=True)

    for idx in range(start_idx, node_count):
        node = nodes[idx]
        enriched_node = (
            node.model_copy(deep=True)
            if hasattr(node, "model_copy")
            else copy.deepcopy(node)
        )

        base_text = enriched_node.metadata.get("original_text", enriched_node.text)
        base_text = _strip_existing_context(base_text)

        source_doc = _clean_doc_name(
            enriched_node.metadata.get("source_title")
            or enriched_node.metadata.get("source_label")
            or enriched_node.metadata.get("file_name")
            or enriched_node.metadata.get("file_path")
        )
        source_authority = str(enriched_node.metadata.get("authority") or "").strip()
        if source_authority:
            source_doc = f"{source_authority}: {source_doc}"
        header_path = enriched_node.metadata.get("header_path")
        section_title = (
            header_path
            if header_path and header_path != "/"
            else _extract_section_title(base_text)
        )
        citation = (
            f"{source_doc} — {section_title}"
            if section_title and section_title != "General Section"
            else source_doc
        )

        excerpt = re.sub(r"\s+", " ", base_text).strip()[:preview_chars]
        cache_key = (source_doc, section_title, excerpt)

        if cache_key in prompt_cache:
            context_tag = prompt_cache[cache_key]
        else:
            prompt = _build_enrichment_prompt(source_doc, section_title, excerpt)
            try:
                response = llm_client.complete(prompt)
                context_tag = " ".join((response.text or "").split())
            except Exception as exc:
                context_tag = _fallback_context(source_doc, section_title)
                print(f"  [fallback] Node {idx + 1}: {exc}")

            if not context_tag:
                context_tag = _fallback_context(source_doc, section_title)
            if context_tag[-1:] not in ".!?":
                context_tag += "."
            prompt_cache[cache_key] = context_tag

        enriched_node.metadata["original_text"] = base_text
        enriched_node.metadata["context_tag"] = context_tag
        enriched_node.metadata["section_title"] = section_title
        enriched_node.metadata["source_label"] = source_doc
        enriched_node.metadata["citation"] = citation
        enriched_node.metadata["context_enriched"] = True
        enriched_node.text = f"[CONTEXT: {context_tag}]\n\n{base_text}"

        enriched_nodes.append(enriched_node)

        current_count = idx + 1
        progress.update(
            current_count,
            extra=f"{citation}: {context_tag[:80]}",
        )

        processed_since_resume = current_count - start_idx
        if processed_since_resume % save_every == 0 or current_count == node_count:
            _save_enrichment_checkpoint(
                checkpoint_root,
                enriched_nodes=enriched_nodes,
                nodes_hash=nodes_hash,
                model_name=model_name,
                prompt_version=prompt_version,
                node_count=node_count,
            )

    print(
        f"Saved enrichment checkpoint to {checkpoint_root} "
        f"({len(enriched_nodes)}/{node_count} nodes)."
    )
    return enriched_nodes
