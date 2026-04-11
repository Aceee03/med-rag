from __future__ import annotations

import argparse
from pathlib import Path

from graphrag_pipeline import merge_graph_checkpoints


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge two cleaned GraphRAG checkpoints into one consistent checkpoint."
    )
    parser.add_argument("--primary-checkpoint-dir", required=True)
    parser.add_argument("--secondary-checkpoint-dir", required=True)
    parser.add_argument("--target-checkpoint-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    merged_artifacts, _, merge_report = merge_graph_checkpoints(
        args.primary_checkpoint_dir,
        args.secondary_checkpoint_dir,
        args.target_checkpoint_dir,
    )

    print("=" * 72)
    print("MERGED GRAPH CHECKPOINT")
    print("=" * 72)
    print(f"Primary:   {Path(args.primary_checkpoint_dir)}")
    print(f"Secondary: {Path(args.secondary_checkpoint_dir)}")
    print(f"Target:    {Path(args.target_checkpoint_dir)}")
    print(f"Entities:  {merged_artifacts.entity_count}")
    print(f"Relations: {merged_artifacts.relation_count}")
    print()
    print(f"Added entities from secondary: {merge_report['added_entities']}")
    print(f"Overlapping entities:          {merge_report['overlapping_entities']}")
    print(f"Type conflicts kept from primary: {merge_report['type_conflicts']}")
    print(f"Added relations from secondary: {merge_report['added_relations']}")
    print(f"Merged duplicate relations:     {merge_report['merged_relations']}")
    print(f"Chunk citation collisions:      {merge_report['chunk_citation_collisions']}")
    cleanup_report = merge_report["cleanup_report"]
    print(
        "Cleanup after merge: "
        f"{cleanup_report['original_entities']} -> {cleanup_report['cleaned_entities']} entities, "
        f"{cleanup_report['original_relations']} -> {cleanup_report['cleaned_relations']} relations"
    )

    if merge_report["type_conflict_entities"]:
        print("\nType conflicts:")
        for item in merge_report["type_conflict_entities"]:
            print(
                f"- {item['name']}: kept {item['primary_label']} "
                f"over {item['secondary_label']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
