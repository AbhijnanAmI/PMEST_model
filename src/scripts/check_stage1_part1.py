"""
Quick manual sanity check for MetadataDecomposer (Stage 1, Part 1).
Run: python scripts/check_stage1_part1.py
"""

import json
import sys
from pathlib import Path

# Adjust this import path to match how your project is structured/installed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.preprocessing.cleaner import MetadataDecomposer


def main():
    field_mappings_path = "config/field_mappings.json"
    sample_record_path = "data/raw/sample_record.json"

    with open(sample_record_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    decomposer = MetadataDecomposer(field_mappings_path)
    result = decomposer.decompose(record, record_id="test_001")

    print("=" * 70)
    print(f"RECORD ID: {result.record_id}")
    print("=" * 70)

    print("\n--- TEXTUAL STREAM ---")
    for f in result.textual_stream:
        print(f"  [{f.field_name}] (salience={f.salience_weight}, bias={f.facet_bias})")
        print(f"    text: {f.text[:80]}{'...' if len(f.text) > 80 else ''}")

    print("\n--- STRUCTURAL STREAM ---")
    for f in result.structural_stream:
        status = "OK" if f.parsed_ok else "FAILED TO PARSE"
        print(f"  [{f.field_name}] (salience={f.salience_weight}, bias={f.facet_bias}) -> {status}")
        print(f"    raw:    {f.raw_value}")
        print(f"    parsed: {f.parsed_value}")

    print("\n--- UNMAPPED FIELDS (fell back to default rule) ---")
    if result.unmapped_fields:
        for name in result.unmapped_fields:
            print(f"  - {name}")
    else:
        print("  (none)")

    print("\n--- FULL DICT (as_dict) ---")
    print(json.dumps(result.as_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()