"""
Quick manual sanity check for Stage 1, Part 2 (tokenization.py).
Chains Part 1 (MetadataDecomposer) -> Part 2 (FieldAwareTokenizer + FieldAwareEmbedding).

Run: python scripts/check_stage1_part2.py
"""
"""
⚠️ DEPRECATED - This file is kept for legacy compatibility only.

This is the OLD version of this module. The NEW version is in 'check_stage1_part3.py'.
This file is scheduled for removal and will be deleted in a future commit.
"""

import warnings

warnings.warn(
    "module_name_deprecated.py is deprecated. Import 'module_name' instead.",
    DeprecationWarning,
    stacklevel=2
)

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.preprocessing.cleaner import MetadataDecomposer
from src.preprocessing.tokenization import build_field_aware_tokenizer_and_embedding


def main():
    field_mappings_path = "config/field_mappings.json"
    sample_record_path = "data/raw/sample_record.json"

    # ---- Step 1: reuse Part 1 to decompose the record ----
    with open(sample_record_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    decomposer = MetadataDecomposer(field_mappings_path)
    decomposed = decomposer.decompose(record, record_id="test_001")

    print("=" * 70)
    print("PART 1 OUTPUT (textual_stream fields feeding into Part 2)")
    print("=" * 70)
    for f in decomposed.textual_stream:
        print(f"  - {f.field_name}: {f.text[:60]}{'...' if len(f.text) > 60 else ''}")

    # ---- Step 2: build tokenizer + embedding layer ----
    tokenizer, embedding_layer = build_field_aware_tokenizer_and_embedding(
        field_mappings_path=field_mappings_path,
        pretrained_tokenizer_name="bert-base-uncased",
        hidden_dim=768,
        max_sequence_length=512,
    )

    print(f"\nField-Type vocab size: {len(tokenizer.field_type_vocab)}")
    print(f"Field-Type vocab entries: {tokenizer.field_type_vocab.itos}")

    # ---- Step 3: tokenize the whole record ----
    tokenized_record = tokenizer.tokenize_record(decomposed)

    print("\n" + "=" * 70)
    print("PART 2 OUTPUT — per field")
    print("=" * 70)

    for tf in tokenized_record.fields:
        print(f"\n--- Field: {tf.field_name} ---")
        print(f"  seq_len: {tf.input_ids.shape[0]}")
        print(f"  tokens: {tf.tokens}")
        print(f"  input_ids: {tf.input_ids.tolist()}")
        print(f"  position_ids: {tf.position_ids.tolist()}")
        print(f"  field_type_ids: {tf.field_type_ids.tolist()} "
              f"(all should be same id -> {tf.field_type_ids[0].item()})")

        # ---- Step 4: run the embedding layer on this field ----
        output = embedding_layer(
            input_ids=tf.input_ids.unsqueeze(0),
            position_ids=tf.position_ids.unsqueeze(0),
            field_type_ids=tf.field_type_ids.unsqueeze(0),
        )
        print(f"  embedding output shape: {tuple(output.shape)}  "
              f"(expect: [1, {tf.input_ids.shape[0]}, 768])")
        print(f"  first token embedding (first 5 dims): {output[0, 0, :5].tolist()}")

    print("\nDone. If shapes and field_type_ids look right, Part 2 is wired correctly.")


if __name__ == "__main__":
    main()