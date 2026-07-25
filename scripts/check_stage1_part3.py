"""
Quick manual sanity check for Stage 1, Part 3 (graph_builder.py + encoders.py),
including verification that salience_weight now flows through the pipeline
via SalienceVocab / salience_ids / salience_embedding.

Run: python scripts/check_stage1_part3.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.preprocessing.cleaner import MetadataDecomposer
from src.preprocessing.tokenization import build_field_aware_tokenizer_and_embedding
from src.preprocessing.graph_builder import MetadataGraphBuilder
from src.models.encoders import DualChannelMetadataEncoder


def main():
    field_mappings_path = "config/field_mappings.json"
    default_config_path = "config/default_config.yaml"
    sample_record_path = "data/raw/sample_record.json"

    with open(default_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(sample_record_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    # ---- Part 1 ----
    decomposer = MetadataDecomposer(field_mappings_path)
    decomposed = decomposer.decompose(record, record_id="test_001")

    print("=" * 70)
    print("PART 1: textual/structural split + salience_weight per field")
    print("=" * 70)
    for f in decomposed.textual_stream:
        print(f"  [textual]    {f.field_name:20s} salience_weight = {f.salience_weight}")
    for f in decomposed.structural_stream:
        print(f"  [structural] {f.field_name:20s} salience_weight = {f.salience_weight}")

    # ---- Part 2 ----
    hidden_dim = 768
    tokenizer, embedding_layer = build_field_aware_tokenizer_and_embedding(
        field_mappings_path=field_mappings_path,
        pretrained_tokenizer_name="bert-base-uncased",
        hidden_dim=hidden_dim,
        max_sequence_length=cfg["encoder"]["max_sequence_length"],
    )

    print(f"\nSalience vocab size: {len(tokenizer.salience_vocab)}")
    print(f"Salience vocab entries: {tokenizer.salience_vocab.itos}")

    # ---- Part 2 salience check: tokenize each textual field, confirm
    # salience_ids are constant per field and map to the right vocab id ----
    print("\n" + "=" * 70)
    print("PART 2 CHECK: salience_ids per textual field")
    print("=" * 70)
    tokenized_record = tokenizer.tokenize_record(decomposed)
    for tf, orig in zip(tokenized_record.fields, decomposed.textual_stream):
        unique_ids = set(tf.salience_ids.tolist())
        expected_id = tokenizer.salience_vocab.id_for_salience(orig.salience_weight)
        ok = unique_ids == {expected_id}
        print(f"  {tf.field_name:20s} salience_weight='{orig.salience_weight}' "
              f"-> salience_id={expected_id}  "
              f"constant_across_tokens={ok}  (found ids: {unique_ids})")

    # ---- Part 3a: graph construction, now check node-level salience too ----
    print("\n" + "=" * 70)
    print("PART 3a: Channel B graph construction + salience on structural nodes")
    print("=" * 70)

    graph_builder = MetadataGraphBuilder(
        tokenizer=tokenizer, embedding_layer=embedding_layer, hidden_dim=hidden_dim
    )
    graph = graph_builder.build(decomposed)

    print(f"  num_nodes: {graph.num_nodes}")
    print(f"  num_edges: {graph.num_edges}  (expect num_nodes*(num_nodes-1))")

    print("\n  Per-node feature + salience check:")
    for i, name in enumerate(graph.node_field_names):
        attr = decomposed.structural_stream[i]
        vec = graph.node_features[i]
        is_zero = torch.allclose(vec, torch.zeros_like(vec))
        print(f"    [{i}] {name:20s} salience_weight='{attr.salience_weight}' "
              f"-> {'ZERO (numeric fallback, salience irrelevant here)' if is_zero else 'EMBEDDED (salience applied)'}")

    # ---- Part 3b: full dual-channel encoder ----
    print("\n" + "=" * 70)
    print("PART 3b: Dual-Channel Encoder (Channel A + Channel B fused)")
    print("=" * 70)

    encoder = DualChannelMetadataEncoder(
        tokenizer=tokenizer,
        embedding_layer=embedding_layer,
        hidden_dim=hidden_dim,
        transformer_layers=2,
        transformer_heads=12,
        gat_layers=2,
        gat_heads=4,
        use_graph_channel=cfg["encoder"]["use_graph_channel"],
    )
    encoder.eval()

    with torch.no_grad():
        output = encoder(decomposed)

    print(f"  use_graph_channel (from config): {cfg['encoder']['use_graph_channel']}")
    print(f"  graph_used (actually applied):   {output.graph_used}")
    print(f"  final token_embeddings shape: {tuple(output.token_embeddings.shape)}")
    print(f"  total tokens across all fields: {len(output.source_field_per_token)}")

    print("\n  Field boundaries in the concatenated sequence:")
    prev_field = None
    for idx, fname in enumerate(output.source_field_per_token):
        if fname != prev_field:
            print(f"    position {idx:4d} -> start of field '{fname}'")
            prev_field = fname

    # ---- Salience ablation: does salience actually change the embedding? ----
    # Same text, same field_name, same position -- only salience_weight differs.
    # If salience_embedding is wired correctly, outputs MUST differ.
    print("\n" + "=" * 70)
    print("SALIENCE ABLATION CHECK: identical text, different salience_weight")
    print("=" * 70)

    test_text = "identical probe text for salience check"
    tok_high = tokenizer.tokenize_field("Title", test_text, salience_weight="high")
    tok_admin = tokenizer.tokenize_field("Title", test_text, salience_weight="administrative")

    with torch.no_grad():
        embed_high = embedding_layer(
            input_ids=tok_high.input_ids.unsqueeze(0),
            position_ids=tok_high.position_ids.unsqueeze(0),
            field_type_ids=tok_high.field_type_ids.unsqueeze(0),
            salience_ids=tok_high.salience_ids.unsqueeze(0),
        )
        embed_admin = embedding_layer(
            input_ids=tok_admin.input_ids.unsqueeze(0),
            position_ids=tok_admin.position_ids.unsqueeze(0),
            field_type_ids=tok_admin.field_type_ids.unsqueeze(0),
            salience_ids=tok_admin.salience_ids.unsqueeze(0),
        )

    identical_output = torch.allclose(embed_high, embed_admin)
    print(f"  same input_ids/position_ids/field_type_ids, salience='high' vs 'administrative'")
    print(f"  embeddings identical: {identical_output}  "
          f"(MUST be False -- if True, salience_embedding is not being applied)")
    print(f"  embed_high  first 5 dims: {embed_high[0, 0, :5].tolist()}")
    print(f"  embed_admin first 5 dims: {embed_admin[0, 0, :5].tolist()}")

    print("\nDone. Confirm: salience_ids constant per field in Part 2 check, "
          "salience vocab entries look right, and the ablation check above "
          "shows 'embeddings identical: False'.")


if __name__ == "__main__":
    main()