"""
Quick manual sanity check for Stage 2, Part 1 (span_extractor.py).
Chains Stage 1 (DualChannelMetadataEncoder -> X_fused) into KeyBERT
candidate proposal, character-offset alignment, and the Boundary +
Soft-Attention Pool + Span Length span representation.

Run: python scripts/check_stage2_part1.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.preprocessing.cleaner import MetadataDecomposer
from src.preprocessing.tokenization import build_field_aware_tokenizer_and_embedding
from src.models.encoders import DualChannelMetadataEncoder
from src.models.span_extractor import KeyBERTCandidateAligner, SpanCandidateBuilder


def main():
    field_mappings_path = "config/field_mappings.json"
    default_config_path = "config/default_config.yaml"
    sample_record_path = "data/raw/sample_record.json"

    with open(default_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(sample_record_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    # ---- Stage 1: Part 1 -> Part 2 -> Part 3 ----
    decomposer = MetadataDecomposer(field_mappings_path)
    decomposed = decomposer.decompose(record, record_id="test_001")

    hidden_dim = 768
    tokenizer, embedding_layer = build_field_aware_tokenizer_and_embedding(
        field_mappings_path=field_mappings_path,
        pretrained_tokenizer_name="bert-base-uncased",
        hidden_dim=hidden_dim,
        max_sequence_length=cfg["encoder"]["max_sequence_length"],
    )

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
        stage1_output = encoder(decomposed)

    x_fused = stage1_output.token_embeddings.squeeze(0)  # (L, hidden_dim)
    print("=" * 70)
    print("STAGE 1 OUTPUT (recap)")
    print("=" * 70)
    print(f"  X_fused shape: {tuple(x_fused.shape)}")
    print(f"  total tokens: {len(stage1_output.source_field_per_token)}")

    # ---- Reconstruct the SAME concatenated text Channel A tokenized ----
    # (needed so KeyBERT and Stage 1's tokenizer operate on identical text)
    full_text = " ".join(f.text for f in decomposed.textual_stream)

    print(f"\n  Full concatenated text (first 150 chars):")
    print(f"  {full_text[:150]}...")

    # ---- Stage 2 Part 1a: KeyBERT candidate proposal + alignment ----
    print("\n" + "=" * 70)
    print("STAGE 2 PART 1a: KeyBERT candidate proposal + char-offset alignment")
    print("=" * 70)

    kw_cfg = cfg["keyword_extraction"]
    aligner = KeyBERTCandidateAligner(
        keybert_model_name=kw_cfg["base_embedding_model"],
        stage1_tokenizer=tokenizer.tokenizer,
        ngram_range=tuple(kw_cfg["ngram_range"]),
        top_n=kw_cfg["top_n"],
        use_mmr=kw_cfg["use_mmr"],
        diversity=kw_cfg["diversity"],
        stopwords=kw_cfg["stopwords"],
    )

    aligned_spans = aligner.propose_and_align(full_text)

    ok_count = sum(1 for s in aligned_spans if s.alignment_ok)
    print(f"  candidates proposed: {len(aligned_spans)}")
    print(f"  candidates successfully aligned: {ok_count}")
    print(f"  candidates FAILED alignment: {len(aligned_spans) - ok_count}")

    print("\n  Per-candidate detail:")
    for s in aligned_spans:
        status = "OK" if s.alignment_ok else "FAILED"
        print(f"    [{status}] '{s.phrase}' (keybert_score={s.keybert_score:.4f}) "
              f"char=({s.char_start},{s.char_end}) token=({s.token_start},{s.token_end})")

    # ---- Stage 2 Part 1b: build span representations ----
    print("\n" + "=" * 70)
    print("STAGE 2 PART 1b: Boundary + Soft-Attention Pool + Span Length")
    print("=" * 70)

    builder = SpanCandidateBuilder(hidden_dim=hidden_dim, max_span_length=16)
    builder.eval()

    with torch.no_grad():
        representations = builder.build_all(x_fused, aligned_spans)

    print(f"  successfully built representations: {len(representations)} "
          f"(should equal 'candidates successfully aligned' above)")

    for rep in representations:
        print(f"\n  --- '{rep.span.phrase}' "
              f"(tokens {rep.span.token_start}-{rep.span.token_end}, "
              f"length={rep.span.token_end - rep.span.token_start + 1}) ---")
        print(f"    boundary_vector shape:  {tuple(rep.boundary_vector.shape)}  (expect {2*hidden_dim},)")
        print(f"    pooled_vector shape:    {tuple(rep.pooled_vector.shape)}    (expect {hidden_dim},)")
        print(f"    length_embedding shape: {tuple(rep.length_embedding.shape)} (expect {hidden_dim},)")
        print(f"    combined_vector shape:  {tuple(rep.combined_vector.shape)}  (expect {4*hidden_dim},)")
        print(f"    combined_vector first 5 dims: {rep.combined_vector[:5].tolist()}")

    print("\nDone. Check: alignment success rate, token spans look sensible "
          "relative to the phrase text, and all shapes match expectations.")


if __name__ == "__main__":
    main()