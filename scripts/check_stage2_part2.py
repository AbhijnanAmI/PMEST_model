"""
Quick manual sanity check for Stage 2, Part 2 (Informativeness Head,
Document Coverage Head, Raw Keyword Score, Dynamic Threshold + NMS).

Chains Stage 1 -> Stage 2 Part 1 (KeyBERT + alignment + span representation)
-> Stage 2 Part 2 (scoring heads + corpus IDF prior + NMS) end to end.

Run: python scripts/check_stage2_part2.py

NOTE: for the corpus IDF prior to be meaningful (not a flat constant),
data/raw/ should contain more than one sample record with genuinely
different vocabulary before running this.
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
from src.models.span_extractor import (
    KeyBERTCandidateAligner,
    SpanCandidateBuilder,
    SalienceInformativenessScorer,
    dynamic_threshold_and_nms,
)
from src.utils.corpus_stats import get_or_build_corpus_stats


def main():
    field_mappings_path = "config/field_mappings.json"
    default_config_path = "config/default_config.yaml"
    raw_dir = "data/raw"
    sample_record_path = "data/raw/sample_record.json"  # the record we run full pipeline on

    with open(default_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with open(sample_record_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    # ---- Corpus stats: build/load once, report corpus size up front ----
    print("=" * 70)
    print("CORPUS STATS (background IDF)")
    print("=" * 70)
    corpus_stats = get_or_build_corpus_stats(raw_dir=raw_dir, cache_path="data/vocab/corpus_stats.json")
    print(f"  total_docs in corpus: {corpus_stats.total_docs}")
    print(f"  unique tokens tracked: {len(corpus_stats.doc_freq)}")
    if corpus_stats.total_docs < 2:
        print("  WARNING: only 1 document in corpus -- IDF will be a flat constant,"
              " not yet discriminative. Add more files to data/raw/ for a meaningful signal.")

    # ---- Stage 1 ----
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
    full_text = " ".join(f.text for f in decomposed.textual_stream)

    print(f"\nSTAGE 1 recap: X_fused shape {tuple(x_fused.shape)}, "
          f"{len(stage1_output.source_field_per_token)} tokens")

    # ---- Stage 2 Part 1: KeyBERT candidates + alignment + span representation ----
    print("\n" + "=" * 70)
    print("STAGE 2 PART 1: candidate proposal + alignment + representation")
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
    ok_spans = [s for s in aligned_spans if s.alignment_ok]
    print(f"  candidates proposed: {len(aligned_spans)}, aligned OK: {len(ok_spans)}")

    builder = SpanCandidateBuilder(hidden_dim=hidden_dim, max_span_length=16)
    builder.eval()
    with torch.no_grad():
        representations = builder.build_all(x_fused, aligned_spans)
    print(f"  span representations built: {len(representations)}")

    # ---- Stage 2 Part 2: scoring heads ----
    print("\n" + "=" * 70)
    print("STAGE 2 PART 2: Informativeness + Coverage + Raw Keyword Score")
    print("=" * 70)

    scorer = SalienceInformativenessScorer(hidden_dim=hidden_dim, mlp_hidden_dim=256)
    scorer.eval()

    with torch.no_grad():
        scored = scorer.score_all(representations, x_fused, corpus_stats)

    # sort by raw score desc for readability, purely for display -- NMS below re-sorts internally
    scored_display = sorted(scored, key=lambda c: c.raw_keyword_score, reverse=True)

    print(f"\n  {'Phrase':45s} {'IDF prior':>10s} {'Inform.':>9s} {'Coverage':>9s} {'RawScore':>9s}")
    print(f"  {'-'*45} {'-'*10} {'-'*9} {'-'*9} {'-'*9}")
    for c in scored_display:
        print(f"  {c.representation.span.phrase[:45]:45s} "
              f"{c.idf_prior:>10.4f} "
              f"{c.informativeness_score:>9.4f} "
              f"{c.coverage_score:>9.4f} "
              f"{c.raw_keyword_score:>9.4f}")

    # ---- Dynamic threshold + NMS ----
    print("\n" + "=" * 70)
    print("DYNAMIC THRESHOLD + NON-MAXIMUM SUPPRESSION")
    print("=" * 70)

    scores = [c.raw_keyword_score for c in scored]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_score = variance ** 0.5
    print(f"  candidate pool size: {len(scored)}")
    print(f"  mean raw_keyword_score: {mean_score:.4f}")
    print(f"  std  raw_keyword_score: {std_score:.4f}")
    print(f"  dynamic threshold (mean + 1.0*std): {mean_score + std_score:.4f}")

    final_keywords = dynamic_threshold_and_nms(
        scored, std_multiplier=1.0, top_k_cap=10, overlap_threshold=0.5
    )

    print(f"\n  FINAL retained keywords after threshold + NMS + top-K cap: {len(final_keywords)}")
    print(f"  {'Phrase':45s} {'Tokens':>10s} {'RawScore':>9s}")
    print(f"  {'-'*45} {'-'*10} {'-'*9}")
    for c in final_keywords:
        span = c.representation.span
        print(f"  {span.phrase[:45]:45s} "
              f"({span.token_start:>3d},{span.token_end:>3d}) "
              f"{c.raw_keyword_score:>9.4f}")

    # ---- Sanity checks worth eyeballing ----
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    # 1. No two final keywords should overlap significantly (NMS should have caught this)
    overlap_violations = 0
    for i in range(len(final_keywords)):
        for j in range(i + 1, len(final_keywords)):
            a = final_keywords[i].representation.span
            b = final_keywords[j].representation.span
            inter_start = max(a.token_start, b.token_start)
            inter_end = min(a.token_end, b.token_end)
            intersection = max(0, inter_end - inter_start + 1)
            union = max(a.token_end, b.token_end) - min(a.token_start, b.token_start) + 1
            iou = intersection / union if union > 0 else 0.0
            if iou >= 0.5:
                overlap_violations += 1
                print(f"  WARNING: overlap violation between "
                      f"'{a.phrase}' and '{b.phrase}' (IoU={iou:.2f})")
    if overlap_violations == 0:
        print("  OK: no overlapping token ranges among final keywords (NMS working correctly)")

    # 2. Final count should never exceed top_k_cap
    print(f"  OK: final count ({len(final_keywords)}) <= top_k_cap (10): {len(final_keywords) <= 10}")

    # 3. All final scores should be >= the dynamic threshold OR be the single fallback survivor
    below_threshold = [c for c in final_keywords if c.raw_keyword_score <= (mean_score + std_score)]
    if below_threshold and len(final_keywords) == 1:
        print(f"  NOTE: single fallback survivor kept despite being below threshold "
              f"(expected when threshold is stricter than any candidate) -- OK")
    elif below_threshold:
        print(f"  WARNING: {len(below_threshold)} final candidates are below the computed "
              f"threshold unexpectedly")
    else:
        print(f"  OK: all final candidates exceed the dynamic threshold")

    print("\nDone.")


if __name__ == "__main__":
    main()