"""
PMEST-Net :: Stage 2, Part 1 — Neural Span Candidate Generation

Given Stage 1's Field-Aware Structural Token Embedding Matrix (X_fused,
shape L x d), and a set of candidate spans proposed by KeyBERT over the
SAME raw concatenated text that Channel A tokenized, this module builds
a candidate representation vector per span by pooling three components:

  - Boundary Vector:      concat(X_fused[i], X_fused[j])
  - Soft-Attention Pool:  self-attention over X_fused[i:j+1]
  - Span Length Embedding: learned lookup keyed by span length (j-i+1)

KeyBERT proposes candidates using its own embedding model and its own
tokenizer, operating on character offsets into the raw string. Stage 1's
X_fused is indexed by Stage 1's tokenizer's token positions. These two
coordinate systems do NOT agree token-for-token, so alignment is done via
character-offset mapping: Stage 1's tokenizer's offset_mapping is used to
translate a KeyBERT candidate's character span into a Stage 1 token span
(i, j).

NOTE: The Soft-Attention Pool and Span Length Embedding are learned,
untrained nn.Module components (consistent with Stage 1's embedding
tables) -- they are correctly wired but not yet semantically meaningful
until Stage 5 multi-task training. The Informativeness Head, Document
Coverage Head, and NMS suppression described in the architecture spec's
Stage 2 Part 2 are NOT part of this module; they are built separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from keybert import KeyBERT
from transformers import PreTrainedTokenizerFast
from src.utils.corpus_stats import CorpusStats


import re

# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------

@dataclass
class AlignedCandidateSpan:
    """
    A KeyBERT-proposed candidate phrase, aligned to Stage 1's token
    coordinate system.
    """
    phrase: str                  # the candidate text, as KeyBERT returned it
    keybert_score: float         # KeyBERT's own cosine-similarity score (kept for audit/debug only)
    char_start: int               # character offset into the raw concatenated text
    char_end: int
    token_start: int              # i -- inclusive, index into X_fused's sequence dim
    token_end: int                # j -- inclusive, index into X_fused's sequence dim
    alignment_ok: bool            # False if offset mapping failed to find a clean token boundary


@dataclass
class SpanCandidateRepresentation:
    """Final pooled representation for one candidate span, ready for the
    (separately-built) Informativeness / Document Coverage scoring heads."""
    span: AlignedCandidateSpan
    boundary_vector: torch.Tensor     # (2*d,)
    pooled_vector: torch.Tensor       # (d,) -- soft-attention pooled span content
    length_embedding: torch.Tensor    # (d,)
    combined_vector: torch.Tensor     # (2*d + d + d,) = (4*d,) concatenation of the above three


# ----------------------------------------------------------------------------
# KeyBERT candidate proposal + character-offset alignment
# ----------------------------------------------------------------------------

class KeyBERTCandidateAligner:
    """
    Runs KeyBERT over the full concatenated record text (matching Channel
    A's concatenation order) to propose candidate keyword spans, then
    aligns each candidate's character span onto Stage 1's tokenizer's
    token boundaries via offset_mapping.

    KeyBERT's own score is retained for audit purposes only -- it is NOT
    used as the final keyword score. Scoring is deferred to the
    Informativeness / Document Coverage heads (built separately).
    """

    def __init__(
        self,
        keybert_model_name: str,
        stage1_tokenizer: PreTrainedTokenizerFast,
        ngram_range: tuple[int, int] = (1, 5),
        top_n: int = 20,
        use_mmr: bool = True,
        diversity: float = 0.5,
        stopwords: str | None = "english",
    ):
        self.keybert = KeyBERT(model=keybert_model_name)
        self.stage1_tokenizer = stage1_tokenizer
        self.ngram_range = ngram_range
        self.top_n = top_n
        self.use_mmr = use_mmr
        self.diversity = diversity
        self.stopwords = stopwords


    def _find_char_span(self, full_text: str, phrase: str, search_start: int = 0) -> tuple[int, int] | None:
        """
        KeyBERT does not return character offsets natively, and its displayed
        phrase may have stopwords stripped even though the original text is
        contiguous (e.g. "Bread Making: A Comprehensive Study" -> KeyBERT
        returns "bread making comprehensive study", dropping "A" and ":").

        We therefore search for the phrase's surviving words IN ORDER,
        allowing arbitrary punctuation/whitespace/stopword text between
        them, rather than requiring an exact literal substring match.
        A true non-contiguous candidate (words KeyBERT stitched together
        from genuinely distant parts of the document) will still fail --
        which is correct behavior, since such a "span" has no valid
        (i, j) token range to align to in the first place.
        """
        words = phrase.split()
        if not words:
            return None

        # Build a regex that matches each word (case-insensitive), allowing
        # up to ~40 characters of "filler" (stopwords/punctuation) between
        # consecutive surviving words -- generous enough to bridge a
        # stripped stopword or two, tight enough to reject a genuinely
        # non-contiguous, scattered candidate.
        escaped_words = [re.escape(w) for w in words]
        pattern = r"\b" + escaped_words[0] + r"\b"
        for w in escaped_words[1:]:
            pattern += r".{0,40}?\b" + w + r"\b"

        match = re.search(pattern, full_text[search_start:], flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            return None

        return (search_start + match.start(), search_start + match.end())

    def propose_and_align(self, full_text: str) -> list[AlignedCandidateSpan]:
        # ---- Step 1: KeyBERT candidate proposal (its own embedder, its own tokenizer) ----
        raw_candidates = self.keybert.extract_keywords(
            full_text,
            keyphrase_ngram_range=self.ngram_range,
            top_n=self.top_n,
            use_mmr=self.use_mmr,
            diversity=self.diversity,
            stop_words=self.stopwords,
        )  # list of (phrase, score) tuples

        # ---- Step 2: recover character offsets for each candidate phrase ----
        # Stage 1's tokenizer needs offset_mapping to translate a character
        # span into a token span, so we tokenize the SAME full_text once here.
        stage1_encoding = self.stage1_tokenizer(
            full_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.stage1_tokenizer.model_max_length,
        )
        offset_mapping = stage1_encoding["offset_mapping"]  # list of (char_start, char_end) per token

        aligned_spans: list[AlignedCandidateSpan] = []
        search_cursor = 0

        for phrase, score in raw_candidates:
            char_span = self._find_char_span(full_text, phrase, search_start=0)
            if char_span is None:
                # Could not locate this candidate's text in the source string at
                # all (can happen if KeyBERT normalizes whitespace/case in a way
                # that breaks a literal substring match). Skip rather than guess.
                aligned_spans.append(
                    AlignedCandidateSpan(
                        phrase=phrase, keybert_score=score,
                        char_start=-1, char_end=-1,
                        token_start=-1, token_end=-1,
                        alignment_ok=False,
                    )
                )
                continue

            char_start, char_end = char_span
            token_start, token_end = self._char_span_to_token_span(
                char_start, char_end, offset_mapping
            )

            alignment_ok = token_start is not None and token_end is not None
            aligned_spans.append(
                AlignedCandidateSpan(
                    phrase=phrase,
                    keybert_score=score,
                    char_start=char_start,
                    char_end=char_end,
                    token_start=token_start if alignment_ok else -1,
                    token_end=token_end if alignment_ok else -1,
                    alignment_ok=alignment_ok,
                )
            )

        return aligned_spans

    def _char_span_to_token_span(
        self,
        char_start: int,
        char_end: int,
        offset_mapping: list[tuple[int, int]],
    ) -> tuple[int | None, int | None]:
        """
        Maps a (char_start, char_end) span onto the inclusive token index
        range (token_start, token_end) in Stage 1's tokenizer's output,
        using its offset_mapping. Special tokens ([CLS], [SEP], padding)
        have offset (0, 0) and are skipped when searching for boundaries.

        A token is considered part of the span if its character range
        overlaps [char_start, char_end) at all -- this correctly captures
        partial subword overlaps at the span's edges.
        """
        token_start, token_end = None, None

        for tok_idx, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
            if tok_char_start == tok_char_end == 0:
                continue  # special token, e.g. [CLS]/[SEP]/pad -- has no real char span

            overlaps = tok_char_start < char_end and tok_char_end > char_start
            if overlaps:
                if token_start is None:
                    token_start = tok_idx
                token_end = tok_idx  # keep advancing; last overlapping token wins

        return token_start, token_end


# ----------------------------------------------------------------------------
# Span Length Embedding
# ----------------------------------------------------------------------------

class SpanLengthEmbedding(nn.Module):
    """
    Learnable embedding keyed by span length (in Stage 1 tokens), counteracting
    bias toward excessively long candidate strings by giving the model an
    explicit length signal rather than requiring it to infer length from
    boundary/pooled content alone.
    """

    def __init__(self, hidden_dim: int = 768, max_span_length: int = 16):
        super().__init__()
        # +1 for an overflow bucket absorbing any span longer than max_span_length,
        # so an unusually long candidate doesn't cause an index-out-of-range error.
        self.max_span_length = max_span_length
        self.length_embedding = nn.Embedding(max_span_length + 1, hidden_dim)

    def forward(self, span_length: int) -> torch.Tensor:
        bucket = min(span_length, self.max_span_length)
        length_id = torch.tensor(bucket, dtype=torch.long)
        return self.length_embedding(length_id)  # (hidden_dim,)


# ----------------------------------------------------------------------------
# Soft-Attention Pool over span-internal tokens
# ----------------------------------------------------------------------------

class SoftAttentionSpanPool(nn.Module):
    """
    Computes a single vector summarizing a span's internal semantic core
    via learned self-attention over the span's own tokens, i.e. an
    attention-weighted mean rather than a plain mean-pool. This lets the
    model learn to weight, e.g., a span's head noun more heavily than a
    determiner within the same span, rather than treating every token in
    the span as equally important.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        # A single learned query vector attends over the span's tokens --
        # this is a compact form of self-attention pooling (additive
        # attention), not a full multi-head Transformer block, since the
        # span-internal pooling task doesn't require the expressive power
        # of a full attention stack.
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, span_tokens: torch.Tensor) -> torch.Tensor:
        """
        span_tokens: (span_len, hidden_dim) -- the slice X_fused[i:j+1]
        Returns:     (hidden_dim,) -- attention-weighted pooled vector
        """
        keys = self.key_proj(span_tokens)                     # (span_len, d)
        scores = (keys @ self.query) / self.scale              # (span_len,)
        weights = torch.softmax(scores, dim=0)                 # (span_len,)
        pooled = (weights.unsqueeze(-1) * span_tokens).sum(dim=0)  # (d,)
        return pooled


# ----------------------------------------------------------------------------
# Top-level: builds the full candidate representation for one span
# ----------------------------------------------------------------------------

class SpanCandidateBuilder(nn.Module):
    """
    Combines Boundary Vector + Soft-Attention Pool + Span Length Embedding
    into the final per-candidate representation, per architecture spec
    Stage 2 Part 1.
    """

    def __init__(self, hidden_dim: int = 768, max_span_length: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_pool = SoftAttentionSpanPool(hidden_dim=hidden_dim)
        self.length_embedding = SpanLengthEmbedding(
            hidden_dim=hidden_dim, max_span_length=max_span_length
        )

    def forward(
        self, x_fused: torch.Tensor, span: AlignedCandidateSpan
    ) -> SpanCandidateRepresentation | None:
        """
        x_fused: (L, hidden_dim) -- Stage 1's output for ONE record
                 (batch dimension already squeezed out by the caller)
        span:    an AlignedCandidateSpan with alignment_ok = True
        """
        if not span.alignment_ok:
            return None

        i, j = span.token_start, span.token_end
        if i < 0 or j < 0 or j >= x_fused.shape[0] or i > j:
            return None  # defensive bounds check -- malformed alignment

        # --- Boundary Vector ---
        boundary_vector = torch.cat([x_fused[i], x_fused[j]], dim=-1)  # (2*d,)

        # --- Soft-Attention Pool Vector ---
        span_tokens = x_fused[i : j + 1]        # (span_len, d)
        pooled_vector = self.attention_pool(span_tokens)  # (d,)

        # --- Span Length Embedding ---
        span_length = (j - i + 1)
        length_embedding = self.length_embedding(span_length)  # (d,)

        # --- Combined candidate vector ---
        combined_vector = torch.cat(
            [boundary_vector, pooled_vector, length_embedding], dim=-1
        )  # (4*d,)

        return SpanCandidateRepresentation(
            span=span,
            boundary_vector=boundary_vector,
            pooled_vector=pooled_vector,
            length_embedding=length_embedding,
            combined_vector=combined_vector,
        )

    def build_all(
        self, x_fused: torch.Tensor, spans: list[AlignedCandidateSpan]
    ) -> list[SpanCandidateRepresentation]:
        results = []
        for span in spans:
            rep = self.forward(x_fused, span)
            if rep is not None:
                results.append(rep)
        return results
    # --- add these imports at the top ---


# ----------------------------------------------------------------------------
# Stage 2, Part 2 — Informativeness Head, Document Coverage Head, Raw Score
# ----------------------------------------------------------------------------

@dataclass
class ScoredCandidateSpan:
    """A candidate span after Part 2 scoring, ready for NMS."""
    representation: SpanCandidateRepresentation
    informativeness_score: float   # sigmoid output, in [0, 1]
    coverage_score: float          # sigmoid output, in [0, 1]
    idf_prior: float               # raw corpus-IDF prior fed into the Informativeness Head, kept for audit
    raw_keyword_score: float       # informativeness * coverage


class InformativenessHead(nn.Module):
    """
    Estimates how much unique information a candidate span carries relative
    to the background corpus. Combines the candidate's own combined_vector
    (from Part 1) with a corpus-derived IDF prior as an explicit auxiliary
    feature, since a purely learned-from-scratch, untrained MLP has no way
    to independently discover "rarity" without either training or an
    explicit statistical signal to lean on.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        # input_dim + 1: the extra +1 slot is the scalar IDF prior,
        # concatenated onto the candidate's combined_vector.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, combined_vector: torch.Tensor, idf_prior: float) -> torch.Tensor:
        idf_tensor = torch.tensor([idf_prior], dtype=combined_vector.dtype)
        x = torch.cat([combined_vector, idf_tensor], dim=-1)
        logit = self.mlp(x)
        return torch.sigmoid(logit).squeeze(-1)  # scalar in [0, 1]


class DocumentCoverageHead(nn.Module):
    """
    Measures how well a candidate span's vector aligns with the overall
    document-level embedding, i.e. how representative/central the
    candidate is to the record as a whole (as opposed to a tangential
    or noisy mention). The document-level embedding is the mean-pooled
    Channel A output (X_A), consistent with the architecture spec's
    "pooled Transformer states."
    """

    def __init__(self, span_dim: int, doc_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(span_dim + doc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, combined_vector: torch.Tensor, document_embedding: torch.Tensor) -> torch.Tensor:
        x = torch.cat([combined_vector, document_embedding], dim=-1)
        logit = self.mlp(x)
        return torch.sigmoid(logit).squeeze(-1)  # scalar in [0, 1]


class SalienceInformativenessScorer(nn.Module):
    """
    Top-level Stage 2 Part 2 module: applies both heads to every candidate
    span representation and computes the Raw Keyword Score as their
    element-wise product, per architecture spec.
    """

    def __init__(self, hidden_dim: int = 768, mlp_hidden_dim: int = 256):
        super().__init__()
        combined_dim = hidden_dim * 4  # matches SpanCandidateRepresentation.combined_vector

        self.informativeness_head = InformativenessHead(
            input_dim=combined_dim, hidden_dim=mlp_hidden_dim
        )
        self.coverage_head = DocumentCoverageHead(
            span_dim=combined_dim, doc_dim=hidden_dim, hidden_dim=mlp_hidden_dim
        )

    def score_all(
        self,
        representations: list[SpanCandidateRepresentation],
        x_fused: torch.Tensor,           # (L, hidden_dim) -- for document embedding
        corpus_stats: CorpusStats,
    ) -> list[ScoredCandidateSpan]:
        document_embedding = x_fused.mean(dim=0)  # (hidden_dim,) -- pooled Transformer states

        scored = []
        for rep in representations:
            idf_prior = corpus_stats.phrase_informativeness_score(rep.span.phrase)

            informativeness = self.informativeness_head(rep.combined_vector, idf_prior)
            coverage = self.coverage_head(rep.combined_vector, document_embedding)

            raw_score = (informativeness * coverage).item()

            scored.append(
                ScoredCandidateSpan(
                    representation=rep,
                    informativeness_score=informativeness.item(),
                    coverage_score=coverage.item(),
                    idf_prior=idf_prior,
                    raw_keyword_score=raw_score,
                )
            )
        return scored


# ----------------------------------------------------------------------------
# Dynamic Thresholding + Non-Maximum Suppression
# ----------------------------------------------------------------------------

def _token_overlap_ratio(a: AlignedCandidateSpan, b: AlignedCandidateSpan) -> float:
    """
    Intersection-over-union of two spans' token ranges [token_start, token_end].
    """
    inter_start = max(a.token_start, b.token_start)
    inter_end = min(a.token_end, b.token_end)
    intersection = max(0, inter_end - inter_start + 1)

    union_start = min(a.token_start, b.token_start)
    union_end = max(a.token_end, b.token_end)
    union = union_end - union_start + 1

    if union == 0:
        return 0.0
    return intersection / union


def dynamic_threshold_and_nms(
    scored_candidates: list[ScoredCandidateSpan],
    std_multiplier: float = 1.0,
    top_k_cap: int = 10,
    overlap_threshold: float = 0.5,
) -> list[ScoredCandidateSpan]:
    """
    Two-stage filtering, per architecture spec:

      1. Dynamic threshold: retain only candidates whose raw_keyword_score
         exceeds mean(scores) + std_multiplier * std(scores) for THIS
         record's own candidate pool (adapts per record rather than using
         a fixed global cutoff).

      2. Non-Maximum Suppression: sort surviving candidates by score
         descending; greedily keep a candidate only if its token-range
         overlap with every already-kept candidate is below
         overlap_threshold, discarding lower-scoring overlapping spans.

      3. top_k_cap is then applied as a hard ceiling on the final list
         length, regardless of how many candidates passed steps 1-2.
    """
    if not scored_candidates:
        return []

    scores = [c.raw_keyword_score for c in scored_candidates]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_score = variance ** 0.5

    threshold = mean_score + std_multiplier * std_score

    above_threshold = [c for c in scored_candidates if c.raw_keyword_score > threshold]

    # Fallback: if the statistical threshold is so strict nothing survives
    # (common with very small candidate pools), fall back to the single
    # best candidate rather than returning an empty result.
    if not above_threshold:
        above_threshold = [max(scored_candidates, key=lambda c: c.raw_keyword_score)]

    above_threshold.sort(key=lambda c: c.raw_keyword_score, reverse=True)

    kept: list[ScoredCandidateSpan] = []
    for candidate in above_threshold:
        overlaps_kept = any(
            _token_overlap_ratio(candidate.representation.span, kept_c.representation.span)
            >= overlap_threshold
            for kept_c in kept
        )
        if not overlaps_kept:
            kept.append(candidate)

    return kept[:top_k_cap]