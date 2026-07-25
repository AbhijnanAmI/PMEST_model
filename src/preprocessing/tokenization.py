"""
PMEST-Net :: Stage 1, Part 2 — Positional and Field-Aware Tokenization

Takes the DecomposedMetadata output of Part 1 (cleaner.py) and produces
per-token embeddings composed of three parts:

  - Token Embedding:      standard subword semantic representation
  - Position Embedding:   absolute positional index within the field
  - Field-Type Embedding: trainable structural identifier (Field_Title,
                           Field_Abstract, Field_Publisher, ...)

NOTE: Field-Type and Position embeddings are randomly initialized here.
They only become meaningful once trained via pipelines/train.py using
multi_task_loss.py. In zero_shot mode (see config/default_config.yaml),
this module is not on the active inference path — Stage 2 zero-shot
(KeyBERT) consumes raw text directly. This module exists so the trained
pipeline can be switched on later without restructuring.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.preprocessing.cleaner import DecomposedMetadata


# ----------------------------------------------------------------------------
# Field-type vocabulary
# ----------------------------------------------------------------------------

# Every field name that can appear in field_mappings.json must map to a
# Field-Type id here. Special ids handle padding and anything encountered
# that isn't in field_mappings.json (routed through unmapped_field_default
# upstream in Part 1, but still needs a concrete Field-Type id here).
class SalienceVocab:
    """
    Maps salience_weight strings (as set in field_mappings.json / cleaner.py)
    to Salience embedding ids. Fixed, small vocabulary — unlike FieldTypeVocab
    this isn't built dynamically from the JSON, since the set of possible
    salience levels is closed (high, medium, administrative, low), but it
    still validates against whatever levels actually appear in your JSON.
    """

    PAD = "[SALIENCE_PAD]"
    UNKNOWN = "[SALIENCE_UNKNOWN]"

    # Fixed canonical levels — extend this list if field_mappings.json
    # ever introduces a new salience_weight value not covered here.
    KNOWN_LEVELS = ["high", "medium", "low", "administrative"]

    def __init__(self):
        reserved = [self.PAD, self.UNKNOWN]
        self.itos: list[str] = reserved + self.KNOWN_LEVELS
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def id_for_salience(self, salience_weight: str | None) -> int:
        if salience_weight is None:
            return self.stoi[self.UNKNOWN]
        return self.stoi.get(salience_weight.lower(), self.stoi[self.UNKNOWN])

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]
class FieldTypeVocab:
    """
    Maps raw metadata field names (e.g. "Title", "Abstract", "Publisher")
    to Field-Type embedding ids (e.g. "Field_Title", "Field_Abstract").

    Built dynamically from field_mappings.json plus a small fixed set of
    reserved ids, so it stays in sync with config/field_mappings.json
    instead of hardcoding field names twice.
    """

    PAD = "[FIELD_PAD]"
    UNKNOWN = "[FIELD_UNKNOWN]"

    def __init__(self, known_field_names: list[str]):
        reserved = [self.PAD, self.UNKNOWN]
        # Normalize to the "Field_X" naming convention used in the spec
        field_ids = [f"Field_{name}" for name in sorted(set(known_field_names))]

        self.itos: list[str] = reserved + field_ids
        self.stoi: dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def id_for_field(self, field_name: str) -> int:
        key = f"Field_{field_name}"
        return self.stoi.get(key, self.stoi[self.UNKNOWN])

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @classmethod
    def from_field_mappings(cls, field_mappings_path: str | Path) -> "FieldTypeVocab":
        import json

        with open(field_mappings_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        field_names = list(raw.get("field_types", {}).keys())
        return cls(field_names)


# ----------------------------------------------------------------------------
# Output container
# ----------------------------------------------------------------------------

@dataclass
class TokenizedField:
    """Tokenization result for a single metadata field (one Title, one Abstract, etc.)."""
    field_name: str
    input_ids: torch.Tensor          # (seq_len,)
    attention_mask: torch.Tensor     # (seq_len,)
    position_ids: torch.Tensor       # (seq_len,)
    field_type_ids: torch.Tensor     # (seq_len,)
    salience_ids: torch.Tensor       # (seq_len,) 
    tokens: list[str] = dc_field(default_factory=list)  # human-readable, for debugging


@dataclass
class TokenizedRecord:
    """All tokenized textual fields for one metadata record."""
    record_id: str
    fields: list[TokenizedField] = dc_field(default_factory=list)

    def get_field(self, field_name: str) -> TokenizedField | None:
        for f in self.fields:
            if f.field_name == field_name:
                return f
        return None


# ----------------------------------------------------------------------------
# Field-aware tokenizer
# ----------------------------------------------------------------------------

class FieldAwareTokenizer:
    """
    Wraps a pretrained HuggingFace BPE/WordPiece tokenizer and augments its
    output with position ids and field-type ids, per metadata field.

    Each field (Title, Abstract, ...) is tokenized independently and tagged
    with its own field-type id, so downstream embedding layers can learn
    field-salience-aware representations.
    """

    def __init__(
        self,
        field_mappings_path: str | Path,
        pretrained_tokenizer_name: str = "bert-base-uncased",
        max_sequence_length: int = 512,
    ):
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            pretrained_tokenizer_name
        )
        self.field_type_vocab = FieldTypeVocab.from_field_mappings(field_mappings_path)
        self.salience_vocab = SalienceVocab()
        self.max_sequence_length = max_sequence_length

    def tokenize_field(self, field_name: str, text: str, salience_weight: str | None = None) -> TokenizedField:
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_sequence_length,
            padding=False,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)          # (seq_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (seq_len,)
        seq_len = input_ids.shape[0]

        position_ids = torch.arange(seq_len, dtype=torch.long)

        field_type_id = self.field_type_vocab.id_for_field(field_name)
        field_type_ids = torch.full((seq_len,), field_type_id, dtype=torch.long)
        salience_id = self.salience_vocab.id_for_salience(salience_weight)
        salience_ids = torch.full((seq_len,), salience_id, dtype=torch.long)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())

        return TokenizedField(
            field_name=field_name,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            field_type_ids=field_type_ids,
            salience_ids=salience_ids,
            tokens=tokens,
          )


    def tokenize_record(self, decomposed: DecomposedMetadata) -> TokenizedRecord:
        """
        Tokenizes every textual field in a DecomposedMetadata object
        (Part 1 output). Structural fields are NOT tokenized here — they
        stay as-is, to be injected via Channel B (graph_builder.py) instead.
        """
        tokenized_fields = [
            self.tokenize_field(f.field_name, f.text, salience_weight=f.salience_weight)  
            for f in decomposed.textual_stream
        ]
        return TokenizedRecord(record_id=decomposed.record_id, fields=tokenized_fields)


# ----------------------------------------------------------------------------
# Embedding layers (Token + Position + Field-Type)
# ----------------------------------------------------------------------------

class FieldAwareEmbedding(nn.Module):
    """
    Produces the final per-token embedding as the sum of three components:

        embedding = TokenEmbedding(input_ids)
                  + PositionEmbedding(position_ids)
                  + FieldTypeEmbedding(field_type_ids)

    This mirrors BERT-style embedding composition, but adds the Field-Type
    channel described in the architecture spec. All three embedding tables
    are randomly initialized and UNTRAINED until pipelines/train.py runs.
    """

    def __init__(
        self,
        vocab_size: int,
        field_type_vocab_size: int,
        salience_vocab_size: int,
        hidden_dim: int = 768,
        max_position_embeddings: int = 512,
        pad_token_id: int = 0,
        field_pad_id: int = 0,
        salience_pad_id: int = 0,
        layer_norm_eps: float = 1e-12,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=pad_token_id
        )
        self.position_embedding = nn.Embedding(
            max_position_embeddings, hidden_dim
        )
        self.field_type_embedding = nn.Embedding(
            field_type_vocab_size, hidden_dim, padding_idx=field_pad_id
        )
        self.salience_embedding = nn.Embedding (
            salience_vocab_size, hidden_dim, padding_idx=salience_pad_id
        ) 

        self.layer_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,       # (batch, seq_len)
        position_ids: torch.Tensor,    # (batch, seq_len)
        field_type_ids: torch.Tensor,
        salience_ids: torch.Tensor,  # (batch, seq_len)
    ) -> torch.Tensor:
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        field_type_embeds = self.field_type_embedding(field_type_ids)
        salience_embeds = self.salience_embedding(salience_ids)  

        embeddings = token_embeds + position_embeds + field_type_embeds + salience_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings  # (batch, seq_len, hidden_dim)


# ----------------------------------------------------------------------------
# Convenience: build both tokenizer + embedding layer from config
# ----------------------------------------------------------------------------

def build_field_aware_tokenizer_and_embedding(
    field_mappings_path: str | Path,
    pretrained_tokenizer_name: str = "bert-base-uncased",
    hidden_dim: int = 768,
    max_sequence_length: int = 512,
) -> tuple[FieldAwareTokenizer, FieldAwareEmbedding]:
    tokenizer = FieldAwareTokenizer(
        field_mappings_path=field_mappings_path,
        pretrained_tokenizer_name=pretrained_tokenizer_name,
        max_sequence_length=max_sequence_length,
    )

    embedding_layer = FieldAwareEmbedding(
        vocab_size=tokenizer.tokenizer.vocab_size,
        field_type_vocab_size=len(tokenizer.field_type_vocab),
        salience_vocab_size=len(tokenizer.salience_vocab),
        hidden_dim=hidden_dim,
        max_position_embeddings=max_sequence_length,
        pad_token_id=tokenizer.tokenizer.pad_token_id or 0,
        field_pad_id=tokenizer.field_type_vocab.pad_id,
        salience_pad_id=tokenizer.salience_vocab.pad_id,
    )

    return tokenizer, embedding_layer