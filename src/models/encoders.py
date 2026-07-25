"""
PMEST-Net :: Stage 1, Part 3 — Dual-Channel Metadata Encoder

Channel A (Contextual Transformer Layer): a multi-layer bidirectional
Transformer encoder computing self-attention across ALL textual metadata
fields concatenated into one sequence, enabling cross-field attention
(e.g. Abstract tokens attending to high-weight Title tokens).

Channel B (Metadata Dependency Graph Network): a Graph Attention Network
(GAT) passing messages between structural metadata nodes built by
graph_builder.py.

Channel A and Channel B outputs are concatenated at the token level to
produce the Field-Aware Structural Token Embedding Matrix.

Controlled by config/default_config.yaml -> encoder.use_graph_channel.
When true (current setting), the GAT is live and its pooled output is
broadcast and concatenated onto every Channel A token. When false, this
falls back to a zero-vector structural contribution instead of skipping
the module entirely, so output shape is identical either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from src.preprocessing.cleaner import DecomposedMetadata
from src.preprocessing.tokenization import (
    FieldAwareTokenizer,
    FieldAwareEmbedding,
    TokenizedRecord,
)
from src.preprocessing.graph_builder import MetadataGraphBuilder, MetadataGraph


# ----------------------------------------------------------------------------
# Channel A: Contextual Transformer
# ----------------------------------------------------------------------------

class ContextualTransformerEncoder(nn.Module):
    """
    Multi-layer bidirectional Transformer encoder over a SINGLE concatenated
    sequence built from all textual fields in a record. Concatenation (rather
    than per-field encoding) is what enables true cross-field self-attention,
    e.g. Abstract tokens attending to Title tokens, per the architecture spec.

    A [FIELD_SEP] boundary is not injected between fields at the text level --
    field-type embeddings (already baked into each token's input embedding
    by Part 2) are what the model uses to distinguish field boundaries,
    rather than a special separator token.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        feedforward_dim: int = 3072,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout_prob,
            batch_first=True,
            norm_first=True,  # pre-LN, generally more stable to train
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        token_embeddings: torch.Tensor,       # (1, total_seq_len, hidden_dim)
        attention_mask: torch.Tensor,          # (1, total_seq_len), 1 = real token
    ) -> torch.Tensor:
        # nn.TransformerEncoder expects a padding mask where True = IGNORE.
        padding_mask = attention_mask == 0  # (1, total_seq_len), bool

        output = self.transformer(
            token_embeddings, src_key_padding_mask=padding_mask
        )  # (1, total_seq_len, hidden_dim)

        return output


def concatenate_tokenized_fields(
    tokenized_record: TokenizedRecord,
    embedding_layer: FieldAwareEmbedding,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Concatenates all per-field token embeddings from Part 2 into a single
    sequence for Channel A. Position ids are NOT re-derived globally --
    each field keeps the local position ids assigned in Part 2 (its
    Field-Type and Salience embeddings are what disambiguate field
    identity/importance for the Transformer, not a re-numbered global
    position).

    Returns: (token_embeddings, attention_mask, source_field_per_token)
    """
    all_input_ids = []
    all_position_ids = []
    all_field_type_ids = []
    all_salience_ids = []
    all_attention_mask = []
    source_field_per_token = []

    for tf in tokenized_record.fields:
        all_input_ids.append(tf.input_ids)
        all_position_ids.append(tf.position_ids)
        all_field_type_ids.append(tf.field_type_ids)
        all_salience_ids.append(tf.salience_ids)
        all_attention_mask.append(tf.attention_mask)
        source_field_per_token.extend([tf.field_name] * tf.input_ids.shape[0])

    input_ids = torch.cat(all_input_ids, dim=0).unsqueeze(0)              # (1, total_len)
    position_ids = torch.cat(all_position_ids, dim=0).unsqueeze(0)        # (1, total_len)
    field_type_ids = torch.cat(all_field_type_ids, dim=0).unsqueeze(0)    # (1, total_len)
    salience_ids = torch.cat(all_salience_ids, dim=0).unsqueeze(0)        # (1, total_len)
    attention_mask = torch.cat(all_attention_mask, dim=0).unsqueeze(0)    # (1, total_len)

    token_embeddings = embedding_layer(
        input_ids=input_ids,
        position_ids=position_ids,
        field_type_ids=field_type_ids,
        salience_ids=salience_ids,
    )  # (1, total_len, hidden_dim)

    return token_embeddings, attention_mask, source_field_per_token

# ----------------------------------------------------------------------------
# Channel B: Metadata Dependency Graph Network (GAT)
# ----------------------------------------------------------------------------

class MetadataGraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network over the structural metadata graph built by
    graph_builder.py. Passes messages between structural entity nodes
    (Author, Publisher, Institution, Location_Meta, Temporal_Meta, ...)
    to generate structural context vectors.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        # GATConv splits hidden_dim across heads then concatenates back,
        # so out_channels = hidden_dim // num_heads keeps dimensionality
        # constant across layers.
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        per_head_dim = hidden_dim // num_heads

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            concat = True
            self.layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=per_head_dim,
                    heads=num_heads,
                    concat=concat,
                    dropout=dropout_prob,
                )
            )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        node_features: (num_nodes, hidden_dim)
        edge_index:    (2, num_edges)
        Returns:       (num_nodes, hidden_dim) -- updated node representations
        """
        if node_features.shape[0] == 0:
            # Empty graph (no structural fields present in this record).
            return node_features

        x = node_features
        for gat_layer in self.layers:
            x = gat_layer(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        x = self.layer_norm(x)
        return x  # (num_nodes, hidden_dim)


# ----------------------------------------------------------------------------
# Dual-Channel Encoder: ties Channel A + Channel B together
# ----------------------------------------------------------------------------

@dataclass
class DualChannelEncoderOutput:
    """Final Stage 1 output: Field-Aware Structural Token Embedding Matrix."""
    record_id: str
    token_embeddings: torch.Tensor      # (1, total_seq_len, combined_dim)
    attention_mask: torch.Tensor        # (1, total_seq_len)
    source_field_per_token: list[str]
    graph_node_embeddings: torch.Tensor  # (num_structural_nodes, hidden_dim), for inspection
    graph_used: bool


class DualChannelMetadataEncoder(nn.Module):
    """
    Stage 1, Part 3 top-level module. Combines:
      - Channel A: ContextualTransformerEncoder over concatenated textual fields
      - Channel B: MetadataGraphAttentionNetwork over structural fields

    Channel B's output is mean-pooled into a single structural context
    vector per record, then broadcast and concatenated onto every Channel A
    token -- this is the "concatenated at the token level" step described
    in the architecture spec's Stage 1 summary.

    use_graph_channel=False still runs this module with identical output
    shape, but substitutes a zero-vector for the structural contribution,
    per config/default_config.yaml. Currently set to True.
    """

    def __init__(
        self,
        tokenizer: FieldAwareTokenizer,
        embedding_layer: FieldAwareEmbedding,
        hidden_dim: int = 768,
        transformer_layers: int = 6,
        transformer_heads: int = 12,
        gat_layers: int = 2,
        gat_heads: int = 4,
        use_graph_channel: bool = True,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.hidden_dim = hidden_dim
        self.use_graph_channel = use_graph_channel

        self.channel_a = ContextualTransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
        )

        self.graph_builder = MetadataGraphBuilder(
            tokenizer=tokenizer,
            embedding_layer=embedding_layer,
            hidden_dim=hidden_dim,
        )

        self.channel_b = MetadataGraphAttentionNetwork(
            hidden_dim=hidden_dim,
            num_layers=gat_layers,
            num_heads=gat_heads,
        )

        # Projects the concatenated [Channel A | Channel B] vector back down
        # to hidden_dim, so output shape stays consistent regardless of
        # use_graph_channel, and downstream Stage 2/3/4 modules don't need
        # to know which mode produced it.
        self.fusion_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fusion_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, decomposed: DecomposedMetadata) -> DualChannelEncoderOutput:
        # ---- Channel A ----
        tokenized_record = self.tokenizer.tokenize_record(decomposed)
        token_embeds, attention_mask, source_fields = concatenate_tokenized_fields(
            tokenized_record, self.embedding_layer
        )
        channel_a_output = self.channel_a(token_embeds, attention_mask)
        # (1, total_seq_len, hidden_dim)

        # ---- Channel B ----
        graph: MetadataGraph = self.graph_builder.build(decomposed)

        if self.use_graph_channel and graph.num_nodes > 0:
            graph_node_embeddings = self.channel_b(graph.node_features, graph.edge_index)
            structural_context = graph_node_embeddings.mean(dim=0)  # (hidden_dim,)
        else:
            graph_node_embeddings = torch.zeros((graph.num_nodes, self.hidden_dim))
            structural_context = torch.zeros(self.hidden_dim)

        # ---- Fusion: broadcast structural context onto every Channel A token ----
        seq_len = channel_a_output.shape[1]
        structural_broadcast = structural_context.unsqueeze(0).unsqueeze(0).expand(
            1, seq_len, self.hidden_dim
        )  # (1, seq_len, hidden_dim)

        fused = torch.cat([channel_a_output, structural_broadcast], dim=-1)
        # (1, seq_len, hidden_dim * 2)

        fused = self.fusion_projection(fused)   # (1, seq_len, hidden_dim)
        fused = self.fusion_layer_norm(fused)

        return DualChannelEncoderOutput(
            record_id=decomposed.record_id,
            token_embeddings=fused,
            attention_mask=attention_mask,
            source_field_per_token=source_fields,
            graph_node_embeddings=graph_node_embeddings,
            graph_used=self.use_graph_channel,
        )