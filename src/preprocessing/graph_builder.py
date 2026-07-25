"""
PMEST-Net :: Stage 1, Part 3 — Metadata Dependency Graph Construction (Channel B input)

Builds a fully-connected local knowledge graph over the structural_stream
fields of a single metadata record (Author, Publisher, Institution,
Location_Meta, Temporal_Meta, etc). Node features are produced by reusing
the SAME FieldAwareEmbedding instance as Channel A, so structural and
textual representations live in the same embedding space before the GAT
and Transformer outputs are concatenated in encoders.py.

Text-like structural fields (Publisher, Author, Institution, File_Format)
get tokenized and mean-pooled through FieldAwareEmbedding, same as any
textual field. Pure numeric/parsed fields (Location_Meta coordinates,
Temporal_Meta date ranges) have no meaningful subword tokens, so they
fall back to a zero-vector node feature — the GAT still processes them
(they still have edges and a position in the graph), they just start
from a neutral point rather than an embedded one.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import torch

from src.preprocessing.cleaner import DecomposedMetadata, StructuralAttribute
from src.preprocessing.tokenization import FieldAwareTokenizer, FieldAwareEmbedding


# Fields whose parsed_value is still meaningful text worth tokenizing/embedding.
# Everything else (coordinates, date ranges, ISO dates) is numeric/structured
# and gets the zero-vector fallback instead.
_TEXT_LIKE_STRUCTURAL_FIELDS = {
    "Author",
    "Publisher",
    "Institution",
    "File_Format",
    "Creator",
}


@dataclass
class MetadataGraph:
    """
    A fully-connected graph over one record's structural fields.

    node_features: (num_nodes, hidden_dim)
    edge_index:    (2, num_edges) in PyTorch Geometric COO format
    node_field_names: node index -> field name, for traceability/debugging
    """
    record_id: str
    node_features: torch.Tensor
    edge_index: torch.Tensor
    node_field_names: list[str] = dc_field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


class MetadataGraphBuilder:
    """
    Constructs a MetadataGraph from a DecomposedMetadata object's
    structural_stream, reusing a shared FieldAwareEmbedding so node
    features are comparable to Channel A's token embeddings.
    """

    def __init__(
        self,
        tokenizer: FieldAwareTokenizer,
        embedding_layer: FieldAwareEmbedding,
        hidden_dim: int = 768,
    ):
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.hidden_dim = hidden_dim

    def _node_feature_for_field(self, attr: StructuralAttribute) -> torch.Tensor:
        """
        Produces a single (hidden_dim,) feature vector for one structural field.
        Text-like fields are tokenized + embedded (now including salience) + mean-pooled.
        Numeric/parsed fields fall back to a zero vector.
        """
        if attr.field_name not in _TEXT_LIKE_STRUCTURAL_FIELDS:
            return torch.zeros(self.hidden_dim)

        raw_text = attr.raw_value
        if not isinstance(raw_text, str) or not raw_text.strip():
            return torch.zeros(self.hidden_dim)

        tokenized = self.tokenizer.tokenize_field(
            attr.field_name, raw_text, salience_weight=attr.salience_weight
        )

        with torch.no_grad():
            token_embeds = self.embedding_layer(
                input_ids=tokenized.input_ids.unsqueeze(0),
                position_ids=tokenized.position_ids.unsqueeze(0),
                field_type_ids=tokenized.field_type_ids.unsqueeze(0),
                salience_ids=tokenized.salience_ids.unsqueeze(0),
            )  # (1, seq_len, hidden_dim)

        # Mean-pool across tokens to collapse the field to one node vector.
        pooled = token_embeds.mean(dim=1).squeeze(0)  # (hidden_dim,)
        return pooled

    def _build_fully_connected_edge_index(self, num_nodes: int) -> torch.Tensor:
        """
        Builds a fully-connected, directed-both-ways edge_index
        (every node connects to every other node, excluding self-loops).
        Format matches PyTorch Geometric's expected (2, num_edges) COO layout.
        """
        if num_nodes <= 1:
            # No edges possible with 0 or 1 nodes.
            return torch.empty((2, 0), dtype=torch.long)

        sources = []
        targets = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    sources.append(i)
                    targets.append(j)

        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        return edge_index

    def build(self, decomposed: DecomposedMetadata) -> MetadataGraph:
        structural_fields = decomposed.structural_stream

        if len(structural_fields) == 0:
            # No structural fields at all -- return an empty graph.
            # encoders.py must handle this case (zero structural contribution).
            return MetadataGraph(
                record_id=decomposed.record_id,
                node_features=torch.zeros((0, self.hidden_dim)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                node_field_names=[],
            )

        node_features = torch.stack(
            [self._node_feature_for_field(attr) for attr in structural_fields]
        )  # (num_nodes, hidden_dim)

        edge_index = self._build_fully_connected_edge_index(len(structural_fields))

        return MetadataGraph(
            record_id=decomposed.record_id,
            node_features=node_features,
            edge_index=edge_index,
            node_field_names=[attr.field_name for attr in structural_fields],
        )