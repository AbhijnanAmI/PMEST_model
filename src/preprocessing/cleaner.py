"""
PMEST-Net :: Stage 1, Part 1 — Metadata Decomposition

Splits a raw, semi-structured metadata record into two distinct streams:

  - Textual Payload Stream:     unstructured natural language fields
                                 (Title, Abstract, Description, Subject_Tags, ...)
  - Structural Attribute Stream: typed key-value fields (dates, coordinates,
                                 file formats, administrative metadata)

Field routing rules come from config/field_mappings.json. Any field not
explicitly listed falls back to `unmapped_field_default`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------

@dataclass
class StructuralAttribute:
    """A single normalized structural (non-textual) metadata field."""
    field_name: str
    raw_value: Any
    parsed_value: Any
    expected_format: str | None
    facet_bias: str | None
    salience_weight: str
    parsed_ok: bool


@dataclass
class TextualField:
    """A single natural-language metadata field."""
    field_name: str
    text: str
    salience_weight: str
    facet_bias: str | None


@dataclass
class DecomposedMetadata:
    """Result of Stage 1 / Part 1 decomposition for one metadata record."""
    record_id: str
    textual_stream: list[TextualField] = field(default_factory=list)
    structural_stream: list[StructuralAttribute] = field(default_factory=list)
    unmapped_fields: list[str] = field(default_factory=list)

    def get_textual_field(self, field_name: str) -> TextualField | None:
        for f in self.textual_stream:
            if f.field_name == field_name:
                return f
        return None

    def get_structural_field(self, field_name: str) -> StructuralAttribute | None:
        for f in self.structural_stream:
            if f.field_name == field_name:
                return f
        return None

    def as_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "textual_stream": [f.__dict__ for f in self.textual_stream],
            "structural_stream": [f.__dict__ for f in self.structural_stream],
            "unmapped_fields": self.unmapped_fields,
        }


# ----------------------------------------------------------------------------
# Field mapping loader
# ----------------------------------------------------------------------------

class FieldMappingRegistry:
    """
    Loads and serves field routing rules from config/field_mappings.json.
    """

    def __init__(self, mapping_path: str | Path):
        self.mapping_path = Path(mapping_path)
        if not self.mapping_path.exists():
            raise FileNotFoundError(
                f"field_mappings.json not found at: {self.mapping_path}"
            )

        with open(self.mapping_path, "r", encoding="utf-8") as fh:
            self._raw: dict = json.load(fh)

        self.field_types: dict[str, dict] = self._raw.get("field_types", {})
        self.parsing_rules: dict[str, dict] = self._raw.get("parsing_rules", {})
        self.unmapped_default: dict = self._raw.get(
            "unmapped_field_default",
            {"stream": "structural", "salience_weight": "administrative", "facet_bias": None},
        )

    def get_rule(self, field_name: str) -> dict:
        """
        Returns the routing rule for a given field name.
        Falls back to `unmapped_field_default` if the field isn't registered.
        Field name matching is case-insensitive and tolerant of spaces/underscores.
        """
        normalized = field_name.strip().replace(" ", "_")

        if normalized in self.field_types:
            return self.field_types[normalized]

        # case-insensitive fallback pass
        for known_field, rule in self.field_types.items():
            if known_field.lower() == normalized.lower():
                return rule

        return self.unmapped_default

    def get_parsing_rule(self, expected_format: str | None) -> dict | None:
        if expected_format is None:
            return None
        return self.parsing_rules.get(expected_format)

    def is_field_mapped(self, field_name: str) -> bool:
        normalized = field_name.strip().replace(" ", "_")
        return normalized in self.field_types or any(
            k.lower() == normalized.lower() for k in self.field_types
        )


# ----------------------------------------------------------------------------
# Structural value parsers (driven by parsing_rules in field_mappings.json)
# ----------------------------------------------------------------------------

def _parse_coordinates(raw_value: str, pattern: str) -> dict | None:
    """
    Parses coordinate strings like '35.705° N, 117.506° W' into decimal degrees.
    Returns None if the pattern doesn't match.
    """
    match = re.search(pattern, raw_value)
    if not match:
        return None

    groups = match.groups()
    # pattern captures: lat_value, lat_hemisphere, lon_value, lon_hemisphere
    if len(groups) < 4:
        return None

    lat_val, lat_hem, lon_val, lon_hem = groups[0], groups[1], groups[2], groups[3]

    lat = float(lat_val)
    lon = float(lon_val)

    if lat_hem.upper() == "S":
        lat = -lat
    if lon_hem.upper() == "W":
        lon = -lon

    return {"latitude": lat, "longitude": lon}


def _parse_date_range(raw_value: str, pattern: str) -> dict | None:
    """
    Parses a date-range string using the configured pattern.
    Returns the two captured date strings as-is (format normalization is
    left to a dedicated date-parsing utility if/when needed downstream).
    """
    match = re.search(pattern, raw_value)
    if not match:
        return None

    groups = match.groups()
    if len(groups) < 2:
        return None

    return {"start": groups[0], "end": groups[1]}


def _parse_iso_date(raw_value: str, pattern: str) -> dict | None:
    match = re.search(pattern, raw_value)
    if not match:
        return None
    return {"date": match.group(0)}


_PARSER_DISPATCH = {
    "coordinates": _parse_coordinates,
    "date_range": _parse_date_range,
    "iso_date": _parse_iso_date,
}


def parse_structural_value(raw_value: Any, expected_format: str | None, pattern: str | None):
    """
    Dispatches a raw structural value to the correct parser based on
    expected_format. Returns (parsed_value, parsed_ok).
    """
    if expected_format is None or pattern is None:
        # No specific parsing rule — pass the raw value through unchanged.
        return raw_value, True

    if not isinstance(raw_value, str):
        return raw_value, False

    parser_fn = _PARSER_DISPATCH.get(expected_format)
    if parser_fn is None:
        # Unknown expected_format — pass through, but flag as not validated.
        return raw_value, False

    parsed = parser_fn(raw_value, pattern)
    if parsed is None:
        return raw_value, False

    return parsed, True


# ----------------------------------------------------------------------------
# Main decomposition entry point
# ----------------------------------------------------------------------------

class MetadataDecomposer:
    """
    Splits raw metadata records into Textual Payload Stream and
    Structural Attribute Stream, per Stage 1 / Part 1 of the PMEST-Net
    architecture.
    """

    def __init__(self, field_mappings_path: str | Path):
        self.registry = FieldMappingRegistry(field_mappings_path)

    def decompose(self, record: dict, record_id: str | None = None) -> DecomposedMetadata:
        """
        record: a flat dict of raw metadata, e.g.
            {
                "Title": "...",
                "Abstract": "...",
                "Location_Meta": "35.705° N, 117.506° W",
                "Temporal_Meta": "2019-07-04 / 2019-08-15",
                ...
            }
        """
        if record_id is None:
            record_id = str(record.get("id") or record.get("ID") or "unknown_record")

        result = DecomposedMetadata(record_id=record_id)

        for raw_field_name, raw_value in record.items():
            if raw_value is None or raw_value == "":
                continue

            rule = self.registry.get_rule(raw_field_name)
            stream = rule.get("stream", "structural")
            salience_weight = rule.get("salience_weight", "administrative")
            facet_bias = rule.get("facet_bias")

            if not self.registry.is_field_mapped(raw_field_name):
                result.unmapped_fields.append(raw_field_name)

            if stream == "textual":
                result.textual_stream.append(
                    TextualField(
                        field_name=raw_field_name,
                        text=str(raw_value).strip(),
                        salience_weight=salience_weight,
                        facet_bias=facet_bias,
                    )
                )
            else:
                expected_format = rule.get("expected_format")
                pattern = None
                if expected_format:
                    parsing_rule = self.registry.get_parsing_rule(expected_format)
                    if parsing_rule:
                        pattern = parsing_rule.get("pattern")

                parsed_value, parsed_ok = parse_structural_value(
                    raw_value, expected_format, pattern
                )

                result.structural_stream.append(
                    StructuralAttribute(
                        field_name=raw_field_name,
                        raw_value=raw_value,
                        parsed_value=parsed_value,
                        expected_format=expected_format,
                        facet_bias=facet_bias,
                        salience_weight=salience_weight,
                        parsed_ok=parsed_ok,
                    )
                )

        return result
