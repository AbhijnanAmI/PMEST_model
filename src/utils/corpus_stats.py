"""
PMEST-Net :: Stage 2, Part 2 support — Background Corpus Statistics

Builds and caches token-level Inverse Document Frequency (IDF) statistics
over whatever sample metadata records exist in data/raw/, for consumption
by the Informativeness Head (span_extractor.py).

IDF is computed per TOKEN (not per exact candidate phrase) because with a
small sample corpus, exact multi-word phrase matches are too sparse to give
a meaningful gradient of rarity -- most phrases would trivially score as
"maximally rare" (appearing in only one document). Token-level IDF degrades
far more gracefully: individual words recur across records even when full
phrases don't.

A candidate span's IDF score is the MINIMUM IDF among its constituent
tokens' IDFs (see `phrase_informativeness_score`), i.e. a phrase's rarity
is bottlenecked by its single rarest word.

Stats are cached to data/vocab/corpus_stats.json, keyed by a hash of
data/raw/'s file listing + modification times, so a stale cache is
automatically detected and rebuilt if sample records are added or changed,
without requiring the person to manually delete a cache file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


_TOKEN_PATTERN = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


def _tokenize_for_idf(text: str) -> set[str]:
    """
    Lowercase, alphabetic-word tokenization for IDF purposes. Deliberately
    simpler than Stage 1's BPE tokenizer -- IDF here is a corpus-frequency
    statistic over whole words, not a neural input, so subword granularity
    would only fragment the count unhelpfully (e.g. 'sourdough' splitting
    into 'sour' + '##dough' would make each subword's DF meaningless on
    its own). Returns a SET (not a list) since document frequency counts
    whether a token appears in a document at all, not how many times.
    """
    return set(m.group(0).lower() for m in _TOKEN_PATTERN.finditer(text))


def _collect_record_text(record: dict) -> str:
    """
    Flattens all string-valued fields of a raw metadata record into one
    text blob for corpus-statistics purposes. Unlike Stage 1's decomposer,
    this does NOT distinguish textual vs. structural fields -- for corpus
    frequency purposes we want the broadest possible sample of vocabulary
    usage across the record, not just the Channel-A-eligible fields.
    """
    parts = []
    for value in record.values():
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts)


def _hash_raw_dir(raw_dir: Path) -> str:
    """
    Cheap fingerprint of data/raw/'s current contents (filenames + mtimes),
    used to detect whether the cached corpus stats are stale.
    """
    entries = []
    for f in sorted(raw_dir.glob("*.json")):
        stat = f.stat()
        entries.append(f"{f.name}:{stat.st_mtime_ns}:{stat.st_size}")
    fingerprint = "|".join(entries)
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


class CorpusStats:
    """
    Holds token document-frequency counts and the total document count
    for a corpus, and exposes IDF lookups.
    """

    def __init__(self, doc_freq: dict[str, int], total_docs: int):
        self.doc_freq = doc_freq
        self.total_docs = max(total_docs, 1)  # guard against div-by-zero on an empty corpus

    def idf(self, token: str) -> float:
        """
        Standard smoothed IDF: log((N + 1) / (df + 1)) + 1
        The +1 smoothing avoids a zero or negative IDF for tokens that
        appear in every document, and avoids division by zero or an
        undefined log for tokens never seen in the corpus at all (they
        simply get the maximum possible IDF for this corpus size).
        """
        df = self.doc_freq.get(token, 0)
        return math.log((self.total_docs + 1) / (df + 1)) + 1.0

    def phrase_informativeness_score(self, phrase: str) -> float:
        """
        A candidate phrase's informativeness prior = MIN idf across its
        constituent tokens (bottlenecked by the rarest word), normalized
        into roughly [0, 1] by dividing by the corpus's maximum possible
        IDF (a phrase composed entirely of never-before-seen tokens).
        """
        tokens = _tokenize_for_idf(phrase)
        if not tokens:
            return 0.0

        token_idfs = [self.idf(t) for t in tokens]
        min_idf = min(token_idfs)

        max_possible_idf = self.idf("__never_seen_token__")  # token guaranteed df=0
        if max_possible_idf == 0:
            return 0.0

        return min(min_idf / max_possible_idf, 1.0)

    def to_dict(self) -> dict:
        return {"doc_freq": self.doc_freq, "total_docs": self.total_docs}

    @classmethod
    def from_dict(cls, d: dict) -> "CorpusStats":
        return cls(doc_freq=d["doc_freq"], total_docs=d["total_docs"])


def build_corpus_stats(raw_dir: str | Path) -> CorpusStats:
    """
    Scans every *.json file in raw_dir, treating each as one document,
    and computes token document-frequency counts across all of them.
    """
    raw_dir = Path(raw_dir)
    doc_freq: dict[str, int] = {}
    total_docs = 0

    for f in sorted(raw_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            record = json.load(fh)

        text = _collect_record_text(record)
        tokens_in_doc = _tokenize_for_idf(text)

        for tok in tokens_in_doc:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

        total_docs += 1

    return CorpusStats(doc_freq=doc_freq, total_docs=total_docs)


def get_or_build_corpus_stats(
    raw_dir: str | Path = "data/raw",
    cache_path: str | Path = "data/vocab/corpus_stats.json",
) -> CorpusStats:
    """
    Loads cached corpus stats if the cache exists AND its recorded
    fingerprint of data/raw/ matches the current contents; otherwise
    rebuilds from scratch and refreshes the cache.
    """
    raw_dir = Path(raw_dir)
    cache_path = Path(cache_path)

    current_fingerprint = _hash_raw_dir(raw_dir)

    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("fingerprint") == current_fingerprint:
            return CorpusStats.from_dict(cached["stats"])

    stats = build_corpus_stats(raw_dir)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {"fingerprint": current_fingerprint, "stats": stats.to_dict()},
            f, indent=2,
        )

    return stats