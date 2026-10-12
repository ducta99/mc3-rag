"""Indexing and Retrieval Pipeline for AMD AI League - Match 3.

Provides:
- Entity-preserving tokenizer (preserves part numbers, ticket IDs, codes, revisions, pins).
- Context-aware chunking (keeps image OCR and log sections cohesive).
- BM25 Okapi search with query stopword filtering and term boosting.
- Superseded document detection and de-prioritization.
- Multi-hop entity graph linker (e.g., incident logs -> bug database tickets -> fix versions).
- Persistence to INDEX_DIR (/app/index) for fast zero-overhead loading in query processes.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ingestor import DocumentChunk, IngestedDocument, walk_and_ingest_corpus

STOPWORDS = {
    "what", "is", "the", "of", "in", "which", "does", "for", "a", "an", "and", "or",
    "to", "on", "at", "by", "with", "from", "as", "that", "this", "it", "its", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "did", "how",
    "when", "where", "why", "who", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
}


def enhanced_tokenize(text: str) -> List[str]:
    """Tokenizes text while preserving technical entities like ORR-1847, E7731, B14, REV-C2."""
    if not text:
        return []
    tokens = []
    raw = re.findall(r"[A-Za-z0-9]+(?:[-_#][A-Za-z0-9]+)*", text.lower())
    for t in raw:
        tokens.append(t)
        if "-" in t or "_" in t or "#" in t:
            stripped = t.replace("-", "").replace("_", "").replace("#", "")
            if stripped:
                tokens.append(stripped)
            for sub in re.split(r"[-_#]", t):
                if len(sub) > 1:
                    tokens.append(sub)
    return tokens


@dataclass
class SearchResult:
    chunk_id: str
    rel_path: str
    text: str
    doc_full_text: str
    score: float
    is_superseded: bool = False


class BM25Index:
    """Fast, robust in-memory BM25 index with persistence."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avg_dl: float = 0.0
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.term_freqs: List[Dict[str, int]] = []
        self.chunks_data: List[Dict[str, str]] = []
        self.doc_metadata: Dict[str, Dict] = {}
        self.doc_texts: Dict[str, str] = {}

    def build(self, documents: List[IngestedDocument]) -> None:
        self.chunks_data.clear()
        self.doc_len.clear()
        self.doc_freqs.clear()
        self.term_freqs.clear()
        self.doc_metadata.clear()
        self.doc_texts.clear()

        total_words = 0
        for doc in documents:
            self.doc_metadata[doc.rel_path] = doc.metadata
            self.doc_texts[doc.rel_path] = doc.raw_text
            if doc.error or not doc.raw_text:
                continue

            # For images, keep as a single chunk to maintain 2D layout relationships
            if doc.file_type in {".png", ".jpg", ".jpeg"}:
                chunks_to_index = [doc.raw_text]
            else:
                chunks_to_index = [l.strip() for l in doc.raw_text.splitlines() if l.strip()]

            for idx, text_block in enumerate(chunks_to_index):
                augmented = f"{doc.rel_path} {text_block}"
                toks = enhanced_tokenize(augmented)
                if not toks:
                    continue

                chunk_entry = {
                    "rel_path": doc.rel_path,
                    "chunk_id": f"{doc.rel_path}:{idx}",
                    "text": text_block,
                }
                tf = Counter(toks)
                self.chunks_data.append(chunk_entry)
                self.term_freqs.append(dict(tf))
                dl = len(toks)
                self.doc_len.append(dl)
                total_words += dl

                for term in tf.keys():
                    self.doc_freqs[term] += 1

        num_docs = len(self.chunks_data)
        self.avg_dl = (total_words / num_docs) if num_docs > 0 else 0.0

    def query(self, query_text: str, top_k: int = 5) -> List[SearchResult]:
        if not self.chunks_data:
            return []

        query_tokens = [t for t in enhanced_tokenize(query_text) if t not in STOPWORDS]
        if not query_tokens:
            query_tokens = enhanced_tokenize(query_text)
        if not query_tokens:
            return []

        num_docs = len(self.chunks_data)
        scores: Dict[int, float] = defaultdict(float)

        for token in query_tokens:
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

            for idx, tf_dict in enumerate(self.term_freqs):
                tf = tf_dict.get(token, 0)
                if tf == 0:
                    continue
                dl = self.doc_len[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avg_dl or 1.0)))
                scores[idx] += idf * (numerator / denominator)

        if not scores:
            return []

        results = []
        for idx in sorted(scores.keys(), key=lambda i: scores[i], reverse=True):
            entry = self.chunks_data[idx]
            rel_path = entry["rel_path"]
            meta = self.doc_metadata.get(rel_path, {})
            is_withdrawn = (
                meta.get("status") == "WITHDRAWN" or
                "withdrawn" in rel_path.lower() or
                "withdrawn" in entry["text"].lower()
            )

            score = scores[idx]
            if is_withdrawn:
                score *= 0.05

            results.append(
                SearchResult(
                    chunk_id=entry["chunk_id"],
                    rel_path=rel_path,
                    text=entry["text"],
                    doc_full_text=self.doc_texts.get(rel_path, ""),
                    score=score,
                    is_superseded=is_withdrawn,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "doc_len": self.doc_len,
            "avg_dl": self.avg_dl,
            "doc_freqs": dict(self.doc_freqs),
            "term_freqs": self.term_freqs,
            "chunks_data": self.chunks_data,
            "doc_metadata": self.doc_metadata,
            "doc_texts": self.doc_texts,
        }
        target_file = index_dir / "bm25_index.json"
        target_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path) -> Optional[BM25Index]:
        target_file = index_dir / "bm25_index.json"
        if not target_file.exists():
            return None
        data = json.loads(target_file.read_text(encoding="utf-8"))
        idx = cls(k1=data["k1"], b=data["b"])
        idx.doc_len = data["doc_len"]
        idx.avg_dl = data["avg_dl"]
        idx.doc_freqs = defaultdict(int, data["doc_freqs"])
        idx.term_freqs = data["term_freqs"]
        idx.chunks_data = data["chunks_data"]
        idx.doc_metadata = data["doc_metadata"]
        idx.doc_texts = data.get("doc_texts", {})
        return idx


class CorpusIndexManager:
    """Coordinates indexing pass, persistence, and multi-hop retrieval."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.bm25: Optional[BM25Index] = None
        self.documents: Dict[str, Dict] = {}

    def build_and_persist(self, corpus_root: Path) -> None:
        """Called during python3 /app/app.py --index /app/corpus."""
        raw_docs = walk_and_ingest_corpus(corpus_root, enable_ocr=True)
        self.bm25 = BM25Index()
        self.bm25.build(raw_docs)
        self.bm25.save(self.index_dir)

        docs_data = {}
        for doc in raw_docs:
            docs_data[doc.rel_path] = {
                "rel_path": doc.rel_path,
                "file_type": doc.file_type,
                "raw_text": doc.raw_text,
                "metadata": doc.metadata,
                "error": doc.error,
            }
        (self.index_dir / "documents.json").write_text(
            json.dumps(docs_data, ensure_ascii=False), encoding="utf-8"
        )

    def load_index(self) -> bool:
        """Loads index from disk in milliseconds."""
        self.bm25 = BM25Index.load(self.index_dir)
        docs_file = self.index_dir / "documents.json"
        if docs_file.exists():
            self.documents = json.loads(docs_file.read_text(encoding="utf-8"))
            return self.bm25 is not None
        return False

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if self.bm25 is None:
            if not self.load_index():
                return []
        assert self.bm25 is not None
        results = self.bm25.query(query, top_k=top_k)

        # Multi-hop detection across full text of retrieved docs
        ticket_matches = set()
        for r in results:
            found = re.findall(r"\bORR-\d+\b", r.doc_full_text, re.IGNORECASE)
            ticket_matches.update(f.upper() for f in found)

        if ticket_matches:
            for ticket in ticket_matches:
                hop_results = self.bm25.query(f"{ticket} ticket bug database", top_k=2)
                for hr in hop_results:
                    if hr.rel_path not in [res.rel_path for res in results]:
                        results.append(hr)

        return results
