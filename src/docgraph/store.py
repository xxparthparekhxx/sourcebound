"""The index: SQLite (documents, chunks, FTS5/BM25, knowledge graph) plus ChromaDB (vectors).

The index records which embedding model built it (name and dimension) and refuses to be opened
with a different one, so vectors from two models never end up in the same collection.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import Config
from .embeddings import Embedder
from .ingest.chunking import ChunkSpec
from .ingest.loaders import ParsedDocument
from .models import Entity, Passage

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  format TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  n_pages INTEGER,
  n_chunks INTEGER NOT NULL DEFAULT 0,
  skipped_pages TEXT NOT NULL DEFAULT '[]',
  ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ord INTEGER NOT NULL,
  page INTEGER,
  page_end INTEGER,
  heading TEXT NOT NULL DEFAULT '',
  text TEXT NOT NULL,
  graph_status INTEGER NOT NULL DEFAULT 0   -- 0 not extracted, 1 done, 2 extraction failed validation
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id, ord);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, heading, title, tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS entities (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  norm TEXT NOT NULL UNIQUE,
  type TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS mentions (
  entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  PRIMARY KEY (entity_id, chunk_id)
);
CREATE INDEX IF NOT EXISTS mentions_chunk ON mentions(chunk_id);

CREATE TABLE IF NOT EXISTS relations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  src INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  dst INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  predicate TEXT NOT NULL,
  chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  UNIQUE (src, dst, predicate, chunk_id)
);
CREATE INDEX IF NOT EXISTS relations_src ON relations(src);
CREATE INDEX IF NOT EXISTS relations_dst ON relations(dst);
"""

BM25_WEIGHTS = "1.0, 0.5, 0.2"  # text, heading, document title


class EmbeddingMismatch(RuntimeError):
    """The configured embedding model differs from the one that built the index."""


@dataclass
class DocRecord:
    id: int
    source: str
    title: str
    format: str
    sha256: str
    n_pages: int | None
    n_chunks: int
    skipped_pages: list[dict]
    ingested_at: str


@dataclass
class VectorHit:
    id: int
    similarity: float
    embedding: np.ndarray | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm_entity(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


class Index:
    def __init__(self, db_path: str | Path, chroma_path: str | Path, embedder: Embedder | None = None):
        self.db_path = Path(db_path)
        self.chroma_path = Path(chroma_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self.conn.executescript(SCHEMA)
        self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
        self.conn.commit()
        self._chroma = None
        self._col = None
        self.embedder: Embedder | None = None
        if embedder is not None:
            self.bind_embedder(embedder)

    @classmethod
    def open(cls, cfg: Config, embedder: Embedder | None = None) -> "Index":
        return cls(cfg.db_path, cfg.chroma_path, embedder)

    # ------------------------------------------------------------------ connections

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys = ON")
            c.execute("PRAGMA journal_mode = WAL")
            self._local.conn = c
            with self._lock:
                self._conns.append(c)
        return c

    def close(self) -> None:
        with self._lock:
            for c in self._conns:
                try:
                    c.close()
                except sqlite3.Error:
                    pass
            self._conns.clear()
        self._local = threading.local()

    @property
    def collection(self):
        if self._col is None:
            import chromadb
            from chromadb.config import Settings

            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(
                path=str(self.chroma_path), settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            self._col = self._chroma.get_or_create_collection(
                "chunks", metadata={"hnsw:space": "cosine"}, embedding_function=None
            )
        return self._col

    def _reset_collection(self) -> None:
        _ = self.collection
        self._chroma.delete_collection("chunks")
        self._col = None
        _ = self.collection

    # ------------------------------------------------------------------ meta + embedding guard

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def embedding_info(self) -> dict | None:
        model = self.get_meta("embed_model")
        if model is None:
            return None
        return {"model": model, "dimension": int(self.get_meta("embed_dim") or 0), "provider": self.get_meta("embed_provider")}

    def bind_embedder(self, embedder: Embedder) -> None:
        """Attach the embedder; record its name/dimension on first use, and refuse a different one later."""
        stored = self.embedding_info()
        has_chunks = self.count("chunks") > 0
        if stored is not None and has_chunks and stored["model"] != embedder.model:
            raise EmbeddingMismatch(self._mismatch_message(stored, embedder.model, None))
        dim = embedder.dimension
        if stored is not None and has_chunks and stored["dimension"] != dim:
            raise EmbeddingMismatch(self._mismatch_message(stored, embedder.model, dim))
        if stored is None or not has_chunks:
            if stored is not None and (stored["model"] != embedder.model or stored["dimension"] != dim):
                self._reset_collection()  # nothing indexed yet: safe to start over with the new model
            self._set_meta("embed_model", embedder.model)
            self._set_meta("embed_dim", str(dim))
            self._set_meta("embed_provider", embedder.provider)
            self.conn.commit()
        self.embedder = embedder

    @staticmethod
    def _mismatch_message(stored: dict, model: str, dim: int | None) -> str:
        now = f"{model}" + (f" (dimension {dim})" if dim is not None else "")
        return (
            f"this index was built with embedding model {stored['model']!r} (dimension {stored['dimension']}), "
            f"but the configuration now uses {now}. Vectors from different models cannot be mixed. "
            "Point data_dir at a new directory, or run `docgraph reset` and ingest again."
        )

    def _need_embedder(self) -> Embedder:
        if self.embedder is None:
            raise RuntimeError("this operation needs an embedder; open the index with one")
        return self.embedder

    # ------------------------------------------------------------------ counts and listings

    def count(self, table: str) -> int:
        assert table in {"documents", "chunks", "entities", "relations", "mentions"}
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def stats(self) -> dict:
        return {t: self.count(t) for t in ("documents", "chunks", "entities", "relations", "mentions")}

    @staticmethod
    def _doc_row(r: sqlite3.Row) -> DocRecord:
        return DocRecord(r["id"], r["source"], r["title"], r["format"], r["sha256"], r["n_pages"], r["n_chunks"],
                         json.loads(r["skipped_pages"] or "[]"), r["ingested_at"])

    def documents(self) -> list[DocRecord]:
        return [self._doc_row(r) for r in self.conn.execute("SELECT * FROM documents ORDER BY title, id")]

    def get_document(self, source: str) -> DocRecord | None:
        r = self.conn.execute("SELECT * FROM documents WHERE source = ?", (source,)).fetchone()
        return self._doc_row(r) if r else None

    def doc_ids_for_sources(self, sources: Iterable[str]) -> list[int]:
        out = []
        for s in sources:
            r = self.conn.execute("SELECT id FROM documents WHERE source = ?", (s,)).fetchone()
            if r:
                out.append(r["id"])
        return out

    # ------------------------------------------------------------------ writing

    def upsert_document(
        self,
        source: str,
        parsed: ParsedDocument,
        chunks: Sequence[ChunkSpec],
        sha256: str,
        *,
        context_header: bool = True,
    ) -> tuple[int, list[int], bool]:
        """Insert a document, replacing any earlier version of the same source.

        Returns (doc_id, chunk_ids, replaced). Embeddings are computed before anything is modified.
        """
        emb = self._need_embedder()
        texts = [self._embed_text(parsed.title, c, context_header) for c in chunks]
        vectors = emb.embed_documents(texts) if texts else np.zeros((0, emb.dimension), dtype=np.float32)
        if len(vectors) and vectors.shape[1] != emb.dimension:
            raise EmbeddingMismatch(f"embedder returned dimension {vectors.shape[1]}, expected {emb.dimension}")

        conn = self.conn
        old = self.get_document(source)
        replaced = old is not None
        try:
            conn.execute("BEGIN")
            if old is not None:
                self._delete_document_rows(old.id, delete_vectors=True)
            cur = conn.execute(
                "INSERT INTO documents(source, title, format, sha256, n_pages, n_chunks, skipped_pages, ingested_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (source, parsed.title, parsed.format, sha256, parsed.n_pages, len(chunks),
                 json.dumps([{"page": s.page, "reason": s.reason} for s in parsed.skipped_pages]), _now()),
            )
            doc_id = cur.lastrowid
            ids: list[int] = []
            for i, c in enumerate(chunks):
                cur = conn.execute(
                    "INSERT INTO chunks(doc_id, ord, page, page_end, heading, text) VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, i, c.page, c.page_end, c.heading, c.text),
                )
                cid = cur.lastrowid
                conn.execute("INSERT INTO chunks_fts(rowid, text, heading, title) VALUES (?, ?, ?, ?)",
                             (cid, c.text, c.heading, parsed.title))
                ids.append(cid)
            if ids:
                for start in range(0, len(ids), 256):
                    sl = slice(start, start + 256)
                    self.collection.upsert(
                        ids=[str(i) for i in ids[sl]],
                        embeddings=vectors[sl].tolist(),
                        metadatas=[{"doc_id": doc_id} for _ in ids[sl]],
                    )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return doc_id, ids, replaced

    @staticmethod
    def _embed_text(title: str, c: ChunkSpec, context_header: bool) -> str:
        if not context_header:
            return c.text
        head = " > ".join(x for x in (title, c.heading) if x)
        return f"{head}\n{c.text}" if head else c.text

    def _delete_document_rows(self, doc_id: int, *, delete_vectors: bool) -> None:
        conn = self.conn
        ids = [r["id"] for r in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            marks = ",".join("?" * len(batch))
            conn.execute(f"DELETE FROM chunks_fts WHERE rowid IN ({marks})", batch)
            conn.execute(f"DELETE FROM chunks WHERE id IN ({marks})", batch)  # cascades to mentions/relations
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.execute("DELETE FROM entities WHERE id NOT IN (SELECT entity_id FROM mentions)")
        if delete_vectors and ids:
            for start in range(0, len(ids), 500):
                self.collection.delete(ids=[str(i) for i in ids[start : start + 500]])

    def delete_document(self, source: str) -> bool:
        doc = self.get_document(source)
        if doc is None:
            return False
        try:
            self.conn.execute("BEGIN")
            self._delete_document_rows(doc.id, delete_vectors=True)
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise
        return True

    def reset(self) -> None:
        """Drop everything: documents, chunks, vectors, graph and the recorded embedding model."""
        conn = self.conn
        for t in ("relations", "mentions", "entities", "chunks_fts", "chunks", "documents"):
            conn.execute(f"DELETE FROM {t}")
        conn.execute("DELETE FROM meta WHERE key IN ('embed_model', 'embed_dim', 'embed_provider')")
        conn.commit()
        self._reset_collection()
        if self.embedder is not None:
            self.bind_embedder(self.embedder)

    # ------------------------------------------------------------------ reading

    def _passage(self, r: sqlite3.Row) -> Passage:
        return Passage(
            id=r["id"], doc_id=r["doc_id"], document=r["title"], source=r["source"], page=r["page"],
            page_end=r["page_end"], heading=r["heading"], text=r["text"], ord=r["ord"],
        )

    _PASSAGE_SQL = (
        "SELECT c.id, c.doc_id, c.ord, c.page, c.page_end, c.heading, c.text, d.title, d.source "
        "FROM chunks c JOIN documents d ON d.id = c.doc_id"
    )

    def get_passage(self, chunk_id: int) -> Passage | None:
        r = self.conn.execute(self._PASSAGE_SQL + " WHERE c.id = ?", (chunk_id,)).fetchone()
        return self._passage(r) if r else None

    def get_passages(self, ids: Sequence[int]) -> dict[int, Passage]:
        out: dict[int, Passage] = {}
        for start in range(0, len(ids), 500):
            batch = list(ids[start : start + 500])
            marks = ",".join("?" * len(batch))
            for r in self.conn.execute(self._PASSAGE_SQL + f" WHERE c.id IN ({marks})", batch):
                out[r["id"]] = self._passage(r)
        return out

    def document_passages(self, doc_id: int) -> list[Passage]:
        rows = self.conn.execute(self._PASSAGE_SQL + " WHERE c.doc_id = ? ORDER BY c.ord", (doc_id,))
        return [self._passage(r) for r in rows]

    # ------------------------------------------------------------------ search primitives

    @staticmethod
    def _doc_filter(doc_ids: Sequence[int] | None, column: str) -> tuple[str, list]:
        if not doc_ids:
            return "", []
        return f" AND {column} IN ({','.join('?' * len(doc_ids))})", list(doc_ids)

    def bm25_search(self, terms: Sequence[str], limit: int, doc_ids: Sequence[int] | None = None) -> list[tuple[int, float]]:
        """Chunks ranked by FTS5 BM25 (best first). Returns (chunk_id, score) with higher = better."""
        terms = list(dict.fromkeys(t for t in terms if t))
        if not terms:
            return []
        query = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        extra, params = self._doc_filter(doc_ids, "c.doc_id")
        sql = (
            f"SELECT chunks_fts.rowid AS id, bm25(chunks_fts, {BM25_WEIGHTS}) AS s FROM chunks_fts "
            "JOIN chunks c ON c.id = chunks_fts.rowid "
            f"WHERE chunks_fts MATCH ?{extra} ORDER BY s LIMIT ?"
        )
        try:
            rows = self.conn.execute(sql, [query, *params, limit]).fetchall()
        except sqlite3.OperationalError:
            return []  # a query FTS5 cannot parse matches nothing
        return [(r["id"], -float(r["s"])) for r in rows]

    def vector_search(
        self, query_vec: np.ndarray, limit: int, doc_ids: Sequence[int] | None = None, *, with_embeddings: bool = False
    ) -> list[VectorHit]:
        col = self.collection
        n = col.count()
        if n == 0:
            return []
        where = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": list(doc_ids)}}
        include = ["distances"] + (["embeddings"] if with_embeddings else [])
        res = col.query(query_embeddings=[np.asarray(query_vec, dtype=np.float32).tolist()],
                        n_results=min(limit, n), where=where, include=include)
        ids = res["ids"][0]
        dists = res["distances"][0]
        embs = res["embeddings"][0] if with_embeddings else [None] * len(ids)
        hits = [
            VectorHit(int(i), 1.0 - float(d), None if e is None else np.asarray(e, dtype=np.float32))
            for i, d, e in zip(ids, dists, embs)
        ]
        # Drop vectors whose chunk no longer exists (e.g. left behind by an interrupted ingest).
        if hits:
            present = set(self.get_passages([h.id for h in hits]))
            hits = [h for h in hits if h.id in present]
        return hits

    # ------------------------------------------------------------------ graph primitives

    def pending_graph_chunks(self, doc_ids: Sequence[int] | None = None) -> list[Passage]:
        extra, params = self._doc_filter(doc_ids, "c.doc_id")
        rows = self.conn.execute(self._PASSAGE_SQL + f" WHERE c.graph_status = 0{extra} ORDER BY c.id", params)
        return [self._passage(r) for r in rows]

    def mark_graph(self, chunk_id: int, status: int) -> None:
        self.conn.execute("UPDATE chunks SET graph_status = ? WHERE id = ?", (status, chunk_id))
        self.conn.commit()

    def store_graph(self, chunk_id: int, entities: Sequence[tuple[str, str]], relations: Sequence[tuple[str, str, str]]) -> None:
        """Write one chunk's extraction. `entities` = (name, type); `relations` = (source, predicate, target)
        where source/target are entity names that appear in `entities`."""
        conn = self.conn
        try:
            conn.execute("BEGIN")
            ids: dict[str, int] = {}
            for name, etype in entities:
                n = norm_entity(name)
                if not n:
                    continue
                conn.execute("INSERT OR IGNORE INTO entities(name, norm, type) VALUES (?, ?, ?)", (name.strip(), n, etype.strip()))
                row = conn.execute("SELECT id, type FROM entities WHERE norm = ?", (n,)).fetchone()
                if etype.strip() and not row["type"]:
                    conn.execute("UPDATE entities SET type = ? WHERE id = ?", (etype.strip(), row["id"]))
                ids[n] = row["id"]
                conn.execute("INSERT OR IGNORE INTO mentions(entity_id, chunk_id) VALUES (?, ?)", (row["id"], chunk_id))
            for src, pred, dst in relations:
                s, d = ids.get(norm_entity(src)), ids.get(norm_entity(dst))
                if s is None or d is None or s == d:
                    continue
                conn.execute("INSERT OR IGNORE INTO relations(src, dst, predicate, chunk_id) VALUES (?, ?, ?, ?)",
                             (s, d, pred.strip().lower(), chunk_id))
            conn.execute("UPDATE chunks SET graph_status = 1 WHERE id = ?", (chunk_id,))
            conn.commit()
        except BaseException:
            conn.rollback()
            raise

    def match_entities(self, text: str, max_ngram: int = 5) -> list[Entity]:
        """Entities whose normalised name occurs in `text` as a whole-word n-gram."""
        words = norm_entity(text).split()
        grams: dict[str, int] = {}
        for n in range(1, max_ngram + 1):
            for i in range(len(words) - n + 1):
                grams.setdefault(" ".join(words[i : i + n]), n)
        if not grams:
            return []
        found: list[Entity] = []
        keys = list(grams)
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            marks = ",".join("?" * len(batch))
            rows = self.conn.execute(
                f"SELECT e.id, e.name, e.type, (SELECT COUNT(*) FROM mentions m WHERE m.entity_id = e.id) AS n "
                f"FROM entities e WHERE e.norm IN ({marks})", batch)
            found.extend(Entity(r["id"], r["name"], r["type"], r["n"]) for r in rows)
        return found

    def find_entity(self, name: str) -> Entity | None:
        n = norm_entity(name)
        if not n:
            return None
        sql = ("SELECT e.id, e.name, e.type, (SELECT COUNT(*) FROM mentions m WHERE m.entity_id = e.id) AS n "
               "FROM entities e WHERE e.norm = ?")
        r = self.conn.execute(sql, (n,)).fetchone()
        if r is None:  # fall back to a prefix/substring match, most-mentioned first
            r = self.conn.execute(sql.replace("WHERE e.norm = ?", "WHERE e.norm LIKE ? ORDER BY n DESC LIMIT 1"),
                                  (f"%{n}%",)).fetchone()
        return Entity(r["id"], r["name"], r["type"], r["n"]) if r else None

    def neighbours(self, entity_id: int, limit: int = 50) -> list[dict]:
        """1-hop neighbours with the relation that links them."""
        rows = self.conn.execute(
            "SELECT r.predicate, 'out' AS direction, e.id, e.name, e.type, r.chunk_id, "
            "(SELECT COUNT(*) FROM mentions m WHERE m.entity_id = e.id) AS n "
            "FROM relations r JOIN entities e ON e.id = r.dst WHERE r.src = ? "
            "UNION ALL "
            "SELECT r.predicate, 'in', e.id, e.name, e.type, r.chunk_id, "
            "(SELECT COUNT(*) FROM mentions m WHERE m.entity_id = e.id) "
            "FROM relations r JOIN entities e ON e.id = r.src WHERE r.dst = ? LIMIT ?",
            (entity_id, entity_id, limit))
        return [dict(r) for r in rows]

    def entity_chunks(self, entity_id: int, doc_ids: Sequence[int] | None = None) -> list[int]:
        extra, params = self._doc_filter(doc_ids, "c.doc_id")
        rows = self.conn.execute(
            f"SELECT m.chunk_id FROM mentions m JOIN chunks c ON c.id = m.chunk_id WHERE m.entity_id = ?{extra} ORDER BY m.chunk_id",
            [entity_id, *params])
        return [r["chunk_id"] for r in rows]
