"""
Two-layer dedup:
  Layer 1: URL check via SQLite (fast, zero cost).
  Layer 2: Semantic similarity via ChromaDB + nomic-embed-text.
           < 0.70 similarity → novel
           0.70–0.85 similarity → include as new angle
           > 0.85 similarity → light-model novelty check
             keep=True  → KEEP_BOTH (mark conflicting)
             keep=False → DUPLICATE (discard)

Embeddings are pre-computed by the caller (Brain.embed_texts) so the
check() + upsert() path never re-embeds the same text twice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from .state import StateReader

if TYPE_CHECKING:
    from .brain import Brain

log = logging.getLogger(__name__)


# ── Disposition ───────────────────────────────────────────────────────────────

class Disposition(str, Enum):
    NOVEL      = "novel"       # similarity < 0.70 — clearly new
    NEW_ANGLE  = "new_angle"   # 0.70–0.85 — similar but worth keeping
    KEEP_BOTH  = "keep_both"   # > 0.85, LLM says keep — mark conflicting
    DUPLICATE  = "duplicate"   # > 0.85, LLM says discard


@dataclass
class DedupeResult:
    disposition: Disposition
    nearest_id:   Optional[str]   = None
    nearest_text: Optional[str]   = None
    distance:     Optional[float] = None
    reason:       str             = ""


# ── URL-level dedup (Layer 1, unchanged from Phase 1) ────────────────────────

class URLDedup:
    def __init__(self, reader: StateReader):
        self._reader = reader

    def is_seen(self, url: str, session_id: str) -> bool:
        return self._reader.url_seen(url, session_id)


# ── Semantic dedup (Layer 2, Phase 3) ────────────────────────────────────────

class SemanticDedup:
    def __init__(
        self,
        brain: "Brain",
        config: dict,
        session_id: str,
    ):
        self._brain = brain
        self._session_id = session_id

        # Distance thresholds (cosine space: distance = 1 − similarity)
        self._novel_dist = 1.0 - config.get("similarity_novel_threshold", 0.70)      # 0.30
        self._dup_dist   = 1.0 - config.get("similarity_duplicate_threshold", 0.85)  # 0.15

        # ChromaDB in-memory client — one collection per session
        try:
            import chromadb
            self._chroma = chromadb.EphemeralClient()
            self._collection = self._chroma.get_or_create_collection(
                name=f"findings_{session_id}",
                metadata={"hnsw:space": "cosine"},
            )
            log.info("SemanticDedup: ChromaDB collection ready (session=%s)", session_id)
        except ImportError:
            log.warning(
                "chromadb not installed — semantic dedup disabled. "
                "Run: pip install chromadb"
            )
            self._chroma = None
            self._collection = None

    @property
    def enabled(self) -> bool:
        return self._collection is not None

    # ── core dedup check ─────────────────────────────────────────────────────

    async def check(
        self,
        finding_text: str,
        finding_id: str,
        source_url: str,
        embedding: list[float],
    ) -> DedupeResult:
        """
        Check a pre-embedded finding against the collection.
        Does NOT add the finding to the collection — call upsert() after
        deciding to keep it.
        """
        if not self.enabled:
            return DedupeResult(disposition=Disposition.NOVEL)

        if self._collection.count() == 0:
            return DedupeResult(disposition=Disposition.NOVEL)

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["distances", "documents", "metadatas"],
        )

        if not results["distances"] or not results["distances"][0]:
            return DedupeResult(disposition=Disposition.NOVEL)

        distance     = results["distances"][0][0]
        nearest_id   = results["ids"][0][0]
        nearest_text = results["documents"][0][0]

        if distance >= self._novel_dist:
            # similarity < 0.70 — clearly novel
            return DedupeResult(
                disposition=Disposition.NOVEL,
                nearest_id=nearest_id,
                distance=distance,
            )

        if distance >= self._dup_dist:
            # similarity 0.70–0.85 — new angle, include with note
            return DedupeResult(
                disposition=Disposition.NEW_ANGLE,
                nearest_id=nearest_id,
                nearest_text=nearest_text,
                distance=distance,
            )

        # similarity > 0.85 — run light-model novelty check
        llm = await self._brain.novelty_check(
            existing_finding=nearest_text,
            new_finding=finding_text,
        )

        if llm["keep"]:
            return DedupeResult(
                disposition=Disposition.KEEP_BOTH,
                nearest_id=nearest_id,
                nearest_text=nearest_text,
                distance=distance,
                reason=llm["reason"],
            )
        else:
            return DedupeResult(
                disposition=Disposition.DUPLICATE,
                nearest_id=nearest_id,
                nearest_text=nearest_text,
                distance=distance,
                reason=llm["reason"],
            )

    # ── collection upsert ────────────────────────────────────────────────────

    def upsert(
        self,
        finding_id: str,
        finding_text: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Add a confirmed-novel finding to the ChromaDB collection."""
        if not self.enabled:
            return
        self._collection.upsert(
            ids=[finding_id],
            embeddings=[embedding],
            documents=[finding_text],
            metadatas=[metadata],
        )

    async def upsert_batch(self, findings: list[dict]) -> None:
        """Pre-populate the collection from existing findings (for Phase 4 resume)."""
        if not self.enabled or not findings:
            return
        batch_size = 32
        for i in range(0, len(findings), batch_size):
            batch = findings[i : i + batch_size]
            texts = [f["finding_text"] for f in batch]
            embeddings = await self._brain.embed_texts(texts)
            self._collection.upsert(
                ids=[f["id"] for f in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[
                    {
                        "source_url": f.get("source_url", ""),
                        "session_id": f.get("session_id", ""),
                    }
                    for f in batch
                ],
            )
        log.info("SemanticDedup: pre-loaded %d findings into ChromaDB", len(findings))
