"""
core/hybrid/centroid_search.py — Cosine nearest-centroid intent classifier.

Architecture role
-----------------
Stage 2 in the hybrid pipeline. Takes a Wav2Vec2 embedding vector and
returns the most similar known phrase by cosine similarity to pre-computed
class centroids. This replaces the softmax classification head for dynamic
phrases: adding a new command requires only appending its centroid, without
retraining any weights.

Design rationale
----------------
Cosine similarity is preferred over Euclidean distance in high-dimensional
embedding spaces because Wav2Vec2 embeddings cluster by direction rather
than by magnitude. L2-normalising both the stored centroids and the query
before computing dot products is numerically equivalent to cosine similarity
but faster (single matrix-vector multiply).

Scalability
-----------
With N centroids of dimension D:
  - Memory:    N × D × 4 bytes  (e.g. 20 phrases × 1024-D = 80 KB)
  - Inference: O(N × D) — dominated by a single numpy matmul, well under 1 ms.

Persistence
-----------
    search = CentroidSearch()
    search.add_centroid("машина", centroid_vec)
    search.save_npz("artifacts/hybrid/centroids.npy",
                    "artifacts/hybrid/centroid_labels.json")

    search2 = CentroidSearch.load_npz(...)
    label, score, all_scores = search2.search(embedding, threshold=0.75)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CentroidSearch:
    """Cosine nearest-centroid lookup over a registry of phrase embeddings.

    The registry maps phrase label strings to their mean L2-normalised
    embedding vectors. At inference, the query embedding is L2-normalised
    and compared to all centroids via a single matrix-vector dot product.

    Args:
        min_cosine_similarity: Default acceptance threshold (can be overridden
                               per label via ``per_label_thresholds``).
        per_label_thresholds:  Dict mapping label → individual threshold.
    """

    def __init__(
        self,
        min_cosine_similarity: float = 0.75,
        per_label_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.min_cosine_similarity: float = float(min_cosine_similarity)
        self.per_label_thresholds: Dict[str, float] = per_label_thresholds or {}

        self._labels: List[str] = []
        self._matrix: Optional[np.ndarray] = None    # (N_labels, D) float32

    # ------------------------------------------------------------------
    # Building the registry
    # ------------------------------------------------------------------

    def add_centroid(self, label: str, centroid: np.ndarray) -> None:
        """Register or update the centroid for *label*.

        The vector is L2-normalised before storage. If *label* already exists
        it is overwritten (update semantics).

        Args:
            label:    Exact phrase string (e.g. ``"самый малый вперед"``).
            centroid: 1-D float32 embedding of arbitrary (but consistent) dimension.

        Raises:
            ValueError: If *centroid* is all-zeros or has dimension mismatch
                        with previously registered centroids.
        """
        vec = np.asarray(centroid, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError(
                f"centroid must be 1-D, got shape {vec.shape} for label '{label}'"
            )
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            raise ValueError(
                f"centroid for '{label}' is effectively zero — "
                "check embedding extraction."
            )
        vec = vec / norm

        if self._matrix is not None and vec.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"Dimension mismatch: registered centroids have D={self._matrix.shape[1]}, "
                f"new centroid for '{label}' has D={vec.shape[0]}."
            )

        if label in self._labels:
            idx = self._labels.index(label)
            self._matrix[idx] = vec
            logger.debug("Updated centroid for '%s'.", label)
        else:
            self._labels.append(label)
            row = vec[None, :]
            self._matrix = (
                row
                if self._matrix is None
                else np.concatenate([self._matrix, row], axis=0)
            )
            logger.debug("Added centroid for '%s' (total=%d).", label, len(self._labels))

    def build_from_embeddings(
        self,
        embeddings: np.ndarray,
        labels: List[str],
    ) -> "CentroidSearch":
        """Compute per-class mean centroids from a labelled embedding array.

        Args:
            embeddings: Float32 array of shape ``(N, D)``.
            labels:     List of N label strings (one per row of *embeddings*).

        Returns:
            Self (enables method chaining).
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        label_arr = np.array(labels)
        for lbl in sorted(set(labels)):
            mask = label_arr == lbl
            centroid = embeddings[mask].mean(axis=0)
            self.add_centroid(lbl, centroid)
        logger.info(
            "CentroidSearch built from %d samples → %d class centroids.",
            len(labels), len(self._labels),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def search(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Find the nearest centroid to *embedding* by cosine similarity.

        Args:
            embedding: 1-D float32 embedding of shape ``(D,)``.
            threshold: Acceptance threshold override. If ``None``, uses
                       ``min_cosine_similarity`` (with per-label overrides).

        Returns:
            A tuple of:
              - ``best_label``: The predicted label, or ``None`` if the top
                score is below the effective threshold.
              - ``best_score``: Cosine similarity to the best centroid (float,
                range approximately [-1, 1] but typically [0, 1] for speech).
              - ``all_scores``: Dict mapping every registered label to its
                cosine similarity score.

        Raises:
            RuntimeError: If the registry is empty (no centroids added yet).
        """
        if self._matrix is None or len(self._labels) == 0:
            raise RuntimeError(
                "CentroidSearch has no registered centroids. "
                "Call add_centroid() or build_from_embeddings() first."
            )

        emb = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm < 1e-12:
            logger.warning("Received near-zero embedding — returning no match.")
            return None, 0.0, {lbl: 0.0 for lbl in self._labels}

        normed = emb / norm

        # Single matrix-vector multiply → all cosine similarities at once
        scores: np.ndarray = (self._matrix @ normed).astype(np.float32)

        all_scores: Dict[str, float] = {
            lbl: float(s) for lbl, s in zip(self._labels, scores)
        }

        best_idx = int(np.argmax(scores))
        best_label = self._labels[best_idx]
        best_score = float(scores[best_idx])

        # Determine effective threshold for the best-matching label
        if threshold is not None:
            effective_threshold = float(threshold)
        else:
            effective_threshold = self.per_label_thresholds.get(
                best_label, self.min_cosine_similarity
            )

        if best_score < effective_threshold:
            logger.debug(
                "CentroidSearch: best_label='%s' score=%.4f < threshold=%.4f → rejected.",
                best_label, best_score, effective_threshold,
            )
            return None, best_score, all_scores

        logger.debug(
            "CentroidSearch: best_label='%s' score=%.4f (threshold=%.4f).",
            best_label, best_score, effective_threshold,
        )
        return best_label, best_score, all_scores

    def scores_as_probs(self, all_scores: Dict[str, float]) -> np.ndarray:
        """Convert a ``{label: score}`` dict to a probability-like numpy vector.

        Applies a shift-and-softmax over cosine scores so the output has the
        same shape as an ``OnnxAudioEngine`` ``probs`` vector, enabling
        direct comparison in logging / benchmarks.

        Args:
            all_scores: Dict of cosine similarity scores (output of ``search()``).

        Returns:
            1-D float32 array aligned with ``self.labels`` ordering.
        """
        raw = np.array([all_scores.get(lbl, 0.0) for lbl in self._labels], dtype=np.float32)
        shifted = raw - raw.max()
        exp = np.exp(shifted)
        return (exp / exp.sum()).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_npz(
        self,
        centroids_path: str | Path,
        labels_path: str | Path,
    ) -> None:
        """Save centroids matrix and label list to disk.

        Args:
            centroids_path: ``.npy`` file for the ``(N, D)`` centroid matrix.
            labels_path:    ``.json`` file for the label list.

        Raises:
            RuntimeError: If no centroids have been registered.
        """
        if self._matrix is None:
            raise RuntimeError("Nothing to save — registry is empty.")
        cp = Path(centroids_path)
        lp = Path(labels_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        lp.parent.mkdir(parents=True, exist_ok=True)

        np.save(str(cp), self._matrix)
        with open(lp, "w", encoding="utf-8") as f:
            json.dump(self._labels, f, ensure_ascii=False, indent=2)
        logger.info(
            "CentroidSearch saved: %d centroids (D=%d) → %s | %s",
            len(self._labels), self._matrix.shape[1], cp, lp,
        )

    @classmethod
    def load_npz(
        cls,
        centroids_path: str | Path,
        labels_path: str | Path,
        min_cosine_similarity: float = 0.75,
        per_label_thresholds: Optional[Dict[str, float]] = None,
    ) -> "CentroidSearch":
        """Load centroids and labels from disk into a new ``CentroidSearch``.

        Args:
            centroids_path:        ``.npy`` file with the centroid matrix.
            labels_path:           ``.json`` file with the label list.
            min_cosine_similarity: Threshold forwarded to the new instance.
            per_label_thresholds:  Per-label thresholds forwarded to instance.

        Returns:
            Populated ``CentroidSearch`` instance.

        Raises:
            FileNotFoundError: If either file is missing.
        """
        cp = Path(centroids_path)
        lp = Path(labels_path)
        if not cp.exists():
            raise FileNotFoundError(f"Centroids file not found: {cp}")
        if not lp.exists():
            raise FileNotFoundError(f"Labels file not found: {lp}")

        matrix = np.load(str(cp)).astype(np.float32)
        with open(lp, "r", encoding="utf-8") as f:
            labels: List[str] = json.load(f)

        if len(labels) != matrix.shape[0]:
            raise ValueError(
                f"Label count ({len(labels)}) != centroid rows ({matrix.shape[0]})"
            )

        instance = cls(
            min_cosine_similarity=min_cosine_similarity,
            per_label_thresholds=per_label_thresholds,
        )
        instance._labels = labels
        instance._matrix = matrix
        logger.info(
            "CentroidSearch loaded: %d labels, D=%d from %s",
            len(labels), matrix.shape[1], cp,
        )
        return instance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def labels(self) -> List[str]:
        """Ordered list of registered label strings."""
        return list(self._labels)

    @property
    def n_labels(self) -> int:
        """Number of registered centroids."""
        return len(self._labels)

    @property
    def embedding_dim(self) -> Optional[int]:
        """Embedding dimension D, or None if no centroids registered yet."""
        return self._matrix.shape[1] if self._matrix is not None else None
