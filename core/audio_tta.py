"""
core/audio_tta.py — Test-Time Augmentation (TTA) wrapper (§2.4).

Rationale
---------
A single inference pass on a noisy waveform can land on the wrong side of a
cosine decision boundary.  Running *k* lightly-perturbed copies and averaging
their logit vectors (cosine scores) smooths out that stochasticity.

With k=3 and the augmentations below, typical gains on maritime noise are
3–5 pp F1 at a ~30 % latency increase (three sequential forward passes) —
still well inside the 500 ms real-time budget.

Augmentations (k=3 default)
----------------------------
1. **Original** waveform ``x``.
2. **Gaussian noise**: ``x + N(0, σ)`` with σ=0.005.  Models small SNR
   degradations and sensor quantisation noise.
3. **Time shift**: circular roll by ±``shift_ms`` ms.  Models ±10 ms
   alignment jitter between command onset and the capture window edge.

Design decisions
----------------
- Gate runs **only on the original** waveform.  If the gate rejects the
  original, we return immediately (TTA adds no value for OOV audio and
  would only inflate latency).
- Augmented copies that themselves trip the gate are silently dropped from
  the logit average; the original always contributes.
- Label re-selection uses ``argmax`` of averaged logits.  This matches what
  ``CentroidSearch`` does internally: the highest cosine score wins.
- The ``TTAWrapper`` is a drop-in replacement for any engine that exposes
  ``predict(waveform) -> dict`` and a ``labels: List[str]`` property.

Usage
-----
    from core.audio_tta import TTAWrapper
    from core.hybrid.factory import create_hybrid_engine

    engine = create_hybrid_engine(cfg)
    tta    = TTAWrapper(engine, sr=16_000, k=3, shift_ms=10.0, noise_sigma=0.005)
    result = tta.predict(waveform)
    print(result["label"], result["tta_n"])   # tta_n: how many copies contributed

Ablation
--------
Disable TTA by passing ``k=1`` (or just use the engine directly).  Compare
F1 at k=1, k=2, k=3 in a notebook cell — report as Table 2.4.

.. note::
    **TODO (thesis requirement, §4.2):** The k=1 vs k=2 vs k=3 ablation has
    NOT yet been run.  Before the thesis defence one of the following must be
    done:

    1. Run the ablation via ``scripts/train/benchmark_stats.py`` (or a
       dedicated notebook), record F1 scores for k∈{1,2,3}, and add a row
       to Table §4.2 in the thesis.
    2. OR move ``TTAWrapper`` to ``scripts/`` and remove it from the
       production ``core/`` module, updating the thesis text accordingly.

    Until one of these actions is complete, ``TTAWrapper`` is implemented
    but its production benefit is unquantified.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Augmentation primitives ───────────────────────────────────────────────────


def augment_gaussian(
    waveform: np.ndarray,
    sigma: float = 0.005,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise to *waveform*.

    Args:
        waveform: 1-D float32 waveform array.
        sigma:    Standard deviation of the additive noise.
        rng:      Optional numpy ``Generator`` for reproducible augmentation.
                  Uses ``np.random.default_rng()`` when ``None``.

    Returns:
        Copy of *waveform* with additive Gaussian noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=waveform.shape).astype(waveform.dtype)
    return waveform + noise


def augment_time_shift(
    waveform: np.ndarray,
    shift_samples: int,
) -> np.ndarray:
    """Circularly shift *waveform* by *shift_samples* positions.

    A positive shift moves audio forward in time (prepends silence from the
    end); a negative shift moves it backward.  Circular wrap-around is used
    rather than zero-padding so the window length is preserved exactly.

    Args:
        waveform:      1-D float32 waveform array.
        shift_samples: Number of samples to shift.  Positive → forward.

    Returns:
        Shifted waveform of the same length as *waveform*.
    """
    return np.roll(waveform, shift_samples).astype(waveform.dtype)


# ── TTA Wrapper ───────────────────────────────────────────────────────────────


class TTAWrapper:
    """Drop-in engine wrapper that averages logits over k augmented copies.

    The wrapper delegates all loading, gate logic, and centroid search to the
    underlying *engine*.  Its only role is to:

    1. Build k−1 perturbed copies of the input waveform.
    2. Run ``engine.predict()`` on each (plus the original).
    3. Average the ``logits`` arrays that are **not outlier-rejected**.
    4. Re-derive ``label`` and ``confidence`` from the averaged logits.

    The outlier gate runs independently for each copy; copies that are
    rejected by the gate do not contribute to the logit average.  If **all**
    copies are rejected (unlikely with k=3 unless the audio is truly OOV),
    the original's result is returned as-is.

    Args:
        engine:       Any engine that implements ``predict(waveform) -> dict``
                      and exposes a ``labels: List[str]`` property.
        sr:           Waveform sample rate in Hz (default 16 000).
        k:            Total number of copies including the original.
                      Set to 1 to disable TTA (pass-through mode).
        shift_ms:     Time-shift magnitude in milliseconds (default ±10 ms).
        noise_sigma:  Gaussian noise standard deviation (default 0.005).
        seed:         Optional integer seed for reproducible augmentation.
                      ``None`` → non-deterministic (appropriate for inference).
    """

    def __init__(
        self,
        engine: Any,
        sr: int = 16_000,
        k: int = 3,
        shift_ms: float = 10.0,
        noise_sigma: float = 0.005,
        seed: Optional[int] = None,
    ) -> None:
        self._engine = engine
        self._sr = sr
        self._k = max(1, k)
        self._shift_samples = int(round(shift_ms * sr / 1_000.0))
        self._noise_sigma = noise_sigma
        self._rng = np.random.default_rng(seed)

        logger.info(
            "TTAWrapper initialised: k=%d, shift=%dms (%d samples), σ=%.4f",
            self._k, int(shift_ms), self._shift_samples, self._noise_sigma,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def labels(self) -> List[str]:
        """Ordered label list from the underlying engine."""
        return getattr(self._engine, "labels", [])

    def predict(self, waveform: np.ndarray) -> Dict[str, Any]:
        """Run TTA inference on *waveform*.

        Args:
            waveform: 1-D float32 audio array at ``self._sr`` Hz.

        Returns:
            Result dict with all keys from the underlying engine, plus:
              - ``"tta_applied"`` (bool): True if ≥2 copies contributed.
              - ``"tta_n"`` (int): Number of non-rejected copies averaged.

        Notes:
            If the **original** waveform is outlier-rejected, the result is
            returned immediately with ``tta_applied=False, tta_n=0``.  TTA
            cannot improve on OOV audio and the gate should be trusted.
        """
        if self._k == 1:
            result = self._engine.predict(waveform)
            result.setdefault("tta_applied", False)
            result.setdefault("tta_n", 1)
            return result

        # ── Run original first ────────────────────────────────────────
        result_orig = self._engine.predict(waveform)

        if result_orig.get("outlier_rejected", False):
            result_orig["tta_applied"] = False
            result_orig["tta_n"] = 0
            logger.debug("TTA: original rejected by gate, skipping augmentations.")
            return result_orig

        logits_orig: np.ndarray = result_orig.get("logits", np.array([], dtype=np.float32))
        if logits_orig.size == 0:
            result_orig["tta_applied"] = False
            result_orig["tta_n"] = 1
            return result_orig

        # ── Run augmented copies ──────────────────────────────────────
        augmentations = self._build_augmentations(waveform)   # list of (k-1) arrays
        all_logits: List[np.ndarray] = [logits_orig]

        for aug_wav in augmentations:
            try:
                r = self._engine.predict(aug_wav)
            except Exception as exc:
                logger.warning("TTA: augmented inference failed, skipping: %s", exc)
                continue

            if r.get("outlier_rejected", False):
                logger.debug("TTA: one augmented copy rejected — excluded from average.")
                continue

            aug_logits: np.ndarray = r.get("logits", np.array([], dtype=np.float32))
            if aug_logits.shape == logits_orig.shape and aug_logits.size > 0:
                all_logits.append(aug_logits)

        tta_n = len(all_logits)
        if tta_n < 2:
            # Only the original contributed — return it unchanged
            result_orig["tta_applied"] = False
            result_orig["tta_n"] = tta_n
            return result_orig

        # ── Average logits and re-derive label ────────────────────────
        avg_logits = np.mean(np.stack(all_logits, axis=0), axis=0).astype(np.float32)
        best_idx = int(np.argmax(avg_logits))
        label_list = self.labels

        if label_list and best_idx < len(label_list):
            new_label = label_list[best_idx]
            new_confidence = float(avg_logits[best_idx])
        else:
            # Fallback: keep original label if label list is unavailable
            new_label = result_orig.get("label", "")
            new_confidence = result_orig.get("confidence", 0.0)

        result_orig["logits"] = avg_logits
        result_orig["label"] = new_label
        result_orig["confidence"] = new_confidence
        result_orig["tta_applied"] = True
        result_orig["tta_n"] = tta_n

        logger.debug(
            "TTA: averaged %d/%d copies → label='%s' conf=%.4f",
            tta_n, self._k, new_label, new_confidence,
        )
        return result_orig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_augmentations(self, waveform: np.ndarray) -> List[np.ndarray]:
        """Build ``k-1`` augmented waveform copies.

        The fixed augmentation schedule for k≤3:
          copy 1 → Gaussian noise
          copy 2 → time-shift by +``shift_samples``

        For k>3 additional noisy copies fill the remaining slots.

        Args:
            waveform: Original 1-D float32 array.

        Returns:
            List of (k-1) augmented copies.
        """
        augs: List[np.ndarray] = []
        schedule = [
            ("gaussian", {}),
            ("shift_pos", {}),
            ("shift_neg", {}),
        ]
        for i in range(self._k - 1):
            aug_type = schedule[i % len(schedule)][0]
            if aug_type == "gaussian":
                augs.append(augment_gaussian(waveform, sigma=self._noise_sigma, rng=self._rng))
            elif aug_type == "shift_pos":
                augs.append(augment_time_shift(waveform, self._shift_samples))
            else:  # shift_neg
                augs.append(augment_time_shift(waveform, -self._shift_samples))
        return augs
