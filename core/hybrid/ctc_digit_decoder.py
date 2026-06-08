"""
core/hybrid/ctc_digit_decoder.py — CTC-based numeric slot decoder (Variant B).

Architecture role
-----------------
Stage 4 (optional) in the hybrid pipeline, replacing / augmenting the MLP
``NumberRegressor`` (Variant A) for open-range slot intents such as
``"курс УГОЛ градусов"`` or ``"скорость УГОЛ узлов"``.

How it works
------------
A single ``Linear(frame_dim, VOCAB_SIZE)`` layer — the **CTC numeral head** —
is applied to the per-frame projected features produced by the Wav2Vec2
encoder (``outputs[2]`` from the updated ONNX bundle).  Greedy CTC decoding
collapses the frame-level token sequence into a sequence of Russian numeral
words, whose additive values are then summed to produce the final integer.

Vocabulary (32 tokens) — additive Russian numeral words
--------------------------------------------------------
    0  → blank      (CTC blank; discarded during decoding)
    1  → ноль       (0)
    2  → один       (1)    3  → одна       (1)   # gender variant
    4  → два        (2)    5  → две        (2)   # gender variant
    6  → три        (3)
    7  → четыре     (4)
    8  → пять       (5)
    9  → шесть      (6)
    10 → семь       (7)
    11 → восемь     (8)
    12 → девять     (9)
    13 → десять     (10)
    14 → одиннадцать(11)
    15 → двенадцать (12)
    16 → тринадцать (13)
    17 → четырнадцать(14)
    18 → пятнадцать (15)
    19 → шестнадцать(16)
    20 → семнадцать (17)
    21 → восемнадцать(18)
    22 → девятнадцать(19)
    23 → двадцать   (20)
    24 → тридцать   (30)
    25 → сорок      (40)
    26 → пятьдесят  (50)
    27 → шестьдесят (60)
    28 → семьдесят  (70)
    29 → восемьдесят(80)
    30 → девяносто  (90)
    31 → сто        (100)
    32 → двести     (200)
    33 → триста     (300)

Decoding rule: SUM all token values.
    "двести восемьдесят пять" → [32, 29, 8] → 200 + 80 + 5 = 285
    "сто"                     → [31]         → 100
    "тридцать"                → [24]         → 30
    "двадцать один"           → [23, 2]      → 20 + 1 = 21
    "ноль"                    → [1]          → 0

Verbalization assumption
------------------------
Commands use **full Russian compound numerals** (not digit-by-digit):
    "курс двести восемьдесят пять градусов"   →  285
    "скорость пятнадцать узлов"               →  15
    "поворот влево на девяносто"              →  90

Training data format (for ``scripts/hybrid/train_ctc_head.py``)
---------------------------------------------------------------
CSV with columns:

    path, label, numerals
    artifacts/data/nums/kurs_100.wav, курс УГОЛ градусов, сто
    artifacts/data/nums/kurs_285.wav, курс УГОЛ градусов, двести восемьдесят пять
    artifacts/data/nums/speed_15.wav, скорость УГОЛ узлов, пятнадцать

``numerals`` is a space-separated sequence of Russian numeral word tokens
from the vocabulary above (lowercase, without surrounding context words like
"курс" or "градусов").

Usage
-----
    # Inference:
    decoder = CTCDigitDecoder.load(
        "artifacts/hybrid/ctc_numeral_head.pt",
        frame_dim=256,
        min_val=0, max_val=359,
    )
    value, confidence = decoder.predict(frames)  # frames: (T, D_proj)
    # e.g.  285.0  0.91

Implementation status (2026-05)
--------------------------------
РЕАЛИЗОВАН, НЕ ОТКАЛИБРОВАН.

Класс полностью реализован и интегрирован в HybridAudioEngine
(Stage 4 Variant B).  Активируется при
``cfg.ctc_decoder.enabled=True`` и наличии артефакта ``head_path``.

Ограничения:

* Обучающий корпус числительных речи (target: ~500 примеров на класс)
  к моменту защиты ВКР не собран в достаточном объёме.
* CTC-head не обучен на реальных данных — только архитектурная готовность.
* Точность CTCDigitDecoder на реальном речевом сигнале не измерялась.

При ``cfg.ctc_decoder.enabled=False`` (по умолчанию в production) пайплайн
использует только NumberRegressor (Variant A) без какого-либо влияния на
производительность или точность основной системы.

Для активации обучения::

    python scripts/hybrid/train_ctc_head.py --help

Per thesis revision notes:

* Previously categorised as a "future direction"; reclassified as
  "implemented, awaiting evaluation" (see ВКР revision 2026-05).
* **NOT included in thesis §4 result tables** until calibration is complete
  and F1 / accuracy metrics are measured.
* ``slot_method="ctc"`` will appear in A/B telemetry only after the head
  is trained and the ONNX bundle is re-exported with ``has_frames=True``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Vocabulary ──────────────────────────────────────────────────────────────
BLANK_IDX: int = 0

# Each entry: token_index → (russian_word, additive_numeric_value)
# Index 0 is reserved for CTC blank (no word, no value).
_VOCAB: Dict[int, Tuple[str, int]] = {
    1:  ("ноль",        0),
    2:  ("один",        1),
    3:  ("одна",        1),     # feminine gender variant (e.g. двадцать одна)
    4:  ("два",         2),
    5:  ("две",         2),     # feminine gender variant (e.g. двести две)
    6:  ("три",         3),
    7:  ("четыре",      4),
    8:  ("пять",        5),
    9:  ("шесть",       6),
    10: ("семь",        7),
    11: ("восемь",      8),
    12: ("девять",      9),
    13: ("десять",      10),
    14: ("одиннадцать", 11),
    15: ("двенадцать",  12),
    16: ("тринадцать",  13),
    17: ("четырнадцать",14),
    18: ("пятнадцать",  15),
    19: ("шестнадцать", 16),
    20: ("семнадцать",  17),
    21: ("восемнадцать",18),
    22: ("девятнадцать",19),
    23: ("двадцать",    20),
    24: ("тридцать",    30),
    25: ("сорок",       40),
    26: ("пятьдесят",   50),
    27: ("шестьдесят",  60),
    28: ("семьдесят",   70),
    29: ("восемьдесят", 80),
    30: ("девяносто",   90),
    31: ("сто",         100),
    32: ("двести",      200),
    33: ("триста",      300),
}

VOCAB_SIZE: int = len(_VOCAB) + 1   # +1 for blank at index 0  (= 34)

# Token index → additive numeric value (blank excluded)
_TOKEN_TO_VALUE: Dict[int, int] = {idx: val for idx, (_, val) in _VOCAB.items()}

# Russian word → token index (lowercase lookup for training data parsing)
WORD_TO_TOKEN: Dict[str, int] = {word: idx for idx, (word, _) in _VOCAB.items()}


def _greedy_ctc_decode(log_probs: np.ndarray) -> List[int]:
    """Greedy CTC decoding: argmax per frame → collapse repeats → remove blank.

    Args:
        log_probs: Float32 array of shape ``(T, VOCAB_SIZE)`` — log-softmax
                   outputs from the CTC head.

    Returns:
        List of decoded token indices (blank-free, repeat-collapsed).
        An empty list means no numeral token was detected.
    """
    best = np.argmax(log_probs, axis=-1)    # (T,)

    decoded: List[int] = []
    prev = BLANK_IDX
    for tok in best:
        tok = int(tok)
        if tok != prev:
            if tok != BLANK_IDX:
                decoded.append(tok)
        prev = tok
    return decoded


def _tokens_to_int(tokens: List[int]) -> Optional[int]:
    """Convert decoded token list to an integer by summing additive values.

    Russian compound numerals are additive:
        [двести, восемьдесят, пять] → 200 + 80 + 5 = 285

    Args:
        tokens: Decoded token list from ``_greedy_ctc_decode``.

    Returns:
        Summed integer value, or ``None`` if the list is empty or contains
        an unknown token.
    """
    if not tokens:
        return None
    total = 0
    for tok in tokens:
        val = _TOKEN_TO_VALUE.get(tok)
        if val is None:
            logger.warning("Unknown CTC token %d — decode aborted.", tok)
            return None
        total += val
    return total


# ── PyTorch head (used during training and optional PyTorch inference) ───────

class DigitCTCHead:
    """Lightweight linear CTC head: ``Linear(frame_dim, VOCAB_SIZE)``.

    Despite the legacy name, this head handles full Russian compound numerals
    (not digit-by-digit).  The name is kept for API compatibility.

    Kept as a plain wrapper so the core module does NOT import torch at
    module-load time — torch is only imported inside methods that need it.
    This preserves the fast-startup contract for the ONNX inference path.

    Args:
        frame_dim: Dimensionality of the projected frame features (``D_proj``).
                   Read from ``onnx_config.json["frame_dim"]`` at load time.
    """

    def __init__(self, frame_dim: int) -> None:
        self.frame_dim: int = frame_dim
        self._net = None        # torch.nn.Linear, populated by _build()

    def _build(self) -> None:
        if self._net is None:
            import torch.nn as nn
            self._net = nn.Linear(self.frame_dim, VOCAB_SIZE)

    def forward(self, frames):  # type: ignore[return]
        """Run the linear head and return log-softmax over VOCAB_SIZE.

        Args:
            frames: Float32 tensor of shape ``(B, T, frame_dim)``.

        Returns:
            Log-softmax tensor of shape ``(B, T, VOCAB_SIZE)``.
        """
        import torch.nn.functional as F
        self._build()
        return F.log_softmax(self._net(frames), dim=-1)  # type: ignore[misc]

    def parameters(self):
        self._build()
        return self._net.parameters()   # type: ignore[union-attr]

    def train(self, mode: bool = True) -> "DigitCTCHead":
        self._build()
        self._net.train(mode)           # type: ignore[union-attr]
        return self

    def eval(self) -> "DigitCTCHead":
        return self.train(False)

    def state_dict(self):
        self._build()
        return self._net.state_dict()   # type: ignore[union-attr]

    def load_state_dict(self, sd) -> None:
        self._build()
        self._net.load_state_dict(sd)   # type: ignore[union-attr]

    def to(self, device):
        self._build()
        self._net = self._net.to(device)    # type: ignore[union-attr]
        return self


# ── Public decoder ───────────────────────────────────────────────────────────

class CTCDigitDecoder:
    """Inference wrapper for the CTC numeral head.

    Consumes per-frame projected features ``(T, D_proj)`` and returns a
    predicted integer (summed from decoded Russian numeral tokens) with a
    confidence score.

    Args:
        frame_dim: Dimensionality of input frame features.
        min_val:   Lower bound for clipping the decoded value.
        max_val:   Upper bound for clipping the decoded value.
        device:    ``"cpu"`` or ``"cuda"`` for the head forward pass.

    Example:
        >>> decoder = CTCDigitDecoder.load(
        ...     "artifacts/hybrid/ctc_numeral_head.pt",
        ...     frame_dim=256, min_val=0, max_val=359,
        ... )
        >>> value, conf = decoder.predict(frames_np)
        >>> print(value, conf)   # e.g.  285.0  0.91
    """

    def __init__(
        self,
        frame_dim: int,
        min_val: float = 0.0,
        max_val: float = 359.0,
        device: str = "cpu",
    ) -> None:
        self.frame_dim: int = frame_dim
        self.min_val: float = float(min_val)
        self.max_val: float = float(max_val)
        self.device: str = device

        self._head = DigitCTCHead(frame_dim)
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, frames: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """Decode a Russian compound numeral from per-frame projected features.

        Args:
            frames: Float32 array of shape ``(T, D_proj)`` — the
                    ``projected_frames`` output from ``OnnxEngine``
                    (already squeezed from batch dimension 0).

        Returns:
            ``(value, confidence)`` where:

            * ``value`` is the decoded integer clipped to ``[min_val, max_val]``
              as a ``float``, or ``None`` if decoding failed.
            * ``confidence`` is the mean max-probability of the winning token
              over non-blank frames, in ``[0, 1]``.  Values ≥ 0.70 indicate
              a clean decode.
        """
        if not self._is_loaded:
            logger.warning(
                "CTCDigitDecoder.predict() called but head is not loaded — "
                "returning (None, 0.0).  Call CTCDigitDecoder.load() first."
            )
            return None, 0.0

        if frames.ndim != 2 or frames.shape[-1] != self.frame_dim:
            logger.warning(
                "CTCDigitDecoder: unexpected frames shape %s (expected (T, %d)).",
                frames.shape, self.frame_dim,
            )
            return None, 0.0

        try:
            import torch

            self._head.eval()
            with torch.no_grad():
                t = torch.from_numpy(frames).unsqueeze(0).to(self.device)   # (1, T, D)
                log_probs = self._head.forward(t).squeeze(0).cpu().numpy()  # (T, V)

            tokens = _greedy_ctc_decode(log_probs)
            value_int = _tokens_to_int(tokens)

            if value_int is None:
                logger.debug(
                    "CTCDigitDecoder: empty or invalid decode for frames %s.", frames.shape
                )
                return None, 0.0

            # Confidence: mean max-prob over non-blank frames
            probs = np.exp(log_probs)                    # (T, V)
            non_blank_mask = np.argmax(probs, axis=-1) != BLANK_IDX
            confidence = (
                float(probs[non_blank_mask].max(axis=-1).mean())
                if non_blank_mask.any()
                else 0.0
            )

            # Clip and penalise out-of-range decodes
            clipped = float(np.clip(value_int, self.min_val, self.max_val))
            if value_int != int(clipped):
                confidence *= 0.5
                logger.debug(
                    "CTCDigitDecoder: decoded %d clipped to %.0f "
                    "(out of [%.0f, %.0f]).",
                    value_int, clipped, self.min_val, self.max_val,
                )

            # Decode string for debug logging
            decoded_words = " ".join(
                _VOCAB[tok][0] for tok in tokens if tok in _VOCAB
            )
            logger.debug(
                "CTCDigitDecoder: '%s' → %d → clipped=%.0f  conf=%.3f",
                decoded_words, value_int, clipped, confidence,
            )
            return clipped, confidence

        except Exception as exc:
            logger.warning("CTCDigitDecoder inference error: %s", exc, exc_info=True)
            return None, 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the head state dict and metadata to *path*.

        Args:
            path: Destination ``.pt`` file path.

        Raises:
            RuntimeError: If the head has not been loaded / trained.
        """
        import torch
        if not self._is_loaded:
            raise RuntimeError(
                "Cannot save a CTCDigitDecoder that has not been loaded."
            )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._head.state_dict(),
                "frame_dim":  self.frame_dim,
                "min_val":    self.min_val,
                "max_val":    self.max_val,
                "vocab_size": VOCAB_SIZE,
                # Store vocab version so mismatched checkpoints are detected
                "vocab_version": "numeral_ru_v1",
            },
            p,
        )
        logger.info("CTCDigitDecoder saved to %s", p)

    @classmethod
    def load(
        cls,
        path: str | Path,
        frame_dim: int,
        min_val: float = 0.0,
        max_val: float = 359.0,
        device: str = "cpu",
    ) -> "CTCDigitDecoder":
        """Load a trained CTC numeral head from *path*.

        Args:
            path:      Path to a ``.pt`` file produced by ``CTCDigitDecoder.save()``.
            frame_dim: Expected frame dimension (must match the saved head).
            min_val:   Lower bound for value clipping.
            max_val:   Upper bound for value clipping.
            device:    Torch device for inference.

        Returns:
            Loaded ``CTCDigitDecoder`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError:        If the saved ``frame_dim`` does not match *frame_dim*.
        """
        import torch
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"CTCDigitDecoder checkpoint not found: {p}"
            )

        ckpt = torch.load(p, map_location=device, weights_only=True)
        saved_dim = int(ckpt.get("frame_dim", frame_dim))
        if saved_dim != frame_dim:
            raise ValueError(
                f"CTCDigitDecoder: frame_dim mismatch — "
                f"checkpoint has {saved_dim}, caller requested {frame_dim}."
            )

        dec = cls(
            frame_dim=frame_dim,
            min_val=float(ckpt.get("min_val", min_val)),
            max_val=float(ckpt.get("max_val", max_val)),
            device=device,
        )
        dec._head.load_state_dict(ckpt["state_dict"])
        dec._head.eval()
        dec._is_loaded = True
        logger.info(
            "CTCDigitDecoder loaded from %s  "
            "(frame_dim=%d, range=[%.0f, %.0f], vocab=%s)",
            p, frame_dim, dec.min_val, dec.max_val,
            ckpt.get("vocab_version", "unknown"),
        )
        return dec
