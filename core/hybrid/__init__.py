"""
core/hybrid — Hybrid C+ inference engine for ShipAssistant.

This package implements the "Approach C+ (Hybrid + Outlier Gate)" architecture
as a fully isolated parallel module. It does NOT modify or depend on the
existing ``core/engine.py`` or ``core/onnx_engine.py`` pipelines.

Public surface
--------------
The only intended entry-point for callers is the factory:

    from core.hybrid.factory import create_hybrid_engine
    engine = create_hybrid_engine(cfg)          # HybridConfig
    result  = engine.predict(audio_np_array)    # same dict shape as OnnxAudioEngine

The ``HybridAudioEngine`` class implements the same abstract ``AudioEngine``
interface as ``OnnxAudioEngine`` so the two backends are drop-in replaceable.

Architecture (pipeline order)
------------------------------
1. **Outlier Gate** (``core.hybrid.outlier_gate``) — Mahalanobis early reject.
   Embeddings that lie far from all known class clusters are rejected before
   any further computation. Reduces false-positive rate in noisy environments.
2. **Centroid Search** (``core.hybrid.centroid_search``) — Cosine nearest-centroid
   lookup. Identifies the intent label without requiring a full softmax retrain.
   Adding a new phrase = appending a centroid; zero retraining required.
3. **Number Regressor** (``core.hybrid.number_regressor``) — Optional slot-fill
   for open-range commands ("Курс [1-360] градусов"). Activated only when the
   predicted intent is registered as a slot intent in the config.

Thread safety
-------------
All three components are stateless after loading (read-only numpy operations).
``HybridAudioEngine.predict()`` is safe to call from multiple threads simultaneously
provided the underlying ONNX session is also thread-safe (which it is by default
with ``IntraOpNumThreads=1`` and the CPUExecutionProvider).
"""

from core.hybrid.engine import HybridAudioEngine
from core.hybrid.factory import create_hybrid_engine

__all__ = ["HybridAudioEngine", "create_hybrid_engine"]
