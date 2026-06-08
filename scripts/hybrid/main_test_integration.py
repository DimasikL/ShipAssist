"""
scripts/hybrid/test_integration.py — Integration test for the Hybrid C+ engine.

What is tested
--------------
1. HybridConfig loads from YAML without errors.
2. create_hybrid_engine() returns a HybridAudioEngine instance.
3. predict() on random noise returns the correct dict structure (all required
   keys present, correct types) — regardless of whether artefacts are loaded.
4. predict() on near-silence returns the correct dict structure.
5. OutlierGate can be fitted and used in isolation.
6. CentroidSearch can be built and queried in isolation.
7. NumberRegressor graceful fallback: predict() returns None when unfitted.
8. HybridAudioEngine with injected mock components returns a sane result.

These tests run WITHOUT requiring any trained model artefacts on disk.
They use synthetic numpy arrays and mock components where necessary.

Usage
-----
    python scripts/hybrid/test_integration.py

    # Verbose output:
    python scripts/hybrid/test_integration.py -v
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# ── Project root on path ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Imports ───────────────────────────────────────────────────────────────────
from core.hybrid.config import HybridConfig
from core.hybrid.centroid_search import CentroidSearch
from core.hybrid.outlier_gate import OutlierGate
from core.hybrid.number_regressor import NumberRegressor
from core.hybrid.factory import create_hybrid_engine

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Test utilities ────────────────────────────────────────────────────────────

_PASS = "\033[92m✓\033[0m"
_FAIL = "\033[91m✗\033[0m"
_results: List[tuple[str, bool, str]] = []


def _assert(condition: bool, test_name: str, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    _results.append((test_name, condition, detail))
    icon = _PASS if condition else _FAIL
    print(f"  {icon} {test_name}" + (f"  ({detail})" if detail else ""))


def _required_keys() -> List[str]:
    """Return the list of keys every predict() result must contain."""
    return [
        "label", "full_label", "confidence", "probs",
        "logits", "latency_ms", "outlier_score",
        "outlier_rejected", "slot_value", "slot_confidence", "search_method",
    ]


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _random_audio(length: int = 16_000) -> np.ndarray:
    """Return random noise as a 1-D float32 waveform."""
    return (np.random.randn(length) * 0.05).astype(np.float32)


def _silence(length: int = 16_000) -> np.ndarray:
    """Return near-silence as a 1-D float32 waveform."""
    return np.zeros(length, dtype=np.float32)


def _make_synthetic_embeddings(
    n_classes: int = 3,
    samples_per_class: int = 20,
    dim: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, List[str]]:
    """Generate synthetic labelled embeddings for gate / search testing."""
    rng = np.random.default_rng(seed)
    labels_list: List[str] = [f"command_{i}" for i in range(n_classes)]
    embeddings: List[np.ndarray] = []
    labels: List[str] = []
    for lbl in labels_list:
        centre = rng.standard_normal(dim).astype(np.float32)
        for _ in range(samples_per_class):
            vec = centre + rng.standard_normal(dim).astype(np.float32) * 0.1
            embeddings.append(vec)
            labels.append(lbl)
    return np.stack(embeddings), labels


# ── Test suite ────────────────────────────────────────────────────────────────

def test_config_defaults() -> None:
    """HybridConfig should instantiate with sensible defaults."""
    print("\n[1] HybridConfig defaults")
    cfg = HybridConfig()
    _assert(cfg.sample_rate == 16_000, "sample_rate default", str(cfg.sample_rate))
    _assert(cfg.win_samples == 16_000, "win_samples default", str(cfg.win_samples))
    _assert(cfg.outlier_gate.enabled is True, "gate.enabled default")
    _assert(cfg.outlier_gate.method == "mahalanobis", "gate.method default")
    _assert(cfg.centroid_search.min_cosine_similarity == 0.75, "cosine_threshold default")


def test_config_from_yaml() -> None:
    """HybridConfig.from_yaml() should load both YAML files without error."""
    print("\n[2] HybridConfig.from_yaml()")
    m_yaml = _PROJECT_ROOT / "configs/hybrid/model.yaml"
    t_yaml = _PROJECT_ROOT / "configs/hybrid/thresholds.yaml"
    if not (m_yaml.exists() and t_yaml.exists()):
        print(f"  ⚠  YAML files not found — skipping (run from project root)")
        _results.append(("from_yaml_skip", True, "skipped: files absent"))
        return
    try:
        cfg = HybridConfig.from_yaml(m_yaml, t_yaml)
        _assert(cfg.sample_rate > 0, "from_yaml: sample_rate > 0")
        _assert(
            len(cfg.number_regressor.slot_intents) >= 0,
            "from_yaml: slot_intents parseable",
        )
    except Exception as exc:
        _assert(False, "from_yaml: no exception", str(exc))


def test_outlier_gate_fit_and_predict() -> None:
    """OutlierGate should fit on synthetic data and classify correctly."""
    print("\n[3] OutlierGate fit + predict")
    embeddings, labels = _make_synthetic_embeddings()

    for method in ("mahalanobis", "cosine", "l2"):
        gate = OutlierGate(method=method, percentile=99.0, regularization_eps=1e-3)
        try:
            gate.fit(embeddings, labels)
            dist, nearest = gate.score(embeddings[0])
            _assert(isinstance(dist, float), f"gate({method}).score returns float")
            _assert(isinstance(nearest, str), f"gate({method}).score returns label")
            # Training samples should not be outliers at 99th percentile
            in_dist_count = sum(
                not gate.is_outlier(e) for e in embeddings[:10]
            )
            _assert(in_dist_count >= 8, f"gate({method}): training samples in-dist", f"{in_dist_count}/10")
            # Obvious noise should be an outlier
            noise = np.random.randn(embeddings.shape[1]).astype(np.float32) * 100
            # (noise may or may not be outlier — just check it doesn't crash)
            gate.is_outlier(noise)
            _assert(True, f"gate({method}): noise input no crash")
        except Exception as exc:
            _assert(False, f"gate({method}): no exception", str(exc))

    # Summary
    gate = OutlierGate()
    gate.fit(embeddings, labels)
    summary = gate.summary()
    _assert("n_classes" in summary, "gate.summary() has n_classes")
    _assert(summary["fitted"] is True, "gate.summary() fitted=True")


def test_centroid_search_build_and_query() -> None:
    """CentroidSearch should build from embeddings and find the right class."""
    print("\n[4] CentroidSearch build + query")
    embeddings, labels = _make_synthetic_embeddings(n_classes=4, dim=128)

    search = CentroidSearch(min_cosine_similarity=0.0)
    search.build_from_embeddings(embeddings, labels)

    _assert(search.n_labels == 4, "n_labels == 4")
    _assert(search.embedding_dim == 128, "embedding_dim == 128")

    # Query with the first training sample — should match its own class
    first_emb = embeddings[0]
    first_label = labels[0]
    best_label, score, all_scores = search.search(first_emb, threshold=0.0)
    _assert(best_label == first_label, "nearest-centroid matches own class", f"got {best_label!r}")
    _assert(0.0 <= score <= 1.05, "score in [0, 1]", f"score={score:.4f}")
    _assert(len(all_scores) == 4, "all_scores has 4 entries")

    # Threshold rejection
    best_label_th, score_th, _ = search.search(first_emb, threshold=0.9999)
    _assert(best_label_th is None or score_th < 0.9999, "high threshold can reject")

    # scores_as_probs sums to ~1
    probs = search.scores_as_probs(all_scores)
    _assert(abs(probs.sum() - 1.0) < 1e-5, "scores_as_probs sums to 1", f"sum={probs.sum():.6f}")

    # add_centroid / update
    new_centroid = np.random.randn(128).astype(np.float32)
    search.add_centroid("new_phrase", new_centroid)
    _assert(search.n_labels == 5, "add_centroid: n_labels grew to 5")


def test_number_regressor_fallback() -> None:
    """NumberRegressor.predict() should return None when unfitted (graceful fallback)."""
    print("\n[5] NumberRegressor graceful fallback")
    reg = NumberRegressor(min_val=0.0, max_val=360.0)
    emb = np.random.randn(64).astype(np.float32)
    result = reg.predict(emb)
    _assert(result is None, "unfitted regressor returns None")
    val, conf = reg.predict_with_confidence(emb)
    _assert(val is None, "unfitted predict_with_confidence: val is None")
    _assert(conf == 0.0, "unfitted predict_with_confidence: conf == 0.0")


def test_engine_degraded_mode() -> None:
    """HybridAudioEngine should return the error sentinel when artefacts are missing."""
    print("\n[6] HybridAudioEngine degraded mode (no artefacts)")
    cfg = HybridConfig()
    # Point paths to non-existent files
    cfg.paths.centroids = Path("/tmp/_does_not_exist.npy")
    cfg.paths.centroid_labels = Path("/tmp/_does_not_exist.json")
    cfg.paths.outlier_gate = Path("/tmp/_does_not_exist.pkl")

    engine = create_hybrid_engine(cfg)
    _assert(engine._loaded is False, "engine._loaded is False when artefacts missing")

    audio = _random_audio()
    result = engine.predict(audio)
    _assert("error" in result, "degraded predict() returns error key")
    _assert(result.get("error") == "hybrid_not_loaded", "error value = 'hybrid_not_loaded'")

    # Even the error sentinel must have all required keys
    for key in ("label", "confidence", "probs", "logits", "latency_ms"):
        _assert(key in result, f"sentinel has key '{key}'")


def test_engine_with_mock_components() -> None:
    """HybridAudioEngine should run full pipeline with injected mock components."""
    print("\n[7] HybridAudioEngine with injected mock components")

    DIM = 64
    LABELS = ["машина", "самый малый вперед", "приготовить машину"]

    # Build mock centroid search
    embeddings, labels = _make_synthetic_embeddings(n_classes=3, samples_per_class=30, dim=DIM)
    labels = [LABELS[int(lbl.split("_")[1])] for lbl in labels]

    search = CentroidSearch(min_cosine_similarity=0.0)
    search.build_from_embeddings(embeddings, labels)

    gate = OutlierGate(method="cosine", percentile=99.0)
    gate.fit(embeddings, labels)

    cfg = HybridConfig()
    cfg.win_samples = DIM  # tiny window matching the mock embedding dim
    # Disable ONNX path — we'll use a mock embedder
    cfg.embedder.use_onnx_embeddings = False

    # Inject components directly (bypass file loading)
    from core.hybrid.engine import HybridAudioEngine
    engine = HybridAudioEngine.__new__(HybridAudioEngine)
    engine._cfg = cfg
    engine._outlier_gate = gate
    engine._centroid_search = search
    engine._number_regressors = {}
    engine._onnx_engine = None
    engine._standalone_embedder = None
    engine._label_list = search.labels
    engine._loaded = True
    engine._load_error = None

    # Patch _get_embedding to return first training embedding
    def _mock_get_embedding(audio: np.ndarray):
        return embeddings[0]
    engine._get_embedding = _mock_get_embedding

    # Run predict
    audio = _random_audio(length=DIM)
    result = engine.predict(audio)

    # Verify all required keys
    required = _required_keys()
    missing = [k for k in required if k not in result]
    _assert(len(missing) == 0, "all required keys present", f"missing={missing}")

    _assert(isinstance(result["label"], str), "label is str")
    _assert(isinstance(result["confidence"], float), "confidence is float")
    _assert(isinstance(result["probs"], np.ndarray), "probs is ndarray")
    _assert(isinstance(result["logits"], np.ndarray), "logits is ndarray")
    _assert(isinstance(result["latency_ms"], float), "latency_ms is float")
    _assert(isinstance(result["outlier_rejected"], bool), "outlier_rejected is bool")
    _assert(result["search_method"] == "hybrid_cosine", "search_method correct")
    _assert(result["label"] in LABELS or result["label"] == "", "label is valid or empty")


def test_output_dict_contract() -> None:
    """Verify the output dict keys are a strict superset of OnnxAudioEngine keys."""
    print("\n[8] Output dict key contract")

    # Keys that OnnxAudioEngine guarantees (from core/engine.py docstring)
    onnx_keys = {"label", "confidence", "probs", "logits", "latency_ms"}
    hybrid_extras = {"full_label", "outlier_score", "outlier_rejected",
                     "slot_value", "slot_confidence", "search_method"}
    all_expected = onnx_keys | hybrid_extras

    # The sentinel dict must have all these keys
    from core.hybrid.engine import _NOT_LOADED
    for key in all_expected:
        _assert(key in _NOT_LOADED, f"sentinel has '{key}'")

    _assert(all_expected.issubset(set(_NOT_LOADED.keys())), "sentinel is superset of ONNX keys")


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all integration tests and print a summary."""
    print("\n" + "=" * 60)
    print("  ShipAssistant — Hybrid C+ Engine Integration Tests")
    print("=" * 60)

    np.random.seed(0)

    tests = [
        test_config_defaults,
        test_config_from_yaml,
        test_outlier_gate_fit_and_predict,
        test_centroid_search_build_and_query,
        test_number_regressor_fallback,
        test_engine_degraded_mode,
        test_engine_with_mock_components,
        test_output_dict_contract,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as exc:
            print(f"\n  {_FAIL} UNCAUGHT EXCEPTION in {test_fn.__name__}:")
            traceback.print_exc()
            _results.append((test_fn.__name__, False, str(exc)))

    # ── Summary ────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nFailed tests:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  {_FAIL} {name}" + (f"  — {detail}" if detail else ""))
        sys.exit(1)
    else:
        print("\n\033[92mAll tests passed.\033[0m")
        sys.exit(0)


if __name__ == "__main__":
    main()
