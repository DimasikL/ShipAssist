# core/hybrid — Hybrid C+ Engine

Parallel implementation of **Approach C+** (Embedding Search + Outlier Gate + Slot Fill) for ShipAssistant.

> **Safety guarantee:** This module adds new files only. It does **not** modify `core/engine.py`, `core/onnx_engine.py`, `core/config.py`, or any existing file. The existing ONNX demo is fully unaffected.

---

## Architecture

```
Raw audio
    │
    ▼
[Stage 0]  prepare_window()              core/audio_utils.py (existing, unchanged)
    │
    ▼
[Stage 1]  OnnxEngine.predict_logits()   Reuses existing ONNX session; grabs outputs[1]
           → Wav2Vec2 embedding (1024-D)
    │
    ▼
[Stage 2]  OutlierGate.is_outlier()      Mahalanobis distance → early reject
           core/hybrid/outlier_gate.py
    │  reject → {"outlier_rejected": True, "label": ""}
    │
    ▼
[Stage 3]  CentroidSearch.search()       Cosine nearest-centroid → intent label
           core/hybrid/centroid_search.py
    │  below threshold → {"label": ""}
    │
    ▼
[Stage 4]  NumberRegressor.predict()     Optional: fills numeric slot
           core/hybrid/number_regressor.py
    │
    ▼
Result dict  {"label", "full_label", "confidence", "probs",
              "logits", "latency_ms", "outlier_score", ...}
```

---

## Module Map

| File | Responsibility |
|------|---------------|
| `config.py` | Pydantic config (self-contained, no dependency on `core.config`) |
| `outlier_gate.py` | Mahalanobis / cosine / L2 early-rejection gate |
| `centroid_search.py` | Cosine nearest-centroid intent lookup |
| `number_regressor.py` | MLP regressor wrapper for numeric slot-filling |
| `engine.py` | `HybridAudioEngine` — orchestrates all stages |
| `factory.py` | `create_hybrid_engine(cfg)` — public entry-point |

---

## Quick Start

### 1. Install no new dependencies
Everything uses existing project deps: `numpy`, `pydantic`, `torch`, `joblib`, `yaml`.

### 2. Prepare training data
Your existing `dataset.csv` is the input. It needs columns: `path` (wav file path) and `label`.

### 3. Build centroids
```bash
python scripts/hybrid/build_centroids.py \
    --csv artifacts/data/dataset.csv \
    --out artifacts/hybrid/
```

### 4. Train the outlier gate
```bash
python scripts/hybrid/train_outlier_gate.py \
    --csv artifacts/data/dataset.csv \
    --out artifacts/hybrid/outlier_gate.pkl
```

### 5. (Optional) Train number regressor
For each slot intent (e.g., "курс УГОЛ градусов"):
```bash
python scripts/hybrid/train_regressor.py \
    --csv artifacts/data/dataset_numbers.csv \
    --intent "курс УГОЛ градусов" \
    --min_val 1 --max_val 360 \
    --out artifacts/hybrid/regressors/
```

### 6. Run the demo
```bash
python scripts/hybrid/demo_hybrid.py --wav path/to/audio.wav
```

### 7. Run integration tests
```bash
python scripts/hybrid/test_integration.py
```

---

## Output Dictionary Contract

`HybridAudioEngine.predict()` returns a strict **superset** of `OnnxAudioEngine.predict()`:

| Key | Type | Description |
|-----|------|-------------|
| `label` | `str` | Top-1 intent label (empty string if rejected) |
| `full_label` | `str` | Label with numeric slot filled, e.g. `"курс 245 градусов"` |
| `confidence` | `float` | Cosine similarity to best centroid |
| `probs` | `np.ndarray` | Softmax over all cosine scores (same shape contract as ONNX) |
| `logits` | `np.ndarray` | Raw cosine scores (pre-softmax) |
| `latency_ms` | `float` | Total wall-clock inference time in ms |
| `outlier_score` | `float` | Mahalanobis distance to nearest centroid |
| `outlier_rejected` | `bool` | True if the gate fired and the pipeline stopped |
| `slot_value` | `float\|None` | Numeric prediction for slot intents |
| `slot_confidence` | `float` | Regressor confidence score (0–1) |
| `search_method` | `str` | Always `"hybrid_cosine"` (for A/B telemetry) |

If artefacts are missing, `predict()` returns `{"error": "hybrid_not_loaded", ...}` instead of raising.

---

## Migration to Hybrid Mode

When you're ready to switch the production path from `OnnxAudioEngine` to `HybridAudioEngine`:

1. **Verify** `scripts/hybrid/test_integration.py` passes on real audio.
2. **Benchmark** `scripts/hybrid/demo_hybrid.py` — confirm latency < 500 ms.
3. In `src/inference.py` (or wherever `create_engine()` is called), change:
   ```python
   # Before:
   from core.engine import create_engine
   engine = create_engine(cfg)

   # After:
   from core.hybrid.factory import create_hybrid_engine
   from core.hybrid.config import HybridConfig
   hybrid_cfg = HybridConfig.from_yaml("configs/hybrid/model.yaml",
                                        "configs/hybrid/thresholds.yaml")
   engine = create_hybrid_engine(hybrid_cfg)
   ```
4. The call to `engine.predict(waveform)` is **identical** — both implement `AudioEngine`.
5. Keep the ONNX engine running as a shadow model for comparison metrics during rollout.

---

## Data Requirements

| Component | Min samples | Notes |
|-----------|------------|-------|
| Centroid (per phrase) | 10–20 | More = more robust centroid; 50+ ideal |
| Outlier gate | All existing samples | Uses full training set |
| Number regressor (per slot) | 60–100 | Distribute uniformly across the numeric range |

For number data, TTS augmentation is acceptable:
```bash
python scripts/generation/main_audio_generate_tts.py \
    --phrase "курс {n} градусов" \
    --range "1,360,10" \
    --out artifacts/data/numbers/
```

---

## Adding a New Phrase (Zero Retraining)

1. Record 15–30 samples of the new phrase.
2. Extract embeddings and compute the centroid:
   ```bash
   python scripts/hybrid/build_centroids.py --csv new_phrase.csv --append
   ```
3. The new centroid is appended to `centroids.npy` — no model retraining needed.
4. Optionally re-fit the outlier gate to include the new class:
   ```bash
   python scripts/hybrid/train_outlier_gate.py --csv full_dataset.csv
   ```

---

*Last updated: 2026-04-28 | Architecture: Approach C+ (Hybrid + Outlier Gate)*
