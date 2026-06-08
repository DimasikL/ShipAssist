"""Check which audio input device actually captures sound at 16000 Hz."""
import sounddevice as sd
import numpy as np

SR = 16000
DURATION = 0.5  # seconds per device test

devices = sd.query_devices()
print(f"Testing input devices at {SR} Hz — speak or make noise during test\n")

for i, dev in enumerate(devices):
    if dev["max_input_channels"] < 1:
        continue
    try:
        data = sd.rec(
            int(DURATION * SR),
            samplerate=SR,
            channels=1,
            dtype="float32",
            device=i,
            blocking=True,
        )
        energy = float(np.mean(data ** 2))
        flag = " ← sound!" if energy > 1e-7 else ""
        print(f"  [{i:2d}] {dev['name'][:50]:<50}  energy={energy:.2e}{flag}")
    except Exception as e:
        print(f"  [{i:2d}] {dev['name'][:50]:<50}  ERROR: {e}")
