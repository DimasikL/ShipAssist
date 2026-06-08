"""
scripts/smoke_test_api.py - Phase 4 gate: API smoke test.

Starts src/api.py in a subprocess, runs GET /health, GET /commands,
POST /recognize (with a synthetic WAV), checks responses, then stops the server.

Usage (from project root, venv activated):
    python scripts\\smoke_test_api.py
    python scripts\\smoke_test_api.py --host 127.0.0.1 --port 8000

Exit codes:
    0 - all checks passed
    1 - one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# ── WAV helper (no external deps) ────────────────────────────────────────────

def _write_wav(path, data, sr=16000):
    """Write a minimal PCM-16 mono WAV file."""
    samples = (data * 32767).clip(-32768, 32767).astype("int16")
    data_chunk = samples.tobytes()
    byte_rate = sr * 2
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data_chunk)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, byte_rate, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data_chunk)))
        f.write(data_chunk)


# ── HTTP client (stdlib only) ─────────────────────────────────────────────────

def _http_get(host, port, path):
    import http.client
    conn = http.client.HTTPConnection(host, port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()
    try:
        return resp.status, json.loads(body)
    except json.JSONDecodeError:
        return resp.status, {"raw": body}


def _http_post_file(host, port, path, filepath):
    import http.client
    import uuid
    boundary = uuid.uuid4().hex
    with open(filepath, "rb") as f:
        file_data = f.read()
    part = (
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + file_data + ("\r\n--" + boundary + "--\r\n").encode()
    headers = {
        "Content-Type": "multipart/form-data; boundary=" + boundary,
        "Content-Length": str(len(part)),
    }
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("POST", path, body=part, headers=headers)
    resp = conn.getresponse()
    resp_body = resp.read().decode()
    conn.close()
    try:
        return resp.status, json.loads(resp_body)
    except json.JSONDecodeError:
        return resp.status, {"raw": resp_body}


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="API smoke test for Phase 4 gate")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--startup_timeout", type=float, default=30.0)
    return p.parse_args()


def wait_for_ready(host, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _ = _http_get(host, port, "/health")
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _print_summary(results):
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print("\n" + "=" * 60)
    print("  Phase 4 - API smoke test summary")
    print("=" * 60)
    for name, ok, detail in results:
        mark = "[PASS]" if ok else "[FAIL]"
        print("  " + mark + "  " + name)
        if not ok:
            print("         " + str(detail))
    verdict = "GATE PASSED" if passed == total else "GATE FAILED"
    print("\n  " + str(passed) + "/" + str(total) + " checks passed  -  " + verdict)
    return 0 if passed == total else 1


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    api_script = project_root / "src" / "api.py"

    if not api_script.exists():
        print("[FAIL] src/api.py not found at " + str(api_script))
        return 1

    results = []

    # start server with PYTHONPATH pointing at project root
    print("Starting API server on " + args.host + ":" + str(args.port) + " ...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [sys.executable, str(api_script)],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    ready = wait_for_ready(args.host, args.port, args.startup_timeout)
    if not ready:
        stdout = proc.stdout.read().decode() if proc.stdout else ""
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        proc.terminate()
        print("[FAIL] Server did not become ready within timeout.")
        print("STDOUT:", stdout[-500:])
        print("STDERR:", stderr[-500:])
        results.append(("server_startup", False, "Timeout"))
        return _print_summary(results)

    print("  Server ready.\n")

    try:
        # GET /health
        status, body = _http_get(args.host, args.port, "/health")
        ok = status == 200
        results.append(("GET /health -> 200", ok, "status=" + str(status) + " " + str(body)))
        print("  " + ("[OK]  " if ok else "[FAIL]") + " GET /health  " + str(status) + "  " + str(body))

        # GET /commands
        status, body = _http_get(args.host, args.port, "/commands")
        ok = status == 200 and isinstance(body, list) and len(body) >= 1
        results.append(("GET /commands -> list", ok, "status=" + str(status) + " " + str(body)))
        print("  " + ("[OK]  " if ok else "[FAIL]") + " GET /commands  " + str(status) + "  " + str(body))

        # POST /recognize with synthetic WAV
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(16000 * 2).astype("float32") * 0.05
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            _write_wav(tmp.name, audio, 16000)
            wav_path = tmp.name
        status, body = _http_post_file(args.host, args.port, "/recognize", wav_path)
        os.unlink(wav_path)
        ok = status == 200
        results.append(("POST /recognize -> 200", ok, "status=" + str(status) + " " + str(body)))
        print("  " + ("[OK]  " if ok else "[FAIL]") + " POST /recognize  " + str(status) + "  " + str(body))

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\n  Server stopped.")

    return _print_summary(results)


if __name__ == "__main__":
    sys.exit(main())
