# app/main.py — Phishing Email Detector, serving API
# ---------------------------------------------------------------------------
# Serves the artifact trained by phishing_v2.ipynb:
#   model/phishing_pipeline.joblib   (TF-IDF + logistic regression Pipeline)
#   model/metadata.json              (version, protocol, held-out metrics)
#
# Endpoints
#   GET  /         minimal browser UI (textarea -> verdict) for manual checks
#   POST /predict  {"text": "..."} -> label, phishing probability, version, latency
#   GET  /health   liveness + model version (for orchestration / uptime checks)
#   GET  /stats    in-memory counters: request volume, class mix, latency —
#                  first-line monitoring for the drift the notebook predicts
#
# Monitoring philosophy: the training data's ham has narrow provenance (see the
# notebook's Skeptic's corner), so live traffic WILL differ from the benchmark.
# Every request is therefore logged as a JSON line (timestamp, latency, verdict,
# probability, text length — never the text itself), and /stats summarises the
# running prediction distribution so drift is visible at a glance.
# ---------------------------------------------------------------------------

import json
import logging
import os
import time
from pathlib import Path
from threading import Lock

import joblib
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ----------------------------- model loading -------------------------------
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model"))
pipeline = joblib.load(MODEL_DIR / "phishing_pipeline.joblib")
metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
MODEL_VERSION = metadata.get("model_version", "unknown")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("phishing-api")

app = FastAPI(title="Phishing Email Detector",
              description="TF-IDF + logistic regression, leakage-audited protocol",
              version=MODEL_VERSION)

# ----------------------------- basic monitoring ----------------------------
_stats_lock = Lock()
_stats = {"n_requests": 0, "n_phishing": 0, "n_safe": 0,
          "latency_ms_sum": 0.0, "started_at": time.time()}


class EmailIn(BaseModel):
    text: str = Field(..., min_length=1, description="Raw email text")


class PredictionOut(BaseModel):
    label: str
    probability_phishing: float
    model_version: str
    latency_ms: float


@app.post("/predict", response_model=PredictionOut)
def predict(email: EmailIn) -> PredictionOut:
    t0 = time.perf_counter()
    proba = float(pipeline.predict_proba([email.text])[0, 1])
    label = "Phishing Email" if proba >= 0.5 else "Safe Email"
    latency = (time.perf_counter() - t0) * 1000

    with _stats_lock:
        _stats["n_requests"] += 1
        _stats["n_phishing" if proba >= 0.5 else "n_safe"] += 1
        _stats["latency_ms_sum"] += latency

    # structured request log: metrics only, never the email content
    log.info(json.dumps({"ts": time.time(), "event": "prediction",
                         "label": label, "proba": round(proba, 4),
                         "text_chars": len(email.text),
                         "latency_ms": round(latency, 2),
                         "model_version": MODEL_VERSION}))
    return PredictionOut(label=label, probability_phishing=round(proba, 4),
                         model_version=MODEL_VERSION, latency_ms=round(latency, 2))


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION,
            "uptime_s": round(time.time() - _stats["started_at"], 1)}


@app.get("/stats")
def stats():
    with _stats_lock:
        n = _stats["n_requests"]
        return {"model_version": MODEL_VERSION,
                "n_requests": n,
                "n_phishing": _stats["n_phishing"],
                "n_safe": _stats["n_safe"],
                "phishing_share": round(_stats["n_phishing"] / n, 3) if n else None,
                "avg_latency_ms": round(_stats["latency_ms_sum"] / n, 2) if n else None,
                "held_out_metrics": metadata.get("held_out_metrics")}


# ----------------------------- minimal UI ----------------------------------
_UI = """<!doctype html><html><head><meta charset="utf-8">
<title>Phishing Email Detector</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:3rem auto;padding:0 1rem;color:#1a1a2e}
 textarea{width:100%;height:220px;font-family:ui-monospace,monospace;font-size:.9rem;
          padding:.6rem;border:1px solid #bbb;border-radius:6px;box-sizing:border-box}
 button{margin-top:.7rem;padding:.55rem 1.4rem;font-size:1rem;border:0;border-radius:6px;
        background:#1f3864;color:#fff;cursor:pointer}
 #out{margin-top:1.2rem;padding:1rem;border-radius:6px;display:none;font-size:1.05rem}
 .phish{background:#fdecea;border:1px solid #d95f02}
 .safe{background:#eaf4ea;border:1px solid #2c7fb8}
 small{color:#666}
</style></head><body>
<h1>Phishing Email Detector</h1>
<p>Paste an email below. The model returns a verdict and its phishing probability.
<small>(demo of a leakage-audited TF-IDF + logistic-regression pipeline — see the
project notebook for the honest evaluation behind it)</small></p>
<textarea id="t" placeholder="Paste email text here..."></textarea><br>
<button onclick="go()">Check email</button>
<div id="out"></div>
<script>
async function go(){
  const text = document.getElementById('t').value;
  if(!text.trim()) return;
  const r = await fetch('/predict', {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
  const d = await r.json();
  const el = document.getElementById('out');
  const phish = d.label === 'Phishing Email';
  el.className = phish ? 'phish' : 'safe';
  el.style.display = 'block';
  el.innerHTML = `<b>${d.label}</b> — phishing probability ${(d.probability_phishing*100).toFixed(1)}%` +
    `<br><small>model ${d.model_version} · ${d.latency_ms} ms</small>`;
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return _UI
