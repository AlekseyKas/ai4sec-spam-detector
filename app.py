"""Spam-detection microservice (Flask).

Endpoints:
  GET  /            - tiny HTML demo page
  GET  /health      - liveness + model info
  POST /predict     - {"text": "..."} -> spam verdict (+ category if model present)
  POST /predict_batch - {"texts": ["...", ...]} -> list of verdicts

Models are scikit-learn Pipelines (TF-IDF + classifier) saved with joblib:
  models/spam_pipeline.joblib      (required)
  models/category_pipeline.joblib  (optional, extra task)
  models/spam_threshold.json       (optional: {"threshold": 0.5})
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for preprocess.py referenced by the pickled pipeline

import joblib
from flask import Flask, jsonify, render_template_string, request

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent / "models"))
SPAM_MODEL = MODEL_DIR / "spam_pipeline.joblib"
CAT_MODEL = MODEL_DIR / "category_pipeline.joblib"
THRESHOLD_FILE = MODEL_DIR / "spam_threshold.json"

app = Flask(__name__)

spam_model = joblib.load(SPAM_MODEL)
cat_model = joblib.load(CAT_MODEL) if CAT_MODEL.exists() else None

if "SPAM_THRESHOLD" in os.environ:
    THRESHOLD = float(os.environ["SPAM_THRESHOLD"])
elif THRESHOLD_FILE.exists():
    THRESHOLD = float(json.loads(THRESHOLD_FILE.read_text())["threshold"])
else:
    THRESHOLD = 0.5

MAX_TEXT_LEN = int(os.environ.get("MAX_TEXT_LEN", 200_000))


def _spam_scores(texts: list[str]) -> list[float]:
    """Probability-like spam score in [0, 1] for each text."""
    if hasattr(spam_model, "predict_proba"):
        return spam_model.predict_proba(texts)[:, 1].tolist()
    # e.g. LinearSVC without calibration: squash decision_function
    import numpy as np

    d = spam_model.decision_function(texts)
    return (1 / (1 + np.exp(-d))).tolist()


def _classify(texts: list[str]) -> list[dict]:
    scores = _spam_scores(texts)
    cats = cat_model.predict(texts).tolist() if cat_model is not None else [None] * len(texts)
    out = []
    for score, cat in zip(scores, cats):
        is_spam = score >= THRESHOLD
        item = {
            "is_spam": bool(is_spam),
            "label": "spam" if is_spam else "ham",
            "spam_probability": round(float(score), 4),
            "threshold": THRESHOLD,
        }
        if cat_model is not None:
            item["category"] = cat if is_spam else "ham"
        out.append(item)
    return out


def _get_texts(payload, key: str):
    if not isinstance(payload, dict) or key not in payload:
        return None, (jsonify(error=f"JSON body with '{key}' field required"), 400)
    value = payload[key]
    if key == "text":
        if not isinstance(value, str):
            return None, (jsonify(error="'text' must be a string"), 400)
        value = [value]
    if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
        return None, (jsonify(error="'texts' must be a list of strings"), 400)
    if any(len(t) > MAX_TEXT_LEN for t in value):
        return None, (jsonify(error=f"text longer than {MAX_TEXT_LEN} chars"), 413)
    return value, None


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        spam_model=type(spam_model.steps[-1][1]).__name__ if hasattr(spam_model, "steps") else type(spam_model).__name__,
        category_model=bool(cat_model is not None),
        threshold=THRESHOLD,
    )


@app.post("/predict")
def predict():
    texts, err = _get_texts(request.get_json(silent=True), "text")
    if err:
        return err
    return jsonify(_classify(texts)[0])


@app.post("/predict_batch")
def predict_batch():
    texts, err = _get_texts(request.get_json(silent=True), "texts")
    if err:
        return err
    return jsonify(results=_classify(texts))


DEMO_HTML = """<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Spam detector demo</title>
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px}
textarea{width:100%;height:220px;font-family:monospace}button{padding:8px 18px;font-size:15px}
#out{margin-top:16px;padding:12px;border-radius:8px;background:#f3f4f6;white-space:pre-wrap}
.spam{background:#fde2e2}.ham{background:#e2f7e6}</style></head><body>
<h2>Антиспам-классификатор (ML в кибербезопасности)</h2>
<p>Вставьте текст письма (тема + тело) и нажмите «Проверить».</p>
<textarea id="t">Subject: Congratulations! You have been selected for a $1000 gift card
Click here now to claim your prize before it expires!</textarea><br><br>
<button onclick="go()">Проверить</button>
<div id="out"></div>
<script>
async function go(){const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({text:document.getElementById('t').value})});const j=await r.json();
const o=document.getElementById('out');o.className=j.is_spam?'spam':'ham';o.textContent=JSON.stringify(j,null,2);}
</script><p style="color:#666;font-size:13px">API: <code>POST /predict {"text": "..."}</code>,
<code>POST /predict_batch {"texts": [...]}</code>, <code>GET /health</code></p></body></html>"""


@app.get("/")
def index():
    return render_template_string(DEMO_HTML)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
