"""Shared text normalisation used both for training and in the service.

Kept deliberately simple so it can be embedded inside a scikit-learn Pipeline
(via FunctionTransformer) and therefore shipped inside the joblib file.
"""
from __future__ import annotations

import re

_URL = re.compile(r"(https?://|www\.)\S+", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_MONEY = re.compile(r"[$€£]\s?\d[\d,.]*|\d[\d,.]*\s?(?:usd|eur|dollars?|руб\.?)", re.I)
_NUM = re.compile(r"\b\d{3,}\b")
_HTML = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase + replace high-variance tokens by placeholders."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = _HTML.sub(" ", text)
    t = _URL.sub(" urltoken ", t)
    t = _EMAIL.sub(" emailtoken ", t)
    t = _MONEY.sub(" moneytoken ", t)
    t = _NUM.sub(" numtoken ", t)
    t = t.lower()
    t = _WS.sub(" ", t).strip()
    return t


def normalize_many(texts):
    return [normalize(t) for t in texts]
