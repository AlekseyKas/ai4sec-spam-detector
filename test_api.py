"""Smoke test for the service. Usage: python test_api.py [http://host:port]"""
import json
import sys
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def call(path, payload=None):
    req = urllib.request.Request(base + path, method="POST" if payload else "GET")
    req.add_header("Content-Type", "application/json")
    data = json.dumps(payload).encode() if payload else None
    with urllib.request.urlopen(req, data=data, timeout=30) as r:
        return json.load(r)


print("health:", call("/health"))
print("spam  :", call("/predict", {"text": "Subject: FREE VIAGRA!!! Buy now, limited offer, click here http://bit.ly/x"}))
print("ham   :", call("/predict", {"text": "Subject: meeting tomorrow\nHi John, can we move the 10am sync to 11? Thanks, Anna"}))
print("batch :", call("/predict_batch", {"texts": ["win a prize now", "see attached report for Q3"]}))
