"""Send a sample request to the running API.

    uv run python app/sample_request.py
"""
import json
import urllib.request

URL = "http://127.0.0.1:8000/predict"
trip = {"PULocationID": 161, "DOLocationID": 236, "trip_distance": 2.5}

req = urllib.request.Request(
    URL,
    data=json.dumps(trip).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as resp:
    print(json.dumps(json.load(resp), indent=2))