"""Minimal watsonx.ai client using the REST API directly (only needs `requests`).

Set these environment variables (copy .env.example to .env and fill it in):
  WATSONX_APIKEY      IBM Cloud API key
  WATSONX_PROJECT_ID  the watsonx.ai project ID (Project > Manage > General)
  WATSONX_URL         region endpoint, e.g. https://us-south.ml.cloud.ibm.com
  WATSONX_MODEL_ID    a chat model available in your region, e.g. ibm/granite-4-h-small
                      (check the exact ID in Prompt Lab: model list > model details)
"""
import json
import os
import re
import time

import requests

_token = {"value": None, "expires": 0}


def _load_dotenv():
    path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _real(v):
    """A value that is set and is not the .env.example placeholder."""
    v = (v or "").strip()
    return v if v and not v.startswith("your-") else ""


def configured():
    _load_dotenv()
    return bool(_real(os.environ.get("WATSONX_APIKEY")) and _real(os.environ.get("WATSONX_PROJECT_ID")))


def _iam_token():
    key = os.environ["WATSONX_APIKEY"]
    if _token["value"] and _token.get("key") == key and time.time() < _token["expires"] - 60:
        return _token["value"]
    r = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": key},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    r.raise_for_status()
    body = r.json()
    _token.update(value=body["access_token"], key=key, expires=time.time() + body.get("expires_in", 3600))
    return _token["value"]


def chat(system, user, max_tokens=900, temperature=0.0):
    """Send one system + user message to a watsonx.ai chat model and return the reply text."""
    _load_dotenv()
    url = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").rstrip("/")
    payload = {
        "model_id": os.environ.get("WATSONX_MODEL_ID", "ibm/granite-4-h-small"),
        "project_id": os.environ["WATSONX_PROJECT_ID"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"{url}/ml/v1/text/chat?version=2024-10-08", json=payload,
                      headers={"Authorization": f"Bearer {_iam_token()}"}, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"watsonx.ai error {r.status_code}: {r.text[:500]}")
    return r.json()["choices"][0]["message"]["content"]


def extract_json(text):
    """Models sometimes wrap JSON in prose or ``` fences; pull out the first {...} block."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"No JSON found in model reply: {text[:300]}")
    return json.loads(m.group(0))


def test_connection():
    """One tiny call, to check the key, project and model ID from the settings panel."""
    return chat("Reply with the single word OK.", "Say OK.", max_tokens=5)
