from __future__ import annotations

import ast
import json
import re
import time
from typing import Any

import requests

from .config import load_yaml


THINK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)


def load_llm_config(path: str) -> dict[str, Any]:
    cfg = load_yaml(path)
    required = ["model", "base_url", "api_key"]
    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"Missing LLM config fields: {missing}")
    cfg.setdefault("temperature", 0.2)
    cfg.setdefault("max_tokens", 4096)
    cfg.setdefault("timeout", 600)
    return cfg


def _chat_url(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    # Most hosted OpenAI-compatible gateways expose chat completions under /v1.
    # Keeping this normalization here makes configs provider-agnostic.
    return base + "/v1/chat/completions"


def _strip_think(text: str) -> str:
    return THINK_RE.sub("", text or "").strip()


def _json_candidates(text: str) -> list[str]:
    cleaned = _strip_think(text)
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1))
    greedy = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
    if greedy:
        candidates.append(greedy.group(1))
    candidates.append(cleaned)
    return candidates


def parse_json_from_text(text: str) -> dict[str, Any]:
    for candidate in _json_candidates(text):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            py = re.sub(r"\btrue\b", "True", candidate, flags=re.IGNORECASE)
            py = re.sub(r"\bfalse\b", "False", py, flags=re.IGNORECASE)
            py = re.sub(r"\bnull\b", "None", py, flags=re.IGNORECASE)
            obj = ast.literal_eval(py)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def chat_json(messages: list[dict[str, str]], llm_config: dict[str, Any]) -> dict[str, Any]:
    url = _chat_url(str(llm_config["base_url"]))
    headers = {
        "Authorization": f"Bearer {llm_config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": llm_config["model"],
        "messages": messages,
        "temperature": float(llm_config.get("temperature", 0.2)),
        "max_tokens": int(llm_config.get("max_tokens", 4096)),
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    timeout = int(llm_config.get("timeout", 600))
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            data = response.json()
            content = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
            parsed = parse_json_from_text(content)
            if parsed:
                parsed["_usage"] = data.get("usage", {})
                return parsed
            raise ValueError("LLM response did not contain a JSON object")
        except Exception as exc:
            last_error = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"LLM JSON request failed after retries: {last_error}")
