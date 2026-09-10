"""HTTP client used by the benchmark reasoning judges."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int | None


def chat_complete_json_text(
    *,
    cfg: LLMConfig,
    messages: list[dict[str, Any]],
) -> str:
    """
    Returns the raw assistant message text. (We keep it simple; caller can JSON-parse/validate.)
    """
    message = chat_complete_message(cfg=cfg, messages=messages)
    return str(message.get("content") or "")


def chat_complete_message(
    *,
    cfg: LLMConfig,
    messages: list[dict[str, Any]],
    include_thoughts: bool = False,
) -> dict[str, Any]:
    """
    Returns the full assistant message object from the first choice.
    """
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
    }
    if cfg.max_tokens is not None:
        payload["max_tokens"] = cfg.max_tokens
    if include_thoughts:
        # payload["include_thoughts"] = True
        # payload["include_reasoning"] = True
        # payload["reasoning"] = {"effort":"high", "max_tokens": cfg.max_tokens}
        payload["reasoning"] = {}

    # print(f"Requesting LLM with payload: {json.dumps(payload, indent=2)}")
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if not r.ok:
        detail = r.text
        if len(detail) > 2000:
            detail = detail[:2000] + "..."
        raise requests.HTTPError(
            f"{r.status_code} Client Error for url: {url}; response={detail}",
            response=r,
        )
    resp = r.json()
    message = ((resp.get("choices") or [{}])[0].get("message") or {})
    if isinstance(message, dict):
        return message
    return {}
