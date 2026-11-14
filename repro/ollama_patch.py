"""Route Ollama-served models through Ollama's native API with schema-constrained output.

Upstream Magentic Marketplace asks for structured output through OpenAI's
`chat.completions.parse(response_format=...)`. Ollama's OpenAI-compatible endpoint
does not reliably enforce that schema, so small local models answer in prose and
every agent step fails to parse. Ollama's native /api/chat endpoint accepts the
JSON schema in its `format` field and constrains generation to it.

When the OpenAI client points at an Ollama server (port 11434), this patch sends
the same request to /api/chat instead, with:
  * `format` set to the response model's JSON schema (structured calls only),
  * `options.num_ctx` set from OLLAMA_NUM_CTX (default 8192) so long agent
    histories are not silently truncated to Ollama's 4096-token default.
Every other provider and model is untouched.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pydantic

_APPLIED = False


def _is_ollama(base_url: str | None) -> bool:
    return bool(base_url) and ":11434" in base_url


def _native_chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base + "/api/chat"


def _plain_message(message: Any) -> dict[str, str]:
    """Convert an OpenAI-style message (dict or SDK object) to {'role', 'content'}."""
    if not isinstance(message, dict):
        message = dict(message)
    content = message.get("content") or ""
    if isinstance(content, list):  # content parts
        content = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    role = message.get("role", "user")
    if role not in ("system", "user", "assistant", "tool"):
        role = "user"
    return {"role": role, "content": str(content)}


def apply() -> None:
    """Patch OpenAIClient._generate once per process."""
    global _APPLIED
    if _APPLIED:
        return
    from magentic_marketplace.marketplace.llm.base import Usage
    from magentic_marketplace.marketplace.llm.clients.openai import OpenAIClient

    original = OpenAIClient._generate
    num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
    timeout = float(os.environ.get("OLLAMA_TIMEOUT_S", "900"))

    async def _generate(
        self,
        *,
        model: str,
        messages,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | int | None = None,
        response_format=None,
        **kwargs: Any,
    ):
        if not _is_ollama(self.config.base_url):
            return await original(
                self,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
                **kwargs,
            )

        options: dict[str, Any] = {"num_ctx": num_ctx}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens:
            options["num_predict"] = max_tokens
        body: dict[str, Any] = {
            "model": model,
            "messages": [_plain_message(m) for m in messages],
            "stream": False,
            "options": options,
        }
        if response_format is not None:
            body["format"] = response_format.model_json_schema()

        errors: list[str] = []
        attempts = 3 if response_format is not None else 1
        async with httpx.AsyncClient(timeout=timeout) as http:
            for _ in range(attempts):
                resp = await http.post(_native_chat_url(self.config.base_url), json=body)
                if resp.status_code >= 400:
                    raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                content = (data.get("message") or {}).get("content", "") or ""
                usage = Usage(
                    token_count=int(data.get("prompt_eval_count", 0)) + int(data.get("eval_count", 0)),
                    provider="openai",
                    model=model,
                )
                if response_format is None:
                    return content, usage
                try:
                    return response_format.model_validate_json(content), usage
                except pydantic.ValidationError as e:
                    errors.append(str(e))
                    body["messages"] = body["messages"] + [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": f"That reply did not match the required JSON schema: {e}. "
                            "Reply again with only the JSON object.",
                        },
                    ]
        raise RuntimeError(
            "Exceeded attempts to parse response_format. Inner exceptions: " + " ".join(errors)
        )

    OpenAIClient._generate = _generate  # type: ignore[method-assign]
    _APPLIED = True
