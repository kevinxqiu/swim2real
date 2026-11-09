"""Gemini model requests through the Google Gen AI SDK."""

from __future__ import annotations

import mimetypes
import os
from dataclasses import asdict
from typing import Sequence

from swim2real.config import VLMConfig
from swim2real.providers.client import CachedClient, ModelResponse


class GeminiClient(CachedClient):
    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self._sdk = None
        super().__init__(
            settings=asdict(config),
            attempts=config.max_transport_attempts,
            retry_delay=config.retry_delay_seconds,
            transport=self._generate,
            retryable=self._retryable,
        )

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        code = code() if callable(code) else code
        names = {kind.__name__ for kind in type(exc).__mro__}
        retry_names = {
            "Aborted",
            "DeadlineExceeded",
            "GatewayTimeout",
            "InternalServerError",
            "ResourceExhausted",
            "ServerError",
            "ServiceUnavailable",
            "TooManyRequests",
            "ReadTimeout",
            "ConnectTimeout",
            "RemoteProtocolError",
            "ConnectError",
            "ReadError",
            "WriteError",
        }
        return (
            isinstance(exc, (TimeoutError, ConnectionError))
            or bool(names & retry_names)
            or any(
                token in str(code).upper()
                for token in ("408", "429", "500", "502", "503", "504", "DEADLINE")
            )
        )

    def _generate(self, prompt: str, media_paths: Sequence[str]) -> ModelResponse:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("Gemini requires pip install 'swim2real[vlm]'") from exc
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY before using the Gemini provider")
        if self._sdk is None:
            self._sdk = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    api_version=self.config.api_version,
                    timeout=int(self.config.timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        parts = [types.Part.from_text(text=prompt)]
        from pathlib import Path

        for filename in media_paths:
            path = Path(filename)
            mime = mimetypes.guess_type(path.name)[0]
            if mime is None:
                raise ValueError(f"Unknown media MIME type: {path}")
            parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
        response = self._sdk.models.generate_content(
            model=self.config.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                seed=self.config.decoding_seed,
                max_output_tokens=self.config.max_output_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=self.config.thinking_budget),
            ),
        )
        usage = response.usage_metadata.model_dump(mode="json") if response.usage_metadata else {}
        return ModelResponse(
            response.text, response.model_version, usage, getattr(response, "response_id", "") or ""
        )
