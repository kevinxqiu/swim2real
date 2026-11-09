"""Cached model responses and transport retries."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from swim2real.config import read_json
from swim2real.media import sha256_file
from swim2real.search.store import atomic_write_json, canonical_sha256, record_redo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model_version: str
    usage: Mapping[str, Any]
    response_id: str = ""


class CachedClient:
    """Cache complete responses using prompt, settings, context and media hashes."""

    def __init__(
        self,
        *,
        settings: Mapping[str, Any],
        attempts: int,
        retry_delay: float,
        transport: Callable[[str, Sequence[str]], ModelResponse],
        retryable: Callable[[Exception], bool],
        retry_multiplier: float = 1.0,
        retry_max_delay: float | None = None,
    ) -> None:
        self.settings = dict(settings)
        self.attempts = attempts
        self.retry_delay = retry_delay
        self.transport = transport
        self.retryable = retryable
        self.retry_multiplier = retry_multiplier
        self.retry_max_delay = retry_max_delay

    def call(
        self,
        prompt: str,
        media_paths: Sequence[str],
        directory: Path,
        *,
        request_identity: Mapping[str, Any],
    ) -> ModelResponse:
        """Return a persisted response, or issue a bounded number of transport attempts."""
        request = {
            "prompt": prompt,
            "media": [{"name": Path(p).name, "sha256": sha256_file(p)} for p in media_paths],
            "settings": self.settings,
            "identity": dict(request_identity),
        }
        digest = canonical_sha256(request)
        directory.mkdir(parents=True, exist_ok=True)
        request_file = directory / "request.json"
        if request_file.exists():
            if read_json(request_file) != request:
                raise ValueError(f"Provider request differs on resume: {request_file}")
        else:
            atomic_write_json(request_file, request)
        pending = directory / "pending.json"
        if pending.exists():
            saved_pending = read_json(pending)
            if not isinstance(saved_pending, dict):
                raise ValueError(f"Provider pending request must be a JSON object: {pending}")
            attempt = saved_pending.get("attempt")
            if (
                set(saved_pending) != {"attempt", "request_sha256"}
                or isinstance(attempt, bool)
                or not isinstance(attempt, int)
                or not 1 <= attempt <= self.attempts
                or saved_pending["request_sha256"] != digest
            ):
                raise ValueError(f"Provider pending request mismatch: {pending}")
        response_file = directory / "response.json"
        if response_file.exists():
            saved = read_json(response_file)
            if saved.get("request_sha256") != digest or canonical_sha256(
                saved["response"]
            ) != saved.get("response_sha256"):
                raise ValueError(f"Provider cache content mismatch: {response_file}")
            response = self._response(saved["response"])
            pending.unlink(missing_ok=True)
            return response
        if pending.exists():
            record_redo(directory / "retries", {"request_sha256": digest})
            logger.warning("Resending interrupted model request: %s", request_file)
        for attempt in range(1, self.attempts + 1):
            atomic_write_json(pending, {"attempt": attempt, "request_sha256": digest})
            try:
                response = self.transport(prompt, media_paths)
                response = self._response(asdict(response))
            except Exception as exc:
                pending.unlink()
                if not self.retryable(exc) or attempt == self.attempts:
                    raise
                logger.warning(
                    "Provider transport attempt %d failed (%s); retrying",
                    attempt,
                    type(exc).__name__,
                )
                delay = self.retry_delay * self.retry_multiplier ** (attempt - 1)
                if self.retry_max_delay is not None:
                    delay = min(delay, self.retry_max_delay)
                time.sleep(delay)
                continue
            value = asdict(response)
            atomic_write_json(
                response_file,
                {
                    "request_sha256": digest,
                    "response_sha256": canonical_sha256(value),
                    "response": value,
                },
            )
            pending.unlink()
            return response
        raise RuntimeError("Provider attempts exhausted")

    @staticmethod
    def _response(value: Mapping[str, Any]) -> ModelResponse:
        response = ModelResponse(**value)
        if (
            not isinstance(response.text, str)
            or not response.text.strip()
            or not isinstance(response.model_version, str)
            or not response.model_version
        ):
            raise ValueError("Provider returned empty text or no model version")
        canonical_sha256(dict(response.usage))
        return response
