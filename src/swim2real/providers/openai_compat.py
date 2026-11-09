"""OpenAI-compatible GLM requests with deterministic clip frame sampling."""

from __future__ import annotations

import base64
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from swim2real.config import OpenAIConfig
from swim2real.providers.client import CachedClient, ModelResponse

FRAME_NOTICE_ANCHOR = "frequencies to identify which parameters need adjustment.\n"
EXTRACTION_METHOD_FENCED = "fenced"
EXTRACTION_METHOD_BALANCED = "balanced"
EXTRACTION_METHOD_NONE = "none"


@dataclass(frozen=True)
class FrameContract:
    frames_per_clip: int
    max_side_pixels: int
    jpeg_quality: int


def sample_frame_indices(frame_count: int, frames_per_clip: int) -> tuple[int, ...]:
    """Select evenly spaced clip frames, including both endpoints."""
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < frames_per_clip
    ):
        raise ValueError(f"clip has too few frames for the media contract: {frame_count}")
    return tuple(
        (index * (frame_count - 1) // (frames_per_clip - 1) for index in range(frames_per_clip))
    )


def encode_clip_frames(path: str | Path, contract: FrameContract) -> list[bytes]:
    """Decode, resize and JPEG-encode a clip at the configured sample indices."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Clip frames require pip install 'swim2real[sim]'") from exc
    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open media clip: {source}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = sample_frame_indices(frame_count, contract.frames_per_clip)
        wanted_set = set(wanted)
        decoded: dict[int, Any] = {}
        index = 0
        while index <= max(wanted_set):
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted_set:
                decoded[index] = frame
            index += 1
    finally:
        capture.release()
    missing = [value for value in wanted if value not in decoded]
    if missing:
        raise ValueError(f"clip decode did not reach the contract frames {missing}: {source}")
    encoded: list[bytes] = []
    for position in wanted:
        frame = decoded[position]
        height, width = (int(frame.shape[0]), int(frame.shape[1]))
        longest = max(height, width)
        if longest > contract.max_side_pixels:
            scale = contract.max_side_pixels / float(longest)
            target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            frame = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), contract.jpeg_quality]
        )
        if not ok:
            raise ValueError(f"JPEG encoding failed for {source}")
        encoded.append(buffer.tobytes())
    return encoded


@dataclass(frozen=True)
class ExtractionResult:
    """A proposal object located within assistant text."""

    method: str
    start: int | None
    end: int | None
    extracted_text: str


def _closed_fenced_spans(text: str) -> Iterable[tuple[int, int]]:
    """Locate the content spans of closed Markdown code fences."""
    offset = 0
    content_start: int | None = None
    for line in text.splitlines(keepends=True):
        is_fence = line.strip().startswith("```")
        if content_start is None:
            if is_fence:
                content_start = offset + len(line)
        elif is_fence:
            yield (content_start, offset)
            content_start = None
        offset += len(line)


def _balanced_object_end(text: str, start: int) -> int | None:
    """Find the end of a JSON object while respecting quoted braces."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def extract_proposal_json(text: str | None) -> ExtractionResult:
    """Locate the first complete fenced or brace-balanced proposal object."""
    value = text or ""
    for start, end in _closed_fenced_spans(value):
        if value[start:end].strip().startswith("{"):
            return ExtractionResult(
                method=EXTRACTION_METHOD_FENCED,
                start=start,
                end=end,
                extracted_text=value[start:end],
            )
    opening = value.find("{")
    if opening >= 0:
        closing = _balanced_object_end(value, opening)
        if closing is not None:
            return ExtractionResult(
                method=EXTRACTION_METHOD_BALANCED,
                start=opening,
                end=closing,
                extracted_text=value[opening:closing],
            )
    return ExtractionResult(
        method=EXTRACTION_METHOD_NONE, start=None, end=None, extracted_text=value
    )


def frame_notice_sentence(frames_per_clip: int) -> str:
    return f"Each video is provided as {int(frames_per_clip)} uniformly sampled frames in temporal order.\n"


def apply_frame_notice(prompt: str, frames_per_clip: int) -> str:
    """Insert the exact frame-sampling sentence after the media paragraph."""
    if prompt.count(FRAME_NOTICE_ANCHOR) != 1:
        raise ValueError("configured prompt does not carry exactly one media-paragraph anchor")
    sentence = frame_notice_sentence(frames_per_clip)
    if sentence in prompt:
        raise ValueError("configured prompt already claims a frame decomposition")
    return prompt.replace(FRAME_NOTICE_ANCHOR, FRAME_NOTICE_ANCHOR + sentence, 1)


class OpenAICompatibleClient(CachedClient):
    def __init__(self, config: OpenAIConfig) -> None:
        self.config = config
        self.contract = FrameContract(
            config.frames_per_clip, config.max_side_pixels, config.jpeg_quality
        )
        self._sdk = None
        super().__init__(
            settings=asdict(config),
            attempts=config.max_transport_attempts,
            retry_delay=config.retry_delay_seconds,
            transport=self._generate,
            retryable=self._retryable,
            retry_multiplier=config.retry_multiplier,
            retry_max_delay=config.retry_max_delay_seconds,
        )

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        return (
            isinstance(exc, (TimeoutError, ConnectionError))
            or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
            or getattr(exc, "status_code", None) in {408, 429, 500, 502, 503, 504}
        )

    def request_body(
        self, prompt: str, frames_by_clip: Sequence[Sequence[bytes]]
    ) -> dict[str, Any]:
        """Build the same text and frame request as the second-model adapter."""
        sent_prompt = (
            apply_frame_notice(prompt, self.config.frames_per_clip) if frames_by_clip else prompt
        )
        content = [{"text": sent_prompt, "type": "text"}]
        for frames in frames_by_clip:
            for frame in frames:
                encoded = base64.b64encode(frame).decode("ascii")
                content.append(
                    {"image_url": {"url": f"data:image/jpeg;base64,{encoded}"}, "type": "image_url"}
                )
        body = {
            "max_tokens": self.config.max_output_tokens,
            "messages": [{"content": content, "role": "user"}],
            "model": self.config.model,
            "seed": int(self.config.decoding_seed),
            "temperature": float(self.config.temperature),
        }
        body[
            "reasoning_effort" if self.config.reasoning_style == "reasoning_effort" else "reasoning"
        ] = (
            self.config.reasoning_effort
            if self.config.reasoning_style == "reasoning_effort"
            else {"effort": self.config.reasoning_effort}
        )
        return body

    def _generate(self, prompt: str, media_paths: Sequence[str]) -> ModelResponse:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAI-compatible providers require pip install 'swim2real[vlm]'"
            ) from exc
        if not self.config.base_url:
            raise ValueError("Set openai_compat.base_url to the provider endpoint")
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise ValueError(f"Set {self.config.api_key_env} before using this provider")
        if self._sdk is None:
            self._sdk = openai.OpenAI(
                api_key=key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_seconds,
                max_retries=0,
            )
        frames = [encode_clip_frames(p, self.contract) for p in media_paths]
        body = self.request_body(prompt, frames)
        reasoning = {k: body.pop(k) for k in ["reasoning_effort", "reasoning"] if k in body}
        response = self._sdk.chat.completions.create(**body, extra_body=reasoning)
        if not response.choices:
            raise ValueError("OpenAI-compatible provider returned no choices")
        text = response.choices[0].message.content
        if not isinstance(text, str) or not text.strip():
            raise ValueError("OpenAI-compatible provider returned empty or null content")
        if self.config.parse_mode == "lenient_fenced_json":
            extracted = extract_proposal_json(text)
            text = extracted.extracted_text
        usage = response.usage.model_dump(mode="json") if response.usage else {}
        return ModelResponse(text, response.model, usage, response.id)
