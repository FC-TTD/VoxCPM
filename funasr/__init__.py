from __future__ import annotations

import json
import mimetypes
import os
import uuid
from typing import Any, Optional
from urllib import error, request


class AutoModel:
    def __init__(
        self,
        model: str,
        disable_update: bool = True,
        log_level: str = "INFO",
        device: Optional[str] = None,
        **_: Any,
    ) -> None:
        self.model = model
        self.disable_update = disable_update
        self.log_level = log_level
        self.device = device
        base_url = (
            os.environ.get("ASR_API_BASE_URL", "http://asrpri-api").strip()
            or "http://asrpri-api"
        )
        self.api_url = f"{base_url.rstrip('/')}/api/v1/asr"
        self.timeout_seconds = float(os.environ.get("ASR_API_TIMEOUT_SECONDS", "60"))

    def _build_multipart_body(
        self, audio_path: str, language: str
    ) -> tuple[bytes, str]:
        filename = os.path.basename(audio_path) or "audio.wav"
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        boundary = f"----VoxCPMFunASR{uuid.uuid4().hex}"
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'.encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                audio_bytes,
                b"\r\n",
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="key"\r\n\r\n',
                filename.encode(),
                b"\r\n",
                f"--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="lang"\r\n\r\n',
                language.encode(),
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        return body, boundary

    def generate(
        self, input: str, language: str = "auto", use_itn: bool = True, **_: Any
    ) -> list[dict[str, str]]:
        body, boundary = self._build_multipart_body(input, language)
        req = request.Request(
            self.api_url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ASR API returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"ASR API request failed: {exc.reason}") from exc

        items = payload.get("result", [])
        if not items:
            return [{"text": ""}]

        first = items[0] if isinstance(items[0], dict) else {}
        text = (
            first.get("text") or first.get("clean_text") or first.get("raw_text") or ""
        ).strip()
        if use_itn:
            return [{"text": text}]
        return [{"text": text}]
