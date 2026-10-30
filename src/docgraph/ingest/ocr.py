"""Optional OCR for scanned PDF pages, through a vision-capable chat model."""

from __future__ import annotations

import base64

from ..config import OcrConfig
from ..llm import ChatClient
from .loaders import OcrFn

OCR_PROMPT = (
    "Transcribe all text on this page exactly as written. Output Markdown: use '#' headings for "
    "headings, blank lines between paragraphs, and plain text for tables. Do not describe the page, "
    "do not add commentary, and do not invent text that is not visible. If the page has no readable "
    "text, output nothing."
)


def make_ocr(client: ChatClient, cfg: OcrConfig) -> OcrFn:
    """Return a function that turns PNG bytes of a page into text using the configured endpoint."""

    def run(png: bytes) -> str:
        data_uri = "data:image/png;base64," + base64.b64encode(png).decode()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
        result = client.complete(
            messages,
            endpoint=cfg.endpoint,
            model=cfg.model,
            params={"temperature": 0.2, "max_tokens": 2048},
        )
        return result.content

    return run
