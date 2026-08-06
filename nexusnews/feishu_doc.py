"""Feishu cloud-document (docx) sync for the digest.

After the card digest is pushed, the same content is written into a
Feishu cloud document (one per day, titled with the date) and the user is
granted edit access so they can open it from the card link.

API reference:
- Create doc:  POST /open-apis/docx/v1/documents
- Add blocks:  POST /open-apis/docx/v1/documents/{id}/blocks/{block_id}/children
- Grant perm:  POST /open-apis/drive/v1/permissions/{token}/members?type=docx
"""

from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .delivery import DeliveryError, _feishu_access_token

# Feishu docx block types (see https://open.feishu.cn/document/server-docs/docs/docs/docx-v1/document-block/list)
_BLOCK_TEXT = 2
_BLOCK_HEADING1 = 3
_BLOCK_HEADING2 = 4
_BLOCK_BULLET = 12
_BLOCK_ORDERED = 13
_BLOCK_QUOTE = 15
_BLOCK_DIVIDER = 22

_BASE = "https://open.feishu.cn/open-apis"


def _api(token: str, path: str, body: dict, *, method: str = "POST", timeout: float = 15) -> dict:
    """One authenticated API call; raises DeliveryError on HTTP/business failure."""
    req = Request(
        f"{_BASE}{path}",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise DeliveryError(f"Feishu doc API {path} failed: {exc}") from exc
    if data.get("code", 0) != 0:
        raise DeliveryError(f"Feishu doc API {path} rejected (code={data.get('code')}): {data}")
    return data


def _run(call: object, *, attempts: int = 3, timeout: float = 15, delay: float = 1) -> dict:
    """Decorator-free retry wrapper: run an API call with exponential backoff."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()  # type: ignore[no-any-return]
        except DeliveryError:
            raise
        except (HTTPError, URLError, TimeoutError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(delay * (2 ** attempt))
    raise DeliveryError(f"Feishu doc API failed after {attempts} attempts: {last}") from last


def _text_run(content: str, url: str | None = None) -> dict:
    run: dict = {"text_run": {"content": content}}
    if url:
        run["text_run"]["text_element_style"] = {"link": {"url": url}}
    return run


def _heading(text: str, level: int = 1) -> dict:
    tag = {1: "heading1", 2: "heading2"}[level]
    return {"block_type": _BLOCK_HEADING1 if level == 1 else _BLOCK_HEADING2, tag: {"elements": [_text_run(text)], "style": {}}}


def _paragraph(text: str) -> dict:
    return {"block_type": _BLOCK_TEXT, "text": {"elements": [_text_run(text)], "style": {}}}


def _bullet(text: str, url: str | None = None) -> dict:
    return {"block_type": _BLOCK_BULLET, "bullet": {"elements": [_text_run(text, url)], "style": {}}}


def _divider() -> dict:
    return {"block_type": _BLOCK_DIVIDER, "divider": {}}


def _quote(text: str) -> dict:
    return {"block_type": _BLOCK_QUOTE, "quote": {"elements": [_text_run(text)], "style": {}}}


def _heading_blocks(title: str) -> list[dict]:
    """A docx heading tree for one digest title."""
    return [_heading(title, 1)]


def markdown_to_docx_blocks(lines: list[str]) -> list[dict]:
    """Convert digest text lines into Feishu docx blocks.

    Understands a small subset of the digest markup:
      - '## '        -> heading 2
      - '1. '        -> ordered bullet (numbered item)
      - '- '         -> bullet
      - '> '         -> quote
      - '标题' lines -> heading 1
      - '[text](url)' -> hyperlink in a paragraph/bullet
    Anything else becomes a plain paragraph.
    """
    import re

    blocks: list[dict] = []
    link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("## "):
            blocks.append(_heading(line[3:].strip(), 2))
            continue
        if line.startswith("# "):
            blocks.append(_heading(line[2:].strip(), 1))
            continue
        if line.startswith("> "):
            blocks.append(_quote(line[2:].strip()))
            continue
        ordered = line.startswith(("1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. "))
        bullet = line.startswith("- ")
        if ordered or bullet:
            text = line[3:].strip() if ordered else line[2:].strip()
            # split "[text](url)" into runs
            parts = link_re.split(text)
            if len(parts) == 1:
                blocks.append(_bullet(text))
            else:
                elements = []
                for i in range(0, len(parts) - 2, 3):
                    elements.append(_text_run(parts[i]))
                    elements.append(_text_run(parts[i + 1], parts[i + 2]))
                elements.append(_text_run(parts[-1]))
                blocks.append(
                    {"block_type": _BLOCK_BULLET, "bullet": {"elements": elements, "style": {}}}
                )
            continue
        # Plain paragraph with optional link
        parts = link_re.split(line)
        if len(parts) == 1:
            blocks.append(_paragraph(line))
        else:
            elements = []
            for i in range(0, len(parts) - 2, 3):
                elements.append(_text_run(parts[i]))
                elements.append(_text_run(parts[i + 1], parts[i + 2]))
            elements.append(_text_run(parts[-1]))
            blocks.append({"block_type": _BLOCK_TEXT, "text": {"elements": elements, "style": {}}})

    if not blocks:
        blocks.append(_paragraph("（今日无内容）"))
    return blocks


def create_doc(title: str, *, app_id_env: str = "FEISHU_APP_ID",
               app_secret_env: str = "FEISHU_APP_SECRET") -> dict:
    """Create an empty Feishu docx document. Returns {'document_id': ..., 'url': ...}."""
    app_id = os.environ.get(app_id_env)
    app_secret = os.environ.get(app_secret_env)
    if not app_id or not app_secret:
        raise DeliveryError(f"Feishu doc sync requires {app_id_env} and {app_secret_env}")

    def _create() -> dict:
        token = _feishu_access_token(app_id, app_secret)
        data = _api(token, "/docx/v1/documents", {"title": title})
        doc = data.get("data", {}).get("document", {})
        document_id = doc.get("document_id", "")
        # The create API does not return a URL; build the canonical docx link.
        url = doc.get("url") or (f"https://www.feishu.cn/docx/{document_id}" if document_id else "")
        return {"document_id": document_id, "url": url}

    result = _run(_create)
    if not result.get("document_id"):
        raise DeliveryError(f"Feishu doc create returned no document_id: {result}")
    return result


def add_blocks(document_id: str, blocks: list[dict], *, app_id_env: str = "FEISHU_APP_ID",
               app_secret_env: str = "FEISHU_APP_SECRET", chunk: int = 20) -> None:
    """Append blocks under the document root (document_id acts as root block id)."""
    app_id = os.environ.get(app_id_env)
    app_secret = os.environ.get(app_secret_env)
    if not app_id or not app_secret:
        raise DeliveryError(f"Feishu doc sync requires {app_id_env} and {app_secret_env}")
    for i in range(0, len(blocks), chunk):
        group = blocks[i:i + chunk]

        def _add() -> dict:
            token = _feishu_access_token(app_id, app_secret)
            return _api(
                token,
                f"/docx/v1/documents/{document_id}/blocks/{document_id}/children",
                {"children": group, "index": -1},
            )

        _run(_add)


def grant_permission(document_id: str, open_id: str, *, perm: str = "edit",
                     app_id_env: str = "FEISHU_APP_ID",
                     app_secret_env: str = "FEISHU_APP_SECRET") -> None:
    """Grant the user edit access to the doc so the shared link works for them."""
    app_id = os.environ.get(app_id_env)
    app_secret = os.environ.get(app_secret_env)
    if not app_id or not app_secret:
        raise DeliveryError(f"Feishu doc sync requires {app_id_env} and {app_secret_env}")

    def _grant() -> dict:
        token = _feishu_access_token(app_id, app_secret)
        return _api(
            token,
            f"/drive/v1/permissions/{document_id}/members?type=docx",
            {"member_type": "openid", "member_id": open_id, "perm": perm},
        )

    _run(_grant)


def sync_digest_to_doc(title: str, lines: list[str], open_id: str,
                       *, app_id_env: str = "FEISHU_APP_ID",
                       app_secret_env: str = "FEISHU_APP_SECRET") -> dict:
    """Create a dated doc, write the digest content into it, grant user access.

    Returns {'document_id': ..., 'url': ...}.
    """
    doc = create_doc(title, app_id_env=app_id_env, app_secret_env=app_secret_env)
    blocks = [_heading(title, 1)] + [_divider()] + markdown_to_docx_blocks(lines)
    add_blocks(doc["document_id"], blocks, app_id_env=app_id_env, app_secret_env=app_secret_env)
    grant_permission(doc["document_id"], open_id, app_id_env=app_id_env, app_secret_env=app_secret_env)
    return doc
