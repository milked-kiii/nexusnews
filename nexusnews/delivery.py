from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class DeliveryError(RuntimeError):
    pass


def _feishu_access_token(app_id: str, app_secret: str, *, timeout: float = 10) -> str:
    try:
        req = Request(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            data=json.dumps({"app_id": app_id, "app_secret": app_secret}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise DeliveryError(f"Feishu access token request failed: {exc}") from exc
    code = data.get("code", -1)
    if code != 0:
        raise DeliveryError(f"Feishu rejected access token request (code={code}): {data}")
    return data["tenant_access_token"]


def send_feishu_dm(text: str, open_id: str, *, app_id_env: str = "FEISHU_APP_ID",
                   app_secret_env: str = "FEISHU_APP_SECRET", attempts: int = 3,
                   timeout: float = 10, delay: float = 1) -> None:
    """Send a text message to a Feishu user via the IM API (DM)."""
    return _send_feishu_im(text, open_id, "open_id", app_id_env, app_secret_env,
                           attempts, timeout, delay)


def send_feishu_chat(text: str, chat_id: str, *, app_id_env: str = "FEISHU_APP_ID",
                     app_secret_env: str = "FEISHU_APP_SECRET", attempts: int = 3,
                     timeout: float = 10, delay: float = 1) -> None:
    """Send a text message to a Feishu chat (DM or group) via the IM API."""
    return _send_feishu_im(text, chat_id, "chat_id", app_id_env, app_secret_env,
                           attempts, timeout, delay)


def _send_feishu_im(text: str, receive_id: str, receive_id_type: str,
                    app_id_env: str, app_secret_env: str,
                    attempts: int, timeout: float, delay: float) -> None:
    """Shared Feishu IM send: open_id or chat_id."""
    app_id = os.environ.get(app_id_env)
    app_secret = os.environ.get(app_secret_env)
    if not app_id or not app_secret:
        raise DeliveryError(f"Feishu delivery requires {app_id_env} and {app_secret_env}")
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            token = _feishu_access_token(app_id, app_secret, timeout=timeout)
            payload = json.dumps({
                "receive_id": receive_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            }, ensure_ascii=False).encode()
            req = Request(
                f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={receive_id_type}",
                data=payload,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=timeout) as resp:
                decoded = json.loads(resp.read())
            if decoded.get("code", 0) != 0:
                raise DeliveryError(f"Feishu rejected (code={decoded.get('code')}): {decoded}")
            return
        except DeliveryError:
            raise
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(delay * (2 ** attempt))
    raise DeliveryError(f"Feishu delivery failed after {attempts} attempts: {last}") from last


def send_feishu_card(card_json: str, receive_id: str, receive_id_type: str,
                     *, app_id_env: str = "FEISHU_APP_ID",
                     app_secret_env: str = "FEISHU_APP_SECRET", attempts: int = 3,
                     timeout: float = 10, delay: float = 1) -> None:
    """Send a Feishu interactive card message."""
    app_id = os.environ.get(app_id_env)
    app_secret = os.environ.get(app_secret_env)
    if not app_id or not app_secret:
        raise DeliveryError(f"Feishu card delivery requires {app_id_env} and {app_secret_env}")
    card_obj = json.loads(card_json)
    card_content = card_obj if "card" not in card_obj else card_obj["card"]
    content_str = json.dumps(card_content, ensure_ascii=False)
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            token = _feishu_access_token(app_id, app_secret, timeout=timeout)
            payload = json.dumps({
                "receive_id": receive_id,
                "msg_type": "interactive",
                "content": content_str,
            }, ensure_ascii=False).encode()
            req = Request(
                f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={receive_id_type}",
                data=payload,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=timeout) as resp:
                decoded = json.loads(resp.read())
            if decoded.get("code", 0) != 0:
                raise DeliveryError(f"Feishu rejected card (code={decoded.get('code')}): {decoded}")
            return
        except DeliveryError:
            raise
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(delay * (2 ** attempt))
    raise DeliveryError(f"Feishu card delivery failed after {attempts} attempts: {last}") from last


def send_feishu(webhook: str, text: str, *, attempts: int = 3, timeout: float = 10, delay: float = 1) -> None:
    payload = json.dumps({"msg_type": "text", "content": {"text": text}}, ensure_ascii=False).encode()
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(Request(webhook, data=payload, headers={"Content-Type": "application/json"}), timeout=timeout) as response:
                decoded = json.loads(response.read())
            if decoded.get("code", decoded.get("StatusCode", 0)) not in (0, None):
                raise DeliveryError(f"Feishu rejected message: {decoded}")
            return
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, DeliveryError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(delay * (2 ** attempt))
    raise DeliveryError(f"Feishu delivery failed after {attempts} attempts: {last}") from last
