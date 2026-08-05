from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class DeliveryError(RuntimeError):
    pass


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
