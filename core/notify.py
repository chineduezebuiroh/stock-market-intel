from __future__ import annotations

# core/notify.py

import json
import os
import urllib.request
from typing import Optional


def send_slack(text: str, webhook_url: Optional[str] = None) -> None:
    """
    Send a simple Slack message via Incoming Webhook.
    Expects SLACK_WEBHOOK_URL in env (recommended via GitHub Secrets).
    """
    url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not url:
        print("[NOTIFY][WARN] SLACK_WEBHOOK_URL not set; skipping Slack notification.")
        return

    payload = {"text": text}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            _ = resp.read()
        print("[NOTIFY] Slack message sent.")
    except Exception as e:
        # Don't fail the job just because Slack had a hiccup
        print(f"[NOTIFY][WARN] Slack send failed: {e}")
