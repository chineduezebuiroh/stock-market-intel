from __future__ import annotations

# core/notify.py

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import requests

from core.paths import DATA
from core import storage


SIGNALS_OF_INTEREST = {"long", "short", "watch"}


@dataclass(frozen=True)
class SignalRow:
    symbol: str
    signal: str
    side: str | None = None
    long_score: float | None = None
    short_score: float | None = None


def _env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def send_telegram_message(text: str) -> None:
    token = _env("TELEGRAM_BOT_TOKEN")
    chat_id = _env("TELEGRAM_CHAT_ID")

    # Telegram has message size limits; keep it safe.
    # If you ever hit limits, we can chunk.
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",  # keep simple; Telegram “MarkdownV2” is picky
        "disable_web_page_preview": True,
    }
    r = requests.post(url, data=payload, timeout=20)
    if not r.ok:
        raise RuntimeError(f"Telegram send failed: {r.status_code} {r.text}")


def _alert_state_path(alert_key: str) -> Path:
    safe = alert_key.replace("/", "_").replace(" ", "_")
    return DATA / "_alerts" / f"{safe}.json"


def load_alert_state(alert_key: str) -> dict[str, Any]:
    p = _alert_state_path(alert_key)
    if not storage.exists(p):
        return {}
    try:
        # storage doesn’t have read_json, so use parquet-less approach:
        # read as bytes via local path assumption won’t work for S3.
        # Easiest: store state as parquet instead of json for backend symmetry.
        # BUT you asked to move fast: we’ll store as parquet below.
        return {}
    except Exception:
        return {}


def _state_parquet_path(alert_key: str) -> Path:
    safe = alert_key.replace("/", "_").replace(" ", "_")
    return DATA / "_alerts" / f"{safe}.parquet"


def load_alert_fingerprint(alert_key: str) -> Optional[str]:
    p = _state_parquet_path(alert_key)
    if not storage.exists(p):
        return None
    df = storage.load_parquet(p)
    if df is None or df.empty or "fingerprint" not in df.columns:
        return None
    return str(df["fingerprint"].iloc[-1])


def save_alert_fingerprint(alert_key: str, fingerprint: str, *, meta: dict[str, Any] | None = None) -> None:
    p = _state_parquet_path(alert_key)
    row: dict[str, Any] = {
        "alert_key": alert_key,
        "fingerprint": fingerprint,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    if meta:
        for k, v in meta.items():
            row[f"meta_{k}"] = str(v)
    storage.save_parquet(pd.DataFrame([row]), p)


def _pick_asof(df: pd.DataFrame) -> str:
    """
    Choose a stable “asof” marker from combo contents.
    Preference: lower_date -> middle_date -> upper_date -> index max.
    """
    for col in ("lower_date", "middle_date", "upper_date"):
        if col in df.columns:
            try:
                ts = pd.to_datetime(df[col]).max()
                if pd.notna(ts):
                    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")
            except Exception:
                pass

    try:
        idx = pd.to_datetime(df.index)
        ts = idx.max()
        if pd.notna(ts):
            return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        pass

    return "unknown_asof"


def extract_signals_from_combo(df: pd.DataFrame) -> list[SignalRow]:
    if df is None or df.empty:
        return []

    # combos might have symbol in index
    if "symbol" not in df.columns:
        df = df.reset_index()

    if "signal" not in df.columns:
        return []

    out: list[SignalRow] = []
    for _, r in df.iterrows():
        sym = str(r.get("symbol", "")).strip()
        sig = str(r.get("signal", "")).strip().lower()
        if not sym or sig not in SIGNALS_OF_INTEREST:
            continue

        side = None
        if "signal_side" in df.columns:
            side = str(r.get("signal_side", "")).strip() or None

        long_score = None
        short_score = None
        if "mtf_long_score" in df.columns:
            try:
                long_score = float(r.get("mtf_long_score"))
            except Exception:
                long_score = None
        if "mtf_short_score" in df.columns:
            try:
                short_score = float(r.get("mtf_short_score"))
            except Exception:
                short_score = None

        out.append(
            SignalRow(
                symbol=sym,
                signal=sig,
                side=side,
                long_score=long_score,
                short_score=short_score,
            )
        )
    return out


def _fingerprint(rows: Iterable[SignalRow]) -> str:
    """
    Stable fingerprint of the current “signal set”.
    If signals/participants change, fingerprint changes.
    """
    items = sorted((r.symbol.upper(), r.signal.lower(), (r.side or "").lower()) for r in rows)
    return json.dumps(items, separators=(",", ":"), ensure_ascii=True)


def format_signal_message(
    *,
    combo_name: str,
    asof: str,
    rows: list[SignalRow],
    max_lines_per_bucket: int = 40,
) -> str:
    """
    Compact, scan-friendly message.
    """
    if not rows:
        return ""

    # bucket by signal
    buckets: dict[str, list[SignalRow]] = {"long": [], "short": [], "watch": []}
    for r in rows:
        buckets[r.signal].append(r)

    def fmt_row(r: SignalRow) -> str:
        parts = [f"`{r.symbol}`"]
        if r.side and r.side.lower() != "none":
            parts.append(f"({r.side})")
        # keep scores short
        if r.long_score is not None or r.short_score is not None:
            ls = "NA" if r.long_score is None else f"{r.long_score:g}"
            ss = "NA" if r.short_score is None else f"{r.short_score:g}"
            parts.append(f"LS:{ls} SS:{ss}")
        return " ".join(parts)

    lines = []
    lines.append(f"*Signals* — `{combo_name}`")
    lines.append(f"As-of: `{asof}`")
    lines.append("")

    for sig in ("long", "short", "watch"):
        rs = sorted(buckets[sig], key=lambda x: x.symbol.upper())
        if not rs:
            continue

        emoji = {"long": "🟢", "short": "🔴", "watch": "🟡"}[sig]
        lines.append(f"{emoji} *{sig.upper()}* ({len(rs)})")

        show = rs[:max_lines_per_bucket]
        for r in show:
            lines.append(f"- {fmt_row(r)}")

        if len(rs) > len(show):
            lines.append(f"- …and {len(rs) - len(show)} more")

        lines.append("")

    return "\n".join(lines).strip()


def notify_combo_signals(
    combo_name: str,
    *,
    only_if_changed: bool,
    alert_key: Optional[str] = None,
) -> None:
    """
    Reads data/combo_<combo_name>.parquet and sends a Telegram message
    if any symbols have signal in {long,short,watch}.

    If only_if_changed=True, suppress if the signal set is unchanged vs last alert.
    """
    combo_path = DATA / f"combo_{combo_name}.parquet"
    if not storage.exists(combo_path):
        print(f"[NOTIFY][SKIP] missing combo parquet: {combo_path}")
        return

    df = storage.load_parquet(combo_path)
    if df is None or df.empty:
        print(f"[NOTIFY][SKIP] empty combo: {combo_name}")
        return

    rows = extract_signals_from_combo(df)
    if not rows:
        print(f"[NOTIFY] no signals for {combo_name}")
        return

    asof = _pick_asof(df)
    fp = _fingerprint(rows)

    key = alert_key or f"{combo_name}"
    if only_if_changed:
        prev = load_alert_fingerprint(key)
        if prev == fp:
            print(f"[NOTIFY] unchanged signals for {combo_name}; suppressing alert")
            return

    msg = format_signal_message(combo_name=combo_name, asof=asof, rows=rows)
    send_telegram_message(msg)

    # record last fingerprint (for change suppression)
    save_alert_fingerprint(key, fp, meta={"combo": combo_name, "asof": asof})
    print(f"[NOTIFY] sent alert for {combo_name} (signals={len(rows)})")
