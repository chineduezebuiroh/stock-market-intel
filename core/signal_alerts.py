from __future__ import annotations

# core/signal_alerts.py

import os
from dataclasses import dataclass
from typing import Iterable, Sequence, Optional
import hashlib
from datetime import datetime

import pandas as pd

from core.paths import DATA
from core import storage
from core.notify import send_slack


@dataclass(frozen=True)
class SignalAlert:
    combo: str
    hits: int
    details: str
    digest: str


def _norm_signal(x) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def _alert_state_path(group: str, combo: str) -> "pd.Timestamp":
    safe_group = str(group).strip().replace(" ", "_")
    safe_combo = str(combo).strip().replace(" ", "_")
    return DATA / "_alerts" / safe_group / f"{safe_combo}.parquet"


def _load_last_digest(group: str, combo: str) -> Optional[str]:
    p = _alert_state_path(group, combo)
    if not storage.exists(p):
        return None
    df = storage.load_parquet(p)
    if df is None or df.empty:
        return None
    val = df.iloc[-1].get("last_digest")
    return str(val) if val is not None else None


def _save_last_digest(group: str, combo: str, digest: str) -> None:
    p = _alert_state_path(group, combo)
    df = pd.DataFrame(
        [{
            "group": group,
            "combo": combo,
            "last_digest": digest,
            "sent_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }]
    )
    storage.save_parquet(df, p)


def scan_combo_for_signals(
    combo_name: str,
    *,
    signal_col: str = "signal",
    symbol_col: str = "symbol",
    allowed: Sequence[str] = ("long", "short", "watch"),
    max_symbols_per_signal: int = 40,
) -> SignalAlert:
    path = DATA / f"combo_{combo_name}.parquet"
    if not storage.exists(path):
        return SignalAlert(combo_name, 0, f"missing combo parquet: {path}")

    df = storage.load_parquet(path)
    if df is None or df.empty:
        return SignalAlert(combo_name, 0, "combo empty")

    if symbol_col not in df.columns:
        df = df.reset_index()

    if signal_col not in df.columns:
        return SignalAlert(combo_name, 0, f"missing '{signal_col}' column")
    

    allowed_set = set(a.lower() for a in allowed)

    tmp = df[[symbol_col, signal_col]].copy()
    tmp["__sig"] = tmp[signal_col].map(_norm_signal)
    hits = tmp[tmp["__sig"].isin(allowed_set)].copy()

    if hits.empty:
        return SignalAlert(combo_name, 0, "no signals", digest="")

    # Build canonical grouped representation
    grouped: list[tuple[str, list[str]]] = []
    lines = []
    for sig in allowed:
        s = sig.lower()
        syms = (
            hits.loc[hits["__sig"] == s, symbol_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .unique()
            .tolist()
        )
        syms = sorted(syms)
        if not syms:
            continue
        grouped.append((s, syms))

        shown = syms[:max_symbols_per_signal]
        suffix = "" if len(syms) <= len(shown) else f" (+{len(syms) - len(shown)} more)"
        lines.append(f"- {sig.upper()}: {', '.join(shown)}{suffix}")

    details = "\n".join(lines) if lines else "signals present"

    # Digest of the canonical (signal -> symbols) set
    canonical = "|".join([f"{sig}:{','.join(syms)}" for sig, syms in grouped])
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return SignalAlert(combo_name, len(hits), details, digest=digest)


def notify_on_signals(
    combos: Iterable[str],
    *,
    title: str,
    allowed: Sequence[str] = ("long", "short", "watch"),
    changed_only: bool = False,
    changed_group: str = "default",
) -> None:
    
    alerts: list[SignalAlert] = []
    
    for c in combos:
        a = scan_combo_for_signals(c, allowed=allowed)
        if a.hits <= 0:
            continue

        if changed_only:
            last = _load_last_digest(changed_group, c)
            if last == a.digest:
                print(f"[SIGNALS] {c}: unchanged; no notification.")
                continue

        alerts.append(a)

    if not alerts:
        print("[SIGNALS] No alerts to send.")
        return

    body_parts = []
    for a in alerts:
        body_parts.append(f"\n*{a.combo}* — {a.hits} hit(s)\n{a.details}")

    msg = f"*{title}*\n" + "\n".join(body_parts)
    send_slack(msg)

    # Only mark digests after successful send attempt
    if changed_only:
        for a in alerts:
            _save_last_digest(changed_group, a.combo, a.digest)
