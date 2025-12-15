from __future__ import annotations

# core/signal_alerts.py

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from core.paths import DATA
from core import storage
from core.notify import send_slack


@dataclass(frozen=True)
class SignalAlert:
    combo: str
    hits: int
    details: str


def _norm_signal(x) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


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
        return SignalAlert(combo_name, 0, "no signals")

    # group symbols by signal
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
        if not syms:
            continue
        shown = syms[:max_symbols_per_signal]
        suffix = "" if len(syms) <= len(shown) else f" (+{len(syms) - len(shown)} more)"
        lines.append(f"- {sig.upper()}: {', '.join(shown)}{suffix}")

    details = "\n".join(lines) if lines else "signals present (but could not format list)"
    return SignalAlert(combo_name, len(hits), details)


def notify_on_signals(
    combos: Iterable[str],
    *,
    title: str = "Signal Alert",
    allowed: Sequence[str] = ("long", "short", "watch"),
) -> None:
    alerts: list[SignalAlert] = []
    for c in combos:
        a = scan_combo_for_signals(c, allowed=allowed)
        if a.hits > 0:
            alerts.append(a)

    if not alerts:
        print("[SIGNALS] No long/short/watch signals found; no notification sent.")
        return

    # Helpful GitHub context if present
    wf = os.getenv("GITHUB_WORKFLOW", "").strip()
    run_id = os.getenv("GITHUB_RUN_ID", "").strip()
    ref = os.getenv("GITHUB_REF_NAME", "").strip()

    header = f"*{title}*"
    ctx = " | ".join([x for x in [wf, ref, (f"run {run_id}" if run_id else "")] if x])
    if ctx:
        header += f"\n_{ctx}_"

    body_parts = []
    for a in alerts:
        body_parts.append(f"\n*{a.combo}* — {a.hits} hit(s)\n{a.details}")

    msg = header + "\n" + "\n".join(body_parts)
    send_slack(msg)
