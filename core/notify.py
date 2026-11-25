from __future__ import annotations

# core/notify.py

import json
import hashlib
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import requests

from core.paths import DATA
from core import storage
import smtplib
from email.message import EmailMessage

SIGNALS_OF_INTEREST = {"long", "short", "watch", "anti"}


class TelegramDeliveryError(RuntimeError):
    """A Telegram POST did not produce a confirmed successful response."""

    def __init__(self, category: str, status: int | None = None):
        self.category = category
        self.status = status
        self.outcome = "HTTP_REJECTED" if status is not None and 400 <= status < 500 else "DELIVERY_UNCONFIRMED"
        self.digest = None
        self.attempt_id = None
        self.revision = None
        super().__init__(f"Telegram delivery unconfirmed: {category} status={status}")


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


def telegram_chat_id_for_route(route: str | None = None) -> str:
    if not route or route == "default":
        return _env("TELEGRAM_CHAT_ID")

    key = f"TELEGRAM_CHAT_ID_{route.upper()}"
    return os.getenv(key, "").strip() or _env("TELEGRAM_CHAT_ID")


def send_telegram_message(text: str, *, chat_id: str | None = None,
                          parse_mode: str | None = "Markdown") -> None:
    token = _env("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or _env("TELEGRAM_CHAT_ID")

    # Telegram has message size limits; keep it safe.
    # If you ever hit limits, we can chunk.
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode is not None:
        payload["parse_mode"] = parse_mode
    try:
        r = requests.post(url, data=payload, timeout=20)
    except requests.exceptions.RequestException as exc:
        raise TelegramDeliveryError(type(exc).__name__) from None
    if not r.ok:
        raise TelegramDeliveryError("HTTPError", r.status_code)


def report_delivery_warning(job: str, kind: str, category: str,
                            status: int | None, processing: str,
                            outcome: str = "DELIVERY_UNCONFIRMED", *,
                            digest: str | None = None, attempt_id: str | None = None,
                            revision: str | None = None) -> None:
    record = {"job": job, "notification": kind, "category": category,
              "http_status": status, "outcome": outcome, "data_health": processing,
              "identity_digest": digest, "attempt_id": attempt_id, "revision": revision}
    line = json.dumps(record, sort_keys=True)
    print(f"[NOTIFICATION_WARNING] {line}")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a", encoding="utf-8") as stream:
                stream.write(f"\nNotification warning: `{line}`\n")
        except OSError as exc:
            print(f"[NOTIFICATION_DIAGNOSTIC_WARNING] summary_write_failed={type(exc).__name__}")


def _attempt_path(job: str, kind: str, identity: str) -> Path:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return _attempt_path_for_digest(job, kind, digest)


def _attempt_path_for_digest(job: str, kind: str, digest: str) -> Path:
    allowed = {("stocks_intraday_4h", "repair_telemetry"),
               ("stocks_intraday_4h", "combo_signal"),
               ("futures_intraday_4h", "combo_signal")}
    if (job, kind) not in allowed or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid job, notification type, or identity digest")
    return DATA / "_alerts" / "delivery_attempts_4h" / job / kind / f"{digest}.json"


def _attempt_read(path: Path) -> tuple[dict | None, str | None]:
    result = storage.read_bytes_with_revision(path)
    if result.data is None:
        return None, None
    record = json.loads(result.data)
    if (not isinstance(record, dict) or record.get("schema") != 1
            or not all(record.get(k) for k in ("job", "notification_type", "identity_digest",
                                                "attempt_id", "created_at", "updated_at", "state"))):
        raise ValueError("Invalid delivery attempt record")
    if _attempt_path_for_digest(record["job"], record["notification_type"], record["identity_digest"]) != path:
        raise ValueError("Delivery attempt identity mismatch")
    return record, result.revision


def _attempt_write(path: Path, record: dict, revision: str | None) -> str:
    return storage.compare_and_swap_bytes(path, revision,
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode(),
        content_type="application/json")


def _transition_attempt(path: Path, record: dict, revision: str, state: str, **extra) -> str:
    updated = dict(record, state=state, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    new_revision = _attempt_write(path, updated, revision)
    record.clear()
    record.update(updated)
    return new_revision


def _confirm_4h_attempt(job: str, kind: str, identity: str, attempt_id: str) -> None:
    path = _attempt_path(job, kind, identity)
    try:
        record, revision = _attempt_read(path)
        if not record or record.get("attempt_id") != attempt_id or record.get("state") != "POST_STARTED":
            raise ValueError("Attempt changed before confirmation")
        _transition_attempt(path, record, revision, "CONFIRMED")
    except Exception as exc:
        report_delivery_warning(job, kind, "AttemptStore" + type(exc).__name__, None,
                                "PENDING", "DELIVERY_CONFIRMED_RECORD_UNCERTAIN",
                                digest=hashlib.sha256(identity.encode()).hexdigest(), attempt_id=attempt_id)


def send_4h_tracked(text: str, *, job: str, kind: str, identity: str,
                    chat_id: str | None = None, defer_confirmation: bool = False) -> str | None:
    """Write an intent and POST_STARTED using CAS; never automatically retry unresolved sends."""
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    path = _attempt_path_for_digest(job, kind, digest)
    try:
        previous, revision = _attempt_read(path)
        if previous and previous.get("state") != "CONFIRMED":
            report_delivery_warning(job, kind, "UnresolvedPreviousAttempt", previous.get("http_status"),
                                    "PENDING", previous.get("state", "DELIVERY_UNCONFIRMED"),
                                    digest=digest, attempt_id=previous.get("attempt_id"), revision=revision)
            return None
        now = datetime.now(timezone.utc).isoformat()
        record = {"schema": 1, "job": job, "notification_type": kind, "identity_digest": digest,
                  "attempt_id": uuid.uuid4().hex, "created_at": now, "updated_at": now,
                  "state": "INTENT_RECORDED", "category": None, "http_status": None}
        revision = _attempt_write(path, record, revision)
        report_delivery_warning(job, kind, "AttemptIntent", None, "PENDING", "INTENT_RECORDED",
                                digest=digest, attempt_id=record["attempt_id"], revision=revision)
        # A crash between this transition and requests.post remains ambiguous.
        revision = _transition_attempt(path, record, revision, "POST_STARTED")
    except Exception as exc:
        report_delivery_warning(job, kind, "AttemptStore" + type(exc).__name__, None,
                                "PENDING", "NOT_ATTEMPTED", digest=digest)
        return None
    try:
        send_telegram_message(text, chat_id=chat_id)
    except TelegramDeliveryError as exc:
        exc.digest, exc.attempt_id, exc.revision = digest, record["attempt_id"], revision
        try:
            exc.revision = _transition_attempt(path, record, revision, exc.outcome,
                                               category=exc.category, http_status=exc.status)
        except Exception as store_exc:
            report_delivery_warning(job, kind, "AttemptStore" + type(store_exc).__name__, None,
                                    "PENDING", "ATTEMPT_UPDATE_FAILED", digest=digest,
                                    attempt_id=record["attempt_id"], revision=revision)
        raise
    if not defer_confirmation:
        _confirm_4h_attempt(job, kind, identity, record["attempt_id"])
    return record["attempt_id"]


def send_email_message(
    *,
    subject: str,
    body: str,
    to_email: str | None = None,
    attachments: list[tuple[str, bytes, str]] | None = None,
) -> None:
    """
    Send an email via SMTP.

    Env vars required:
      - EMAIL_SMTP_HOST
      - EMAIL_SMTP_PORT   (optional, default 465)
      - EMAIL_USERNAME
      - EMAIL_PASSWORD
      - EMAIL_FROM
      - EMAIL_TO          (optional if to_email passed explicitly)

    attachments:
      list of tuples: (filename, content_bytes, mime_type)
      e.g. ("sample.csv", csv_bytes, "text/csv")
    """
    smtp_host = _env("EMAIL_SMTP_HOST")
    smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "465"))
    username = _env("EMAIL_USERNAME")
    password = _env("EMAIL_PASSWORD")
    from_email = _env("EMAIL_FROM")
    recipient = to_email or _env("EMAIL_TO")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = recipient
    msg.set_content(body)

    for attachment in attachments or []:
        filename, content_bytes, mime_type = attachment
        maintype, subtype = mime_type.split("/", 1)
        msg.add_attachment(
            content_bytes,
            maintype=maintype,
            subtype=subtype,
            filename=filename,
        )

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(username, password)
        server.send_message(msg)


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
    max_lines_per_bucket: int = 75,
) -> str:
    """
    Compact, scan-friendly message.
    """
    if not rows:
        return ""

    # bucket by signal
    buckets: dict[str, list[SignalRow]] = {"long": [], "short": [], "watch": [], "anti": []}
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

    for sig in ("long", "short", "watch", "anti"):
        rs = sorted(buckets[sig], key=lambda x: x.symbol.upper())
        if not rs:
            continue

        emoji = {"long": "🟢", "short": "🔴", "watch": "🟡", "anti": "🔘"}[sig]
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
    route: str = "default",
    tracked_4h_job: str | None = None,
) -> None:
    """
    Reads data/combo_<combo_name>.parquet and sends a Telegram message
    if any symbols have signal in {long,short,watch,anti}.

    If only_if_changed=True, suppress if the signal set is unchanged vs last alert.
    """
    combo_path = DATA / f"combo_{combo_name}.parquet"
    if not storage.exists(combo_path):
        if tracked_4h_job:
            raise FileNotFoundError(f"Missing combo parquet: {combo_name}")
        print(f"[NOTIFY][SKIP] missing combo parquet: {combo_path}")
        return

    df = storage.load_parquet(combo_path)
    if df is None or df.empty:
        if tracked_4h_job:
            raise ValueError(f"Empty combo: {combo_name}")
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
    if tracked_4h_job:
        attempt_id = send_4h_tracked(msg, job=tracked_4h_job, kind="combo_signal",
                               identity=fp, chat_id=telegram_chat_id_for_route(route),
                               defer_confirmation=True)
        if not attempt_id:
            return
    else:
        send_telegram_message(msg, chat_id=telegram_chat_id_for_route(route))

    # record last fingerprint (for change suppression)
    save_alert_fingerprint(key, fp, meta={"combo": combo_name, "asof": asof})
    if tracked_4h_job:
        _confirm_4h_attempt(tracked_4h_job, "combo_signal", fp, attempt_id)
    print(f"[NOTIFY] sent alert for {combo_name} (signals={len(rows)})")



def notify_stock_eod_failure(reason: str, coverage: dict | None = None,
                             health_details: Iterable[str] = ()) -> None:
    """One explicit failure attempt, independent of signal presence/health success."""
    lines = ["Stock EOD degraded/failure"]
    if coverage is not None:
        lines.append("Family: " + coverage["family_run_id"])
        for tf, label in (("daily", "D"), ("weekly", "W"), ("monthly", "M")):
            lines.append(f"{label}: {coverage['final_counts'][tf]} / "
                         f"{coverage['expected_count']} ({coverage['coverage'][tf]:.2%})")
        failing = ", ".join(coverage["failing_timeframes"]) or "none"
        lines.append(f"Catastrophic guardrail: {'PASS' if coverage['passed'] else 'FAIL'}; "
                     f"floor={coverage['threshold_percent']}%; failing={failing}")
        missing = coverage.get("shortlist_missing")
        lines.append("Shortlist missing from final DWM support: " +
                     ("unknown" if missing is None else ", ".join(missing) or "none"))
        lines.append("Publication/mirror/product stages may already have written usable results; see run logs.")
    else:
        lines.append("Family/coverage unavailable for this invocation; see run logs.")
    lines.extend(health_details)
    lines.append("Failure: " + reason)
    text = "\n".join(lines)
    print("[STOCK_EOD_FAILURE] " + text, flush=True)
    if len(text) > 3900:
        text = text[:3800] + "\n[truncated; full details in run logs]"
    send_telegram_message(text, chat_id=telegram_chat_id_for_route("stocks_eod"), parse_mode=None)
