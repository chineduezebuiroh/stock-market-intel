"""Read-only persisted-evidence audit of strict R3 monthly repairability."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from core import storage
from core.paths import DATA
from core.stock_eod_family import artifact_path
from etl.eod_reconciliation import (
    TerminalObservation,
    _metadata_price_tolerance,
    canonical_terminal_session_close,
)

FAMILY_RUN_ID = "prod-20261002T003510Z-85b4ac62"
PREVIOUS_FAMILY_RUN_ID = "prod-20261001T002500Z-c924d845"
TARGET_SESSION = date(2026, 10, 1)
EXPECTED_AFFECTED = 2355
SURVIVORS = frozenset({"ASA", "DBRG", "DSL", "GBTG", "KYN", "LBRDA", "MBWM", "MSDL", "UVSP"})
MAX_SAMPLE = 20
UNKNOWN = "UNKNOWN_BECAUSE_EVIDENCE_NOT_PERSISTED"

REQUIRED = {
    "record_type", "symbol", "interval", "acquisition_status", "quality_status",
    "quality_reason", "close_is_null", "raw_open", "raw_high", "raw_low",
    "raw_close", "raw_adj_close", "raw_volume", "terminal_session",
    "session_evidence", "adjustment_mode", "session_semantics",
    "provider_metadata_json", "finalization_status", "processing_status",
    "processing_reason",
}


class DiagnosticFailure(RuntimeError):
    """Persisted evidence is missing or inconsistent with the incident contract."""


@dataclass(frozen=True)
class EvidenceResult:
    detail: pd.DataFrame
    summary: dict[str, int]
    primary_reasons: pd.Series


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _date_value(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("Time") or value.get("time")
    try:
        stamp = pd.Timestamp(value, unit="s", tz="UTC") if isinstance(value, (int, float)) else pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York")
    return stamp.date()


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED - set(manifest.columns))
    if missing:
        raise DiagnosticFailure(f"manifest missing required columns: {missing}")
    intervals = manifest.loc[manifest["record_type"].eq("interval")].copy()
    duplicates = intervals.duplicated(["symbol", "interval"], keep=False)
    if duplicates.any():
        raise DiagnosticFailure("manifest has duplicate symbol/interval rows")
    return intervals


def select_affected_monthly(intervals: pd.DataFrame) -> pd.DataFrame:
    monthly = intervals.loc[intervals["interval"].eq("1mo")]
    numeric = monthly[["raw_open", "raw_high", "raw_low", "raw_volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    mask = (
        monthly["acquisition_status"].eq("success")
        & monthly["quality_status"].eq("invalid")
        & monthly["quality_reason"].eq("invalid terminal required fields: close")
        & monthly["close_is_null"].eq(True)
        & monthly["raw_close"].isna()
        & monthly["raw_adj_close"].isna()
        & numeric.apply(lambda column: column.map(math.isfinite)).all(axis=1)
        & monthly["finalization_status"].eq("invalid")
        & monthly["processing_status"].eq("excluded")
        & monthly["processing_reason"].eq("finalized observation is not eligible")
    )
    affected = monthly.loc[mask].copy()
    if len(affected) != EXPECTED_AFFECTED:
        raise DiagnosticFailure(
            f"affected monthly population mismatch: expected={EXPECTED_AFFECTED} observed={len(affected)}"
        )
    return affected


def _eligibility(row: pd.Series, interval: str) -> tuple[str, str]:
    failures = []
    unknown = []
    if row is None or row.empty:
        return "INELIGIBLE", "missing interval observation"
    checks = {
        "acquisition not successful": row.get("acquisition_status") != "success",
        "wrong interval": row.get("interval") != interval,
        "raw Close not finite": _finite(row.get("raw_close")) is None,
        "structural quality invalid": row.get("quality_status") != "valid",
        "non-raw adjustment mode": row.get("adjustment_mode") != "raw_unadjusted",
        "non-regular session": row.get("session_semantics") != "regular",
        "wrong terminal session": str(row.get("terminal_session")) != TARGET_SESSION.isoformat(),
        "missing session evidence": not isinstance(row.get("session_evidence"), str)
        or not row.get("session_evidence"),
    }
    if interval == "1wk" and row.get("session_evidence") != "yfinance_interval_merge_lastTrade":
        checks["weekly evidence is not price-bound lastTrade"] = True
    failures.extend(reason for reason, failed in checks.items() if failed)
    # revision_ambiguous is not persisted, but current observations construct it
    # with the explicit default False. Record that limitation without converting
    # every otherwise-eligible row to UNKNOWN.
    if "revision_ambiguous" in row.index and pd.isna(row.get("revision_ambiguous")):
        unknown.append("revision ambiguity unavailable")
    elif "revision_ambiguous" in row.index and bool(row.get("revision_ambiguous")):
        failures.append("revision ambiguity")
    if failures:
        return "INELIGIBLE", "; ".join(failures)
    if unknown:
        return UNKNOWN, "; ".join(unknown)
    return "ELIGIBLE", "all persisted current-contract gates pass; revision ambiguity defaults false in code"


def evaluate_eligibility(intervals: pd.DataFrame, affected_symbols: set[str], interval: str) -> pd.DataFrame:
    rows = intervals.loc[
        intervals["interval"].eq(interval) & intervals["symbol"].isin(affected_symbols)
    ].set_index("symbol")
    output = []
    for symbol in sorted(affected_symbols):
        row = rows.loc[symbol] if symbol in rows.index else pd.Series(dtype=object)
        state, reason = _eligibility(row, interval)
        output.append({"symbol": symbol, "state": state, "reason": reason})
    return pd.DataFrame(output).set_index("symbol")


def _observation(row: pd.Series, interval: str) -> TerminalObservation:
    metadata = parse_metadata(row.get("provider_metadata_json"))
    return TerminalObservation(
        source_interval=interval,
        interval_label="not_persisted",
        terminal_market_session_date=_date_value(row.get("terminal_session")),
        session_evidence=row.get("session_evidence"),
        close=_finite(row.get("raw_close")),
        adjustment_mode=str(row.get("adjustment_mode")),
        session_semantics=str(row.get("session_semantics")),
        price_tolerance=_metadata_price_tolerance(metadata),
    )


def analyze(intervals: pd.DataFrame, affected: pd.DataFrame) -> EvidenceResult:
    symbols = set(affected["symbol"])
    by_key = intervals.set_index(["symbol", "interval"])
    d_state = evaluate_eligibility(intervals, symbols, "1d")
    w_state = evaluate_eligibility(intervals, symbols, "1wk")
    records = []
    for m in affected.set_index("symbol").itertuples():
        symbol = m.Index
        record: dict[str, Any] = {
            "symbol": symbol,
            "d_state": d_state.loc[symbol, "state"],
            "d_reason": d_state.loc[symbol, "reason"],
            "w_state": w_state.loc[symbol, "state"],
            "w_reason": w_state.loc[symbol, "reason"],
        }
        o, h, low, volume = map(_finite, (m.raw_open, m.raw_high, m.raw_low, m.raw_volume))
        record["m_substrate_valid"] = bool(
            None not in (o, h, low, volume) and volume >= 0 and low <= o <= h
        )
        record["m_interval_label"] = None  # not persisted in the health manifest
        session = _date_value(m.terminal_session)
        evidence = isinstance(m.session_evidence, str) and bool(m.session_evidence)
        if session == TARGET_SESSION and evidence:
            record["m_current_period"] = "PROVEN"
            record["m_current_through_target"] = "PROVEN"
        elif session is not None and session != TARGET_SESSION:
            record["m_current_period"] = "FALSE"
            record["m_current_through_target"] = "FALSE"
        else:
            record["m_current_period"] = "UNKNOWN"
            record["m_current_through_target"] = "UNKNOWN"

        record.update(dw_status="NOT_EVALUATED", dw_abs_diff=None, dw_rel_diff=None,
                      tolerance=None, candidate=None, m_range_pass=False)
        if record["d_state"] == record["w_state"] == "ELIGIBLE":
            d = by_key.loc[(symbol, "1d")]
            w = by_key.loc[(symbol, "1wk")]
            d_obs, w_obs = _observation(d, "1d"), _observation(w, "1wk")
            tolerance = max(d_obs.price_tolerance or 1e-4, w_obs.price_tolerance or 1e-4)
            result = canonical_terminal_session_close(
                [d_obs, w_obs], target_session=TARGET_SESSION, price_tolerance=tolerance
            )
            d_close, w_close = float(d_obs.close), float(w_obs.close)
            record["dw_abs_diff"] = abs(d_close - w_close)
            record["dw_rel_diff"] = abs(d_close - w_close) / max(abs(d_close), abs(w_close), 1e-300)
            record["tolerance"] = tolerance
            record["dw_status"] = {
                "canonical": "AGREE", "conflict": "CONFLICT",
            }.get(result.status, "UNCOMPARABLE")
            if result.status == "canonical":
                record["candidate"] = result.canonical_close
                record["m_range_pass"] = bool(low <= result.canonical_close <= h)
        records.append(record)

    detail = pd.DataFrame(records).set_index("symbol")
    detail["strict_r3_repairable"] = (
        detail["d_state"].eq("ELIGIBLE")
        & detail["w_state"].eq("ELIGIBLE")
        & detail["dw_status"].eq("AGREE")
        & detail["m_substrate_valid"]
        & detail["m_current_period"].eq("PROVEN")
        & detail["m_current_through_target"].eq("PROVEN")
        & detail["m_range_pass"]
    )
    reasons = detail.apply(_primary_reason, axis=1)
    summary = _summary(detail)
    return EvidenceResult(detail, summary, reasons)


def _primary_reason(row: pd.Series) -> str:
    if row.strict_r3_repairable:
        return "REPAIRABLE"
    if UNKNOWN in {row.d_state, row.w_state}:
        return "UNKNOWN_DW_EVIDENCE"
    if row.d_state != "ELIGIBLE":
        return "MISSING_OR_INELIGIBLE_D"
    if row.w_state != "ELIGIBLE":
        return "MISSING_OR_INELIGIBLE_W"
    if row.dw_status == "UNCOMPARABLE":
        return "DW_UNCOMPARABLE"
    if row.dw_status == "CONFLICT":
        return "DW_CONFLICT"
    if not row.m_substrate_valid:
        return "MALFORMED_M_SUBSTRATE_BEYOND_CLOSE"
    if row.m_current_period == "FALSE":
        return "STALE_M_PERIOD"
    if row.m_current_period == "UNKNOWN":
        return "M_CURRENT_PERIOD_UNKNOWN"
    if row.m_current_through_target != "PROVEN":
        return "M_CURRENTNESS_UNPROVEN"
    if not row.m_range_pass:
        return "CANDIDATE_OUTSIDE_M_RANGE"
    return "AMBIGUITY"


def _summary(d: pd.DataFrame) -> dict[str, int]:
    count = lambda mask: int(mask.sum())
    both = d.d_state.eq("ELIGIBLE") & d.w_state.eq("ELIGIBLE")
    return {
        "TOTAL_AFFECTED_M": len(d),
        "D_ELIGIBLE": count(d.d_state.eq("ELIGIBLE")),
        "D_INELIGIBLE": count(d.d_state.eq("INELIGIBLE")),
        "D_UNKNOWN": count(d.d_state.eq(UNKNOWN)),
        "W_ELIGIBLE": count(d.w_state.eq("ELIGIBLE")),
        "W_INELIGIBLE": count(d.w_state.eq("INELIGIBLE")),
        "W_UNKNOWN": count(d.w_state.eq(UNKNOWN)),
        "DW_BOTH_ELIGIBLE": count(both),
        "DW_AGREE": count(d.dw_status.eq("AGREE")),
        "DW_CONFLICT": count(d.dw_status.eq("CONFLICT")),
        "DW_UNCOMPARABLE": count(d.dw_status.eq("UNCOMPARABLE")),
        "M_CURRENT_PERIOD_PROVEN": count(d.m_current_period.eq("PROVEN")),
        "M_CURRENT_PERIOD_FALSE": count(d.m_current_period.eq("FALSE")),
        "M_CURRENT_PERIOD_UNKNOWN": count(d.m_current_period.eq("UNKNOWN")),
        "M_CURRENT_THROUGH_TARGET_PROVEN": count(d.m_current_through_target.eq("PROVEN")),
        "M_CURRENT_THROUGH_TARGET_FALSE": count(d.m_current_through_target.eq("FALSE")),
        "M_CURRENT_THROUGH_TARGET_UNKNOWN": count(d.m_current_through_target.eq("UNKNOWN")),
        "DW_AGREE_AND_M_RANGE_PASS": count(d.dw_status.eq("AGREE") & d.m_range_pass),
        "STRICT_R3_REPAIRABLE": count(d.strict_r3_repairable),
        "STRICT_R3_NOT_REPAIRABLE": count(~d.strict_r3_repairable),
    }


def metadata_analysis(affected: pd.DataFrame, result: EvidenceResult) -> None:
    rows = []
    for row in affected.itertuples():
        metadata = parse_metadata(row.provider_metadata_json)
        last = metadata.get("lastTrade") if isinstance(metadata.get("lastTrade"), dict) else {}
        rows.append({
            "symbol": row.symbol,
            "last_trade_present": bool(last),
            "last_trade_price": _finite(last.get("Price")),
            "last_trade_date": _date_value(last.get("Time")),
            "regular_market_price": _finite(metadata.get("regularMarketPrice")),
            "regular_market_date": _date_value(metadata.get("regularMarketTime")),
            "dataGranularity": metadata.get("dataGranularity"),
            "exchangeTimezoneName": metadata.get("exchangeTimezoneName"),
            "currentTradingPeriod_present": metadata.get("currentTradingPeriod") is not None,
            "metadata_error": metadata.get("metadata_error"),
        })
    meta = pd.DataFrame(rows).set_index("symbol")
    print("[PROVIDER_METADATA]")
    for name, mask in {
        "LAST_TRADE_PRESENT": meta.last_trade_present,
        "LAST_TRADE_PRICE_FINITE": meta.last_trade_price.notna(),
        "LAST_TRADE_DATE_TARGET": meta.last_trade_date.eq(TARGET_SESSION),
        "REGULAR_MARKET_PRICE_PRESENT": meta.regular_market_price.notna(),
        "REGULAR_MARKET_TIME_PRESENT": meta.regular_market_date.notna(),
        "REGULAR_MARKET_DATE_TARGET": meta.regular_market_date.eq(TARGET_SESSION),
    }.items():
        print(f"{name}={int(mask.sum())}")
    print("LAST_TRADE_DATE_DISTRIBUTION")
    print(meta.last_trade_date.astype("string").value_counts(dropna=False).sort_index().to_string())
    print("REGULAR_MARKET_DATE_DISTRIBUTION")
    print(meta.regular_market_date.astype("string").value_counts(dropna=False).sort_index().to_string())
    for field in (
        "dataGranularity", "exchangeTimezoneName", "currentTradingPeriod_present",
        "metadata_error",
    ):
        print(f"{field.upper()}_DISTRIBUTION")
        print(meta[field].value_counts(dropna=False).to_string())
    comparable = result.detail.candidate.notna()
    for field in ("last_trade_price", "regular_market_price"):
        joined = meta.loc[comparable, field].dropna()
        differences = (joined - result.detail.loc[joined.index, "candidate"]).abs()
        print(f"{field.upper()}_CONSENSUS_COMPARABLE={len(differences)}")
        if not differences.empty:
            tolerances = result.detail.loc[differences.index, "tolerance"]
            print(f"{field.upper()}_CONSENSUS_AGREE={int(differences.le(tolerances).sum())}")
            print(differences.describe().to_string())


def _snapshot_labels(frame: pd.DataFrame) -> dict[str, Any]:
    if "symbol" not in frame:
        raise DiagnosticFailure("affected monthly snapshot missing symbol column")
    labels = frame.index if isinstance(frame.index, pd.DatetimeIndex) else frame.get("date")
    if labels is None:
        raise DiagnosticFailure("affected monthly snapshot has no date index/column")
    return dict(zip(frame["symbol"].astype(str).str.upper(), pd.to_datetime(labels)))


def print_survivors(intervals: pd.DataFrame, snapshot: pd.DataFrame) -> None:
    labels = _snapshot_labels(snapshot)
    missing = sorted(SURVIVORS - set(labels))
    if missing:
        raise DiagnosticFailure(f"affected monthly snapshot missing survivors: {missing}")
    monthly = intervals.loc[
        intervals.interval.eq("1mo") & intervals.symbol.isin(SURVIVORS)
    ].set_index("symbol")
    print("[SURVIVORS]")
    survivor_labels = pd.Series({symbol: labels[symbol].date() for symbol in SURVIVORS})
    print("SURVIVOR_SNAPSHOT_DATE_DISTRIBUTION")
    print(survivor_labels.value_counts().sort_index().to_string())
    for symbol in sorted(SURVIVORS):
        row = monthly.loc[symbol]
        metadata = parse_metadata(row.provider_metadata_json)
        label = labels[symbol]
        belongs = label.year == 2026 and label.month == 10
        current = str(row.terminal_session) == TARGET_SESSION.isoformat() and bool(row.session_evidence)
        stale_conclusion = "NOT_STALE" if belongs else "D_EVIDENCE_INSUFFICIENT"
        values = {field: row[field] for field in (
            "raw_open", "raw_high", "raw_low", "raw_close", "raw_adj_close", "raw_volume",
            "terminal_session", "session_evidence", "quality_status", "finalization_status",
            "processing_status",
        )}
        print(json.dumps({
            "symbol": symbol,
            "snapshot_label": label.isoformat(),
            "native_m_interval_label": "NOT_PERSISTED_SEPARATELY",
            **values,
            "lastTrade": metadata.get("lastTrade"),
            "regularMarketPrice": metadata.get("regularMarketPrice"),
            "regularMarketTime": metadata.get("regularMarketTime"),
            "belongs_to_october_2026": belongs,
            "current_through_target_proven": current,
            "stale_provider_explanation": stale_conclusion,
        }, default=str, sort_keys=True))


def run_diagnostic(*, data_root: Path = DATA) -> EvidenceResult:
    manifest_path = data_root / "_health" / "stocks_eod_family" / FAMILY_RUN_ID / "manifest.parquet"
    intervals = validate_manifest(storage.load_parquet(manifest_path))
    affected = select_affected_monthly(intervals)
    result = analyze(intervals, affected)
    print("[SUMMARY]")
    for name, value in result.summary.items():
        print(f"{name}={value}")
    print("[MONTHLY_SUBSTRATE]")
    numeric = affected[["raw_open", "raw_high", "raw_low", "raw_volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    for field in numeric:
        print(f"{field.upper()}_UNAVAILABLE={int((~numeric[field].map(math.isfinite)).sum())}")
    print(f"RAW_CLOSE_NULL={int(affected.raw_close.isna().sum())}")
    print(f"RAW_ADJ_CLOSE_NULL={int(affected.raw_adj_close.isna().sum())}")
    print(f"OPEN_OUTSIDE_LOW_HIGH={int((~numeric.raw_open.between(numeric.raw_low, numeric.raw_high)).sum())}")
    print(f"HIGH_BELOW_LOW={int((numeric.raw_high < numeric.raw_low).sum())}")
    print(f"VOLUME_NEGATIVE={int((numeric.raw_volume < 0).sum())}")
    print("M_INTERVAL_LABEL_NOT_PERSISTED=" + str(len(affected)))
    print("REVISION_AMBIGUITY_FIELD_NOT_PERSISTED=" + str(len(affected)))
    print("[INDEPENDENT_GATE_FAILURE_COUNTS]")
    gates = {
        "D_NOT_ELIGIBLE": ~result.detail.d_state.eq("ELIGIBLE"),
        "W_NOT_ELIGIBLE": ~result.detail.w_state.eq("ELIGIBLE"),
        "DW_NOT_AGREEING": ~result.detail.dw_status.eq("AGREE"),
        "M_SUBSTRATE_INVALID": ~result.detail.m_substrate_valid,
        "M_PERIOD_NOT_PROVEN": ~result.detail.m_current_period.eq("PROVEN"),
        "M_CURRENTNESS_NOT_PROVEN": ~result.detail.m_current_through_target.eq("PROVEN"),
        "M_RANGE_NOT_PASSING": ~result.detail.m_range_pass,
    }
    for name, mask in gates.items():
        print(f"{name}={int(mask.sum())}")
    print("[PRIMARY_REASON_PARTITION]")
    print(result.primary_reasons.value_counts().sort_index().to_string())
    comparable = result.detail.loc[result.detail.dw_status.isin(["AGREE", "CONFLICT"])]
    print("[DW_DIFFERENCES]")
    print(comparable[["dw_abs_diff", "dw_rel_diff"]].describe().to_string())
    agreeing = comparable.loc[comparable.dw_status.eq("AGREE"), "dw_abs_diff"]
    print(f"MAX_AGREEING_ABS_DIFF={agreeing.max() if not agreeing.empty else 'unavailable'}")
    print("[DW_CONFLICT_SAMPLE]")
    print(comparable.loc[comparable.dw_status.eq("CONFLICT")].head(MAX_SAMPLE).to_string())
    metadata_analysis(affected, result)
    snapshot = storage.load_parquet(artifact_path(FAMILY_RUN_ID, "monthly", data_root))
    print_survivors(intervals, snapshot)
    print(
        "[PRE_NORMALIZATION_LIMITATION] persisted evidence cannot distinguish "
        "provider Close present-but-null from absent/unrecognized Close normalized to pd.NA"
    )
    return result


def main() -> int:
    try:
        run_diagnostic()
    except (DiagnosticFailure, FileNotFoundError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
