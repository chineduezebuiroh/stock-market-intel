"""Engine regime ledger loading and deterministic observation assignment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LEDGER_PATH = Path(__file__).with_name("engine_regimes.json")


@dataclass(frozen=True)
class EngineRegime:
    engine_regime_id: str
    effective_from: datetime | None
    effective_to: datetime | None
    affected_combo: str
    affected_component: str
    change_type: str
    old_definition: str | None
    new_definition: str
    commit_sha: str | None
    adr_or_reference: str
    evidence_strength: str
    notes: str
    observed_logic_era: str | None = None


def load_regimes(path: Path = LEDGER_PATH) -> tuple[EngineRegime, ...]:
    def instant(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None

    records = json.loads(path.read_text())
    return tuple(
        EngineRegime(
            **{
                **record,
                "effective_from": instant(record.get("effective_from")),
                "effective_to": instant(record.get("effective_to")),
            }
        )
        for record in records
    )


def assign_engine_regime(
    combo: str,
    execution_timestamp: datetime,
    *,
    observed_logic_era: str | None = None,
    regimes: tuple[EngineRegime, ...] | None = None,
) -> str:
    """Assign exactly one regime; schema-era evidence disambiguates legacy rows."""
    if execution_timestamp.tzinfo is None:
        execution_timestamp = execution_timestamp.replace(tzinfo=timezone.utc)
    candidates = []
    for regime in regimes or load_regimes():
        if regime.affected_combo not in ("*", combo):
            continue
        if observed_logic_era is None and regime.observed_logic_era is not None:
            continue
        if (
            observed_logic_era is not None
            and regime.observed_logic_era != observed_logic_era
        ):
            continue
        if regime.effective_from and execution_timestamp < regime.effective_from:
            continue
        if regime.effective_to and execution_timestamp >= regime.effective_to:
            continue
        candidates.append(regime)
    if len(candidates) != 1:
        raise ValueError(
            f"expected one engine regime for {combo} at {execution_timestamp}; "
            f"found {[item.engine_regime_id for item in candidates]}"
        )
    return candidates[0].engine_regime_id
