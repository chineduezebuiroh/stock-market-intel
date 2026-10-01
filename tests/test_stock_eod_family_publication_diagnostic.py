from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd
import pytest

from core import storage
from core import stock_eod_family as family
from diagnostics import validate_stock_eod_family_publication as diagnostic


PREVIOUS = "prod-20260927T045518Z-da16331f"
CURRENT = "prod-20260929T035216Z-1ecf0088"


def _frame(*symbols: str) -> pd.DataFrame:
    return pd.DataFrame({"symbol": symbols, "close": range(1, len(symbols) + 1)})


def _publish(root: Path, run_id: str, state: family.StockEODPointerState):
    return family.publish_stock_eod_family(
        run_id,
        _frame("AAPL", "MSFT"),
        _frame("AAPL"),
        _frame("AAPL", "MSFT", "NVDA"),
        expected_pointer_revision=state.revision,
        expected_previous_family_run_id=state.previous_family_run_id,
        data_root=root,
    )


@pytest.fixture(autouse=True)
def local_backend(monkeypatch):
    monkeypatch.setattr(storage, "_DATA_BACKEND", "local")


@pytest.fixture
def publication(tmp_path: Path) -> Path:
    _publish(tmp_path, PREVIOUS, family.StockEODPointerState(None, None))
    state = family.capture_current_pointer_state(tmp_path)
    current = _publish(tmp_path, CURRENT, state)
    combo = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "stock_eod_family_run_id": [CURRENT],
            "stock_eod_family_committed_at_utc": [current.pointer.committed_at_utc],
        }
    )
    for name in diagnostic.COMBOS:
        combo.to_parquet(tmp_path / f"combo_{name}.parquet")
    for timeframe, filename in diagnostic.LEGACY_MIRRORS.items():
        _frame("AAPL").to_parquet(tmp_path / filename)
    return tmp_path


def _rewrite_current_manifest(root: Path, mutate) -> None:
    path = family.manifest_path(CURRENT, root)
    value = json.loads(path.read_bytes())
    mutate(value)
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(data)
    pointer_path = family.current_pointer_path(root)
    pointer = json.loads(pointer_path.read_bytes())
    pointer["manifest_sha256"] = sha256(data).hexdigest()
    pointer_path.write_bytes(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )


def _validate(root: Path):
    return diagnostic.validate_publication_contract(
        data_root=root,
        expected_family_run_id=CURRENT,
        expected_previous_family_run_id=PREVIOUS,
    )


def test_happy_path_complete_contract(publication, capsys):
    report = _validate(publication)
    assert report.family_run_id == CURRENT
    assert report.previous_family_run_id == PREVIOUS
    assert "[PASS] publication contract validated read-only" in capsys.readouterr().out


def test_pointer_to_manifest_family_mismatch(publication):
    _rewrite_current_manifest(
        publication,
        lambda value: value.update(family_run_id="prod-20260929T035216Z-aaaaaaaa"),
    )
    with pytest.raises(diagnostic.DiagnosticFailure, match="manifest schema/family"):
        _validate(publication)


def test_manifest_byte_hash_mismatch(publication):
    family.manifest_path(CURRENT, publication).write_bytes(b"{}")
    with pytest.raises(diagnostic.DiagnosticFailure, match="manifest byte hash"):
        _validate(publication)


def test_immutable_artifact_hash_mismatch(publication):
    path = family.artifact_path(CURRENT, "daily", publication)
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(data)
    with pytest.raises(diagnostic.DiagnosticFailure, match="daily artifact byte hash"):
        _validate(publication)


def test_artifact_row_count_mismatch(publication):
    _rewrite_current_manifest(
        publication,
        lambda value: value["artifacts"]["daily"].update(row_count=999),
    )
    with pytest.raises(diagnostic.DiagnosticFailure, match="daily artifact row_count"):
        _validate(publication)


def test_artifact_symbol_count_mismatch(publication):
    _rewrite_current_manifest(
        publication,
        lambda value: value["artifacts"]["daily"].update(symbol_count=999),
    )
    with pytest.raises(diagnostic.DiagnosticFailure, match="daily artifact symbol_count"):
        _validate(publication)


def test_missing_immutable_artifact(publication):
    family.artifact_path(CURRENT, "weekly", publication).unlink()
    with pytest.raises(diagnostic.DiagnosticFailure, match="weekly immutable artifact exists"):
        _validate(publication)


def _rewrite_combo(root: Path, mutate) -> None:
    path = root / "combo_stocks_c_dwm_shortlist.parquet"
    frame = pd.read_parquet(path)
    mutate(frame)
    frame.to_parquet(path)


def test_combo_missing_lineage_column(publication):
    _rewrite_combo(publication, lambda frame: frame.drop(columns=["stock_eod_family_run_id"], inplace=True))
    with pytest.raises(diagnostic.DiagnosticFailure, match="combo lineage column"):
        _validate(publication)


def test_combo_has_null_lineage(publication):
    _rewrite_combo(publication, lambda frame: frame.__setitem__("stock_eod_family_run_id", None))
    with pytest.raises(diagnostic.DiagnosticFailure, match="non-null"):
        _validate(publication)


def test_combo_contains_multiple_family_ids(publication):
    path = publication / "combo_stocks_c_dwm_shortlist.parquet"
    frame = pd.read_parquet(path)
    second = frame.copy()
    second["stock_eod_family_run_id"] = PREVIOUS
    pd.concat([frame, second], ignore_index=True).to_parquet(path)
    with pytest.raises(diagnostic.DiagnosticFailure, match="single family lineage"):
        _validate(publication)


def test_combo_family_differs_from_current_pointer(publication):
    _rewrite_combo(publication, lambda frame: frame.__setitem__("stock_eod_family_run_id", PREVIOUS))
    with pytest.raises(diagnostic.DiagnosticFailure, match="current family lineage"):
        _validate(publication)


def test_expected_family_assertion_mismatch(publication):
    with pytest.raises(diagnostic.DiagnosticFailure, match="expected current family"):
        diagnostic.validate_publication_contract(
            data_root=publication,
            expected_family_run_id="prod-20260929T035216Z-aaaaaaaa",
        )


def test_expected_previous_family_assertion_mismatch(publication):
    with pytest.raises(diagnostic.DiagnosticFailure, match="expected previous family"):
        diagnostic.validate_publication_contract(
            data_root=publication,
            expected_previous_family_run_id="prod-20260927T045518Z-aaaaaaaa",
        )


def test_previous_generation_missing(publication):
    family.manifest_path(PREVIOUS, publication).unlink()
    with pytest.raises(diagnostic.DiagnosticFailure, match="previous manifest exists"):
        _validate(publication)


def test_legacy_mirror_missing(publication):
    (publication / diagnostic.LEGACY_MIRRORS["monthly"]).unlink()
    with pytest.raises(diagnostic.DiagnosticFailure, match="legacy monthly mirror exists"):
        _validate(publication)


def test_read_only_behavior_no_write_apis_invoked(publication, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("diagnostic invoked a storage write API")

    monkeypatch.setattr(storage, "save_parquet", forbidden)
    monkeypatch.setattr(storage, "compare_and_swap_bytes", forbidden)
    monkeypatch.setattr(storage, "create_bytes_if_absent", forbidden)
    _validate(publication)
