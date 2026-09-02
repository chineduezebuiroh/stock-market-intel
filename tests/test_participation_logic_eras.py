import pandas as pd

from diagnostics.analyze_stock_options_participation import (
    DERIVED_BOOL_FIELDS,
    FIVE_COMPONENT_ERA,
    KNOWN_CASES,
    PRE_PARTICIPATION_ERA,
    classify_modern_scores,
    run_known_case_validation,
    score_pattern_summary,
)


def scored_row(*, participation, long_delta=0, short_delta=0):
    return {
        "reconstructed_long_score": 5,
        "mtf_long_score": 5 - long_delta,
        "reconstructed_short_score": 3,
        "mtf_short_score": 3 - short_delta,
        "participation_pass": participation,
        "either_score_mismatch": bool(long_delta or short_delta),
    }


def test_pre_participation_pattern_allows_exact_nonparticipation_rows():
    rows = pd.DataFrame(
        [
            scored_row(participation=True, long_delta=1, short_delta=1),
            scored_row(participation=False),
        ]
    )

    assert classify_modern_scores(rows) == PRE_PARTICIPATION_ERA
    summary = score_pattern_summary(rows)
    assert summary["nonparticipation_mismatch_count"] == 0
    assert summary["participation_true_rows"] == 1
    assert summary["participation_false_rows"] == 1


def test_unexpected_delta_remains_quarantined():
    rows = pd.DataFrame([scored_row(participation=True, long_delta=1, short_delta=2)])

    assert classify_modern_scores(rows) == "MODERN_QUARANTINED_SCORE_MISMATCH"


def test_exact_modern_scores_are_five_component_supported():
    rows = pd.DataFrame([scored_row(participation=True)])

    assert classify_modern_scores(rows) == FIVE_COMPONENT_ERA


def test_all_known_case_assertions_still_pass():
    records = []
    for fixture in KNOWN_CASES:
        row = {field: False for field in DERIVED_BOOL_FIELDS}
        row.update(fixture["expected"])
        row.update(
            {field: expected for field, (expected, _) in fixture["approx"].items()}
        )
        row.update(
            {
                "symbol": fixture["symbol"],
                "lower_date": pd.Timestamp(fixture["lower_date"]),
                "source_s3_key": f"fixture/{fixture['case_id']}",
                "artifact_execution_utc": pd.Timestamp("2026-09-01", tz="UTC"),
                "reconstructed_long_score": fixture["expected"][
                    "reconstructed_long_score"
                ],
                "mtf_long_score": fixture["expected"]["reconstructed_long_score"],
                "reconstructed_short_score": 0,
                "mtf_short_score": 0,
                "long_score_match": True,
                "short_score_match": True,
            }
        )
        records.append(row)
    supported = pd.DataFrame(records)

    validation, failures = run_known_case_validation(supported, supported, None, None)

    assert failures == []
    assert validation["assertions_pass"].all()
