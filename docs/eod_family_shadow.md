# Stock EOD family shadow runner (PR A)

This diagnostic implements the non-authoritative PR-A boundary described in
`eod_reconciliation_family_handoff.md`. It sequentially acquires one native
Yahoo observation for each of `1d`, `1wk`, and `1mo`, reconciles the daily Close
from those already acquired frames, and optionally runs the existing indicator
engine. Reconciliation cannot fetch provider data.

## Safety boundary

All output is written below:

```text
data/_shadow/eod_family/<family_run_id>/
  manifest.parquet
  indicator_terminals.parquet  # when at least one finalized interval is valid
```

The runner does not import or call execution-registry, combo, notification, or
canonical snapshot publication functions. It does not update rolling bars. The
manifest is evidence, not the future authoritative family commit/pointer.

## Progressive validation commands

Start with a tiny set:

```bash
python jobs/run_stock_eod_family_shadow.py \
  --symbols AAPL MSFT \
  --run-id tiny-$(date -u +%Y%m%dT%H%M%SZ)
```

Run the established live-pattern set:

```bash
python jobs/run_stock_eod_family_shadow.py \
  --live-pattern \
  --run-id seven-$(date -u +%Y%m%dT%H%M%SZ)
```

After reviewing those results, run the shortlist or configured broad universe:

```bash
python jobs/run_stock_eod_family_shadow.py --universe shortlist_stocks
python jobs/run_stock_eod_family_shadow.py --universe options_eligible
```

Use `--no-indicators` to capture acquisition/reconciliation lineage only.

## Manifest contract

There are three deterministic interval rows per symbol plus one run-summary row.
Interval rows contain acquisition state/error, UTC fetch timestamps, terminal
session evidence, a compact provider-metadata JSON value, native/finalized
terminal OHLC/Adj Close/Volume, native/finalized hashes, quality and Close-null
states, reconciliation eligibility/decision/tier/source/value/reason, request
counts, and per-family timing gaps. The run summary records the first fetch
start, last fetch completion, wall duration, and aggregate application-level
request counts.

For a complete family, application request accounting is exactly three attempts:
one each for D, W, and M. Failed intervals remain explicit and do not erase
siblings. `reconciliation_provider_requests` is always zero because the pure
family finalizer accepts no loader or provider dependency.

Indicator output is deliberately limited to finalized terminal rows. It enables
column-level comparisons using the existing indicator implementation without
publishing snapshot-, combo-, score-, or signal-equivalent artifacts. Full
authoritative snapshot/combo lineage and scoring parity remain part of the
future cutover work; no second scoring implementation is introduced here.
