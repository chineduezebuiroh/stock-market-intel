# Stock EOD family production cutover (PR-B1)

## Authoritative route

The guarded `stocks_eod` profile invokes `jobs/run_stock_eod_family.py`. It first
acquires the bounded SPY/QQQ D/W/M reference history required by the configured
volume-ratio indicator. For each routed symbol the runner then independently
acquires native Yahoo daily, weekly, and monthly observations exactly once,
reconciles only from those already-acquired observations, and sends eligible
finalized frames through the same quality-aware rolling merge, indicators, and
persistence path used by standalone ingestion. The run-scoped reference context
prevents indicators from falling back to provider access during this processing.

After all three canonical snapshots are written, control returns to the existing
profile sequence: weekly and daily ETF trends, both D/W/M combos, health checks,
notifications, and finally execution-registry success.

Generic `jobs/run_timeframe.py` commands are unchanged. In particular, standalone
daily, weekly, and monthly runs and the generic `stocks daily --cascade` retain
their previous behavior. Stock 4H/130m, futures, and Q/Y paths are outside the
family cutover.

## Startup and request invariants

Daily, weekly, and monthly production universes are resolved independently with
the existing timeframe-universe function. Acquisition begins only if all three
sets are exactly equal. A mismatch fails closed with counts and representative
daily-relative differences; the runner never infers a union or intersection.

Production window lengths come from `config/timeframes.yaml` and are passed to
the family acquisition API. Reference history retains the indicator's existing
300-bar loader contract. For `N` routed symbols, the runner requires:

* primary family acquisition attempts: exactly `3N`;
* reference-data attempts: exactly `SPY + QQQ` × `D/W/M` = `6`;
* total provider attempts: exactly `3N + 6`;
* reconciliation provider requests: exactly `0`.
* downstream provider requests after explicit acquisition: exactly `0`.

An invariant violation fails the family subprocess before snapshot publication.

## Partial families

Failures remain symbol-local. Each valid or repaired interval proceeds through
the normal downstream processing path. An invalid or unavailable interval is
rejected, its prior rolling file is not overwritten, and the symbol is omitted
from the newly generated snapshot for that interval. Existing D/W/M combo inner
joins consequently exclude symbols missing any required snapshot role.

The job fails if any required D/W/M snapshot has no valid rows, if request
accounting fails, or if the normal processing/snapshot pipeline raises. It does
not require every routed symbol to have a complete family.

## Telemetry

Each run writes a non-authoritative health artifact at:

```text
data/_health/stocks_eod_family/<family_run_id>/manifest.parquet
```

It contains PR-A acquisition, provenance, native/finalized hashes, quality,
reconciliation, evidence, request, and timing fields, explicit reference-data
rows, plus B1 finalization and processing status/reason fields and a run-summary
row separating primary, reference, reconciliation, downstream, and total request
counts. This artifact is audit telemetry, not an atomic publication manifest or
a committed-family pointer.

## Persistence limitation and rollback

PR-B1 intentionally retains sequential writes to per-symbol rolling files and
the canonical daily, weekly, and monthly snapshot paths. A process or storage
failure between snapshot writes can leave mixed-generation canonical snapshots
until the next successful run. Versioned snapshots and atomic family commits are
deferred to PR-B2. Combos remain sequenced after successful completion of the
family subprocess, limiting the normal in-profile exposure.

The rollback seam is the first command in
`.github/scripts/run_stocks_eod_guarded.py`: restore
`jobs/run_timeframe.py stocks daily --cascade` in place of the family runner. A
revert of PR-B1 restores the complete prior route.
