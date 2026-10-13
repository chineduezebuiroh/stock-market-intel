# Stock EOD timeframe coverage incident diagnostic

The manually dispatched **Stock EOD Timeframe Coverage Incident Diagnostic**
workflow reads the five fixed October incident generations and their health
telemetry. It does not read the current publication pointer, invoke production
jobs, acquire market data, or use storage write APIs. Reports are written under
`RUNNER_TEMP` and uploaded as a GitHub Actions artifact. Any missing object,
identity/integrity/count mismatch, or retained-log coverage mismatch fails the
workflow before report generation.

The workflow uses the same AWS secret bindings as the existing read-only
publication diagnostic. Repository permissions are `contents: read`. The
repository does not establish the underlying AWS credentials' IAM permissions;
read-only operation is enforced by the diagnostic's use of GetObject only, not
by assuming those credentials cannot write. No credential or IAM change is made.

A new `workflow_dispatch` workflow must be registered on the default branch
before GitHub permits dispatch. This implementation does not modify an existing
bridge. To execute before default-branch registration would require a separately
authorized bridge; otherwise register the new workflow on the default branch and
select the desired ref when dispatching it.

For an already-configured production read environment:

```sh
PYTHONPATH=. python diagnostics/stock_eod_timeframe_coverage_incident.py --output-dir /tmp/stock-eod-coverage-incident
```

The output directory must be local and outside the production data root.
The CLI has no universe, provider, production runner, or publication options.

## Reports

The artifact contains the eight requested CSVs and `incident_summary.md`, plus
`daily_reason_recurrence.csv` and `daily_loss_timing.csv`. Missing symbols are
one row per symbol/timeframe, with separate intersection rows. Each observation
retains all original telemetry columns; absent schema fields are `UNKNOWN`,
while persisted nulls are `NULL`. Raw-field presence flags distinguish these.
Reason partitions are mutually exclusive and include original supporting reason
strings. Stage counts are never inferred from artifact membership; partially
missing fields yield `UNKNOWN` with observed and unknown counts separately.

## Evidence limits

- Interval rows repeat the three-timeframe family request counters. The
  diagnostic counts attempted acquisition from per-interval acquisition status,
  rather than summing those repeated counters.
- `terminal_session` is derived session evidence, not the raw interval label.
  The current telemetry schema does not store the raw terminal interval label.
- Production telemetry normalizes nonfinite terminal numbers to null. A null
  alone cannot establish whether the provider originally returned null or
  infinity. The reason category preserves that ambiguity.
- Weekly/monthly `reconciliation_status=not_needed` does not prove they were
  valid: production leaves those observations unmodified.
- Acquisition order is reconstructed only with complete, unique, timezone-aware
  UTC fetch-start timestamps. Telemetry symbol order is the persisted runner
  enumeration; it is not an explicitly stored processing index.
- Midnight counts use acquisition start and completion independently, with an
  explicit unknown bucket. `CONFIRMED` means an exact observed membership split
  at midnight UTC, not proof of calendar or provider causation. Counterexamples
  reject that strict split even when other timestamps are unknown.
- Artifact SHA-256, byte size, and row/symbol counts are checked against the
  generation manifest. Its own byte hash is recorded, but historical manifest
  authenticity is not independently verified against a historical pointer.
- Matching the earlier Close/Adj Close missing-field signature proves a shared
  observation pattern, not an identical upstream root cause.
