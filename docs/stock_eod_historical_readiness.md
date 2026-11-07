# Historical 5 PM Stock EOD diagnostic

This offline diagnostic analyzes **already retained** evidence for intended sessions
2026-10-12 through 2026-11-04 inclusive. It imports no production application,
storage backend, credentials, provider, or network client. There is no workflow,
dispatch, automatic discovery, production write, or live acquisition implementation.
Do not execute it against real evidence until separately authorized. Running its
synthetic unit tests is separate from running the diagnostic.

## Evidence inputs

Prepare a local directory outside production `data/` with a JSON catalog, unchanged
health-manifest Parquet downloads, and optional decoded Actions job logs. Obtain
these through separately authorized read-only extraction; this command does not
fetch them. Preserve source URLs/S3 keys, download byte sizes and SHA-256 values.
Hashes verify byte identity, not authenticity or completeness of the archive.

Each catalog entry contains the raw GitHub workflow-run object and a `jobs` array
from its **latest attempt** jobs API, an explicitly reviewed `intended_session`,
and optional manifest/log file specifications. The intended session must come from
investigation/calendar context, not the UTC family creation date. A late recovery
may start the next calendar day. Holidays require explicit reviewer interpretation.
Do not put credentials or unrelated logs in this catalog.

Example shape (replace placeholders with verified evidence; not executable as-is):

```json
{
  "runs": [{
    "intended_session": "2026-11-02",
    "workflow_run": {
      "id": 41551677411,
      "name": "Stocks Daily EOD",
      "status": "completed",
      "conclusion": "failure",
      "event": "workflow_dispatch",
      "head_sha": "a80ac60e34aa2aa5f0463b2268cfc5defec41bcc",
      "created_at": "2026-11-03T00:16:32Z",
      "run_started_at": "2026-11-03T00:16:32Z"
    },
    "jobs": ["REPLACE WITH RAW LATEST-ATTEMPT JOB OBJECTS"],
    "manifest": {
      "path": "failed/manifest.parquet",
      "family_run_id": "prod-20261103T001713Z-044f683b",
      "size_bytes": "REPLACE WITH INTEGER",
      "sha256": "REPLACE WITH VERIFIED DIGEST",
      "source": "s3://stock-intel-data-prod/data/_health/stocks_eod_family/prod-20261103T001713Z-044f683b/manifest.parquet"
    },
    "log": {
      "path": "failed/job.log",
      "size_bytes": "REPLACE WITH INTEGER",
      "sha256": "REPLACE WITH VERIFIED DIGEST",
      "source": "GitHub Actions job-log URL"
    }
  }]
}
```

The verified November control family is
`prod-20261103T031823Z-b2cccde3`, run `41568225348`. Its S3 key uses the same
production health path contract. Include other completed runs, failed families,
and registry skips from the requested period; missing manifests/logs are allowed
and remain explicit unknowns. Do not include a manifest for a skipped invocation.

Optional `scheduled_at` must be an independently verified, timezone-qualified
intended trigger instant. Omit it if unknown. **Do not infer a cron slot from run
creation time.** Retain the trigger provenance with the evidence package.

Bounds: 100 catalog runs, 1 MiB catalog, 128 MiB aggregate referenced files,
30,000 rows and 96 columns per manifest. All referenced files must resolve inside
the catalog directory, including symlinks. Duplicate run IDs, family IDs, native
symbol/interval rows, identity mismatches, naive timestamps, reversed timestamp
pairs and checksum/size mismatches fail the analysis. The run/job API objects
must describe completed Stocks EOD executions and the matching job. Logs, when
provided, must corroborate the manifest family ID. No inferred IDs are fetched.

## Proposed usage after approval

From the repository root, with pandas and pyarrow already installed:

```bash
PYTHONPATH=. python diagnostics/stock_eod_historical_readiness.py \
  --catalog /tmp/stock-eod-retained/catalog.json \
  --output-dir /tmp/stock-eod-readiness-report
```

Output must be new and outside both the evidence directory and repository `data/`.
Inputs are never modified. Analysis completes in memory before output creation;
write errors can leave a partial new report directory. No production APIs execute.

Reports are JSON: `inventory`, `runtime`, `observations`, `field_quality`,
`revisions`, and `summary`. Inventory records input sizes, hashes and sources.
Runtime separates supplied scheduled time, workflow creation/start, actual job
start/end, family-ID creation time, exact **recorded loader-call** weekly
start/completion envelope, and publication log time. Timestamp summaries include
UTC and America/New_York offsets. Weekly duration is the first-start to
last-completion envelope, including intervening D/M work, not summed HTTP time.
Missing timestamps never fall back to family creation or manifest-write time.

Created-to-job and workflow-to-job delays are measured, but their scheduler,
runner and concurrency components cannot be separated by these timestamps alone.
`concurrency_delay_seconds` therefore remains unknown. Registry skips are detected
only from explicit log lines and excluded from execution runtime distributions.
The preceding verified registry-success run is reported only within the supplied
catalog; without complete logs/history it is not proof of the persisted registry
row or the exact cooldown decision. Runtime distributions exclude entries whose
execution-versus-skip status cannot be verified from logs.

The 19:30 target uses the explicitly supplied intended session. The counterfactual
assumes **actual job start** at 17:00 and the same measured setup/acquisition
elapsed time; it excludes scheduler and queue delay. It is neither a forecast nor
a safe-start guarantee. Timing completeness covers recorded weekly rows only,
not the routed universe; acquisition failures can still have complete timestamps.

## Readiness interpretation and decision criteria

`raw_*` health columns are reported as **normalized native** values, never HTTP
fields. Missing/nonfinite values and unrecorded columns are separately labeled.
Session states distinguish stale, future, unverified, and matching-session
observations. Matching evidence and finite OHLCV are never promoted to finality.
Full provider metadata JSON is retained as evidence, without deriving completion
from generic quote timestamps. Partial-period status remains unknown.

Adjacent observations are compared only within the same supplied intended session,
symbol and native interval, ordered by fetch completion. Close/Volume deltas and
missing-to-finite transitions are recorded. Pairs with missing/stale session proof
are explicitly unverified; their differences are not asserted to be provider
revisions to the same bar. W/M period identity remains unknown because this health
schema omits terminal interval labels. Daily matching-session evidence still does
not establish finalized volume. Adjusted/raw modes and provider versions remain
available for reviewers; differing modes/versions invalidate a causal comparison.

The summary conservatively returns NO-GO for **proven 17:00 provider finality**:
the existing schema has no authoritative finality flag, full terminal labels,
raw HTTP/retry/cache lineage, exchange calendar verification, or revision policy.
This does not conclude that 17:00 is unsafe. It separates observed operational
facts from an unproven data-readiness claim.

Before production GO, require a complete routed-universe timing sample, measured
queue/setup/acquisition tails leaving a declared margin before 19:30, current-session
regular-hours/raw-price provenance, and an accepted Close/Volume correction policy.
Review missing runs and evidence gaps explicitly. Required scheduling corrections
include DST-aware local eligibility, late-start behavior, and session-aware
cooldown/idempotency after late recovery. Preserve serialization and the existing
90% catastrophic coverage floor; this diagnostic changes none of them.

If historical evidence is insufficient, the minimum next step is a separately
approved **nonpublishing** bounded panel at 17:00/17:30/18:00 with reliable later
and next-session controls. Include Monday failures and exceptions, verify exchange
sessions, and compare Close and Volume. A panel screens readiness; universe-wide
acceptance is still needed. Include Friday, holidays/half-days and month-end
controls before generalizing. No such acquisition or workflow is included here.

## Validation

```bash
python -m pytest -q tests/test_stock_eod_historical_readiness.py
```

Tests use synthetic local Parquet/log fixtures only and do not invoke the CLI,
network, provider, production runner or storage backend.
