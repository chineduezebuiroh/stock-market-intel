# Stock EOD aggregate timing (Track E Phase 2B)

Each `run_stock_eod_family.run_family` invocation emits one flushed JSON log line
prefixed `[STOCK_EOD_PERF]`, including on failure (partial aggregates). There is
no new S3 object, per-symbol timing output, provider hook, or acquisition timer.
The collector is inactive outside this runner, including standalone timeframes
and downstream ETF/combo jobs. Scope is restored on exit.

Schema version 1:

```text
family_run_id: existing run identity
timing_errors: count of failed timing observations (0 means complete timing)
processing: daily / weekly / monthly:
  count: process_one_preloaded invocations, including rejection/exception
  total_seconds: inclusive monotonic elapsed time in those calls
  storage_seconds: sum of existing storage-call durations inside processing
  indicator_seconds: inclusive elapsed time in apply_core
  residual_seconds: max(0, total_seconds - storage_seconds - indicator_seconds)
  storage: exists / read / write:
    count: calls attempted, including calls that raise
    total_seconds: inclusive elapsed time in those storage calls
snapshot: daily / weekly / monthly:
  count: snapshot scan invocations (normally 1 each)
  total_seconds: inclusive monotonic elapsed time in each full scan
```

All durations use `time.perf_counter()`. Storage includes initial exists/read,
write, post-write exists, and the existing first-symbol verification read.
Storage sub-bins and indicator execution do not overlap each other; both are
nested inside processing total. Do **not** add storage/indicator to processing
total. Snapshot scans run later and their I/O is not included in processing's
storage bin. Residual includes all other processing work and tiny instrumentation
overhead; it does not measure reconciliation or manifest work outside processing.
It must not be labeled CPU time or pure local computation. Storage elapsed is
inclusive SDK/pandas/serialization/network time, not wire latency.

Counts are attempted invocations, not accepted symbols. On production failure,
completed and partially elapsed calls are included, and later bins may be zero.
If `timing_errors` is nonzero, durations and derived residuals are incomplete;
do not use them for causal attribution. Diagnostic setup, timer, aggregation,
serialization, and output exceptions do not fail production or replace its errors.
If summary output itself fails, the line may be absent; production continues.
Output size is independent of universe size (three fixed timeframe entries).

Existing acquisition manifest/capture timestamps remain the loader-span source.
The existing flushed `[STOCK_EOD_COVERAGE]` boundary follows snapshot assembly
and precedes publication; its Actions timestamp through `[FAMILY_PUBLISH][COMMIT_OK]`
provides the publication/stored-validation/CAS envelope. No production dispatch
is needed: inspect the next naturally scheduled run after deployment.

This is measurement only. Storage call order, arguments, results and exception
handling, provider behavior, indicator calculations, snapshot contents, B2/CAS,
legacy mirrors, and Track D remain unchanged. No timing is used as policy input.
