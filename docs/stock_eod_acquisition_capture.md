# Bounded Stock EOD acquisition diagnostics

This adds observation only. It does not repair data, change observation policies,
add coverage thresholds, or write production S3 objects. Independent D/W/M remain
primary observations; diagnostic dictionaries are never attached to dataframe
attrs or supplied to reconciliation. PR #41 and the investigation bridges are
unchanged.

## Controls and activation

Both controls default OFF:

- `STOCK_EOD_CAPTURE_APPLICATION=1`: capture application boundaries B/C.
- `STOCK_EOD_CAPTURE_RAW=1`: request a safety assessment of Layer 2. **Raw
  capture remains disabled**, even for the supported implementation (see below).
- `STOCK_EOD_CAPTURE_DIR`: local output root, outside the production data root.
- `STOCK_EOD_CAPTURE_SYMBOLS`: optional comma-separated selected symbols (max 32).

The existing Stocks Daily EOD workflow gains two manual boolean inputs, both
false by default, and a conditional seven-day artifact upload. Scheduled runs
remain OFF. Upload failure is `continue-on-error` and does not alter the production
job outcome. No new provider execution or workflow dispatch was performed. Any
future manual production execution/capture still requires separate authorization.
The upload is the only durable diagnostic publication; the capture module uses
local filesystem APIs only and has no S3 writer.

## Exact B/C boundaries

B is immediately after the existing `Ticker.history()` returns and before the
empty-frame/adapter logic in `etl.sources.load_eod`. C is after the existing
sanitization and provenance attachment, immediately before the native frame
returns to quality validation. Empty results also have a C record. B preserves
original columns, dtypes, index type/name/timezone, row count, terminal two labels
and quote fields; C records their normalized equivalents and inserted columns /
duplicate-removal facts. Only allowlisted existing provenance is observed.

Each native loader call has a context-scoped identity:
`family_run_id + symbol + native_interval + acquisition_sequence`. Reference
loader calls are outside this context. The sequence advances before acquisition;
D/W/M remain separate identities. Snapshot capture itself does not mutate the
frame. The existing adapter intentionally changes its returned frame's index;
tests compare those existing side effects with diagnostics OFF rather than
requiring the adapter to stop doing so.

Values distinguish absent columns, None, pd.NA, NaN, +/-infinity, integers, finite
floating values (including hexadecimal representation), timestamps and unsupported
or redacted types. Numeric arrays are not passed through the production `_number`
conversion. Unknown quote strings/object values are not rendered with repr.
Unrestricted attrs, cookies, tokens, crumbs, URLs and response headers are not
retained. This is intentionally not a full arbitrary-attrs dump.

## Layer 2 blocker

The optional capability assessment checks yfinance `1.7.0` and exact source hashes
of unwrapped `YfData.get` / `cache_get`. Unsupported versions, source drift or
inspection failure disable raw capture while leaving B/C active.

Even when fingerprints pass, get/cache_get does not expose the individual
retry/authentication response lineage hidden inside `_make_request`. Cached
responses may be shared. Installing an observer while claiming the approved
retry-attribution contract would weaken that contract. The implementation therefore
installs **no hooks** and records
`get_cache_get_seam_does_not_expose_internal_retry_lineage` in summary.json.
It does not wrap or replace the session, cache, response or library methods and
cannot emit boundary A evidence or subordinate HTTP-request identities. Capturing
raw response data remains a separately designed/tested follow-up; B/C can identify
adapter-induced changes, but cannot attribute Yahoo versus yfinance origin.

## Bounds and sampling

`Config` exposes conservative, validated Python-API budgets without allowing
unbounded increases. Defaults:

- 64 KiB encoded per native fetch;
- 16 MiB accounted in-memory budget (retained encodings use at most one quarter);
- 32 MiB run output ceiling, with a reserved 64 KiB summary allowance;
- categories: selected 96 interval slots, boundary 192, malformed 192, controls
  96, deterministic background 24;
- +/-120 seconds around UTC midnight or retained pre/regular/post boundaries;
- deterministic background hash sample: one in 100 symbols per run.

Category byte capacity is reserved separately; early failures cannot consume
controls or boundary capacity. Complete acquired D/W/M groups are admitted together.
A valid family immediately before a malformed transition is buffered as a control,
and the first valid family after a malformed group is a control. Selected symbols
and boundary sampling have priority, then malformed, controls, background. A quota
is a sampling cap, not a coverage/eligibility threshold. Overflow is counted by
bounded gap categories. A selected family can be omitted at quota exhaustion;
there is no guarantee every malformed or selected symbol is retained.

The timed path reads at most 32 columns and two rows per boundary and constrained
metadata into detached dictionaries. Frames wider than 32 columns yield a gap.
No disk writes, JSON encoding, compression, hashing of entire frames, raw parsing,
provider calls or background workers occur there. Encoding is after the loader
(and its SIGALRM) returns; family promotion is after all three acquisitions.
The run context writes retained evidence after the family runner returns or raises.
Directories are unique/private (0700), files 0600. No pre-existing evidence file
is overwritten. Record allocations are structurally bounded; byte budgets track
encodings, not exact Python heap usage. Tracemalloc tests verify the configured
memory envelope on the synthetic fixture; this is not a process-wide RSS limit.

## Failure and timing contract

Capture errors, serialization errors, invalid capture config, output failures and
quota exhaustion fail open for diagnostics. Original production exceptions are
unchanged. Genuine TimeoutException / TimeoutError, KeyboardInterrupt and SystemExit
are propagated through diagnostic guards; capture must not swallow an acquisition
alarm. An injected real SIGALRM exercises the existing safe_load_eod timeout path.

Instrumentation overhead can move requests across a wall-clock boundary or consume
part of the 30-second alarm. It cannot mathematically guarantee identical outcomes
against a changing live provider. Offline tests establish identical request count,
parameters, order, frames and deterministic production publication bytes. Record
capture time is measured in summary.json. Before authorized activation, the measured
budget must be assessed on the target runner. JSON transformations between native
calls and file writes after family publication still have bounded runtime cost.

## Evidence files

Each run creates a unique local acquisition directory with five category JSONL files
and summary.json. Summary contains capture timing, admitted counts, limits, overflow
counts and raw capability status. Failures during output may leave a partial diagnostic
artifact; they never change family success/failure. A consumer must inspect summary
and gap counts rather than assuming complete evidence.

## Local validation measurements

A 1,000-iteration six-quote-column B+C snapshot benchmark without tracemalloc
measured median 0.471 ms, p99 1.530 ms and maximum 2.002 ms on the implementation
workspace. The 300-iteration traced backlog test measured median 3.924 ms,
maximum 7.694 ms and peak traced memory 5,283,912 bytes (below the 16 MiB budget).
These are synthetic offline measurements, not production service-level guarantees.
Both exclude normal provider latency and post-acquisition encoding/artifact writing.
The focused test enforces a conservative 100 ms per-B+C CI ceiling under tracing;
it does not add a timeout or eligibility rule to production.

History start/end evidence is the exact application-level Ticker.history kwargs;
HTTP period1/period2, cache consumption and individual retry identities remain
unobserved while Layer 2 is disabled. Already-retained normalized-frame hashes
are labeled as such and never represented as Yahoo response hashes. Capture
failures can leave an incomplete D/W/M evidence group, explicitly counted as a
capture gap; quota admission itself never splits a group.
