# Targeted Stock EOD readiness pilot (draft, not dispatched)

This is a standalone acquisition diagnostic. It imports no production runner,
registry, storage, publication, scoring or notification module. PR #57 is not
changed. No recurring trigger exists. Each phase requires separate authorization
and a manual dispatch; merging alone does not acquire anything.

## Proposed panel and cohort evidence

Fixed controls: **SPY, QQQ, IWM, GLD, XLF, AAPL, MSFT, NVDA**. These cover broad,
small-cap, sector and gold ETFs plus liquid equities. SPY is always included.

Add three November 2 affected symbols and three of the 13 valid post-boundary
exceptions: lowest, median and highest **incident share volume** within each
cohort, breaking ties by symbol. This spans observed trading activity without
pretending share volume is spread, dollar liquidity or a stable liquidity ranking.
Union/deduplicate with controls: **at most 14 symbols**. The actual added names
must be reviewed in the emitted panel before drawing conclusions.

The exception names are not available in the present task's verified inputs.
They are intentionally not guessed or hardcoded. Before acquisition, the tool
requires the unchanged failed health manifest from
`prod-20261103T001713Z-044f683b`, an independently verified SHA-256, all 2,335 unique
weekly records and the verified partition using native **fetch_started_at**:
686 valid pre-01:00Z, 13 finite Close/Adj Close post-01:00Z, and 1,636 post-boundary
missing Close/Adj Close with finite O/H/L/Volume. Ambiguous timestamp attribution,
wrong family, checksum failure, missing controls or a different partition stops
before any provider request. Manifest size/row/column limits apply.

The manual workflow downloads the already existing extraction artifact
`stock-eod-nov02-health-manifest-<extraction_run_id>` using read-only Actions
permission. Its fixed failed-manifest path is the family-ID subdirectory created
by PR #56. No AWS credentials, S3 access or fresh extraction are included.

## Exact timing and request budget

Use an explicitly reviewed XNYS session date, preferably a Monday first. The
exchange calendar rejects non-session target dates and supplies the actual close,
including early closes. There is no automatic waiting, cron or repeated dispatch.
Dispatch early enough for setup/artifact download to finish in the window;
a delayed job does not silently shift the experiment.

| Phase | Actual collector start (New York) | Finish deadline |
|---|---|---|
| primary | target date 17:00:00–17:05:00 | 17:15:00 |
| later | target date 18:00:00–18:05:00 | 18:15:00 |
| reference | next calendar date 08:00:00–08:05:00 | 08:15:00 |

For a Friday the reference is Saturday morning, not an implicit Monday trading
session. Every phase fixes history start at target minus 90 calendar days, and
uses `end=min(fetch start, midnight immediately after target)` (exclusive).
The reference therefore cannot request the next session's bar. A returned terminal
label outside the target daily session/week/month is explicitly rejected anyway.

For N unique panel symbols, each phase attempts **exactly 3N native history calls
if it completes** (1d, 1wk, 1mo). N <= 14: **42 maximum per phase, 126 across the
three authorized phases**. There is no diagnostic retry or extra metadata getter.

A separately bounded shared curl_cffi session counts all outgoing Session.request
calls, including yfinance timezone/authentication/retry requests: **60 maximum per
phase; 180 across three phases**. curl_cffi automatic retries are disabled and
redirect following is disabled so hidden transfers cannot evade the budget.
Unexpected redirects fail closed. This changes transport behavior from ordinary
production defaults and may prevent collection; that is an experiment limitation,
not evidence of provider unreadiness. It must not be worked around by raising
budgets or silently changing transport. Native history calls and HTTP transfers
are different counts; both are retained.

HTTP timeout is at most 10 seconds and bounded by remaining phase time. No new
history call starts after the phase deadline. Late completion is marked failed.
The workflow has a 20-minute job timeout, separate diagnostic concurrency group,
no cancellation of an active phase and no production concurrency/registry coupling.
Forced runner termination may prevent artifact completion; missing/partial
artifacts are never a readiness pass. Repeated manual dispatches are additional
experiments requiring their own authorization, not part of this budget.

## Evidence and native semantics

Pin yfinance 1.7.0 and curl_cffi 0.16.3. Record their exact versions plus pandas
and exchange_calendars versions; comparisons require identical version sets.
Calendar dependency is constrained to major version 4, not claimed reproducible
across installations; reject mismatched versions rather than combine silently.

Explicit history options: `auto_adjust=False`, `back_adjust=False`,
`prepost=False`, `actions=False`, `repair=False`, `keepna=True`. The last option
retains native missing rows for diagnostics instead of applying the production
loader's default cleanup. Native frame evidence is captured without modifying
frames. No adapter, Close repair or stale fallback is applied.

Each observation records symbol/interval, start/completion timestamps, requested
bounds, raw-unadjusted mode, full native returned columns/dtypes/index timezone
and every bounded row including OHLCV/Adj Close. Finite floats retain hexadecimal
representations; missing/null/infinity distinctions and integer values are
preserved. SHA-256 covers the canonical native-frame serialization (not a Yahoo
HTTP response). Maximum 256 rows/16 columns and 128 KiB per frame, 32 KiB of
allowlisted metadata per observation. Oversized evidence fails closed.

Only already-fetched private history metadata is read, with no getter call:
exchange timezone/type, quote precision, generic market timestamp/price,
currentTradingPeriod and synthesized lastTrade. Generic metadata is retained as
evidence, never promoted to interval completion proof. Arbitrary attrs, cookies,
crumbs, authentication headers, response bodies and URLs are not archived.
This is **raw native yfinance-return evidence**, not raw HTTP attribution evidence.

The calendar classifies W/M as WEEK_TO_DATE/MONTH_TO_DATE or COMPLETED_WEEK/
COMPLETED_MONTH according to whether the target is the calendar's last applicable
session. Completion is a calendar expectation, not proof the provider incorporated
that close. Daily requires the target label; W/M also require finite price-bound
interval lastTrade with the target date for current-session attribution. Unsupported
timezones, invalid OHLCV/Adj Close, stale/future labels, or absent W/M session proof
are NO_GO_OR_UNVERIFIED. A finite observation never establishes finalized volume.

## Artifacts, comparison and acceptance

Each phase emits `observations.json`; the workflow hashes it into
`observations.sha256` and uploads one `stock-eod-readiness-pilot-<run_id>` artifact
for 30 days, including diagnostic failures where evidence exists. Retain GitHub's
artifact digest and verify file SHA-256 after download. Artifact provenance must
be reviewed externally; internal payload hashes alone do not authenticate files.

Exit 2 means unverified/invalid observations, not necessarily a failed HTTP request.
Partial exceptions retain completed observations and an exception type, without
exception text that might expose URLs. Missing prerequisites or a missed start
window produces no provider requests and no passing report. Upload failures fail
the workflow; failed collection is not masked by a successful upload.

Offline comparison (after authorized collection, no requests):

```bash
python diagnostics/stock_eod_readiness_pilot.py compare \
  --primary /tmp/primary/observations.json \
  --later /tmp/later/observations.json \
  --reference /tmp/reference/observations.json
```

Comparison requires one complete report per phase, identical target session,
incident hash/panel/versions and unique matching symbol/interval sets. It verifies
native hashes, recomputes assessments from preserved native evidence, and rejects
out-of-plan timestamps. It compares terminal label identity plus absolute Close
and Volume revisions versus primary at both comparison points. Period/label
ambiguity cannot be called a same-bar revision. Stable observations remain
STABLE_OBSERVED_NOT_FINALITY_PROOF, never an automatic production GO.

Pilot acceptance for review: all requested symbols/intervals collected within
budget/window; target-session/period identity established; finite internally
consistent OHLCV/Adj Close; no unexplained Close or Volume revision across phases;
explicit reviewed treatment of provider corrections; and no missing evidence.
Absent W/M freshness evidence, a revised value, or any next-session ambiguity
blocks readiness approval. A later/next-morning observation is a comparison,
not an authoritative source of finality. Validate against an independent regular
session reference if finality remains ambiguous. A small panel cannot establish
universe-wide coverage/runtime or future guarantees.

Collect separate Monday, Friday, half-day and month-end cases before generalizing.
Keep the existing production 90% guard, schedule, registry, signal/publication and
S3 state untouched. No production changes are proposed by this pilot PR.

## Validation

Offline tests exercise deterministic cohort selection, budgets, timestamps/DST,
calendar boundaries, native frame immutability, wrong-session rejection, partial
failure preservation, revision comparison, tamper detection and manual-only workflow
permissions. They invoke fake histories and a fake transport/calendar, never live
acquisition. The real provider/calendar/transport combination remains unexercised
until separately authorized.
