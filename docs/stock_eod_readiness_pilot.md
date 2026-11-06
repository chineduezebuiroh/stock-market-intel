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
including early closes. There is no cron or repeated dispatch. The collector can be armed up to 30 minutes
early and waits locally in at-most-one-second increments until the target time.
Dispatch early enough for setup/artifact download to finish before arming.
A delayed job does not silently shift the experiment. Arrivals after the five-minute
window emit MISSED_OBSERVATION_WINDOW with zero calls, not provider unreadiness.
More than 30 minutes early is rejected before acquisition. Queue delays remain
unbounded; no finite dispatch lead can guarantee a sample.

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
three authorized phases**. There is no diagnostic retry or additional metadata acquisition. The public
metadata getter is transport-locked; if it needs a fetch, evidence is unavailable
and session attribution remains unverified.

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
The workflow has a 55-minute job timeout (setup + up to 30 minutes arming + bounded capture), separate diagnostic concurrency group,
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

Metadata uses the public `Ticker.get_history_metadata(repair=False)` getter,
with transport I/O locked out while accessing only allowlisted non-lazy keys:
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
timezones, invalid required OHLCV, stale/future labels, or absent W/M session proof
are NO_GO_OR_UNVERIFIED. A finite observation never establishes finalized volume.

## Artifacts, comparison and acceptance

Each phase emits `observations.json`; the workflow hashes it into
`observations.sha256` and uploads one `stock-eod-readiness-pilot-<run_id>` artifact
for 30 days, including diagnostic failures and missed-window reports where evidence exists. Retain GitHub's
artifact digest and verify file SHA-256 after download. Artifact provenance must
be reviewed externally; internal payload hashes alone do not authenticate files.

Exit 2 means unverified/invalid observations, not necessarily a failed HTTP request.
Partial exceptions retain completed observations and an exception type, without
exception text that might expose URLs. Missing prerequisites or a missed start
window produces no provider requests and no passing report; missed windows have
a distinct MISSED_OBSERVATION_WINDOW artifact. Upload failures fail
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
out-of-plan timestamps. It compares terminal label identity plus absolute OHLCV/Adj Close
revisions versus primary at both comparison points, highlighting Close and Volume and whether the full
native payload hash changed. Period/label
ambiguity cannot be called a same-bar revision. Stable observations remain
STABLE_OBSERVED_NOT_FINALITY_PROOF, never an automatic production GO.

Pilot acceptance for review: all requested symbols/intervals collected within
budget/window; target-session/period identity established; finite required OHLCV with Low <= High and Low <= Close <= High; no unexplained Close or Volume revision across phases;
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


## Pre-merge contract and compatibility verification

Open must be finite but need not fall within Low/High. The required fields are
finite O/H/L/Close/Volume, Volume >= 0, Low <= High and Low <= Close <= High.
Adj Close is separately observed, not a required production field. Its absence
does not exclude a core Close/Volume comparison: deltas for Adj Close are null,
`adj_close_comparison_eligible=false`, and unchanged core data is reported as
CORE_STABLE_ADJ_CLOSE_UNAVAILABLE. This is not an adjusted-price readiness pass.

Offline inspection of installed yfinance 1.7.0 and curl_cffi 0.16.3 verified:

- `Ticker(ticker, session=None)` accepts the bounded curl_cffi Session subclass.
- Session accepts `impersonate='chrome', retry=0`; Session.request supports timeout,
  allow_redirects and max_redirects. Offline tests replace the actual transport
  send seam and verify GET reaches the bounded override with redirects off and
  retry count zero. No HTTP connection occurs.
- Ticker.history forwards kwargs to PriceHistory.history; the latter explicitly
  supports end, keepna, prepost, auto_adjust, back_adjust, actions and repair.
- Private `_price_history._history_metadata` exists but is not used by the pilot.
  The supported public metadata getter can acquire base metadata when absent,
  and its mapping has a lazy tradingPeriods key. The pilot locks transport while
  calling it and reads only the allowlist, never copies the full lazy mapping.
- The actual pinned history parser converts end to period2 and removes a trailing
  quote at/after exclusive end. Offline fabricated chart-response tests exercise
  that real parser for 1d/1wk/1mo with next-morning rows and Monday/Tuesday,
  Friday/Saturday, holiday-adjacent, half-day and month-end dates. The sentinel
  next-date quote is removed and target quotes retained. These tests prove client
  filtering behavior, not that Yahoo will supply a correct historical aggregate.

If Yahoo returns an end-bounded W/M aggregate incorporating a later session while
retaining an earlier period label, the label alone cannot detect it. Price-bound
target-session evidence remains mandatory, and absence is UNVERIFIED. Reference
availability and provider finality require separately authorized live validation;
never fall back to an unbounded end or admit the next session to obtain data.

Serialization uses tagged dictionaries/lists/tuples, booleans, timestamps, exact
integers, missing/nonfinite values and deterministic sorted dictionary keys.
Unsupported objects are type-marked without repr, never converted unsafely.
Depth/item/string truncation is explicit. Sensitive nested keys and URL/credential
strings are redacted. Original 128 KiB native/32 KiB metadata limits still apply;
oversized evidence fails closed. No arbitrary attrs or authentication data are
read. Captured schema remains diagnostic-only; use artifacts from one revision.

Production parity: production's keepna=False can drop all-empty rows, whereas
this pilot deliberately retains them. Production uses default transport behavior;
this pilot disables redirects/retries and enforces hard transfer/deadline limits.
Production calls a metadata getter without this pilot's I/O lock. End bounds in
production use acquisition time; this reference freezes the end at target midnight.
The pilot has a small panel, no indicators or persistence workload and stricter
explicit diagnostic session/period checks. Its observed usability, transport errors,
missing rows/metadata and runtime cannot be presented as equivalent production
readiness, failure probabilities or full-universe throughput. Versions are recorded.

Recommendation: diagnostic infrastructure can be reviewed for merge after these
offline checks; merging is not authorization to dispatch. No real provider calls,
calendar package integration, runner scheduling or market-data finality have been
validated live. Missed windows, transport-blocked metadata and unverified reference
aggregates are inconclusive experiments, never production-readiness failures.


Corrected artifacts use schema version 2 and explicitly record production_parity=false
and their acquisition-profile differences; the comparator rejects old schema versions
and mismatched acquisition profiles. Local validation: 44 focused offline tests,
compilation, YAML/manual-only/read-permission checks, shell syntax and whitespace
checks passed under Python 3.12. The workflow remains Python 3.11; that interpreter
was not exercised locally. Suggested dispatch lead is roughly 25 minutes before
the phase, subject to setup/queue variance; actual acquisition still has the fixed
five-minute start window and fifteen-minute deadline.
