# Stock D/W/M family orchestration design

Status: design only. The current supplemental reconciliation path remains the
interim implementation. This document does not authorize W/M repair or change
signals, scoring, ETFs, combos, notifications, or dashboards.

## Recommendation

Use **acquire / reconcile / process**, implemented by an internal family runner
while preserving the external production command `stocks daily --cascade`.
Build one small in-memory family per symbol, not one giant all-symbol object:

1. independently fetch native D, W, and M;
2. retain each raw frame and request provenance as separate observations;
3. validate and reconcile the available family;
4. merge and run indicators once for each finalized timeframe;
5. accumulate versioned D/W/M snapshot rows;
6. publish a committed family manifest only after the snapshot set is ready;
7. allow D/W/M combos only from a single committed family.

An interval acquisition failure must not discard valid independent intervals.
It does prevent publication of downstream products that require that interval.
Standalone `stocks daily`, `stocks weekly`, and `stocks monthly` remain
independent commands; only `stocks daily --cascade` selects the family runner.

## Minimal observation model

`StockEODObservationFamily(run_id, symbol, observations)` owns exactly three
optional native members (`1d`, `1wk`, `1mo`). Each member retains:

- raw native frame (never overwritten by repair);
- provider interval and interval label;
- fetch start/completion timestamps;
- provider/yfinance version and request semantics;
- raw/unadjusted and regular-session declarations;
- provider metadata, terminal-session evidence, and partial-period state;
- payload hash and quality result;
- raw terminal values;
- optional finalized frame plus repair provenance.

Q/Y are descendants produced after finalized M and are never family evidence.
Stock 4H is outside this family.

Use a per-symbol in-memory family so memory is bounded by three rolling frames
plus snapshot accumulators. Persist a compact, append-safe run manifest with
per-symbol payload hashes, acquisition/quality/reconciliation status, evidence,
and errors. Before implementation, measure actual peak memory and manifest/S3
cost with the current 2,381-symbol universe; repository snapshots do not measure
the S3 rolling-frame footprint.

## Current call graph and ownership

The `stocks_eod` GitHub workflow invokes the guarded profile. The execution
registry guards the whole `stocks_eod` function, not individual D/W/M runs. The
profile starts `run_timeframe.py stocks daily --cascade`; `run()` resolves the
daily universe, `ingest_one()` fetches/repairs/merges/runs indicators/persists,
and `run()` writes the daily snapshot. Recursive child calls then fully process
weekly and monthly independently. Only after that subprocess succeeds does the
profile update ETF trends, build D/W/M combos, run health, and notify. The guard
marks `stocks_eod` successful only after the entire profile returns.

Weekly and monthly rollup workflows have separate `stocks_weekly` and
`stocks_monthly` registry identities. They normally build Q or Y and their
combos; they invoke standalone W/M only to bootstrap missing shortlist symbols.
`run_timeframe.py` itself does not mutate the execution registry.

## Registry and family state

Keep `stocks_eod` as the family-run registry owner. Do not create child registry
success rows that can falsely imply a complete family. Record interval states in
the family manifest instead:

| Situation | Persist independent valid bars? | Publish family snapshots? | Run D/W/M combos? | Mark `stocks_eod` success? |
|---|---:|---:|---:|---:|
| D/W acquired, M acquisition fails | yes, D/W | no complete family | no | no |
| all acquired; D and M process, W fails | yes, D/M | no complete family | no | no |
| all required intervals finalize and snapshots commit | yes | yes | yes | only after combos/health/notifications return |
| retry after partial failure | reuse only immutable staged observations whose run/session identity is still valid; otherwise refetch | after completion | after completion | after completion |

Standalone weekly/monthly profile registry timestamps continue to describe
their rollup profiles, not family interval freshness. The family manifest is the
source of truth for per-interval state.

## Snapshot and combo transaction boundary

Current snapshots have no family/run identity and combos inner-join whichever
three canonical files exist. Sequential orchestration prevents normal in-profile
early combo execution but cannot prove lineage after a partial run. The target
must write versioned snapshot artifacts carrying `family_run_id`, then atomically
commit a small manifest/pointer naming the complete D/W/M set. Combo loading must
resolve one committed manifest (or at minimum verify identical family IDs) before
scoring. Canonical aliases may remain for dashboards, but are not sufficient as
the combo transaction boundary.

## Q/Y

Derive Q/Y from the finalized same-run native M series rather than refetching M.
Persist lineage (`derived_from_interval=1mo`, family ID, M payload/finalized
hash). Q/Y remain non-independent and cannot vote in reconciliation. Q/Y failure
does not retroactively invalidate a valid M observation, but blocks rollup
products that require the failed descendant.

## Provider request model

For universe size `N=2,381`, the interim daily cascade makes `3N=7,143` normal
native requests plus at most 200 duplicate supplemental W/M requests: 7,343
worst case. Calls are sequential, one symbol/request at a time; there is no batch
or parallel fetch. The family runner makes exactly `3N=7,143` normal requests
and eliminates up to 200 reconciliation duplicates. No concurrency change is
required for correctness. The family path removes the 100-symbol cap because
same-run evidence is already acquired; the cap/hash policy remains only for an
interim standalone supplemental path.

## Discarded yfinance live row

No public stable yfinance API exposes the full dropped live row. Metadata exposes
only its Close/Time. Options, in increasing maintenance risk, are:

1. contribute/upstream or maintain a pinned minimal yfinance patch that exposes
   an immutable full live-row copy without changing the returned W/M frame;
2. directly capture and parse Yahoo chart JSON before yfinance cleanup, assuming
   responsibility for crumb/cookie, schema, timezone, actions, and repair logic;
3. wrap/monkeypatch the private cleanup function (brittle global/internal hook;
   not recommended);
4. make a second request, which does not necessarily reproduce the discarded
   interval row and defeats request reduction.

Full live-row Time/O/H/L/Close/Adj Close/Volume would show which extrema and
volume were available for merge and make anomalous `W.Close=null` plus finite
`lastTrade.Price` diagnosable. It still would not by itself resolve later
provider revisions. Do not implement W repair until this evidence is captured
and the anomaly is explained.

## Migration and shadow validation

Implement in reviewable stages with rollback after each:

1. Add family/observation types and acquisition helpers; no production routing.
2. Add a read-only family shadow runner and compact manifest.
3. Make reconciliation consume family evidence in shadow and compare decisions.
4. Process finalized D/W/M once into versioned shadow rolling/snapshot outputs.
5. Add committed-manifest lineage validation to combo loading.
6. Route `stocks daily --cascade` to the family runner; retain standalone paths.
7. Derive Q/Y from finalized M.
8. Remove family-path supplemental calls and its 100-symbol cap after production
   shadow acceptance.

For unaffected symbols compare native terminal rows/hashes, final rolling rows,
all indicator columns, snapshots, combo inputs, and signal outputs. For repaired
symbols preserve native malformed values, evidence, repair, indicators, combo
eligibility, and signal. Count requests, validity patterns, tiers, conflicts,
exclusions, and avoided supplemental calls. Do not tune signal policy during the
comparison.
