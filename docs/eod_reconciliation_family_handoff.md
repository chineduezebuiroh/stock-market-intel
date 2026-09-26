# EOD reconciliation → D/W/M family orchestration handoff

Read this first in the next Codex thread. This is the durable boundary between
the current reconciliation PR and the separate family-orchestration work.

## 1. Purpose and current state

This work began after CRWD's persisted stock EOD data showed null daily/weekly
Close values, degraded indicators, and a suppressed/non-evaluable D/W/M signal.
The current PR adds a fail-closed stock EOD quality floor plus narrowly supported
daily Close reconciliation, provenance, health telemetry, bounded provider load,
and tests. It is considered **safe to merge as an interim architecture**.

The immediate next effort is a separate **shadow D/W/M family acquisition and
lineage PR**. The current supplemental W/M evidence fetches and 100-symbol cap
are temporary technical debt, not the target architecture.

At handoff, the working branch is `work` and the repository's squashed PR head
before this document is `c6b800b` (`Add EOD reconciliation/repair for native
D/W/M, telemetry, and yfinance pin`). The investigation previously referred to
head `40943566f0f1418b137af439e52ee3d9328b9876`; use repository history rather
than assuming those hashes survive merge. **After this PR merges, start the next
implementation from fresh `main`, on a fresh branch and fresh Codex thread.**

## 2. Why we got here: CRWD forensics

The historical persisted shape was approximately:

- D: native O/H/L/V present, Close and Adj Close null;
- W: Close and Adj Close null;
- M: valid Close.

Malformed D/W state degraded downstream indicators and left the D/W/M stock
signal non-evaluable or suppressed. A strict as-of reconstruction strongly
indicated that corrected daily state could materially change indicator/scoring
inputs, but it was a present-day reconstruction—not immutable proof of the exact
historical provider payload or final signal.

The retained historical artifacts lacked terminal-session, fetch, revision, and
interval-specific completion provenance. Therefore the historical CRWD row
cannot be repaired retrospectively under the production evidence contract.
Contemporaneous repair is different: a current fetch can carry the provenance
needed to authorize a repair before indicators run.

## 3. Live Yahoo/yfinance findings

A repeated seven-symbol diagnostic (`AAPL MSFT SPY CRWD NVDA JPM XOM`) observed:

- native D terminal O/H/L/V valid;
- native D Close/Adj Close null;
- native W and M Close populated and agreeing for each symbol;
- the state persisted across two observations five minutes apart;
- W/M interval metadata included `lastTrade` for the target session and price;
- daily quote metadata such as `regularMarketTime`/`regularMarketPrice` was
  ticker-level context, not proof of W/M aggregate completion.

This established that malformed D was not CRWD-specific.

Pinned yfinance 1.7.0 handles a separate W/M live row by treating the preceding
row as the aggregate, filling aggregate Open only if missing, conditionally
updating High/Low when live values are non-null, replacing Close and Adj Close,
adding Volume, dropping the live row, and then synthesizing `lastTrade.Price`
and `lastTrade.Time` from that dropped row.

Preserve these unresolved cases:

- **W.Close null + finite W.lastTrade.Price:** anomalous relative to the normal
  traced path, which should have assigned live Close to W Close. Investigate the
  inconsistency before authorizing W repair.
- **W.Close null + absent/null lastTrade:** not contradictory in the same way,
  but provides no positive proof that W incorporated target session T.

Because metadata exposes only dropped-row Close/Time—not full live-row O/H/L/V—
W/M target repair remains intentionally deferred.

## 4. Current quality and reconciliation contract

### PR #23 quality floor

- Required raw fields are Open, High, Low, Close, Volume; values must be finite
  and Volume nonnegative.
- Malformed fetched rows cannot overwrite a valid retained row.
- Fields from different rows are not coalesced; retention is whole-row.
- Malformed terminal state cannot reach indicators or snapshots.
- Daily same-label retention remains possible because the label identifies its
  session; unproven partial W/M same-label retention is disabled.
- Unsupported states fail closed.

### Supported daily Close repair

- Only the raw **daily Close** target is currently repairable.
- **Tier 1:** two independent eligible native D/W/M observations prove the same
  terminal session, have compatible raw/unadjusted regular-session semantics,
  agree within deterministic tolerance, and have no eligible contradiction.
- **Tier 2:** one native W or M source may suffice only when interval-specific
  `lastTrade` is price-bound to that native Close and session-bound to target T,
  with no contradictory eligible source or known revision ambiguity.
- Target native O/H/L/V must be structurally valid; Open and candidate Close
  must lie within Low/High.
- Adjustment/session semantics must match.
- Every repaired Close is an actually observed eligible raw Close. No midpoint
  or synthetic price is allowed.
- Adj Close is never repaired or synthesized.
- Repair occurs before merge completion and `apply_core`, so all daily indicators
  see the finalized series.

Tier-1 source precedence is:

1. finer provider-declared price precision;
2. if equal, finer native interval;
3. return that source's actual observed raw Close;
4. outside-tolerance disagreement remains fail-closed.

Q/Y inherit M lineage and are not independent votes. Stock 4H is outside the
regular-session EOD reconciliation family. W/M target repair is not implemented.

## 5. Interim operational architecture

Current production is conceptually:

```text
D fetch/process
  -> eligible malformed D only: supplemental W + M fetch, reconcile D
  -> D snapshot
  -> normal W fetch/process -> W snapshot
  -> normal M fetch/process -> M snapshot
  -> combos/scoring/notifications
```

For the configured `N=2,381` stock universe:

- normal D/W/M application-level requests: `3N = 7,143`;
- maximum supplemental reconciliation requests: `200`;
- maximum total: `7,343`.

`MAX_EOD_RECONCILIATION_SYMBOLS=100` is a provider-load circuit breaker.
Allocation uses deterministic `SHA256(UTC run date + symbol)` order, rotating by
date instead of alphabetic order. It is operational allocation only—not market
cap, options liquidity, signal, scoring, or business eligibility—and must not
become part of the target family contract.

## 6. Decided target architecture

Use **ACQUIRE → RECONCILE → PROCESS**, not late quarantine/reprocessing:

```text
stocks daily --cascade
  -> run_stock_eod_family(...)
       -> independently acquire native D/W/M
       -> construct one per-symbol observation family
       -> independently validate D/W/M
       -> reconcile explicitly eligible target fields
       -> process each finalized timeframe once
       -> indicators
       -> versioned snapshots
       -> committed family manifest
  -> combos from the committed family
```

Invariants:

- D/W/M remain independent native observations.
- Native observations are immutable; finalized/repaired observations are
  separate and provenance-bearing.
- Reconciliation is ingestion/data quality, not indicator repair.
- Indicators consume finalized inputs once.
- Canonical prices are always actually observed raw values.
- Failure of one interval does not erase valid independent intervals.
- Products requiring D/W/M together run only after a complete family commit.
- Standalone `stocks daily`, `stocks weekly`, and `stocks monthly` remain
  independent; `stocks daily --cascade` becomes the family path.

## 7. Snapshot/combo atomicity

Today the combo loader reads canonical D/W/M snapshot filenames without common
run/session/family validation. A partial failure can theoretically leave `new D
+ new W + old M` even though the guarded normal path runs combos only after its
cascade succeeds.

Target layout:

```text
snapshots/stocks/eod/<family_run_id>/daily.parquet
snapshots/stocks/eod/<family_run_id>/weekly.parquet
snapshots/stocks/eod/<family_run_id>/monthly.parquet
```

Commit a small manifest only after all required artifacts exist. It must name
the family, snapshot paths, sessions, and lineage hashes. Combos resolve inputs
from that committed family and fail closed on mismatch. Canonical aliases may
remain for compatibility, but are not the combo transaction boundary.

## 8. Q/Y lineage target

```text
finalized native M
  |- M
  |- Q derived from the same finalized M
  `- Y derived from the same finalized M
```

Q/Y should stop independently refetching M after cutover. They remain derived,
never vote in reconciliation, and carry explicit `derived_from=1mo`, family ID,
and source/finalized-M hash lineage.

## 9. Immediate next implementation: PR A only

Begin **SHADOW FAMILY ACQUISITION + LINEAGE** from fresh `main` after merge.
Do not cut over production in PR A.

1. **Family types/helpers:** independently acquire native D/W/M per symbol;
   preserve provenance and partial acquisition; no authoritative routing change.
2. **Compact run-scoped evidence manifest:** hashes, terminal values, fetch
   timestamps, metadata subset, quality/reconciliation states, and errors; no
   authoritative output replacement.
3. **Shadow reconciliation:** feed family observations into the existing
   reconciliation contract and compare decisions; no supplemental calls inside
   family reconciliation.
4. **Shadow finalized processing:** generate enough rolling, indicator, snapshot,
   combo-input, and signal output to compare; production remains authoritative.

## 10. PR A measurements and acceptance

For unaffected symbols compare native terminal rows, metadata/hashes, finalized
rolling terminal rows, every indicator, snapshots, combo inputs, scores, and
signals. Expected invariant: **no repair → parity with current production**.

For repaired symbols retain native malformed state, evidence, target session,
tier, canonical source/value, raw-versus-finalized diff, repair provenance,
indicators, combo inclusion, and score/signal under unchanged policy.

Measure provider request counts, malformed D/W prevalence, all D/W/M validity
patterns, tier counts, conflicts, exclusions, supplemental calls avoided, family
commit failures, and retry/refetch behavior.

Timing must include:

- per symbol: D fetch complete → W fetch start;
- per symbol: W fetch complete → M fetch start;
- per symbol: D first fetch → M final fetch;
- whole run: first family fetch → last family fetch.

Use this to detect sequential acquisition crossing meaningful session/provider
update boundaries. Do not introduce concurrency unless correctness requires it.

## 11. PR B: future cutover, not PR A

After shadow acceptance, a separate PR B may add:

- committed/versioned family snapshots;
- combo family-lineage gating;
- production routing of `stocks daily --cascade` to the family orchestrator;
- preserved standalone D/W/M commands;
- Q/Y derivation from finalized M;
- removal of family-path supplemental duplicate fetches;
- removal of the family-path 100-symbol cap;
- a reversible cutover/rollback switch.

## 12. Full dropped live row and future W repair

No stable public yfinance 1.7.0 API exposes the complete discarded live row.
Future approaches, in order of preference/risk, include an upstream contribution
or pinned minimal fork, direct Yahoo chart-response capture, private-function
interception (not preferred), or a second request (not preferred).

Required evidence is live-row timestamp, O/H/L/C, Adj Close, and Volume, captured
without changing the provider-observed returned W/M frame. Do not authorize W
repair until positive aggregate-freshness evidence exists and the finite
`lastTrade.Price`/null-W-Close anomaly is understood.

## 13. Separate options-liquidity roadmap

`docs/options_liquidity_eligibility_roadmap.md` is a separate future business-
actionability/universe-quality project. It is not a solution to provider request
allocation. Do not reopen or implement it during PR A.

## 14. Other deferred work

- Post-fix signal-incidence attribution:
  - A: prior persisted state/current rules;
  - B: corrected EOD state/pre-participation-change rules;
  - C: corrected EOD state/current participation rules.
- No scoring or participation changes before that attribution is measured.
- Dashboard Stage A/B remains pending.
- The historical ETF `significant_volume` versus `sig_vol_current_bar` mismatch
  was resolved separately by aligning the scorer with the canonical persisted
  `sig_vol_current_bar` indicator-instance key.
- W/M target repair remains deferred.
- Provider concurrency/batching remains deferred unless needed for correctness.

## 15. Read these first

1. `docs/eod_reconciliation_family_handoff.md` (this document)
2. `docs/eod_family_orchestration_design.md`
3. `docs/eod_reconciliation.md`
4. `docs/options_liquidity_eligibility_roadmap.md`

Before coding, inspect:

- `etl/eod_reconciliation.py`
- `etl/eod_quality.py`
- `etl/sources.py`
- `jobs/run_timeframe.py`
- `jobs/run_combo.py`
- `.github/scripts/run_stocks_eod_guarded.py`
- `core/guard.py`
- `core/storage.py`
- `tests/test_eod_reconciliation.py`
- `tests/test_eod_reconciliation_integration.py`
- `tests/test_stock_eod_quality.py`
- `tests/test_execution_registry_guard.py`

The next thread should verify fresh `main` and current production history before
assuming this handoff's hashes, universe count, or workflow details are unchanged.
