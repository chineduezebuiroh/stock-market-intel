# Stock EOD terminal-close reconciliation contract

## `lastTrade` investigation

### Code-established

In installed yfinance 1.7.0, chart `result[0].meta` is first retained as history
metadata. `lastTrade` is **not** a Yahoo chart metadata field: yfinance adds it
after `fix_Yahoo_returning_live_separate()` finds a separate live row in the
requested interval, merges that row into the preceding W/M aggregate, and
returns the dropped row. Its `Price` is that row's raw `Close` and its `Time` is
that row's index. Consequently it is interval-request-specific evidence, not
ticker quote metadata repeated across interval requests. yfinance does not
transform `lastTrade` in `format_history_metadata`; it does transform Yahoo's
`regularMarketTime`, `firstTradeDate`, and `currentTradingPeriod` timestamps.

`regularMarketPrice` and `regularMarketTime` are ticker quote metadata.
`currentTradingPeriod` describes market periods. `dataGranularity` and `range`
describe the chart request/response, and `exchangeTimezoneName` supplies its
timezone. None of those fields alone establishes completion of a particular
coarse aggregate.

### Observed

Run 35941806060 reported identical W/M `lastTrade` dates/prices and terminal raw
closes for the seven named symbols, alongside malformed daily closes. The run's
W non-close aggregate corroboration passed while exact M corroboration did not.
The two observations were stable five minutes apart.

### Inferred and not established

When `lastTrade.Time` is session T and `lastTrade.Price` exactly matches the
returned terminal aggregate Close, the implementation path establishes that
yfinance merged session T into that specific interval aggregate. It does not
prove exchange finality, absence of later provider revisions, or absence of a
corporate-action revision. `lastTrade` may be absent when Yahoo did not emit a
separate row, so absence is not evidence of staleness. Its dropped-row time can
lead a bucket label by days; its relationship to ticker-level
`regularMarketTime` can also differ. Therefore production uses it only when the
time and price bind it to the requested aggregate and never as standalone price
consensus.

## Pipeline and evidence contract

Native 1d, 1wk, and 1mo remain separate Yahoo observations. The pipeline is:
validate each row; establish its incorporated session; reconcile eligible raw
closes; repair only malformed raw Close; rerun the normal full indicator engine;
then persist snapshots and combos. Q/Y (monthly lineage) and stock 4H are never
eligible.

A tier-1 canonical close uses two independent native D/W/M observations, both
raw/unadjusted regular-session values, both demonstrably incorporating the
target session, agreeing within half the provider display unit, with no eligible
contradiction or revision ambiguity. Tier 2 permits one native W or M source
symmetrically when its interval-specific `lastTrade` is time-bound to the target
session and price-bound to its returned raw Close; `lastTrade` is provenance,
not a second vote. This resolves the otherwise unjustified asymmetry with an
observed D Close, which production accepts from one provider observation. A
target additionally needs finite native
O/H/L/V, nonnegative volume, and `Low <= Close <= High`. Adj Close is not
repaired. The current production activation is deliberately narrower: malformed
daily Close with valid daily O/H/L/V may use eligible independently fetched W/M
evidence under either tier. Other states remain behind the PR #23 fail-closed
floor.

Volume has two deliberately separate roles. Target native Volume must be finite
and nonnegative for structural validity. Exact cross-timeframe or daily-derived
Volume agreement is not required: price-bound interval-specific `lastTrade`
provides the completion evidence, while coarse and daily volume aggregation can
legitimately differ.

## Decision matrix

| Case | Decision |
|---|---|
| D malformed; W+M eligible and agree | Repair D Close only. |
| D malformed; only W or only M has price/time-bound `lastTrade` | Tier-2 repair after target range/semantic/ambiguity gates. |
| W malformed; D+M eligible and agree | Contract permits repair, but production activation is deferred pending joint-family orchestration; currently retain only a proven compatible old row or exclude. |
| D+W malformed; only M valid but session provenance absent | Retry, then exclude. |
| D/W/M valid and agree | Use each as observed; no repair needed. |
| One eligible valid source disagrees | Conflict; retry, then exclude rather than silently choose. |
| Only one otherwise-valid source without price/time-bound coarse provenance | Retry, then exclude. |
| W+M agree without session-T evidence | Retry, then exclude. Labels/timing are insufficient. |
| Candidate outside target Low/High | Exclude. |
| Corporate-action/revision ambiguity | Retry after ambiguity clears; otherwise exclude. |

## Retention, provenance, and health

Current native fetch and reconciliation precede the existing whole-row merge.
PR #23 remains the outer safety floor. A same-label old W/M row is not considered
fresh merely because its bucket label matches; without its actual incorporated
session it cannot participate in reconciliation. The existing merge still
prevents malformed observations from overwriting a valid stored row, but this
narrow activation does not promote an unproven old row to cross-timeframe
evidence. Production therefore disables same-label retention for partial W/M
rows; daily retention remains safe because its label identifies the session.

Frames carry compact in-memory provenance: native interval, fetch time, session
and adjustment semantics, yfinance version, provider metadata, and payload hash.
Repair health rows persist target interval/label/session, status and reason, raw
and repaired close, source combination, conflicts, and tolerance. This is enough
to distinguish observed, repaired, conflicting, and excluded inputs without
adding columns to indicator parquet schemas.

The insertion point is `ingest_one`, after native fetch and before
`merge_stock_eod_window` and `apply_core`; thus repaired series are fully
recomputed and no indicator output is patched. Health output is written beneath
`data/_health/`. A future bounded family orchestrator can activate symmetric W/M
repair and explicit retry/revision gates without changing the consensus
primitive.

Supplemental evidence requests are made only for structurally valid, close-only
malformed daily rows. Each attempted symbol makes one W and one M request using
the ordinary bounded loader; neither result is persisted. A per-process default
budget of 100 symbols caps a systemic incident at 200 supplemental requests;
later symbols fail closed and are recorded as budget-exhausted. Health writes
are best-effort: failures emit `[HEALTH][WARN]` but do not invalidate correctly
processed market data.

The historical persisted CRWD D-null/W-null/M-valid shape does not retain the
interval-specific metadata needed to prove that M incorporated the D session,
so a faithful persisted-state replay still fails closed. The same shape repairs
at tier 2 only if a contemporaneous M (or, symmetrically, W) fetch supplies the
required price/time-bound `lastTrade`. Exact historical indicator values and
signal admission cannot be reconstructed from the repository's snapshot-only
artifacts; exclusion means no repaired D indicators or regained D/W/M combo.

## Weekly target investigation

In yfinance 1.7.0, `fix_Yahoo_returning_live_separate()` identifies the final two
rows as belonging to the same requested W/M interval. It treats the penultimate
row as the aggregate and the last row as the separate live row. It fills
aggregate Open from live Open only when aggregate Open is null; updates High/Low
with max/min only when the corresponding live value is non-null; replaces Close
and Adj Close with the live values; adds live Volume; then drops the live row.
The caller subsequently stores that dropped row's Close/Time as `lastTrade` and
returns the modified aggregate frame.

This establishes that the merge path ran, but `lastTrade` exposes only the live
Close and timestamp. It does not expose whether live High/Low were non-null, and
Open normally remains the period's first open. A structurally valid returned
High/Low can therefore be the pre-live aggregate values when the live extrema
were missing. Moreover, with `repair=False` and `auto_adjust=False`, a finite
live Close is explicitly assigned to aggregate Close before `lastTrade` is
synthesized; a returned null W Close paired with finite `lastTrade.Price` is an
unexpected downstream/provider inconsistency, not a normal output guaranteed by
that merge contract. Consequently `lastTrade` alone does not prove that every W
O/H/L/V field is complete through T strongly enough to repair W. Weekly-target
repair remains deferred; D repair can still use an eligible coarse Close.

Tier-1 selection always returns an observed value. It first prefers the eligible
source with finer provider-declared price precision, then the finer native
interval for equal precision because it minimizes aggregation/revision surface.
Thus equal-precision W=100.00 and M=100.01 within tolerance selects observed W
100.00. Outside-tolerance disagreement remains a conflict.

## Fetch architecture and bounded allocation

The daily cascade currently completes all D symbols before launching separate
weekly and monthly ingestion. Persisted rolling rows do not preserve a reliable
current-session observation-family identity, and partial W/M labels are
ambiguous. There is no reusable current-session D/W/M intermediate artifact.
Supplemental W/M requests are therefore correct and isolated in the short term,
but the request cap is technical debt. A later joint-family orchestrator should
fetch each native interval independently, retain one current-session observation
family in memory or a run-scoped artifact, reconcile, and only then feed the
existing independent timeframe persistence paths.

Budget allocation is neutral rather than business-ranked. Daily symbols are
ordered by SHA-256 of UTC run date plus symbol; the first 100 eligible malformed
rows in that rotating deterministic order are attempted. This removes alphabetic
preference and rotates opportunity across dates without introducing market-cap,
liquidity, or strategy preferences. Per-symbol telemetry records the date, hash,
policy and skipped identity; a run-summary row records eligible, attempted,
skipped and configured-budget counts.
