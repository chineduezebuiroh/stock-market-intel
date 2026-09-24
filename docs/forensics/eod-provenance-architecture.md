# Durable stock EOD provenance architecture

This is a design investigation only. It does not change production behavior.

## Current lineage

| TF | Lineage | Fetch/window | Persistence and update |
|---|---|---|---|
| D | Yahoo `1d` -> canonicalize -> merge -> indicators -> snapshot | `2 * 260` calendar days | `data/bars/stocks_daily/<symbol>.parquet`; same label replaced |
| W | Yahoo `1wk` -> canonicalize -> merge -> indicators -> snapshot | `2 * 210 * 7` days | `data/bars/stocks_weekly/...`; current week label updated |
| M | Yahoo `1mo` -> canonicalize -> merge -> indicators -> snapshot | `2 * 250 * 30` days | `data/bars/stocks_monthly/...`; current month label updated |
| Q | fresh Yahoo `1mo` -> local `QS` aggregation -> merge -> indicators -> snapshot | `4 * 200` monthly bars requested | `data/bars/stocks_quarterly/...` |
| Y | fresh Yahoo `1mo` -> local `YS` aggregation -> merge -> indicators -> snapshot | `12 * 200` monthly bars requested | `data/bars/stocks_yearly/...` |

All Yahoo calls use `auto_adjust=False`, `actions=False`; `prepost` is omitted
(false). Provider indexes are converted to America/New_York and made naive.
Indicators run after rolling merge; snapshots select the terminal row; combos
inner-join snapshots by symbol. Persisted bars retain OHLCV, indicators, and the
interval index, but not request parameters, fetch timestamps, provider metadata,
payload identity, terminal session, partial/completion state, or lineage.

## What Yahoo exposes versus what the application knows

The DataFrame directly exposes an interval label, OHLCV/adjusted close, and an
exchange-localized index. After `history`, yfinance separately exposes metadata
such as exchange timezone and current trading-period/regular-market timestamps,
but repository normalization discards it. Neither inspected code nor metadata
establishes a provider revision ID or, for a coarse row, a per-row “last session
incorporated” field.

Application-derived facts include request/fetch time, parameters, payload hash,
the exchange-calendar expected session, and the latest daily observation. It is
reasonable—but not proof—to expect a post-close current-period W/M bar to include
that session. Fetch time alone, matching period labels, or cumulative H/L/V
changes cannot prove completion; interval updates may be asynchronous.

## Minimum provenance contract

| Field | Class | Reason |
|---|---|---|
| symbol, timeframe, interval_label | Required | Observation identity |
| provider, provider_interval | Required | Semantics/source |
| fetch_id, completed_at | Required | Groups exact observed payloads and freshness |
| fetch_started_at | Useful | Latency/overlap diagnosis |
| request parameters, session semantics, adjustment mode | Required | Comparability |
| expected_terminal_session | Required | Calendar freshness target |
| last_market_session_date | Required, nullable until proven | Reconciliation key |
| is_complete_through_session, is_partial_period | Required | Prevents stale substitution |
| canonical payload hash | Required | Identity for bytes/data observed by us |
| raw response hash | Useful | Stronger audit if safely retained |
| provider revision ID | Unnecessary unless provider supplies one | Yahoo does not expose one |
| data_quality_status | Required | Publication gate |
| observed/derived/repaired status | Required | Never conflate evidence classes |
| source lineage and repair sources | Required for derived/repaired; otherwise useful | Audit/recompute |

An execution ID is useful above fetch ID for job-wide correlation but is not
needed to identify one payload. Production-created hashes identify the exact
payload we observed; they do not claim a Yahoo revision identity.

## Proving native W/M terminal session

| Evidence | Strength | Failure mode |
|---|---|---|
| Provider per-row terminal-session metadata | Proof if ever exposed/documented | Not currently found |
| Fetch time + exchange calendar | Weak corroboration | Coarse endpoint may update late |
| Same-run daily through T | Strong corroboration, not alone proof of W/M | Independent W/M can be stale |
| Native W/M close agreement | Strong corroboration | Same stale close can repeat |
| H/L/V evolution | Weak corroboration | Session may not set extremes; zero/similar volume |
| Daily-derived W/M | Proof for the **derived** row | Does not alone prove native row |
| Native equals derived O/H/L/C and compatible volume | Strongest practical evidence | Shared upstream revisions/rounding; still application attestation |

The durable application rule should treat full native-versus-derived agreement
through expected T as sufficient production proof under an explicit policy,
while preserving that it is application-established rather than provider-stated.
Close-only agreement is useful but cannot prove incorporation: T and T-1 may
have identical closes. H/L/C agreement is stronger; volume is useful with a
declared tolerance because provider aggregation and revisions can differ.

## Architecture comparison

**A—native D/W/M:** preserves independent failure domains and coverage, but has
ambiguous coarse lineage and asynchronous-state risk. It needs the provenance
contract, derived expectations, reconciliation, and hard publication gates.

**B—D-derived W/M:** provides deterministic session lineage, calendar-correct
partial aggregation, and one revision model, but makes all EOD horizons depend
on daily correctness/history. Daily nulls, outages, gaps, volume errors, or a
local aggregation bug can eliminate every higher-timeframe signal. It would
have discarded CRWD's valuable native monthly evidence.

**C—hybrid:** retain native D/W/M and independently build read-only D-derived
W/M expectations. This preserves redundancy while adding deterministic lineage,
staleness detection, and eventual recovery. Native and derived are not wholly
independent—they share Yahoo and corporate-action revisions—but interval
endpoints and local aggregation are distinct enough to detect the incident
class. Complexity is justified if evidence classes remain separate.

## Failure-domain matrix

Legend: D=detect, R=recover candidate, L=signal loss, B=bad-signal exposure.

| Failure | A native | B derived | C hybrid |
|---|---|---|---|
| D close null/request fails | D; W/M survive, D L | all horizons L | D; native W/M survive, possible R |
| W close null/request fails | D; W L | unaffected if D valid | D; derived W R candidate |
| M close null/request fails | D; M/Q/Y L | unaffected if D valid | D; derived M R candidate |
| all native closes null | D; all L | all L if D affected | D; derived survives only if canonical D survives |
| D stale T-1 | hard to detect | correlated stale W/M; B | derived-v-native conflict D; native evidence survives |
| W/M stale T-1 | hard without lineage; B | impossible independently | derived comparison D; possible R |
| revision/split mismatch | cross-native conflict D if compared | one correlated revision | native-derived conflict D, fail closed |
| malformed O/H/L/V or volume | field validation D; interval L | can propagate upward | both paths D; unaffected path survives |
| legitimate partial period | ambiguous session | deterministic | derived lineage validates native |
| local aggregation bug | unaffected | all coarse B/L | native conflict D; fail closed |
| persistence failure | affected artifact L | downstream correlated | fetch IDs/manifests detect; alternate evidence retained |
| manual rerun/prior partial label | overwrite ambiguity | deterministic recompute | fetch lineage distinguishes observations |

## Native/derived field expectations

When both use identical eligible daily sessions and raw conventions, O/H/L/C
should agree: first open, max high, min low, final close. Volume should normally
agree as a sum, but comparison should support a documented tolerance and flag,
not silently normalize, provider differences. Full OHLCV agreement is powerful
evidence native includes T; H/L/C agreement is strong; C alone is insufficient
to prove freshness because consecutive sessions can share a close.

## CRWD and recommendation

Under A plus provenance, malformed D/W would be rejected and native M retained,
but the historical metadata is insufficient to use M to repair. Under B, the D
defect could remove D/W/M and lose independent monthly evidence. Under C, native
M survives, the D-derived path exposes its own failure, and—had lineage/full
comparison existed—M could become corroborating recovery evidence. C best
detects the anomaly, prevents corrupted indicators, preserves evidence, and
minimizes eventual signal loss.

Adopt **C, hybrid**, but do not implement repair yet:

1. Merge the current fail-closed safety floor unchanged. Same-label W/M
   whole-row retention remains only when the old row's terminal-session
   freshness is known; until metadata exists it must not be treated as recovery
   proof.
2. Separate PR: provenance schema, fetch manifests/hashes, immutable diagnostic
   evidence, and exchange-calendar expected session.
3. Separate PR: daily-derived W/M expectations stored under distinct diagnostic
   keys; native/derived validation and telemetry only.
4. Separate PR: reconciliation policy/tolerances and canonical-session status.
5. Separate PR: same-timeframe retry, then close-only repair with invariants,
   provenance, and full indicator recomputation.
6. Separate PR: alerts, dashboards/health, rollout shadow mode, and promotion.

Fetch-after-close is never itself completion proof. Run the read-only
`compare_native_derived_eod.yml` workflow repeatedly on Monday, midweek, Friday,
first-session, month-end, and boundary days; its artifact explicitly identifies
current direct-provider evidence and never writes production storage.
