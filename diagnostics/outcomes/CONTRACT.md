# Phase 4A outcome data and regime contract

This contract is deliberately read-only. It does not change scoring, participation
thresholds, indicators, schedules, alerts, or production storage.

## Truth boundaries

* **Entry-state truth** is copied unchanged from the canonical immutable
  `data/combo_history/stocks/<combo>/combo_<combo>_asof=<UTC>.parquet` object selected
  by the validated Phase 3B reconstruction. `lower_close` is the entry close;
  `lower_date` is the entry market date; the object key and execution timestamp are
  mandatory provenance.
* **Outcome-price truth** is a separate, explicitly revisable observation sourced
  from `data/bars/stocks_daily/<SYMBOL>.parquet`. It must never replace an entry
  value. The outcome artifact records its key and retrieval/as-of time.
* **Engine truth** is assigned by `engine_regimes.json`. Supported modern rows use
  combo-specific time boundaries. Earlier rows require the immutable artifact's
  schema/score-derived `observed_logic_era`; unknown/mixed rows fail closed.
* **Policy truth** has two named modes. `HISTORICAL_PRODUCTION` accepts no overrides.
  `COUNTERFACTUAL_PARTICIPATION` may vary only participation parameters and stores
  the unchanged `EntryState`; it cannot recompute indicators or entry prices.

## Outcome-price source reconnaissance

| Candidate | Convention and granularity | Retention / revision | Semantics and fitness |
|---|---|---|---|
| Per-symbol daily bars (recommended with limitations) | `data/bars/stocks_daily/<SYMBOL>.parquet`; one daily OHLC, raw close, adjusted close, volume row per market-date index | Fixed rolling window of 260 rows. Every ingest re-downloads and duplicate dates keep the newest provider row, so history is rolling and revisable rather than append-only. An overwritten row has no application-level correction history. | Yahoo is requested with `auto_adjust=False` and actions disabled. Raw OHLC supports terminal close and high/low excursions; `adj_close` is separate. Corporate actions are not stored and no split/dividend adjustment is applied to OHLC. Indexes are converted from provider timezone to America/New_York then made naive. Current/incomplete daily bars are not explicitly excluded. Removed-universe symbols are no longer refreshed and may eventually be absent, although an existing object survives unless separately purged. At roughly one trading year, this supports D/W/M horizons and only part of W/M/Q; it cannot support 2x/3x M/Q/Y and generally cannot resolve 2x/3x W/M/Q from early-2026 entries at a September-2026 snapshot. |
| Single-timeframe daily snapshot | `data/snapshot_stocks_daily.parquet`; latest row per currently ingested symbol | Replaced on each run; no history | OHLC exists but only the latest row. Not an outcome history and loses removed-universe symbols. |
| Higher-timeframe rolling bars/snapshots | `data/bars/stocks_{weekly,monthly,quarterly,yearly}/<SYMBOL>.parquet` and corresponding snapshots | Fixed rolling/revisable bars; snapshots latest-only | Coarser OHLC cannot deterministically resolve calendar targets or daily-window excursions. Quarterly/yearly are derived from monthly; not preferred. |
| Immutable combo history | `data/combo_history/stocks/<combo>/combo_<combo>_asof=<UTC>.parquet`; point-in-time lower/middle/upper values | Append-only by design and canonical for entry state | Excellent entry provenance, but universe snapshots omit a symbol after removal and are therefore insufficient for forward prices. Repeated execution artifacts are observations, not a complete daily tape. |
| S3 object versions | Version IDs for a rolling daily key, if bucket versioning and IAM permit | Potential provider-revision evidence, not a documented canonical dataset | Existing forensic tooling can list versions, but availability/retention is not guaranteed and versions are not exposed by the normal storage abstraction. Useful for audit, not the baseline source contract. |

The rolling daily store is the best **existing** outcome source because it is the
only repository store with daily high, low, and close. Its values are outcome-price
truth as retrieved, not immutable entry truth. The CAKE 2026-07-08 forensic finding
is direct evidence that a later rolling OHLC value can differ from the immutable
production artifact. Raw (unadjusted) outcomes can also be distorted by splits or
other corporate actions. Phase 4 must flag windows with suspected corporate actions
before interpretation; the present store contains no actions table with which to
repair them.

Local checkout data contain snapshots and old sample combo histories but no
`data/bars/stocks_daily` directory, and this environment has no configured AWS
credentials. Therefore no trustworthy cross-combo smoke artifact is emitted in
Phase 4A. A smoke run is blocked until read-only S3 credentials are available; it
must not fall back to a new external provider.

## Horizon and alignment convention

Horizon targets are exact calendar offsets from `entry_market_date`, based on the
upper timeframe convention already used by project loaders (month 30, quarter 90,
year 365 days): D/W/M = 30/60/90, W/M/Q = 90/180/270, and M/Q/Y =
365/730/1095 calendar days. All horizons are calculated for both directions. The 1x
horizon is SHORT-primary; 2x and 3x are LONG-primary.

The exit is the first valid symbol-specific daily market close on or after the
target, bounded to seven calendar days inclusive. This is the sole permitted
post-target alignment and is recorded as actual elapsed calendar days. No earlier
session is substituted. Trading sessions are the number of observed daily bars
strictly after entry through and including exit. Exchange-calendar assumptions are
not invented: sessions are determinable from the symbol's actual valid bars.

## Maturity and censoring

* `MATURE`: a valid exit exists in `[target, target + 7 days]` and outcome values can
  be computed.
* `IMMATURE`: the price dataset's explicit as-of date is before the target, or is
  inside the alignment window with no exit yet. Wall-clock time is not consulted.
* `MISSING_PRICE_HISTORY`: no daily history exists for the symbol.
* `UNRESOLVABLE_TARGET_DATE`: the data as-of date has passed the full alignment
  window but no valid bar resolves the target.

Null, never zero, is stored for censored outcome values. Reports group maturity by
combo, direction, horizon, and entry month/date. Long M/Q/Y horizons are expected to
remain immature until sufficient actual data accrue.

## Metrics

The future window excludes the entry bar and includes all valid daily bars through
the resolved exit. Prices must be positive.

* LONG return = `exit_close / entry_close - 1`; MFE = maximum
  `high / entry_close - 1`; MAE = minimum `low / entry_close - 1`.
* SHORT return = `entry_close / exit_close - 1`; MFE = maximum
  `entry_close / low - 1`; MAE = minimum `entry_close / high - 1`.

Thus favorable excursion is normally positive and adverse excursion normally
negative for either direction. No good/bad label or arbitrary outcome threshold is
defined.

## Canonical table

`OUTCOME_COLUMNS` is the ordered observation-outcome schema. It contains immutable
entry provenance and state, historical scores and participation route, named policy
mode and evaluated route, engine regime, target/resolution/censoring fields, all
return/excursion values and elapsed measures, plus outcome source key and retrieval
as-of. There is one row per immutable observation, direction, and horizon.

## Next implementation step

With read-only S3 access, implement a bounded orchestrator that consumes the
canonical Phase 3B rows, loads each unique rolling daily key once, rejects/flags
corporate-action windows, materializes all censored rows, emits maturity summaries,
and writes only local smoke artifacts. Full threshold calibration remains a later
step.

## Phase 4A.1 coverage audit

The manual `Analyze Stock Options Participation` workflow accepts
`phase=phase4a_coverage`, requires `validation_scope=cross_combo_phase3b`, and
requires blank date bounds. It first regenerates and validates all three canonical
populations, then uses only S3 `GetObject` for daily outcome bars. Results are local
files under `diagnostic_artifacts/phase4a_coverage` inside the ephemeral workflow
artifact. The audit distinguishes terminal coverage from complete high/low path
coverage and never replaces immutable entry OHLC.

Potential corporate-action contamination is descriptive. An event is flagged when
adjacent valid `adj_close / close` ratios change by more than 5%. A corroborating
split-like condition also flags a raw close move above 40% when the adjustment ratio
moves in the opposite direction by more than 20%. Flags remain separate from
coverage status and do not adjust or exclude prices.
