# Terminal-session close pressure test

This note distinguishes the behavior on `main` before the stock-EOD-quality PR,
the PR behavior, and a possible later recovery design. It is analysis only; it
does not change ingestion, scoring, ETF, or dashboard behavior.

## Cadence and duplicate relevance

The EOD workflow is scheduled at 21:00, 22:00, and 23:00 UTC each weekday, but
the execution registry normally admits one run per 20-hour window. Manual
dispatch bypasses that registry. The admitted run downloads an overlapping
history on every execution: approximately twice the configured window (daily
520 calendar days, weekly 2,940, monthly 15,000), then merges by interval label.

On the first ordinary successful run after session T, yesterday's object should
not already contain daily T. It can already contain a weekly/monthly T-labelled
partial period only when the provider uses the same period label across updates;
that is not evidence that the old version incorporated session T. Same-label
duplicates are nevertheless normal on later days within a week/month, scheduled
or manual reruns, recovery after a partial job, and provider revisions.

There is no immutable evidence that a valid CRWD daily or weekly September 21
row existed before the malformed fetch. Consequently, the PR's exact-timestamp
retention is duplicate-refresh resilience, not demonstrated incident recovery.
Absent such an old row, the PR would exclude CRWD.

The PR retains the **entire old row**. Given old `(100,105,99,104,1000)` and new
`(100,107,99,NULL,1500)`, it retains the old high, close, and volume; it does not
produce `(100,107,99,104,1500)`.

## Meaning of raw terminal close

For regular-hours, unadjusted OHLC bars, Close is the price of the final market
observation incorporated into the aggregation period. Therefore, if D/W/M bars
are independently proven to contain data through the same final regular-hours
session, their terminal raw closes should agree within deterministic provider
precision. Monday versus Friday, holidays, shortened sessions, and week/month
boundaries alter the aggregation's open/high/low/volume and label, not that
terminal event. Corporate actions can change adjusted histories and provider
revisions, so this contract is for raw Close with a common provider revision
semantics, not adjusted close.

| D through | W through | M through | Equality expected? | Recovery candidate? |
|---|---|---|---|---|
| same session T | T | T | Yes | Yes, after freshness/session proof |
| T | T-1 | T | No | No; W is stale |
| T | T | prior month end | No | No; M is stale |
| Friday T | completed Friday T | month-to-date T | Yes | Yes |
| first session of month T | partial week T | new partial month T | Yes | Yes |
| month-end T | week containing T through T | completed month T | Yes | Yes |
| any labels that differ but all incorporate T | T | T | Yes | Labels alone do not decide |
| unknown incorporated session | unknown | unknown | Not provable | Validation only/fail closed |

The unsafe boundary is different information sets—not calendar aggregation.
Provider interval labels cannot by themselves prove the information sets.
Required evidence is an exchange-calendar-normalized
`last_market_session_date`, provider fetch time, interval and label, partial or
complete status, regular-hours/raw-price semantics, and a positive assertion
that the payload is complete through that session.

## Timeframe hierarchy

| Family | Same-session raw close expected? | Validation safe? | Recovery safe? |
|---|---|---|---|
| D/W, D/M, W/M | Yes | With terminal-session proof | Conditionally |
| W/M/Q, M/Q | Yes | With proof | Conditionally |
| M/Q/Y, D/W/M/Q/Y | Yes | With proof | Conditionally |
| Extended-hours 4H versus EOD | Not generally | No | No |

Quarterly and yearly stock bars exist and are locally aggregated from directly
fetched monthly bars using last close; they are production-used by WMQ/MQY
rollups rather than independently fetched quarterly/yearly observations.
Consequently Q/Y close is semantically compatible, but it is not independent
corroboration from monthly: it inherits the same monthly payload and freshness.
Crossing quarter or year boundaries does not change terminal-close equality.

Stock 4H is different. It downloads 60-minute extended-hours data, patches the
current hour from shorter intervals, and resamples on a 17:00 ET anchor into
17:00/21:00/01:00/05:00/09:00/13:00 buckets. The terminal 4H bar may include
post-market observations, and its close is not the official 16:00 close. A
16:00 observation inside the 13:00 bucket could be separately preserved as a
session-close observation, but the current 4H aggregate does not expose that
provenance. Keep 4H outside the EOD equivalence/recovery family.

## Deriving the terminal session

Daily can derive `last_market_session_date` from its exchange-local bar date,
subject to verifying the fetch occurred after the session and the row is not a
premature partial. Coarser provider bars cannot derive it reliably from their
period-start labels alone. The robust design is to retain provider metadata or
derive the session from the latest valid daily observation fetched in the same
run, validate it against an NYSE/Nasdaq calendar, and stamp each interval only
after confirming that its terminal close reflects that session. O/H/L/V
evolution is corroboration, not proof.

Each observation should carry `interval`, `interval_label`,
`last_market_session_date`, `provider_fetch_time`, `is_partial_period`,
`is_complete_through_session`, regular/extended session semantics, adjustment
mode, and `data_quality_status`. Q/Y inherit session and lineage from monthly.

## Conditional recovery contract

No counterexample was found once all proposed preconditions are genuinely
satisfied **and** the following are added: the same instrument/listing/currency,
same raw-price convention and provider revision epoch, target O/H/L/V validity,
`low <= canonical_close <= high`, `low <= open <= high`, nonnegative volume,
and no split/corporate-action ambiguity. Under that strengthened contract,
cross-timeframe close-only recovery is technically sound: Close represents the
shared terminal event, while O/H/L/V remain target-period aggregates. This is
not arbitrary field coalescing.

A concrete apparent counterexample is a split effective at T where one interval
has incorporated a provider backfill/revision and another has not. It disappears
only if “compatible raw semantics/provider revision epoch” is actually proven;
session-date equality alone is insufficient. Likewise, a source close outside
target `[low, high]` proves inconsistent vintages and must reject recovery.

Validation, reconciliation, and repair are separate operations:

1. **Validation** compares eligible, proven-same-session raw closes.
2. **Reconciliation** selects a canonical close only when valid evidence agrees
   within a declared absolute-plus-relative tolerance.
3. **Repair** writes only that close into a malformed target before recomputing
   every target indicator, with immutable provenance.

Evidence confidence is strongest with two independently fetched eligible
intervals agreeing. Q/Y derived from M do not increase source independence. A
single coarser source can be sufficient only with strong session-completeness,
freshness, revision, and range proof; source count alone is not authority.

## Recommended ordering

1. Use a valid fetched target row.
2. Refetch the same timeframe; use it if valid.
3. Establish target and sources' terminal-session/freshness metadata.
4. Reconcile a canonical raw close from compatible evidence; disagreement
   fails closed.
5. If target O/H/L/V and OHLC range/order invariants pass, repair only Close,
   record source(s), session, fetches, original state, tolerance, and confidence,
   then recompute all target indicators.
6. Otherwise consider an old same-timeframe/same-label whole row only if its
   incorporated session and revision freshness are proven; mark it retained.
7. Otherwise exclude/fail visibly.

Canonical recovery should precede stale whole-row retention when the new
target O/H/L/V are proven current and canonical-close evidence is stronger.
Same-timeframe refetch remains first because it avoids reconciliation. This is
a generic EOD-family contract, not monthly-to-daily hardcoding.

## CRWD conclusion

The production artifacts establish one valid monthly close but do not record a
normalized terminal session, provider fetch time/revision, or completion proof.
They therefore do not prove that production monthly was current through
September 21 under the proposed contract. `249.350006` is within the
reconstructed daily range `[230.850006, 250.309998]`; the checked-in evidence
does not expose the exact persisted production daily high/low for an independent
range check. If production had recorded the missing metadata and range checks
passed, close-only repair followed by full recomputation would have been
technically defensible. With the evidence actually retained, CRWD must be
excluded rather than retrospectively repaired, and no exact signal is claimed.

## Empirical limitation and PR recommendation

A direct Yahoo sample was attempted for AAPL, MSFT, and SPY at `1d`, `1wk`, and
`1mo`, but this environment's proxy rejected Yahoo with HTTP 403, so no direct
interval empirical result is claimed. The existing CRWD daily-derived
reconstruction supports the invariant but is not independent interval evidence.

The current PR should remain unchanged for this merge: it is a valid safety
floor and the repository cannot yet prove the metadata needed for repair.
Follow it with a revision that first adds read-only lineage/session metadata and
direct-interval empirical telemetry, then adds retry and conditional canonical
close recovery behind an observable policy. Do not replace the present gate
with an unproven monthly fallback.
