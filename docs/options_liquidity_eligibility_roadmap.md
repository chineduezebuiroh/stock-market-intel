# Options-liquidity eligibility investigation roadmap

This is a design-only follow-up. It does not change the EOD reconciliation
request budget, production universes, signals, scoring, thresholds, ETFs,
notifications, or dashboards.

## Evidence to acquire first

Inventory current provider support for full option chains, bid/ask timestamps,
exchange/market-state metadata, expirations, strikes, volume, and open interest.
Run read-only captures during market hours and after close to determine quote
freshness and whether crossed, zero, stale, or missing quotes can be identified.
No eligibility rule should be activated until provider reliability and timing
are measured across liquid and sparse underlyings.

## Candidate assessment model (not policy)

For each signal family, map the upper timeframe to an expiration search window.
The existing hypotheses—long expirations near two-to-three times the upper
timeframe in days and short expirations near one times that timeframe—must be
validated and assigned explicit tolerances before use. Eligibility should likely
be keyed by `(symbol, signal_family)` rather than one global ticker boolean, so
4H/D/W and D/W/M can select different expiration regions.

Select a representative strike neighborhood relative to spot and available
strike spacing: ATM and near-ATM contracts first, then systematic distance/delta
buckets. Multiples of 5/10/20/25/100 may be recorded as candidate liquidity
landmarks, but must not be hard-coded without evidence that they add information
beyond the exchange's strike grid. Compute relative spread only for valid quotes:

```text
(ask - bid) / ((ask + bid) / 2)
```

The historical roughly-10% heuristic is an investigation hypothesis, not a
production threshold. Define explicit treatment for zero bid/ask, crossed
markets, stale timestamps, cheap-option denominator effects, missing contracts,
deep ITM/OTM contracts, sparse strikes, and multiple expirations. Evaluate
spread alongside—not yet weighted with—open interest, contract volume,
underlying dollar volume, strike density, and quote freshness.

## Proposed architectural boundary

```text
raw provider option chain + quote provenance
    -> normalized options-liquidity assessment
    -> signal-family eligibility artifact
    -> existing scoring/admission pipeline
```

The assessment should preserve raw quotes and emit reasons/coverage rather than
silently removing symbols. A shadow study should compare eligibility outcomes by
family and time of day before any pre-scoring gate is proposed.

This business-actionability investigation is independent of EOD repair request
allocation. Any eventual reduction in scoring workload is secondary and must not
be used to justify or prioritize the reconciliation budget.
