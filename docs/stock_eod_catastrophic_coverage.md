# Stock EOD catastrophic coverage guardrail v1

The operational floor is defined once as
`core.health.STOCK_EOD_CATASTROPHIC_COVERAGE_PERCENT = 90`.
Each final D/W/M snapshot is evaluated independently using unique normalized
symbols intersected with this run's normalized routed universe. The comparison
is integer-exact: `final_count * 100 >= threshold_percent * expected_count`.
Exactly 90% passes. Native-invalid counts, pre-repair rows and processing attempts
are not coverage inputs. Existing reconciliation and row-quality rules are unchanged.

## Execution seam and availability

The family runner computes/logs `[STOCK_EOD_COVERAGE]` from final snapshots,
writes the invocation-scoped local coverage report, and enforces the floor before
immutable generation staging, pointer CAS, or legacy snapshot writes. Health
telemetry and acquisition diagnostics remain available on failure. Rolling-bar
writes made during acquisition/processing are not rolled back.

The official profile supplies a fresh temporary `--coverage-report` path. Both
family and profile enforce coverage; the profile checks immediately after the
family returns, before ETF/combo stages. A failed family subprocess stops those
stages. Missing/unreadable coverage reports fail closed before dependent work;
report-write failure in the family also stops publication.

Weekly and daily ETF stages are mandatory, but configured ETFs are not a
per-signal required universe. Isolated empty acquisitions/merges are logged and
omitted from the new snapshot; a wholly empty universe or snapshot fails the
stage. Processing and persistence exceptions remain fatal. No new ETF coverage
percentage is imposed. Transient
`eod_provenance` attrs are detached from both ETF merge inputs before indicators
and persistence; unrelated attrs remain, with new values taking precedence.

The existing overlay contract uses primary/optional-secondary mappings, left
joins and available-data aggregation. An omission leaves only its dependent
stocks with less overlay evidence. There is no documented minimum ETF coverage
or per-stock overlay-readiness gate. In particular, all-absent ETF scores bypass
ETF guards in the existing scorer; an ingestion success alone does not prove
that every signal has ETF confirmation. Also, `load_etf_trend_scores` prefers an
existing cached score object over the current snapshot, and ingestion does not
refresh that cache. Before recovery, verify cache freshness/coherence or address
it in a separately reviewed bounded change. These gaps are not silently changed
into new signal/completeness policies by this hotfix.

Stock-family technical publication is not end-to-end trading-signal readiness.
ETF or combo failure still propagates, prevents normal signal notifications and
prevents registry-success marking. Stock publication and subsequent ETF/combo
writes are not one transaction: a later failure can leave a valid stock family
published while products remain stale or partially written. A separate governed
candidate/readiness promotion contract is required for full-cycle atomicity.

## Failure notification

One explicit failure-notification attempt uses the existing Stock EOD Telegram
route, independent of signal presence or health success. It includes this
invocation's family ID, expected count, D/W/M counts/coverage, floor, failing
timeframes, missing shortlist support where available, product health details,
and the original failure. Plain text avoids Markdown parsing failures from error
messages. Full details are logged; overly long Telegram text is bounded.

A notification error is logged and cannot replace the original run exception.
A failed profile does not reach normal registry-success marking. Successful runs
retain their two existing signal notifications and default Markdown formatting.
Shortlist completeness remains a product requirement; options health remains
nonemptiness. Neither is folded into the family coverage calculation.

## Deliberately deferred

This v1 does not promise notification across process termination, runner timeout,
or external Telegram outage; it guarantees an attempt for caught run failures,
not delivery. No retry/ledger or exactly-once machinery is introduced.
There is no accepted/candidate pointer split, dashboard warning/selector,
product framework, relative policy, or semantic-state model. Streamlit continues
to read existing mirrors/current combos; previously retained combo
files can therefore still be visible after a failed product stage. These adjacent
consumer/transactional issues are deferred; no performance optimization is included.
