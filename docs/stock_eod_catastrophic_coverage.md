# Stock EOD catastrophic coverage guardrail v1

The operational floor is defined once as
`core.health.STOCK_EOD_CATASTROPHIC_COVERAGE_PERCENT = 90`.
Each final D/W/M snapshot is evaluated independently using unique normalized
symbols intersected with this run's normalized routed universe. The comparison
is integer-exact: `final_count * 100 >= threshold_percent * expected_count`.
Exactly 90% passes. Native-invalid counts, pre-repair rows and processing attempts
are not coverage inputs. Existing reconciliation and row-quality rules are unchanged.

## Execution seam and availability

The family runner computes/logs `[STOCK_EOD_COVERAGE]` from the final snapshots
immediately before publication, when the universe and candidate are frozen.
It deliberately does not use that result to block B2 publication/CAS or legacy
mirrors. A guardrail failure is an operational run failure, not rejection of
family authority in v1. Immutable artifacts and partial dashboard results remain
available under the existing architecture; nothing is deleted or rolled back.
Existing structural failures (including a wholly empty required snapshot) still
fail in the existing path and do not gain permission to publish.

The official profile gives its family subprocess a fresh temporary local
`--coverage-report` path. The child writes the same report only after successful
CAS, before mirror writes, so a later mirror failure can still be explained.
The profile never rereads a possibly newer `current.json` to reconstruct counts.
A missing/unreadable report fails the profile rather than sending normal signals.
Report-write errors are logged without preventing mirror writes.

The profile retains existing ETF/combo/health ordering. After successful product
health, it enforces the coverage result before ordinary signal notifications.
If product work or health fails first, that original failure remains primary;
the explicit failure notification still includes the coverage result when
available. Existing combo-stage failures still stop subsequent product work.
The standalone family CLI, without `--coverage-report`, enforces the floor after
publication and mirrors. Python `run_family` remains the acquisition/publication
function returning the telemetry path; CLI/profile callers own terminal coverage
enforcement.

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
to read existing mirrors/current combos; partial or previously retained combo
files can therefore still be visible. The explicit run failure and Telegram
attempt make the problem visible without hiding usable results. These adjacent
consumer/transactional issues are deferred; no performance optimization is included.
