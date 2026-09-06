# ADR 0001: Conservatively reduce stock/options participation thresholds

- **Status:** Accepted
- **Date:** 2026-09-05
- **Decision owners:** Production policy governance
- **Scope:** Stock/options participation scoring only

## Context and problem

The stock/options scorer first evaluates four trend, price-action, and momentum
components. Participation is the fifth component and is deliberately a restrictive
quality filter, not a mechanism for maximizing admissions. A qualifying lower **or**
middle route adds the same participation point to both directional scores. The gate
uses strict `>` comparisons.

The production policy introduced three thresholds. Strong significant volume
(`sigvol == 2`) required a ratio above 5% when upper Wyckoff history was available
and above 10% when it was unavailable. Moderate significant volume (`sigvol == 1`)
required a ratio above 25% in either upper-history state. Repository history
establishes these values and their route hierarchy, but does not establish a
statistical optimization or a more detailed original rationale. Phase 3 through
Phase 4C therefore treated them as the historical production contract.

## Empirical evidence considered

The following are findings from the completed diagnostics. They characterize the
available samples and do not establish a universally optimal policy.

### Phase 3 population

Phase 3 reconstructed the five-component production era and isolated opportunities
that met four non-participation components. In D/W/M, the existing gate was
structurally restrictive. It rejected a large majority of otherwise qualified
opportunities, and the normalized participation-ratio distributions showed the
25% MODERATE route to be especially demanding.

### Phase 3B cross-combo characterization

Phase 3B applied the same reconstruction across the supported stock/options combos
and directions. Across those cells, approximately 70–82% of otherwise
four-of-five-qualified opportunities were rejected by participation. The degree of
restriction varied by combo and direction. Evidence for the strong route without
upper history was comparatively thin. These were population and sensitivity
findings, not outcome-based threshold recommendations.

### Phase 4 outcome validation

Phase 4A established immutable entry, outcome, maturity, coverage, and corporate-
action handling contracts. Phase 4B then compared subsequent outcomes for current
passes, current blocks, and marginal admissions under predeclared counterfactual
threshold families. It used first qualified observations in episodes as the primary
unit, retained observation-level results as secondary robustness, and reported
coverage, censoring, chronological slices, corporate-action sensitivity, and
symbol-cluster bootstrap intervals. It did not select production policy.

Phase 4B found little evidence that the existing gate improves subsequent D/W/M
LONG outcomes. D/W/M SHORT supplied more evidence that participation still has
value as a quality filter. Estimates were sample- and horizon-dependent, so this is
not a claim of statistical certainty.

### Phase 4C calibration and untouched holdout

Phase 4C used a precommitted chronological design: calibration observations were on
or before the fixed cutoff, holdout observations were after it, candidate selection
used calibration aggregates only, and the resulting table was frozen before the
holdout was evaluated. D/W/M was policy-decisive; W/M/Q was exploratory because of
limited temporal breadth, and M/Q/Y was excluded as immature. The untouched D/W/M
holdout provided strong evidence that materially looser candidates can admit useful
LONG opportunities. SHORT evidence was more mixed. The design reduces selection
leakage but does not eliminate sampling uncertainty, market-regime dependence, or
common-date dependence.

## Decision

We will halve every governed production threshold while preserving its hierarchy:

| Participation route | Previous | New |
| --- | ---: | ---: |
| Strong + upper history available | 5% | 2.5% |
| Strong + upper history unavailable | 10% | 5% |
| Moderate | 25% | 12.5% |

The comparisons remain strict `>` comparisons. Lower and middle routes remain ORed,
significant-volume and upper-history routing remain unchanged, and participation
continues to add the fifth point to both scores.

This is a **governance judgment informed by the totality of Phase 3–4 evidence**, not
an optimization result. In particular, **12.5% was not in the precommitted Phase 4
candidate grid**. It is not empirically selected, Phase-4C validated, optimal, or a
best-performing threshold. It is the policy consequence of applying a consistent
50% reduction to the historical 25% value.

## Why one LONG/SHORT policy

LONG holdout evidence supports meaningful relaxation, while mixed SHORT evidence
supports retaining a meaningful quality filter. The evidence does not yet justify
the operational and governance complexity of two directional policies, especially
for thinner routes. A common conservative reduction permits prospective comparison
without premature fragmentation. Direction-specific thresholds remain a possible
future refinement if prospective evidence demonstrates materially different LONG
and SHORT requirements.

## Why a conservative 50% reduction

Some empirically evaluated candidates were more permissive, particularly in the
LONG results. Adopting the loosest supported candidate would give disproportionate
weight to one direction and finite historical samples. Halving all thresholds is a
measured first production revision: it increases access, preserves the relative
route hierarchy, leaves participation restrictive, and treats limited SHORT and
upper-unavailable evidence cautiously. The uniform reduction is an explicit policy
rule, not a fitted result.

## Consequences and tradeoffs

- More otherwise qualified stock/options signals will receive the fifth point;
  equality at each new boundary will still fail.
- The policy may capture useful LONG opportunities previously blocked, but lower
  selectivity may also admit weaker signals.
- A common policy is simple and comparable, but may be suboptimal if LONG and SHORT
  outcome relationships genuinely diverge.
- The strong upper-unavailable route changes despite thin evidence so that the
  hierarchy remains coherent; this increases the importance of monitoring it.
- Historical Phase 3/3B/4 contracts, candidate grids, and outputs remain frozen at
  the policy under which those studies were designed and run.

## Prospective monitoring and revisit criteria

This is the first calibrated production revision. A future governed review should
compare pre- and post-change periods while accounting for combo mix and market
regime, and should examine:

- admission-rate changes by combo and direction;
- realized outcome quality of newly admitted signals;
- divergence between LONG and SHORT;
- whether MODERATE 12.5% remains appropriately restrictive;
- whether strong + upper available at 2.5% is too permissive;
- whether strong + upper unavailable has accumulated adequate evidence; and
- whether direction-specific thresholds have become justified.

No automatic rollback boundary is created by this decision. Revisit the policy
through the governed review process when adequate prospective exposure and mature
outcomes exist, or sooner if a material implementation defect or clear degradation
is identified.
