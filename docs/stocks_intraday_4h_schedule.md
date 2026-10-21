# Stocks Intraday 4h eligibility

The `stocks_4h` job in `.github/workflows/intraday_4h.yml` invokes
`.github/scripts/run_stocks_intraday_4h_guarded.py`. Its scheduled eligibility is
owned by `core/intraday_cadence.py`, in `America/New_York` local time.

The weekday target hours change from **09/13/17 ET** to **05/09/13/17 ET**.
The 09, 13, and 17 windows remain unchanged. The first target hour now begins at
05:00, with nominal scheduler opportunities at **05:01, 05:21, and 05:41**.
As with every other target hour, delayed starts are accepted through :19, :39,
and :59 respectively; :00, :20, and :40 are ineligible. Non-target hours and
weekends remain ineligible. This policy does not add holiday-calendar filtering.

The existing shared cron, `1,21,41 * * * 0-5`, is UTC and already reaches every
weekday 05 ET target: 09 UTC during EDT and 10 UTC during EST. New York conversion
uses `zoneinfo`, so no seasonal cron changes are required. GitHub Actions can
delay or drop scheduled invocations; this remains an opportunity-based schedule.

The runner retains the standardized registry guard, including authoritative
existing activation settings and cooldowns, the default 3.4-hour cooldown, and
success-only execution updates. Retries after success are suppressed. A late
success may suppress an early retry in the next target hour, as before. Manual
dispatch retains its existing eligibility and registry bypass. Job concurrency
continues to serialize Stocks 4h invocations.

This earlier opportunity supports preparation ahead of the stated 07:30 options
premarket expansion. It does not assert that an options session is open at 05 ET
or that providers supply new bars then. Provider availability, freshness checks,
signal calculations, and health thresholds are unchanged. Futures, other stocks
cadences, daily jobs, and ETF jobs retain their existing policies. No production
registry migration is required.
