# Data Sources and Governance

This document records why each source exists in the system. Re-check provider
terms before a public deployment because access tiers and conditions can change.

## Selected sources

### Football-Data.co.uk — primary historical source

- EPL provider code: `E0`.
- La Liga provider code: `SP1`.
- URL pattern: `https://www.football-data.co.uk/mmz4281/{SEASON}/{CODE}.csv`.
- Example season code: `2024-25` becomes `2425`.
- Includes full-time/half-time results, selected match statistics, and bookmaker
  odds depending on league and season.
- No API key is required.

Why selected: long history, directly downloadable CSV files, useful match odds,
and a low-friction reproducible backtesting workflow.

Risk controls:

- retain raw files locally but do not commit them;
- validate required columns on every load;
- select odds columns by an explicit priority list;
- normalize implied probabilities to remove bookmaker overround;
- record season and source with every transformed row.

Official pages:

- [England results and odds](https://www.football-data.co.uk/englandm.php)
- [Spain results and odds](https://www.football-data.co.uk/spainm.php)
- [Column notes](https://www.football-data.co.uk/notes.txt)

### football-data.org v4 — live fixture source

The free tier lists both Premier League and La Liga. The adapter uses `PL` and
`PD` respectively and authenticates through the `X-Auth-Token` header.

Why selected: clean official API interface for current results and scheduled
fixtures, which historical CSVs do not always provide consistently.

Operational controls:

- keep `FOOTBALL_DATA_API_TOKEN` outside code and Git;
- cache responses and respect the documented free-tier rate limit;
- display the required “Data provided by football-data.org” attribution in a
  public application;
- keep provider match IDs for reconciliation and de-duplication.

Official pages:

- [Free competition coverage](https://www.football-data.org/coverage)
- [API reference](https://www.football-data.org/documentation/api)
- [Attribution FAQ](https://www.football-data.org/documentation/faq)

### OpenFootball — CC0 fallback

OpenFootball publishes English and Spanish league results under CC0. It is a
good continuity fallback and a useful source for testing provider abstraction,
but it has fewer modeling fields than Football-Data.co.uk. Its adapter is a
planned extension and is not part of the current MVP.

- [England repository](https://github.com/openfootball/england)
- [Spain repositories](https://github.com/openfootball)
- [Project documentation](https://openfootball.github.io/)

### StatsBomb Open Data — event-level extension

StatsBomb provides selected matches with lineups, events, and some 360 data.
This enables later xG, shot-map, passing-network, xT, and player-action modules.
It is not the primary forecasting source because open coverage is selective,
and the event-data module is not implemented in the current MVP.

Public analysis must identify StatsBomb as the source and follow its published
logo/attribution instructions.

- [StatsBomb Open Data](https://github.com/hudl/open-data)

## Sources deliberately excluded from the MVP

- Unofficial scraping from FBref, Understat, or Transfermarkt: brittle and
  dependent on site terms and HTML changes.
- Kaggle mirrors as the system of record: provenance, update frequency, and
  redistribution rights vary by dataset.
- Paid feeds: unnecessary for proving the architecture and modeling discipline.

They may be added only through a separate adapter after licensing and data
quality are reviewed.
