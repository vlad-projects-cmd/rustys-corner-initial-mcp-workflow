# rustys-corner-initial-mcp-workflow

# Premier League Gameweek Outlook (Baseline Predictor)

A reproducible, explainable baseline model that generates a weekly Premier League "Gameweek Outlook":
- lists all fixtures in the selected gameweek
- computes rolling team strength from historical results (goals for/against)
- predicts scoreline probabilities using a Poisson model
- derives W/D/L probabilities from the scoreline grid
- outputs a Markdown report (and optionally JSON) for publishing.

## Scope (Phase 1)
We only generate a Monday "Gameweek Outlook" for the upcoming Premier League fixtures.

**Not in scope yet:**
- injuries/suspensions
- 24h pre-match updates
- post-match reviews & evaluation metrics
- shots/corners modeling
- automated publishing

## Inputs
- `season`: e.g. `2025`
- `competition`: Premier League
- `gameweek`: integer (matchday)
- `as_of`: timestamp used to freeze data for reproducibility

## Outputs (Gameweek Outlook)
For each fixture:
- Match: Home vs Away
- Kickoff date/time (UTC)
- Rolling averages (last N matches, default N=5):
  - goals_for_avg
  - goals_against_avg
  - optional: home/away weighted splits (v1: minimal)
- Expected goals proxy:
  - lambda_home
  - lambda_away
- W/D/L probabilities:
  - P(HomeWin), P(Draw), P(AwayWin)
- Most likely scorelines (top 3–5) from a 0–5 goal grid
- Short notes (data-driven; no betting language)

### Example output block
Match: Arsenal vs Everton
Kickoff: 2026-02-07 15:00 UTC

Rolling (last 5):
Arsenal GF/GA: 1.80 / 0.80
Everton GF/GA: 1.00 / 1.60

Expected goals:
Arsenal λ: 1.65
Everton λ: 0.92

Outcome probabilities:
Home 56% | Draw 25% | Away 19%

Top scorelines:
1-0 (14%), 2-0 (12%), 2-1 (11%)

Notes:
- Arsenal strong home attack signal (GF trend).
- Everton concede above league avg away (GA trend).

## Method (Baseline)
1) Build rolling team metrics from prior matches only (no leakage).
2) Convert team metrics into expected goals (λ_home, λ_away).
3) Generate Poisson scoreline probabilities for 0..5 goals each side.
4) Sum the grid into W/D/L probabilities.
5) Render a gameweek report.

## Repo structure
- `src/`
  - `cli.py`              # command line entrypoint
  - `mcp_server.py`       # MCP server (exposes tools for AI agents)
  - `fetch.py`            # API client + caching
  - `features.py`         # rolling metrics, league averages
  - `model_poisson.py`    # Poisson grid + W/D/L aggregation
  - `model_strength.py`   # attack/defence strength model (gradient descent)
  - `render.py`           # markdown/json output + prediction orchestration
  - `evaluate.py`         # predictions vs actuals scoring
  - `performance.py`      # cumulative performance artifacts
  - `data_loader.py`      # shared data loading utilities
  - `metrics.py`          # shared evaluation metrics + plotting
- `tests/`                # pytest test suite
- `data/`
  - `raw/`                # cached API responses
  - `processed/`          # normalized per-season CSVs
  - `curated/`            # merged multi-season datasets
  - `predictions/`        # saved prediction JSONs/CSVs
  - `evaluation/`         # evaluation ledger

## Setup

### Prerequisites
- Python 3.10+
- A free API token from [football-data.org](https://www.football-data.org)

### Install dependencies

```bash
pip install -r requirements.txt
# or for development:
pip install -e ".[dev]"
```

### Set your API token

```bash
export FOOTBALL_DATA_TOKEN=your_token_here
```

## CLI Usage

All commands are run from the project root:

```bash
cd rustys-corner-initial-mcp-workflow
```

### List supported leagues

```bash
python -m src.cli leagues
```

Output:
```
Code            ID     League                    Country         Season
---------------------------------------------------------------------------
pl              2021   Premier League            England         split (Aug-May)
championship    2016   Championship              England         split (Aug-May)
laliga          2014   Primera Division          Spain           split (Aug-May)
bundesliga      2002   Bundesliga                Germany         split (Aug-May)
seriea          2019   Serie A                   Italy           split (Aug-May)
ligue1          2015   Ligue 1                   France          split (Aug-May)
eredivisie      2003   Eredivisie                Netherlands     split (Aug-May)
primeira        2017   Primeira Liga             Portugal        split (Aug-May)
brasileirao     2013   Campeonato Brasileiro     Brazil          calendar (Jan-Dec)
ucl             2001   UEFA Champions League     Europe          split (Aug-May)
euro            2018   European Championship     Europe          split (Aug-May)
libertadores    2152   Copa Libertadores         South America   calendar (Jan-Dec)
worldcup        2000   FIFA World Cup            World           split (Aug-May)
```

### Fetch match data

```bash
# Premier League (default)
python -m src.cli fetch --season 2025

# Any league using --league code
python -m src.cli fetch --season 2025 --league laliga
python -m src.cli fetch --season 2025 --league bundesliga
python -m src.cli fetch --season 2026 --league brasileirao

# Multiple seasons
python -m src.cli fetch --seasons 2023 2024 2025 --league pl

# Inclusive range
python -m src.cli fetch --season-range 2022 2025 --league championship

# Force re-download (ignore cache)
python -m src.cli fetch --season 2025 --league pl --force-refresh
```

### Curate (merge seasons into one dataset)

```bash
python -m src.cli curate --seasons 2023 2024 2025 --league laliga
```

### Generate gameweek outlook (predictions)

```bash
# Rolling model (default) - Premier League
python -m src.cli outlook --season 2025 --gameweek 3 --save-predictions

# Bundesliga
python -m src.cli outlook --season 2025 --gameweek 20 --league bundesliga --save-predictions

# Brazilian Serie A with strength model (calendar-year season)
python -m src.cli outlook --season 2026 --gameweek 10 --league brasileirao --model strength --save-predictions

# With Dixon-Coles correction and previous seasons for training
python -m src.cli outlook --season 2025 --gameweek 5 --league pl \
  --model strength --dc-rho -0.10 --include-prev-seasons 2 --save-predictions
```

### Evaluate predictions vs actual results

```bash
python -m src.cli evaluate --season 2025 --gameweek 3 --league pl --append --refresh-cumulative
python -m src.cli evaluate --season 2025 --gameweek 20 --league bundesliga --append
```

### Regenerate performance artifacts

```bash
python -m src.cli performance --season 2025
```

### Verify CLI help

```bash
python -m src.cli --help
python -m src.cli fetch --help
python -m src.cli outlook --help
```

### Notes on season numbering

- **Split-year leagues** (PL, La Liga, etc.): `--season 2025` = the 2025/26 season
- **Calendar-year leagues** (Allsvenskan, Eliteserien, Veikkausliiga): `--season 2026` = the 2026 season (Jan-Dec)

## Running Tests

```bash
python -m pytest tests/ -v
```
