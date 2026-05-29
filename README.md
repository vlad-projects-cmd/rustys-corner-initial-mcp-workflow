# rustys-corner-initial-mcp-workflow

# Football Gameweek Outlook (Multi-Model Predictor)

A reproducible, explainable prediction system that generates football "Gameweek Outlook" reports:
- supports 13 leagues (Premier League, La Liga, Bundesliga, Serie A, etc.)
- four prediction models: rolling averages, attack/defence strength, Elo ratings, ensemble
- predicts scoreline probabilities using Poisson distribution
- derives W/D/L probabilities from the scoreline grid
- saves predictions per-model for comparison and backtesting
- evaluates predictions against actual results (Brier score, log-loss, accuracy)

## Models

| Model | Description | Key params |
|-------|-------------|------------|
| **rolling** | Rolling GF/GA averages with home/away venue splits + Poisson grid | `--window`, `--venue-weight` |
| **strength** | Attack/defence ratings via gradient descent with time decay + Poisson grid | `--half-life-days`, `--l2`, `--lr`, `--max-iter` |
| **elo** | Elo rating system with home advantage, K-factor, goal-diff multiplier | `--elo-k`, `--elo-home-advantage`, `--elo-season-carryover` |
| **ensemble** | Weighted average of all three models above | `--ensemble-weight-*` (via MCP/code) |

All models only use data from **before** the predicted gameweek (no data leakage).

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

### Fetch match data

```bash
# Premier League (default)
python -m src.cli fetch --season 2025

# Any league using --league code
python -m src.cli fetch --season 2025 --league laliga
python -m src.cli fetch --season 2025 --league bundesliga

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

---

## Generating Predictions

```bash
# Rolling model (default)
python -m src.cli outlook --season 2025 --gameweek 10 --save-predictions

# Strength model
python -m src.cli outlook --season 2025 --gameweek 10 --model strength --save-predictions

# Elo model
python -m src.cli outlook --season 2025 --gameweek 10 --model elo --save-predictions

# Ensemble (combines all three)
python -m src.cli outlook --season 2025 --gameweek 10 --model ensemble --save-predictions
```

Each `--save-predictions` call writes to a **model-specific file** so they don't overwrite each other:
```
data/predictions/season_2025/gameweek_10_rolling_w5_vw0.5_g5.json
data/predictions/season_2025/gameweek_10_elo_k30_ha65_co0.6.json
data/predictions/season_2025/gameweek_10_strength_hl60_l21.00_lr0.050_it250_g5.json
data/predictions/season_2025/gameweek_10_ensemble_r35%_s35%_e30%_g5.json
```

### List saved models for a gameweek

```bash
python -m src.cli models --season 2025 --gameweek 10
```

---

## Evaluating Predictions vs Actual Results

Once the gameweek has been played, evaluate each model:

```bash
# Evaluate a specific model
python -m src.cli evaluate --season 2025 --gameweek 10 \
  --model-id "rolling_w5_vw0.5_g5" --append --refresh-cumulative

python -m src.cli evaluate --season 2025 --gameweek 10 \
  --model-id "elo_k30_ha65_co0.6" --append

python -m src.cli evaluate --season 2025 --gameweek 10 \
  --model-id "strength_hl60_l21.00_lr0.050_it250_g5" --append

python -m src.cli evaluate --season 2025 --gameweek 10 \
  --model-id "ensemble_r35%_s35%_e30%_g5" --append
```

Flags:
- `--append` — adds results to the cumulative ledger (`data/evaluation/all_matches.csv`)
- `--refresh-cumulative` — regenerates performance summary reports

This produces per-model review reports in `reports/` with:
- Outcome accuracy (correct H/D/A picks)
- Brier score (probability calibration, lower = better)
- Log-loss (penalizes confident wrong predictions, lower = better)
- Goals MAE (expected goals vs actual)

### View cumulative performance

```bash
python -m src.cli performance --season 2025
```

---

## Tweaking Model Parameters

### Rolling model

```bash
# Larger rolling window (more stable, less reactive)
python -m src.cli outlook --season 2025 --gameweek 10 --model rolling \
  --window 8 --save-predictions

# Smaller window (more reactive to recent form)
python -m src.cli outlook --season 2025 --gameweek 10 --model rolling \
  --window 3 --save-predictions

# More weight on venue-specific stats (1.0 = home/away only, 0.0 = overall only)
python -m src.cli outlook --season 2025 --gameweek 10 --model rolling \
  --venue-weight 0.8 --save-predictions

# Less venue weight (treat home/away the same)
python -m src.cli outlook --season 2025 --gameweek 10 --model rolling \
  --venue-weight 0.2 --save-predictions
```

### Strength model

```bash
# Shorter memory (recent form matters more)
python -m src.cli outlook --season 2025 --gameweek 10 --model strength \
  --half-life-days 30 --save-predictions

# Longer memory (more stable ratings)
python -m src.cli outlook --season 2025 --gameweek 10 --model strength \
  --half-life-days 120 --save-predictions

# Stronger regularization (pull ratings toward league average)
python -m src.cli outlook --season 2025 --gameweek 10 --model strength \
  --l2 2.0 --save-predictions

# Include previous seasons in training data
python -m src.cli outlook --season 2025 --gameweek 10 --model strength \
  --include-prev-seasons 2 --save-predictions

# Dixon-Coles correction (adjusts low-scoring probabilities, typical rho: -0.05 to -0.15)
python -m src.cli outlook --season 2025 --gameweek 10 --model strength \
  --dc-rho -0.10 --save-predictions
```

### Elo model

```bash
# Higher K-factor (ratings change faster after each match)
python -m src.cli outlook --season 2025 --gameweek 10 --model elo \
  --elo-k 40 --save-predictions

# Lower K-factor (more stable ratings)
python -m src.cli outlook --season 2025 --gameweek 10 --model elo \
  --elo-k 20 --save-predictions

# More home advantage
python -m src.cli outlook --season 2025 --gameweek 10 --model elo \
  --elo-home-advantage 80 --save-predictions

# Less season carryover (fresh start each season)
python -m src.cli outlook --season 2025 --gameweek 10 --model elo \
  --elo-season-carryover 0.4 --save-predictions
```

### Scoreline grid size

```bash
# Higher grid (more accurate for high-scoring leagues, slower)
python -m src.cli outlook --season 2025 --gameweek 10 --model rolling \
  --max-goals-grid 7 --save-predictions

# Lower grid (faster, fine for low-scoring matches)
python -m src.cli outlook --season 2025 --gameweek 10 --model rolling \
  --max-goals-grid 4 --save-predictions
```

### Comparing tweaked models

Since each parameter combination generates a unique model_id in the filename, you can run many variants and compare:

```bash
# Run multiple rolling configs
python -m src.cli outlook --season 2025 --gameweek 10 --model rolling --window 3 --save-predictions
python -m src.cli outlook --season 2025 --gameweek 10 --model rolling --window 5 --save-predictions
python -m src.cli outlook --season 2025 --gameweek 10 --model rolling --window 8 --save-predictions

# See what's saved
python -m src.cli models --season 2025 --gameweek 10
# Output:
#   - rolling_w3_vw0.5_g5
#   - rolling_w5_vw0.5_g5
#   - rolling_w8_vw0.5_g5

# Evaluate each
python -m src.cli evaluate --season 2025 --gameweek 10 --model-id "rolling_w3_vw0.5_g5" --append
python -m src.cli evaluate --season 2025 --gameweek 10 --model-id "rolling_w5_vw0.5_g5" --append
python -m src.cli evaluate --season 2025 --gameweek 10 --model-id "rolling_w8_vw0.5_g5" --append

# Check cumulative performance to see which config wins
python -m src.cli performance --season 2025
```

---

## Full Backtest Workflow (Example)

Test all models across multiple gameweeks to find the best performer:

```bash
# 1. Fetch data
python -m src.cli fetch --season 2025 --league pl

# 2. Generate predictions for GW 5-15 with each model
for gw in 5 6 7 8 9 10 11 12 13 14 15; do
  python -m src.cli outlook --season 2025 --gameweek $gw --model rolling --save-predictions
  python -m src.cli outlook --season 2025 --gameweek $gw --model strength --save-predictions
  python -m src.cli outlook --season 2025 --gameweek $gw --model elo --save-predictions
  python -m src.cli outlook --season 2025 --gameweek $gw --model ensemble --save-predictions
done

# 3. Evaluate all
for gw in 5 6 7 8 9 10 11 12 13 14 15; do
  python -m src.cli models --season 2025 --gameweek $gw
  # evaluate each model_id shown above
done

# 4. View overall performance
python -m src.cli performance --season 2025
```

---

## Other Leagues

```bash
# La Liga
python -m src.cli fetch --season 2025 --league laliga
python -m src.cli outlook --season 2025 --gameweek 20 --league laliga --model elo --save-predictions

# Bundesliga
python -m src.cli fetch --season 2025 --league bundesliga
python -m src.cli outlook --season 2025 --gameweek 15 --league bundesliga --model ensemble --save-predictions

# Brazilian Serie A (calendar-year season)
python -m src.cli fetch --season 2026 --league brasileirao
python -m src.cli outlook --season 2026 --gameweek 10 --league brasileirao --model rolling --save-predictions
```

### Notes on season numbering

- **Split-year leagues** (PL, La Liga, etc.): `--season 2025` = the 2024/25 season
- **Calendar-year leagues** (Brasileirao, Libertadores): `--season 2026` = the 2026 season (Jan-Dec)

---

## Additional Data Sources

Beyond football-data.org, the project supports two additional free data sources that require **no API key** and cover leagues currently in-season.

### OpenFootball (GitHub — historical data, no API key)

Fetches data directly from the [openfootball/football.json](https://github.com/openfootball/football.json) GitHub repo. Public domain. Covers 12 leagues across England, Germany, Spain, Italy, and France with 15+ years of history.

```bash
# List available OpenFootball leagues
python -m src.cli leagues --source openfootball

# Fetch Premier League 2024-25
python -m src.cli fetch --season 2024 --league en-pl

# Fetch multiple seasons of Bundesliga (2015-2025)
python -m src.cli fetch --season-range 2015 2025 --league de-bundesliga

# Fetch La Liga, Serie A, Ligue 1
python -m src.cli fetch --season 2024 --league es-laliga
python -m src.cli fetch --season 2024 --league it-seriea
python -m src.cli fetch --season 2024 --league fr-ligue1

# Fetch English lower leagues
python -m src.cli fetch --season 2024 --league en-championship
python -m src.cli fetch --season 2024 --league en-league1

# Generate predictions using OpenFootball data
python -m src.cli outlook --season 2025 --gameweek 30 --league en-pl \
  --source openfootball --model ensemble --save-predictions

# Evaluate
python -m src.cli evaluate --season 2025 --gameweek 30 --league en-pl \
  --source openfootball --model-id "ensemble_r35%_s35%_e30%_g5" --append
```

**Available leagues:** `en-pl`, `en-championship`, `en-league1`, `en-league2`, `de-bundesliga`, `de-2bundesliga`, `es-laliga`, `es-segunda`, `it-seriea`, `it-serieb`, `fr-ligue1`, `fr-ligue2`

### TheSportsDB (free API — currently active calendar-year leagues)

Fetches from [TheSportsDB](https://www.thesportsdb.com) free API. No registration or API key needed. Covers calendar-year leagues that are **currently active** (March–November).

```bash
# List available TheSportsDB leagues
python -m src.cli leagues --source thesportsdb

# Fetch Swedish Allsvenskan 2025 (currently in-season)
python -m src.cli fetch --season 2025 --league se-allsvenskan --no-verify-ssl

# Fetch Norwegian Eliteserien
python -m src.cli fetch --season 2025 --league no-eliteserien --no-verify-ssl

# Fetch Finnish Veikkausliiga
python -m src.cli fetch --season 2025 --league fi-veikkausliiga --no-verify-ssl

# Fetch Japanese J1 League
python -m src.cli fetch --season 2025 --league jp-j1league --no-verify-ssl

# Fetch Korean K League 1
python -m src.cli fetch --season 2025 --league kr-kleague --no-verify-ssl

# Fetch MLS (USA)
python -m src.cli fetch --season 2025 --league us-mls --no-verify-ssl

# Fetch previous seasons for training
python -m src.cli fetch --season 2024 --league se-allsvenskan --no-verify-ssl
python -m src.cli fetch --season 2023 --league se-allsvenskan --no-verify-ssl

# Generate predictions for Allsvenskan matchday 15
python -m src.cli outlook --season 2025 --gameweek 15 --league se-allsvenskan \
  --model strength --save-predictions

# Predict with Elo model for J1 League
python -m src.cli outlook --season 2025 --gameweek 20 --league jp-j1league \
  --model elo --save-predictions

# Evaluate predictions
python -m src.cli evaluate --season 2025 --gameweek 15 --league se-allsvenskan \
  --model-id "strength_hl60_l21.00_lr0.050_it250_g5" --append --refresh-cumulative
```

**Available leagues:** `se-allsvenskan`, `no-eliteserien`, `fi-veikkausliiga`, `jp-j1league`, `kr-kleague`, `us-mls`

**Note:** `--no-verify-ssl` is needed on machines with corporate proxy/SSL certificate issues. The free API has soft rate limits — if fetching many seasons consecutively, allow a few minutes between fetches.

### Full workflow example (TheSportsDB)

```bash
# 1. Fetch data for Swedish league (current + previous seasons for training)
python -m src.cli fetch --season 2023 --league se-allsvenskan --no-verify-ssl
python -m src.cli fetch --season 2024 --league se-allsvenskan --no-verify-ssl
python -m src.cli fetch --season 2025 --league se-allsvenskan --no-verify-ssl

# 2. Generate predictions for multiple matchdays
for gw in 5 6 7 8 9 10 11 12 13 14 15; do
  python -m src.cli outlook --season 2025 --gameweek $gw --league se-allsvenskan \
    --model rolling --include-prev-seasons 2 --save-predictions
  python -m src.cli outlook --season 2025 --gameweek $gw --league se-allsvenskan \
    --model elo --include-prev-seasons 2 --save-predictions
done

# 3. Evaluate all predictions
for gw in 5 6 7 8 9 10 11 12 13 14 15; do
  python -m src.cli evaluate --season 2025 --gameweek $gw --league se-allsvenskan \
    --model-id "rolling_w5_vw0.5_g5" --append
  python -m src.cli evaluate --season 2025 --gameweek $gw --league se-allsvenskan \
    --model-id "elo_k30_ha65_co0.6" --append
done

# 4. View performance summary
python -m src.cli performance --season 2025
```

### Data source auto-detection

The `--source` flag is usually not needed — the CLI auto-detects the correct source from the league code:

| League code prefix | Auto-detected source |
|-------------------|---------------------|
| `pl`, `laliga`, `bundesliga`, etc. | football-data.org |
| `en-pl`, `de-bundesliga`, `es-laliga`, etc. | openfootball |
| `se-allsvenskan`, `no-eliteserien`, `us-mls`, etc. | thesportsdb |

You can override with `--source football-data`, `--source openfootball`, or `--source thesportsdb`.

---

## Repo Structure

```
src/
  cli.py              # command line entrypoint
  mcp_server.py       # MCP server (exposes tools for AI agents)
  fetch.py            # football-data.org API client + caching
  fetch_apifootball.py # API-Football (api-sports.io) fetcher
  fetch_openfootball.py # OpenFootball GitHub data fetcher (no API key)
  fetch_thesportsdb.py # TheSportsDB free API fetcher (no API key)
  features.py         # rolling metrics, venue splits
  model_poisson.py    # Poisson grid + W/D/L aggregation
  model_strength.py   # attack/defence strength model (gradient descent)
  model_elo.py        # Elo rating system
  model_ensemble.py   # weighted model combination
  render.py           # markdown/json output + prediction orchestration
  evaluate.py         # predictions vs actuals scoring
  performance.py      # cumulative performance artifacts
  data_loader.py      # shared data loading utilities
  metrics.py          # shared evaluation metrics + plotting
  competitions.py     # league registry (codes, IDs, patterns)
tests/               # pytest test suite
data/
  raw/               # cached API responses
  processed/         # normalized per-season CSVs
  curated/           # merged multi-season datasets
  predictions/       # saved prediction JSONs/CSVs (per-model)
  evaluation/        # evaluation ledger
reports/             # generated markdown reports + plots
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## CLI Help

```bash
python -m src.cli --help
python -m src.cli outlook --help
python -m src.cli evaluate --help
```
