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

## Repo structure (suggested)
- `src/`
  - `fetch.py`            # API client + caching
  - `features.py`         # rolling metrics, league averages
  - `model_poisson.py`    # Poisson grid + W/D/L aggregation
  - `render.py`           # markdown/json output
  - `cli.py`              # command line entrypoint
- `data/`
  - `raw/`                # cached API responses
  - `processed/`
