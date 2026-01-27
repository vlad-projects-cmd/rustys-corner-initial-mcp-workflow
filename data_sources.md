# Data Sources

## Primary API (recommended for testing): football-data.org (v4)
Reason:
- Free plan available (rate-limited) and includes fixtures + league tables. 
- Enough for Phase 1 gameweek outlook using goals-based rolling averages.

Notes:
- Free plan has request throttling and may have delayed scores/schedules.
- We can upgrade later if we need faster post-game stats.

Docs:
- API documentation / quickstart: https://www.football-data.org/documentation/api
- v4 docs: https://www.football-data.org/docs/v1/index.html
- Pricing / rate limits: https://www.football-data.org/pricing

### Authentication
- Register and obtain an API token.
- Use HTTP header:
  `X-Auth-Token: <YOUR_TOKEN>`

### Competition identifiers
Premier League is a competition resource. Commonly used PL id is `2021` (confirm via API if needed).

---

## Endpoints needed for Phase 1 (Gameweek Outlook)

### 1) Fixtures for a gameweek
Goal: list all fixtures (home/away teams, kickoff time, match id) for the selected gameweek.

Endpoint:
- `GET /v4/competitions/{competitionId}/matches?matchday={N}`

Example:
- `/v4/competitions/2021/matches?matchday=24`

Data used:
- `matches[].id`
- `matches[].utcDate`
- `matches[].homeTeam.id`, `matches[].homeTeam.name`
- `matches[].awayTeam.id`, `matches[].awayTeam.name`

### 2) Historical matches (to compute rolling averages)
Goal: build rolling goals-for/goals-against from prior matches only.

Endpoint options:
A) Pull all season matches once, then compute locally:
- `GET /v4/competitions/{competitionId}/matches?season={YYYY}`

B) Pull team matches on demand:
- `GET /v4/teams/{teamId}/matches?status=FINISHED&limit={K}`
  (Use only if needed; prefer option A + local cache.)

Data used:
- final score fields: full-time home/away goals
- match date (to avoid leakage)

### 3) League-wide average goals (normalization)
Goal: compute league average goals to normalize expected goals proxy.

Data source:
- derived locally from the historical matches dataset pulled above.

---

## Minimal dataset fields we will store (normalized)
Table: `matches`
- `match_id`
- `season`
- `matchday`
- `utc_date`
- `home_team_id`, `home_team_name`
- `away_team_id`, `away_team_name`
- `home_goals_ft`, `away_goals_ft`
- `status` (SCHEDULED/FINISHED/etc.)

---

## Optional (later phases)
### Standings (for narrative context in previews)
- `GET /v4/competitions/{competitionId}/standings`

### Team details / squad (not needed for baseline)
- `GET /v4/teams/{teamId}`

### Alternative APIs (if we outgrow football-data.org free tier)
- API-FOOTBALL (api-sports): broader stats, but free plan has daily request limits and plan constraints.
- TheSportsDB: good for metadata/artwork, less ideal for rigorous stats baselines.
