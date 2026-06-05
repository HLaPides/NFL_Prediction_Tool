# Jaginator

NFL spread prediction and player prop model. Desktop app built on XGBoost, logistic regression, and ELO ratings — trained on 25 seasons of play-by-play data.

**Backtest results (walk-forward, 2005–2025):**
- 66.4% winner accuracy
- 58.7% ATS on ~22 filtered bets per season
- +$5,418 ROI over 20 seasons at $100/bet
- Playoff model: 68.9% accuracy, +5.5% vs Vegas

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | React + TypeScript, Electron, Vite |
| Backend | Python 3.11, FastAPI, SQLite |
| Models | XGBoost, scikit-learn LogisticRegression |
| Data | nflverse (nflreadpy), ESPN API |

---

## Repo structure

```
Jaginator/
├── src/
│   ├── backend/
│   │   ├── api.py                    # FastAPI server — entry point
│   │   ├── data/
│   │   │   ├── sql_helpers.py        # DB connection and read/write helpers
│   │   │   ├── getter.py             # Historical data loader (seeding only)
│   │   │   ├── depth_chart_getter.py # ESPN depth chart fetcher
│   │   │   └── injury_getter.py      # ESPN injury report fetcher
│   │   ├── engine/
│   │   │   ├── model.py              # XGBoost spread model
│   │   │   ├── predictor.py          # Weekly prediction runner
│   │   │   ├── playoff_model.py      # Logistic regression playoff model
│   │   │   ├── elo_analyzer.py       # ELO rating system
│   │   │   ├── defense.py            # Opponent-adjusted defensive ratings
│   │   │   └── props/
│   │   │       ├── qb_props.py       # QB stat predictions
│   │   │       ├── rb_props.py       # RB stat predictions
│   │   │       └── wr_props.py       # WR stat predictions
│   │   └── scripts/
│   │       ├── weekly_update.py      # Full 9-step weekly pipeline
│   │       ├── seed.py               # Build rolling stats + player features
│   │       ├── backtest.py           # Season-start model sanity check
│   │       └── predict_season.py     # Monte Carlo Super Bowl simulator
│   └── frontend/
│       └── src/
│           └── App.tsx               # React UI
└── README.md
```

---

## Setup from scratch

### Prerequisites
- Python 3.11
- Node.js 18+
- Git

### 1. Clone and install

```bash
git clone https://github.com/HLaPides/Jaginator.git
cd Jaginator

# Backend
cd src/backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### 2. Seed the database

The DB is not in the repo — build it from scratch:

```bash
cd src/backend

# Seed base tables from nflverse (schedules, odds, conditions, personnel)
python3.11 -c "from data.sql_helpers import seed_database; seed_database()"

# Build ELO history (1999–present)
python3.11 engine/elo_analyzer.py

# Build rolling team stats and player features (~15 min)
python3.11 scripts/seed.py

# Build opponent-adjusted defense ratings
python3.11 engine/defense.py

# Train spread model and generate initial predictions
python3.11 engine/model.py
python3.11 scripts/weekly_update.py
```

### 3. Run

```bash
# Backend (from src/backend)
python3.11 api.py

# Frontend (from src/frontend)
npm run dev

# Or build the Electron desktop app
npm run build
```

---

## Weekly cadence

Each Tuesday during the season, hit the refresh button in the app (or run manually):

```bash
python3.11 scripts/weekly_update.py
```

This runs the full 9-step pipeline:
1. Fetch schedules, player stats, team stats, rosters from nflverse
2. Fetch ESPN depth charts
3. Fetch ESPN injury report
4. Save all current data to `*_current` tables
5. Merge completed games into historical tables
6. Rebuild ELO + rolling stats + player features
7. Retrain spread model on all historical data
8. Rebuild defense ratings
9. Generate game picks and QB/RB/WR props for the upcoming week

---

## Model architecture

### Spread model (`engine/model.py`)
Pure XGBoost trained on **ATS margin** (`result - vegas_spread`) as the target. Vegas spread is included as a peer feature — not a stacking anchor. The model learns market inefficiency directly.

**Key finding — asymmetric edge filter:**
Underdog bets are profitable across all gap sizes. Favorite bets are only profitable when the model's edge is small (gap ≤ 3.0, spread ≤ 7). The model overcorrects on large-gap heavy favorites.

```
Underdog bets: 251 bets, 60.6% ATS, +$3,918
Favorite bets (filtered): 195 bets, 55.4% ATS, +$1,500
```

**Params (Optuna-tuned, confirmed ceiling):**
```python
n_estimators=403, max_depth=2, learning_rate=0.0339,
subsample=0.7768, colsample_bytree=0.7109, min_child_weight=6,
gamma=1.7893, reg_alpha=3.0354, reg_lambda=7.8139
```

**Features (21):** vegas_spread, elo_diff, rolling pass/rush EPA and yards diffs, injury score, QB EPA, WR/TE/RB yards, defensive TFL/sacks/ints, first downs, penalties, clutch rate, rest differential, dome flag, wind flag.

### Playoff model (`engine/playoff_model.py`)
Logistic regression trained on playoff games only. Silently replaces the spread model during weeks 19–22. Tuned to weight Conference Championship and Super Bowl accuracy 2x.

```
Overall: 68.9%  Wild Card: 62.0%  Divisional: 72.7%
Conference: 77.3%  Super Bowl: 68.2%
```

### ELO (`engine/elo_analyzer.py`)
Standard 400-point scale with margin-of-victory K-factor adjustment and 30% regression to mean each offseason. Home advantage: +48 ELO points.

### Defense ratings (`engine/defense.py`)
Opponent-adjusted defensive EPA (DVOA-style). Separate ratings for pass defense, rush defense, and position-specific coverage (WR, TE, RB targets). Used by both the spread model and the props models.

---

## Props models

Three separate XGBoost models (QB, RB, WR), each predicting multiple stat targets:

| Position | Stats predicted |
|---|---|
| QB | passing_yards, passing_tds, attempts, completions, interceptions |
| RB | rushing_yards, rushing_tds, carries, receptions, receiving_yards |
| WR | receiving_yards, receiving_tds, receptions, targets |

All use 4-game rolling player stats + opponent defense quality as features. Walk-forward validated. Models have room for improvement

---

## DB tables

| Category | Tables |
|---|---|
| Historical | schedules, game_odds, game_conditions, game_personnel, player_stats, team_stats_weekly, injuries_historical, elo_history, team_rolling_stats, player_features, defense_ratings |
| Current (weekly replace) | schedules_current, player_stats_current, team_stats_current, rosters_current, depth_charts_current, injuries_current |
| Predictions | weekly_predictions, model_predictions, playoff_predictions, qb/rb/wr_prop_predictions |

---

## Season simulator

Monte Carlo Super Bowl odds across 10,000 simulations, using current ELO ratings adjusted for QB EPA, strength of schedule, and injury penalties.

```bash
python3.11 scripts/predict_season.py
```

---

## What doesn't work

Things tested extensively that produced no improvement over the baseline:

- Stacking sub-models (QB-only, defense-only, ELO-only) — Ridge meta-layer negatively weighted the main model
- Recency weighting on rolling stats — equal weighting is optimal
- Season weighting, split-season models, rolling window sweep — all flat
- LightGBM / CatBoost / ensemble — flat or worse
- Underdog-specific model — no additional signal beyond the main model
- Feature interactions, QB-adjusted ELO — flat

The asymmetric favorite/underdog filter was the single biggest improvement (+$2,036 ROI vs unfiltered).