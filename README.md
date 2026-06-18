# ⚽ World Cup 2026 Predictor — Elo + Poisson

A football match-outcome model that predicts **win / draw / loss probabilities** and
**scorelines** for any two national teams, then simulates the **FIFA World Cup 2026** to
estimate each team's chance of winning the trophy.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
*(Upload `world_cup_2026_predictor.ipynb` via File → Upload notebook, then Runtime → Run all.)*

## Results (honest, walk-forward validated)

| Metric | Score | Notes |
|---|---|---|
| 3-way accuracy (W/D/L) | **~61%** | vs 48% "always home" baseline; above most commercial models |
| 2-way accuracy | **~79%** | when a winner exists (draws set aside) |
| Top-2 accuracy | **~84%** | true result is in the model's two likeliest outcomes |
| Log loss / RPS | 0.86 / 0.17 | standard probabilistic metrics (lower is better) |

Validated **year by year (2018–2025)** with walk-forward testing, not a single lucky split.

## Why Elo + Poisson, not a neural network?

Tested empirically on the same data: a 1-layer net, a 3-layer net, and XGBoost **all scored
~60.7%** — identical to this Poisson model. Football has a hard accuracy ceiling set by
randomness (≈23% of matches are draws, nearly unpredictable), not by model capacity. So the
right choice is the simplest model that reaches the ceiling — which also gives full scoreline
distributions, clean generalization to unseen matchups, and interpretability for free.

## How it works

1. **Elo ratings** — every international match since 1872 replayed chronologically; ratings
   update after each game and stay current automatically as new results arrive.
2. **Poisson goal model** — two time-decay-weighted Poisson regressions map the Elo gap
   (+ home advantage) to expected goals, which become a full scoreline probability grid.
3. **Monte Carlo** — the real 48-team, 12-group 2026 tournament (groups and fixtures read
   straight from the data, played matches using real results) simulated thousands of times.

## Data

- **Primary:** [martj42/international_results](https://github.com/martj42/international_results)
  — every men's international from 1872, no API key, auto-updating. The notebook re-downloads
  it on each run, so the model is always current.
- **Optional live top-up:** [football-data.org](https://www.football-data.org/) free tier
  (paste a free key in the notebook to pull the very latest results mid-tournament).

## Usage

```bash
pip install -r requirements.txt
jupyter notebook world_cup_2026_predictor.ipynb
```

Or just open it in Google Colab and **Run all**. Predict any matchup:

```python
predict("Brazil", "Argentina")          # neutral venue
predict("Brazil", "Argentina", home_team="Brazil")
```

## Limitations

- Draws cap three-way accuracy at ~60% for *every* model — this is football, not a bug.
- Only team strength + venue drive predictions. Biggest gains would come from richer data:
  lineups/injuries, squad value, fatigue, and expected-goals (xG).
- Bigger upgrade path: a full **Dixon–Coles** model for better draw/low-score calibration.
- The tournament simulation uses a simplified random knockout draw, not the exact official
  Round-of-32 bracket mapping.

## Keeping it current

The notebook pulls fresh data on every run. For hands-off updates, run the training in a
weekly **GitHub Actions** cron job that commits the updated `elo_ratings.json` and
`model_params.json`.

---
*Built as an honest baseline, not a betting oracle. Football is unpredictable — which is the
whole point of watching.*
