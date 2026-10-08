# ACRS ML review

```
python ml/acrs_ml.py              # after each weekly update -> ACRS_ML_Review.html, ml/ml_review.json
python -m unittest ml/test_acrs_ml.py
```

Needs numpy only. It reads the data the builder already embeds in `ACRS_Simulation.html`
(330-trade ledger with entry features `f_*`, equity curve, live "since" book, sensitivity grid)
and changes nothing in the live model.

## How it maps to the research -> code -> backtest -> live -> post-mortem -> fine-tune loop

| Step | What ACRS already had | What this adds |
|---|---|---|
| 3 Backtest | walk-forward, bootstrap, sensitivity grid, survivorship measurement | **Deflated Sharpe ratio** (Bailey & López de Prado): how many independent rule variants the backtest can survive before its Sharpe is explainable by luck |
| 5 Post-mortem | survival curve, shadow book | **Critic over the ledger**: R by setup, year and exit cause; stop/gap leakage; a censoring-correct live-vs-backtest test; entry-feature drift |
| 6 Fine-tune | rules documented in page text | **Experiment registry** (`experiments.json`, the database of failed rules) feeding the DSR trial count, plus a **record-only meta-label model** with a pre-registered promotion rule |

Steps 1, 2 and 4 (an LLM writes the code and one click deploys it) are deliberately not
automated. A loop that rewrites rules and re-backtests until something looks better is
a trial generator. At the current Sharpe the backtest stays significant (DSR ≥ 0.95) for
only about **10 independent tries**. The registry already holds 19.

## What it found (data as of 2026-10-07)

1. **Setup A is the edge.** 57 of 330 trades (17%) make +167R of +179R. Without its 3 best
   trades it still makes +68R, and it was positive in 5 of 6 years. Setup C is 63% of entries for
   +4.6R (−21R without its best 3). Setup B makes +7R. Logged as `setup-c-demotion` (proposed).
   It needs a portfolio-level A/B in the builder, because per-trade R ignores slot competition.
2. **The live book isn't broken.** 29 closed trades, −5.4R, PF 0.36. Compared against
   backtest trades that could have closed within the same 69-day window, a result this bad
   happens 36% of the time. Comparing against all backtest trades would say 5%, but that
   comparison is wrong: young books have their winners still open. The live window also got
   only 2 Setup A entries (5 expected).
3. **No entry-feature drift.** The live entries look like backtest entries on all 11 features.
4. **ML entry filtering doesn't help.** This was tested with purged annual walk-forward, ridge
   on winsorised R and logistic on win/loss. Take-all keeps +171R, the filters +159R and +126R,
   and the mean out-of-sample IC is negative. The top 5% of trades make 68% of gross profit,
   so any filter that skips entries mostly cuts winners. The model stays record-only and is
   re-scored every run.
5. **Stop and gap exits cost 30R.** Those entries were extended (12% above MA50 vs 5%) and had
   high relative volume. That is a candidate for a future registered test, not a rule change.

## Rules for using it

- Add a row to `experiments.json` **before** running a test. Never delete rows.
- Set `untracked_prior_trials` honestly. It is the number of v1.0–v1.4 variants tried before
  the registry existed.
- The promotion rule in `acrs_ml.py` (`PROMOTION`) was fixed before the first run. Do not loosen it after seeing a result.

## About the open-source frameworks

- **Qlib / FreqAI**: good if you later want a feature store and model zoo. They don't fix the
  core problem: 330 fat-tailed trades is a small dataset for supervised ML.
- **FinRL** (deep RL): would replace a rule system that has been measured carefully with a policy
  that is far harder to audit. It is not a good fit for this sleeve.
- **TradingAgents-style LLM desks**: useful for step 1 (research and hypothesis drafting) and for
  writing up the critic's findings. Any hypothesis they produce still goes into the registry and
  through the builder's same-universe A/B.
