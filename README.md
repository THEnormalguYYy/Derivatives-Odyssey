# 📈 Derivatives Odyssey

**A research pipeline for forecasting market volatility and backtesting a systematic options strategy that trades the Volatility Risk Premium.**

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg">
  <img alt="Style" src="https://img.shields.io/badge/code%20style-OOP%20%7C%20vectorized-informational">
  <img alt="Status" src="https://img.shields.io/badge/status-research-orange.svg">
</p>

---

## 🎯 Overview — The Volatility Risk Premium (VRP) Thesis

Option prices embed the market's expectation of how much the underlying will
move: the **Black-Scholes Implied Volatility (IV)**. Historically, IV tends to
trade *richer* than the volatility the underlying actually realizes — sellers of
options demand a premium for bearing gap and tail risk. That structural gap,

$$\text{VRP} \;=\; \sigma_{\text{implied}} \;-\; \sigma_{\text{realized}},$$

is the **Volatility Risk Premium**.

Most of the time the premium favors option *sellers*. But it is not constant — it
compresses, and occasionally **inverts**, precisely when the market has
*under-priced* future turbulence. Derivatives Odyssey attacks that inefficiency
from the long side: instead of blindly shorting volatility, it trains a machine
learning model to **forecast realized volatility** and only takes a position
when the model's forecast diverges sharply from what the market has priced in.

> **Signal.** For each date, compute the spread between the model's predicted
> realized volatility and the at-the-money implied volatility:
>
> ```
> spread = predicted_RV  −  implied_volatility
> ```
>
> When `spread > threshold`, the model believes the market has under-priced
> future movement → **BUY the ATM straddle** (long 1 call + 1 put) to profit
> from the anticipated move, regardless of direction.

---

## 🧠 How It Works

```
   Raw Options Chain
          │
          ▼
┌─────────────────────┐   Black-Scholes IV (Newton–Raphson),
│   data_processor.py │   Moneyness (S/K), Time-to-Expiry,
│    DataProcessor    │   Rolling Realized Volatility
└─────────────────────┘
          │  clean, feature-rich panel
          ▼
┌─────────────────────┐   ML forecaster  (Random Forest / Ridge)
│      models.py      │   Baseline       (GARCH(1,1) via `arch`)
│  Forecast & Eval    │   Metrics        (MSE · MAE · R²)
└─────────────────────┘
          │  predicted realized volatility
          ▼
┌─────────────────────┐   Signal → ATM straddle execution → PnL
│   backtester.py     │   Total Return · Sharpe · Max Drawdown
│  StraddleBacktester │
└─────────────────────┘
          │
          ▼
   Performance Report
```

### The three core modules

| Module | Class | Responsibility |
| :--- | :--- | :--- |
| `src/data_processor.py` | `DataProcessor` | Validate & cleanse dirty market data; solve **Black-Scholes IV** with a vectorized Newton–Raphson root finder; engineer moneyness, time-to-expiry, and rolling realized volatility. |
| `src/models.py` | `MLVolatilityModel`, `GARCHVolatilityModel`, `ModelEvaluator` | Forecast **future** realized volatility from the engineered features (crucially including IV); benchmark the ML model against a classical GARCH(1,1) baseline. |
| `src/backtester.py` | `StraddleBacktester` | Turn forecasts into trades: generate the IV-vs-RV signal, execute ATM straddles, compute PnL at expiration, and report performance. |

---

## 📂 Directory Structure

```
derivatives-odyssey/
├── data/                  # (input datasets — gitignored in practice)
├── notebooks/             # exploratory research & visualization
├── src/
│   ├── __init__.py
│   ├── data_processor.py  # cleansing, Black-Scholes IV, feature engineering
│   ├── models.py          # ML + GARCH volatility forecasting & evaluation
│   └── backtester.py      # systematic straddle-wager execution engine
├── main.py                # end-to-end orchestration (mock data → report)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Requires **Python 3.11+**. From the project root:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the complete pipeline — it generates a mock options chain, engineers
features, trains the models, backtests the strategy, and prints the report:

```bash
python main.py
```

### Using the modules directly

```python
from src.data_processor import DataProcessor
from src.models import MLVolatilityModel, build_supervised_frame, train_test_split_by_date
from src.backtester import StraddleBacktester

# 1. Clean data & engineer features (Black-Scholes IV, moneyness, RV, …)
processed = DataProcessor(raw_chain).process(feature_only=False)

# 2. Build the forward-RV target and split chronologically (no look-ahead)
supervised = build_supervised_frame(processed, horizon=21)
train, test, split_date = train_test_split_by_date(supervised, test_size=0.30)

# 3. Train the ML forecaster and predict out-of-sample realized volatility
model = MLVolatilityModel(model_type="rf").fit(train)
test = test.assign(predicted_rv=model.predict(test))

# 4. Backtest the straddle wager on the model's forecasts
result = StraddleBacktester(threshold=0.02, cost_pct=0.01).run(test)
print(result.metrics)
```

---

## 🔬 Methodology Notes

- **Implied Volatility** is solved per-quote with a **vectorized Newton–Raphson**
  iteration (`scipy.optimize.newton` with analytic vega). Non-convergent or
  economically invalid quotes gracefully degrade to `NaN`.
- **Realized Volatility** is the annualized 21-day rolling standard deviation of
  log returns of the underlying — a property of the *spot* series, broadcast to
  every option on each date.
- **Target** = *future* realized volatility (the RV window shifted forward by the
  forecast horizon). The train/test split is strictly **chronological** to
  prevent look-ahead bias.
- **PnL** settles each straddle at its intrinsic value `|S_expiry − K|` minus the
  premium paid, net of configurable transaction costs.
- **Sharpe** is annualized by the number of non-overlapping holding periods per
  year; **Max Drawdown** is computed on a fixed-capital equity curve that is
  robust to individual trades losing 100% of premium.

---

## 📊 Backtesting Results

> _Placeholder — populate after running `python main.py` on your dataset._

**Forecast accuracy (out-of-sample):**

| Model            | MSE   | MAE   | R²    |
| :--------------- | :---: | :---: | :---: |
| GARCH(1,1) — baseline | `0.00XX` | `0.00XX` | `0.XX` |
| Random Forest — ML    | `0.00XX` | `0.00XX` | `0.XX` |

**Strategy performance (straddle wager):**

| Metric                 | Value    |
| :--------------------- | :------: |
| Total Return           | `+XX.X%` |
| Annualized Sharpe      | `X.XX`   |
| Max Drawdown           | `−XX.X%` |
| Win Rate               | `XX.X%`  |
| Number of Trades       | `XX`     |
| Avg. Holding (days)    | `21`     |

<p align="center"><em>Equity curve plot → <code>notebooks/</code></em></p>

---

## 🗺️ Roadmap

- [ ] Ingest a real options dataset (e.g. OptionMetrics / CBOE) via `data/`.
- [ ] Add a short-volatility sleeve to harvest the premium symmetrically.
- [ ] Mark-to-model daily PnL and delta-hedging for the holding-period exit.
- [ ] Hyperparameter search & walk-forward cross-validation.
- [ ] Richer feature set: term-structure slope, skew, volume/open-interest.

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. It uses
simulated data and simplifying assumptions (European exercise, intrinsic-value
settlement, no early assignment or dividends). It is **not** investment advice
and makes no representation of live trading performance.
