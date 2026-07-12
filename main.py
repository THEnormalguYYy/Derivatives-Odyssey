"""End-to-end orchestration of the Derivatives Odyssey pipeline.

Wires the three core modules into a single runnable workflow:

    mock options chain
        -> DataProcessor        (clean data, Black-Scholes IV, features)
        -> models.py            (train ML forecaster; GARCH baseline)
        -> backtester.py        (trade the IV vs predicted-RV spread)
        -> performance report

The strategy harvests the **Volatility Risk Premium**: when the model's
predicted realized volatility sits meaningfully above the market's implied
volatility, options look cheap and we buy the at-the-money straddle.

Run with::

    python main.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.data_processor import DataProcessor
from src.models import (
    TARGET_COLUMN,
    GARCHVolatilityModel,
    MLVolatilityModel,
    ModelEvaluator,
    build_supervised_frame,
    daily_returns,
    to_daily,
    train_test_split_by_date,
)
from src.backtester import StraddleBacktester

RANDOM_SEED = 42


# ====================================================================== #
# 1. Mock data generation
# ====================================================================== #
def generate_mock_options_chain(
    n_days: int = 504,
    strikes_pct: tuple[float, ...] = (0.90, 0.95, 1.00, 1.05, 1.10),
    dte_days: int = 21,
    base_spot: float = 100.0,
    risk_free_rate: float = 0.03,
    vol_risk_premium: float = 0.03,
    seed: int = RANDOM_SEED,
    dirty: bool = True,
) -> pd.DataFrame:
    """Generate a realistic mock options chain.

    A single underlying is simulated with a *time-varying* instantaneous
    volatility (calm and stressed regimes), and a strip of European calls and
    puts is priced each day with Black-Scholes. Market implied volatility is set
    to the local realized vol plus a ``vol_risk_premium`` and a mild strike
    smile, so IV and future RV genuinely diverge — the signal the strategy
    trades. When ``dirty`` is ``True`` a few NaNs and outliers are injected to
    exercise the processor's cleansing.

    Returns
    -------
    pandas.DataFrame
        Columns: ``date, strike, spot_price, option_type, expiration_date,
        option_price, risk_free_rate``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    # Regime-switching instantaneous vol: a calm baseline with periodic stress.
    t = np.arange(n_days)
    inst_vol = 0.14 + 0.06 * (np.sin(2 * np.pi * t / 90.0) ** 2)
    inst_vol[120:150] += 0.18  # a volatility shock
    inst_vol[330:360] += 0.12  # a second, milder shock

    daily_ret = rng.normal(0.0, inst_vol / np.sqrt(252.0))
    spot = base_spot * np.exp(np.cumsum(daily_ret))
    T = dte_days / 252.0

    def bs_price(S, K, T, r, sigma, is_call):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        disc = np.exp(-r * T)
        if is_call:
            return S * norm.cdf(d1) - K * disc * norm.cdf(d2)
        return K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)

    records: list[list] = []
    for d, S, v in zip(dates, spot, inst_vol):
        expiry = d + pd.offsets.BDay(dte_days)
        for pct in strikes_pct:
            K = round(S * pct, 2)
            # Implied vol = local vol + risk premium + smile + noise.
            smile = 0.20 * (pct - 1.0) ** 2
            iv = v + vol_risk_premium + smile + rng.normal(0.0, 0.01)
            iv = float(np.clip(iv, 0.03, 2.0))
            for is_call, otype in ((True, "call"), (False, "put")):
                price = bs_price(S, K, T, risk_free_rate, iv, is_call)
                records.append([d, K, S, otype, expiry, price, risk_free_rate])

    df = pd.DataFrame(
        records,
        columns=[
            "date", "strike", "spot_price", "option_type",
            "expiration_date", "option_price", "risk_free_rate",
        ],
    )

    if dirty:
        # Inject a handful of missing values and extreme outliers.
        df.loc[df.sample(frac=0.01, random_state=seed).index, "option_price"] = np.nan
        df.loc[df.sample(frac=0.005, random_state=seed + 1).index, "spot_price"] = -1.0
        df.loc[df.sample(frac=0.005, random_state=seed + 2).index, "strike"] = 0.0

    return df


# ====================================================================== #
# 2. Pipeline stages
# ====================================================================== #
def print_header(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def run_pipeline() -> None:
    """Execute the full research-to-backtest workflow and print results."""
    warnings.filterwarnings("ignore")  # keep the console report clean
    np.random.seed(RANDOM_SEED)

    # ---- 1. Data ----------------------------------------------------- #
    print_header("1. GENERATING MOCK OPTIONS CHAIN")
    raw = generate_mock_options_chain()
    print(f"Raw rows: {len(raw):,}  |  trading dates: {raw['date'].nunique()}")

    # ---- 2. Processing & feature engineering ------------------------- #
    print_header("2. PROCESSING: BLACK-SCHOLES IV + FEATURES")
    processor = DataProcessor(raw)
    processed = processor.process(feature_only=False)
    print(f"Clean rows after cleansing: {len(processed):,}")
    print(
        "Feature preview:\n"
        + processed[["date", "moneyness", "time_to_expiry",
                     "realized_volatility", "implied_volatility"]]
        .head()
        .to_string(index=False)
    )

    # ---- 3. Supervised framing & chronological split ---------------- #
    print_header("3. BUILDING SUPERVISED DATASET (FORWARD RV TARGET)")
    supervised = build_supervised_frame(processed, horizon=21)
    train, test, split_date = train_test_split_by_date(supervised, test_size=0.30)
    print(f"Split date: {split_date.date()}")
    print(f"Train rows: {len(train):,}  |  Test rows: {len(test):,}")

    # ---- 4. Models: ML forecaster + GARCH baseline ------------------ #
    print_header("4. TRAINING MODELS")
    ml = MLVolatilityModel(model_type="rf").fit(train)
    test = test.copy()
    test["predicted_rv"] = ml.predict(test)
    print("Random Forest feature importances:")
    print(ml.feature_importances_.round(3).to_string())

    garch = GARCHVolatilityModel(horizon=21).fit(
        daily_returns(processed), last_obs=split_date
    )
    garch_daily = garch.forecast_vol(start=split_date)

    # ---- 5. Forecast-accuracy comparison (per-date) ----------------- #
    print_header("5. FORECAST ACCURACY: ML vs GARCH BASELINE")
    actual_daily = to_daily(test[TARGET_COLUMN], test["date"])
    ml_daily = to_daily(test["predicted_rv"], test["date"])
    garch_aligned = garch_daily.reindex(pd.to_datetime(actual_daily.index))

    evaluator = ModelEvaluator()
    evaluator.evaluate("GARCH(1,1)", actual_daily.values, garch_aligned.values)
    evaluator.evaluate("RandomForest", actual_daily.values, ml_daily.values)
    print(evaluator.summary().round(5).to_string())

    # ---- 6. Backtest the straddle wager on out-of-sample dates ------ #
    print_header("6. BACKTESTING THE VOLATILITY STRADDLE WAGER")
    backtester = StraddleBacktester(
        threshold=0.02,       # require a 2 vol-point edge
        cost_pct=0.01,        # 1% round-trip transaction cost
        annual_risk_free=0.03,
    )
    result = backtester.run(test)

    print(f"Signals evaluated: {len(result.signals)}  |  "
          f"Straddles executed: {result.metrics['num_trades']}")
    print("\nStrategy performance report:")
    _print_metrics(result.metrics)


def _print_metrics(metrics: dict) -> None:
    """Pretty-print the performance-report dictionary."""
    label_fmt = {
        "total_return": ("Total Return", "{:.2%}"),
        "annualized_sharpe": ("Annualized Sharpe", "{:.2f}"),
        "max_drawdown": ("Max Drawdown", "{:.2%}"),
        "win_rate": ("Win Rate", "{:.2%}"),
        "avg_trade_return": ("Avg Trade Return", "{:.2%}"),
        "num_trades": ("Number of Trades", "{:.0f}"),
        "avg_holding_days": ("Avg Holding (days)", "{:.1f}"),
        "avg_spread": ("Avg Vol Spread", "{:.4f}"),
        "total_pnl": ("Total PnL ($)", "{:.2f}"),
    }
    for key, (label, fmt) in label_fmt.items():
        value = metrics.get(key, float("nan"))
        try:
            shown = fmt.format(value)
        except (ValueError, TypeError):
            shown = str(value)
        print(f"  {label:<22}: {shown}")


if __name__ == "__main__":
    run_pipeline()
