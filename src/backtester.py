"""Systematic backtesting engine for a volatility straddle strategy.

The strategy monetizes the gap between the market's Black-Scholes implied
volatility (IV) and our model's *predicted* realized volatility (RV):

* **Signal.** For each trading date, compute the volatility spread
  ``predicted_rv - atm_iv`` on the at-the-money (ATM) straddle. When the spread
  exceeds ``threshold`` the model expects the underlying to move more than the
  market has priced in — options look cheap — so we go **long the straddle**.
* **Execution.** A straddle buys one ATM call and one ATM put at market. The
  premium paid (call + put) is the capital at risk and the return denominator.
* **PnL.** Each straddle is held to expiration (or a fixed holding period).
  Its exit value is the intrinsic straddle payoff ``|S_exit - K|`` — exact at
  expiration — and PnL is that value minus the premium (and any costs).
* **Metrics.** :meth:`StraddleBacktester.performance_report` returns Total
  Return, Annualized Sharpe Ratio, and Maximum Drawdown (plus supporting
  statistics).

All logic is fully vectorized with pandas/numpy — no per-row loops. ATM
selection uses a grouped ``idxmin``, leg pairing uses a merge, and exit-price
lookup uses positional indexing into the sorted date axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Sequence

import numpy as np
import pandas as pd

__all__ = ["StraddleBacktester", "BacktestResult"]


@dataclass
class BacktestResult:
    """Container for the artifacts of a single backtest run.

    Attributes
    ----------
    trades:
        One row per executed straddle, with entry/exit prices, PnL and return.
    signals:
        Per-date ATM straddle table with spread and signal (executed or not).
    equity_curve:
        Cumulative equity (starting at 1.0) of compounded trade returns,
        indexed by trade exit date.
    metrics:
        The performance-report dictionary (Total Return, Sharpe, Max Drawdown…).
    """

    trades: pd.DataFrame
    signals: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict[str, float]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        m = self.metrics
        return (
            f"BacktestResult(trades={m.get('num_trades', 0)}, "
            f"total_return={m.get('total_return', float('nan')):.2%}, "
            f"sharpe={m.get('annualized_sharpe', float('nan')):.2f}, "
            f"max_drawdown={m.get('max_drawdown', float('nan')):.2%})"
        )


class StraddleBacktester:
    """Backtest a long-volatility ATM straddle strategy on an options chain.

    Parameters
    ----------
    threshold:
        Minimum volatility spread (``predicted_rv - atm_iv``) required to open a
        long straddle. Expressed in annualized vol points (e.g. ``0.02`` = 2 vol
        points). Defaults to ``0.0`` (trade any positive edge).
    holding_period:
        If ``None`` (default), each straddle is held to expiration and settled at
        intrinsic value ``|S_T - K|`` (exact). If an integer, the position is
        exited after that many trading days (capped at expiration) and marked at
        intrinsic value at the exit-date spot — a conservative estimate that
        ignores residual time value.
    pred_col:
        Column holding the model's predicted realized volatility.
    iv_col:
        Per-option implied-volatility column, used to derive the ATM straddle IV.
    cost_pct:
        Round-trip transaction cost as a fraction of premium (e.g. ``0.01`` =
        1%). Deducted from every trade's PnL. Defaults to ``0.0``.
    allow_short:
        If ``True``, also sell straddles when the spread is below
        ``-threshold`` (market vol looks rich). Defaults to ``False`` (the
        requested long-only behavior).
    trading_days:
        Trading days per year, used to annualize the Sharpe ratio.
    annual_risk_free:
        Annual risk-free rate subtracted from returns when computing Sharpe.
        Defaults to ``0.0``.
    """

    _BASE_REQUIRED: Final[tuple[str, ...]] = (
        "date",
        "strike",
        "spot_price",
        "expiration_date",
        "option_price",
        "moneyness",
    )

    def __init__(
        self,
        threshold: float = 0.0,
        holding_period: int | None = None,
        pred_col: str = "predicted_rv",
        iv_col: str = "implied_volatility",
        cost_pct: float = 0.0,
        allow_short: bool = False,
        trading_days: int = 252,
        annual_risk_free: float = 0.0,
    ) -> None:
        self.threshold = float(threshold)
        self.holding_period = None if holding_period is None else int(holding_period)
        self.pred_col = pred_col
        self.iv_col = iv_col
        self.cost_pct = float(cost_pct)
        self.allow_short = bool(allow_short)
        self.trading_days = int(trading_days)
        self.annual_risk_free = float(annual_risk_free)

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Execute the full backtest and return a :class:`BacktestResult`."""
        self._validate(df)
        legs = self._pair_legs(df)
        signals = self._select_atm(legs)
        signals = self._generate_signals(signals)
        trades = self._evaluate_pnl(signals, df)
        equity = self._equity_curve(trades)
        metrics = self.performance_report(trades)
        return BacktestResult(
            trades=trades, signals=signals, equity_curve=equity, metrics=metrics
        )

    # ------------------------------------------------------------------ #
    # Straddle construction
    # ------------------------------------------------------------------ #
    def _pair_legs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pair each call with the put at the same (date, strike).

        Returns one row per (date, strike) carrying both leg prices and IVs, so
        a straddle can be priced directly. An inner join guarantees both legs
        exist for every candidate strike.
        """
        d = df.copy()
        d["_is_call"] = self._is_call(d)

        shared = ["date", "strike", "spot_price", "expiration_date", "moneyness"]
        if self.pred_col in d.columns:
            shared.append(self.pred_col)

        calls = (
            d.loc[d["_is_call"], shared + ["option_price", self.iv_col]]
            .rename(columns={"option_price": "call_price", self.iv_col: "call_iv"})
        )
        puts = (
            d.loc[~d["_is_call"], ["date", "strike", "option_price", self.iv_col]]
            .rename(columns={"option_price": "put_price", self.iv_col: "put_iv"})
        )
        legs = calls.merge(puts, on=["date", "strike"], how="inner")
        return legs

    def _select_atm(self, legs: pd.DataFrame) -> pd.DataFrame:
        """Pick the single ATM straddle (moneyness closest to 1) per date."""
        legs = legs.copy()
        legs["atm_dist"] = (legs["moneyness"] - 1.0).abs()
        atm = legs.loc[legs.groupby("date")["atm_dist"].idxmin()].reset_index(drop=True)

        atm["premium"] = atm["call_price"] + atm["put_price"]
        atm["atm_iv"] = 0.5 * (atm["call_iv"] + atm["put_iv"])
        # Keep only economically valid straddles (positive premium).
        return atm.loc[atm["premium"] > 0].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Signal generation
    # ------------------------------------------------------------------ #
    def _generate_signals(self, atm: pd.DataFrame) -> pd.DataFrame:
        """Compute the vol spread and map it to BUY / SELL / FLAT signals."""
        atm = atm.copy()
        atm["spread"] = atm[self.pred_col] - atm["atm_iv"]

        signal = np.where(atm["spread"] > self.threshold, "BUY", "FLAT")
        if self.allow_short:
            signal = np.where(atm["spread"] < -self.threshold, "SELL", signal)
        atm["signal"] = signal
        atm["direction"] = np.select(
            [atm["signal"] == "BUY", atm["signal"] == "SELL"],
            [1.0, -1.0],
            default=0.0,
        )
        return atm

    # ------------------------------------------------------------------ #
    # PnL evaluation
    # ------------------------------------------------------------------ #
    def _evaluate_pnl(self, signals: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Settle every executed straddle and compute per-trade PnL and return.

        Exit spot is located by positional indexing into the sorted date axis,
        so the whole book is priced without a Python loop.
        """
        executed = signals.loc[signals["direction"] != 0.0].copy()
        if executed.empty:
            return self._empty_trades()

        spot_by_date = df.groupby("date")["spot_price"].first().sort_index()
        didx = pd.DatetimeIndex(spot_by_date.index)
        spot_values = spot_by_date.to_numpy(dtype=float)
        last_date = didx[-1]

        entry_pos = didx.get_indexer(pd.DatetimeIndex(executed["date"]))
        # Position of the last trading day at or before expiration.
        exp_pos = didx.get_indexer(
            pd.DatetimeIndex(executed["expiration_date"]), method="ffill"
        )

        if self.holding_period is None:
            exit_pos = exp_pos
        else:
            exit_pos = np.minimum(entry_pos + self.holding_period, exp_pos)

        # A trade is settleable only if entry and expiration fall inside the
        # sample; otherwise we cannot observe the exit spot without look-ahead.
        valid = (
            (entry_pos >= 0)
            & (exp_pos >= 0)
            & (executed["expiration_date"].to_numpy() <= np.datetime64(last_date))
        )
        executed = executed.loc[valid].copy()
        if executed.empty:
            return self._empty_trades()

        entry_pos, exit_pos = entry_pos[valid], exit_pos[valid]
        exit_pos = np.clip(exit_pos, 0, len(didx) - 1)

        exit_spot = spot_values[exit_pos]
        straddle_value = np.abs(exit_spot - executed["strike"].to_numpy(dtype=float))

        premium = executed["premium"].to_numpy(dtype=float)
        direction = executed["direction"].to_numpy(dtype=float)
        costs = self.cost_pct * premium
        pnl = direction * (straddle_value - premium) - costs

        executed["exit_date"] = didx[exit_pos]
        executed["exit_spot"] = exit_spot
        executed["straddle_value"] = straddle_value
        executed["holding_days"] = exit_pos - entry_pos
        executed["cost"] = costs
        executed["pnl"] = pnl
        executed["trade_return"] = pnl / premium

        return executed.sort_values("exit_date").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def performance_report(self, trades: pd.DataFrame) -> dict[str, float]:
        """Summarize a set of trades into a performance dictionary.

        Returns Total Return (return on capital deployed), Annualized Sharpe
        Ratio and Maximum Drawdown, along with supporting statistics.

        Total Return and drawdown use the fixed-capital additive equity model
        described in :meth:`_equity_curve`. The Sharpe ratio is computed on the
        equal-weighted per-trade return distribution and annualized by
        ``trading_days / mean_holding_days`` — how many non-overlapping holding
        periods fit in a year — net of the per-period ``annual_risk_free``.
        """
        n = int(len(trades))
        if n == 0:
            return self._empty_metrics()

        pnl = trades["pnl"].to_numpy(dtype=float)
        premium = trades["premium"].to_numpy(dtype=float)
        r = trades["trade_return"].to_numpy(dtype=float)

        total_premium = float(premium.sum())
        total_return = float(pnl.sum() / total_premium) if total_premium > 0 else np.nan

        mean_hold = float(trades["holding_days"].mean())
        periods_per_year = self.trading_days / mean_hold if mean_hold > 0 else np.nan
        rf_period = self.annual_risk_free * (mean_hold / self.trading_days)

        std = float(np.std(r, ddof=1)) if n > 1 else np.nan
        if n > 1 and std > 0 and np.isfinite(periods_per_year):
            sharpe = (float(np.mean(r)) - rf_period) / std * np.sqrt(periods_per_year)
        else:
            sharpe = np.nan

        # Fixed-capital additive equity curve (base 1.0); safe against <=-100%
        # trade returns that would break geometric compounding.
        equity = 1.0 + np.cumsum(pnl) / total_premium
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = float(drawdown.min())

        return {
            "num_trades": n,
            "total_return": total_return,
            "annualized_sharpe": float(sharpe),
            "max_drawdown": max_drawdown,
            "win_rate": float(np.mean(r > 0)),
            "avg_trade_return": float(np.mean(r)),
            "return_volatility": std,
            "avg_holding_days": mean_hold,
            "avg_spread": float(trades["spread"].mean()),
            "total_pnl": float(trades["pnl"].sum()),
            "total_premium": float(trades["premium"].sum()),
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _equity_curve(trades: pd.DataFrame) -> pd.Series:
        """Capital-normalized equity curve (base 1.0), ordered by exit date.

        Uses a fixed-capital model: each trade deploys its own premium and PnL
        accrues additively, normalized by total premium deployed. This avoids
        the pathologies of geometric compounding here — trade returns can be
        ``<= -100%`` (a max-loss long straddle plus costs) and 21-day holds
        opened daily overlap, so reinvesting a compounding book is ill-defined.
        """
        if trades.empty:
            return pd.Series(dtype=float, name="equity")
        total_premium = trades["premium"].sum()
        eq = 1.0 + trades["pnl"].cumsum() / total_premium
        eq.index = pd.DatetimeIndex(trades["exit_date"])
        return eq.rename("equity")

    def _is_call(self, df: pd.DataFrame) -> pd.Series:
        """Boolean call/put flag, preferring an existing ``is_call`` column."""
        if "is_call" in df.columns:
            return df["is_call"].astype(bool)
        if "option_type" in df.columns:
            return (
                df["option_type"].astype(str).str.strip().str.lower().str.startswith("c")
            )
        raise ValueError("Input must contain either 'is_call' or 'option_type'.")

    def _validate(self, df: pd.DataFrame) -> None:
        required = list(self._BASE_REQUIRED) + [self.iv_col, self.pred_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[StraddleBacktester] missing required columns: {missing}.")
        if "is_call" not in df.columns and "option_type" not in df.columns:
            raise ValueError("Input must contain either 'is_call' or 'option_type'.")

    @staticmethod
    def _empty_trades() -> pd.DataFrame:
        cols = [
            "date", "strike", "spot_price", "expiration_date", "premium",
            "atm_iv", "spread", "signal", "direction", "exit_date", "exit_spot",
            "straddle_value", "holding_days", "cost", "pnl", "trade_return",
        ]
        return pd.DataFrame(columns=cols)

    @staticmethod
    def _empty_metrics() -> dict[str, float]:
        return {
            "num_trades": 0,
            "total_return": 0.0,
            "annualized_sharpe": float("nan"),
            "max_drawdown": 0.0,
            "win_rate": float("nan"),
            "avg_trade_return": float("nan"),
            "return_volatility": float("nan"),
            "avg_holding_days": float("nan"),
            "avg_spread": float("nan"),
            "total_pnl": 0.0,
            "total_premium": 0.0,
        }
