"""Volatility-forecasting models and evaluation framework.

This module turns the feature matrix produced by
:class:`src.data_processor.DataProcessor` into a supervised learning problem —
forecasting *future* realized volatility — and provides two competing models
plus a shared evaluation harness:

* :class:`GARCHVolatilityModel` — a classical GARCH(1,1) baseline (via the
  ``arch`` library) that forecasts volatility purely from the underlying's
  return history.
* :class:`MLVolatilityModel` — a scikit-learn pipeline (Random Forest or Ridge)
  that forecasts future realized volatility from the engineered option
  features, crucially including Black-Scholes implied volatility.
* :class:`ModelEvaluator` — computes MSE / MAE / R² so the ML model can be
  benchmarked against the statistical baseline on an identical test set.

Design notes
------------
* **No look-ahead.** The train/test split is strictly time-ordered
  (:func:`train_test_split_by_date`); the target is a *forward* realized-vol
  window built by shifting the daily RV series into the future.
* **Comparable surface.** Realized volatility is a property of the underlying,
  so both models are compared at the *daily* level: the GARCH model is
  natively per-date, and per-option ML predictions are aggregated to a per-date
  forecast via :func:`to_daily`.
* **Defensive.** Dirty inputs, missing dependencies, and non-converged fits
  raise clear errors rather than failing silently.
"""

from __future__ import annotations

from typing import Final, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # arch is an optional-at-import-time dependency; required to fit GARCH.
    from arch import arch_model

    _HAS_ARCH = True
except ImportError:  # pragma: no cover - exercised only when arch is absent
    arch_model = None  # type: ignore[assignment]
    _HAS_ARCH = False

__all__ = [
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "build_supervised_frame",
    "daily_returns",
    "train_test_split_by_date",
    "to_daily",
    "GARCHVolatilityModel",
    "MLVolatilityModel",
    "ModelEvaluator",
]

# Engineered features consumed by the ML model (see data_processor.py).
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "moneyness",
    "time_to_expiry",
    "realized_volatility",
    "implied_volatility",
)

# Forward realized-volatility target built by :func:`build_supervised_frame`.
TARGET_COLUMN: Final[str] = "future_realized_volatility"


# ====================================================================== #
# Dataset preparation
# ====================================================================== #
def build_supervised_frame(
    processed: pd.DataFrame,
    horizon: int = 21,
    date_col: str = "date",
    rv_col: str = "realized_volatility",
) -> pd.DataFrame:
    """Attach the forward realized-volatility target to a processed chain.

    The target for a given trading date ``t`` is the realized volatility
    observed ``horizon`` trading days later — i.e. the RV computed over the
    window ending at ``t + horizon``. It is built from the *daily* RV series
    (one value per date) and broadcast back to every option row on that date.
    Rows for which no future value exists (the tail of the sample) are dropped.

    Parameters
    ----------
    processed:
        Output of :meth:`DataProcessor.process`; must contain ``date`` and
        ``realized_volatility``.
    horizon:
        Forecast horizon in trading days. Match this to the RV window used in
        ``DataProcessor`` (default 21) so target and forecast are consistent.

    Returns
    -------
    pandas.DataFrame
        Copy of ``processed`` with a ``future_realized_volatility`` column and
        no NaN targets.
    """
    _require_columns(processed, (date_col, rv_col), context="build_supervised_frame")
    if horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon}.")

    out = processed.copy()
    rv_by_date = out.groupby(date_col)[rv_col].first().sort_index()
    future_rv = rv_by_date.shift(-horizon)
    out[TARGET_COLUMN] = out[date_col].map(future_rv)
    return out.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)


def daily_returns(
    processed: pd.DataFrame,
    date_col: str = "date",
    spot_col: str = "spot_price",
) -> pd.Series:
    """Daily log-return series of the underlying, indexed by date.

    Required to fit the GARCH baseline. One spot per date is taken (spot is a
    property of the underlying, not the option), so the input frame must retain
    ``spot_price`` — i.e. call ``DataProcessor.process(feature_only=False)``.
    """
    _require_columns(processed, (date_col, spot_col), context="daily_returns")
    spot = processed.groupby(date_col)[spot_col].first().sort_index()
    rets = np.log(spot / spot.shift(1)).dropna()
    return rets.rename("returns")


def train_test_split_by_date(
    df: pd.DataFrame,
    test_size: float = 0.2,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Chronological train/test split that never leaks the future.

    Unique dates are ordered and the last ``test_size`` fraction of *dates*
    (not rows) forms the test set, so every option observed on a given date
    lands in the same split.

    Returns
    -------
    (train, test, split_date)
        ``split_date`` is the first date belonging to the test set.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}.")

    dates = np.sort(df[date_col].unique())
    if len(dates) < 2:
        raise ValueError("Need at least two distinct dates to split.")

    cut = int(np.floor(len(dates) * (1.0 - test_size)))
    cut = min(max(cut, 1), len(dates) - 1)  # guarantee both sides non-empty
    split_date = pd.Timestamp(dates[cut])

    train = df.loc[df[date_col] < split_date].copy()
    test = df.loc[df[date_col] >= split_date].copy()
    return train, test, split_date


def to_daily(
    values: pd.Series,
    dates: pd.Series,
    agg: str = "mean",
) -> pd.Series:
    """Collapse a per-row series to one value per date.

    Used to reduce per-option ML predictions (or targets) to the per-date level
    at which the GARCH baseline lives, giving both models an identical
    evaluation surface.

    Parameters
    ----------
    values:
        Per-row quantity (e.g. predictions), aligned to ``dates``.
    dates:
        Trading date for each row, same index/length as ``values``.
    agg:
        Aggregation applied within each date (``"mean"`` by default).
    """
    frame = pd.DataFrame({"date": np.asarray(dates), "value": np.asarray(values)})
    return frame.groupby("date")["value"].agg(agg).sort_index()


# ====================================================================== #
# Baseline: GARCH(1,1)
# ====================================================================== #
class GARCHVolatilityModel:
    """GARCH(1,1) volatility forecaster (statistical baseline).

    Wraps :func:`arch.arch_model`. Returns are rescaled by ``rescale`` (100 by
    default, per ``arch`` guidance) for numerical stability; forecasts are
    converted back to decimal, averaged over the horizon, and annualized so
    they are directly comparable to the annualized realized-volatility target.

    Parameters
    ----------
    horizon:
        Forecast horizon in trading days; should match the target horizon used
        in :func:`build_supervised_frame`.
    p, q:
        GARCH lag orders (defaults give the standard GARCH(1,1)).
    mean, dist:
        Mean model and error distribution passed through to ``arch``.
    trading_days:
        Annualization factor for volatility.
    rescale:
        Multiplicative scaling applied to returns before fitting.
    """

    def __init__(
        self,
        horizon: int = 21,
        p: int = 1,
        q: int = 1,
        mean: str = "Constant",
        dist: str = "normal",
        trading_days: int = 252,
        rescale: float = 100.0,
    ) -> None:
        if not _HAS_ARCH:
            raise ImportError(
                "The 'arch' package is required for GARCHVolatilityModel. "
                "Install it with `pip install arch`."
            )
        self.horizon = int(horizon)
        self.p = int(p)
        self.q = int(q)
        self.mean = mean
        self.dist = dist
        self.trading_days = int(trading_days)
        self.rescale = float(rescale)
        self.result_ = None  # populated by fit()
        self.returns_: pd.Series | None = None

    def fit(
        self,
        returns: pd.Series,
        last_obs: pd.Timestamp | None = None,
    ) -> "GARCHVolatilityModel":
        """Estimate GARCH parameters on the return history.

        Parameters
        ----------
        returns:
            Daily *decimal* log returns indexed by date.
        last_obs:
            If given, parameters are estimated only on observations strictly
            before this date (the rest are held out for forecasting). This is
            the canonical ``arch`` pattern for out-of-sample evaluation.
        """
        rets = pd.Series(returns).dropna().astype(float).sort_index()
        if len(rets) < 2 * (self.p + self.q) + 10:
            raise ValueError(
                f"Insufficient return history ({len(rets)} obs) to fit "
                f"GARCH({self.p},{self.q})."
            )
        self.returns_ = rets
        am = arch_model(
            rets * self.rescale,
            vol="GARCH",
            p=self.p,
            q=self.q,
            mean=self.mean,
            dist=self.dist,
            rescale=False,
        )
        self.result_ = am.fit(disp="off", last_obs=last_obs)
        return self

    def forecast_vol(self, start: pd.Timestamp | None = None) -> pd.Series:
        """Annualized volatility forecast per date from ``start`` onward.

        For each date ``t`` the ``arch`` analytic forecast produces the
        conditional variance for steps ``1..horizon``; these are averaged
        (expected variance over the window), de-scaled, and annualized.

        Returns
        -------
        pandas.Series
            Annualized vol forecast indexed by date, named ``garch_forecast``.
        """
        self._check_fitted()
        fc = self.result_.forecast(horizon=self.horizon, start=start, reindex=False)
        # Mean daily variance over the horizon, undo the return rescaling.
        avg_daily_var = fc.variance.mean(axis=1) / (self.rescale**2)
        annual_vol = np.sqrt(avg_daily_var * self.trading_days)
        return annual_vol.rename("garch_forecast")

    def predict(self, dates: Sequence[pd.Timestamp]) -> pd.Series:
        """Return annualized vol forecasts aligned to ``dates``."""
        target = pd.DatetimeIndex(pd.to_datetime(list(dates))).unique().sort_values()
        forecast = self.forecast_vol(start=target.min())
        forecast.index = pd.to_datetime(forecast.index)
        return forecast.reindex(target)

    def _check_fitted(self) -> None:
        if self.result_ is None:
            raise RuntimeError("Model is not fitted; call fit() first.")


# ====================================================================== #
# Machine-learning model
# ====================================================================== #
class MLVolatilityModel:
    """Supervised ML forecaster of future realized volatility.

    A scikit-learn :class:`~sklearn.pipeline.Pipeline` of
    :class:`~sklearn.preprocessing.StandardScaler` followed by either a Random
    Forest (default) or Ridge regressor. Scaling is a no-op for the forest but
    essential for Ridge, so the pipeline is uniform across estimators.

    Parameters
    ----------
    model_type:
        ``"rf"`` for :class:`RandomForestRegressor` or ``"ridge"`` for
        :class:`Ridge`.
    features:
        Feature columns to consume. Defaults to :data:`FEATURE_COLUMNS`, which
        includes Black-Scholes implied volatility.
    target:
        Target column name (defaults to :data:`TARGET_COLUMN`).
    **model_kwargs:
        Passed through to the underlying estimator, overriding the defaults.
    """

    _DEFAULTS: Final[dict[str, dict]] = {
        "rf": dict(n_estimators=300, max_depth=None, min_samples_leaf=2,
                   random_state=42, n_jobs=-1),
        "ridge": dict(alpha=1.0, random_state=42),
    }

    def __init__(
        self,
        model_type: Literal["rf", "ridge"] = "rf",
        features: Sequence[str] = FEATURE_COLUMNS,
        target: str = TARGET_COLUMN,
        **model_kwargs,
    ) -> None:
        self.model_type = model_type
        self.features = list(features)
        self.target = target
        self.pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", self._build_estimator(model_type, model_kwargs)),
            ]
        )
        self.fitted_: bool = False

    @classmethod
    def _build_estimator(cls, model_type: str, overrides: dict) -> RegressorMixin:
        if model_type not in cls._DEFAULTS:
            raise ValueError(
                f"model_type must be one of {sorted(cls._DEFAULTS)}, got {model_type!r}."
            )
        params = {**cls._DEFAULTS[model_type], **overrides}
        if model_type == "rf":
            return RandomForestRegressor(**params)
        # Ridge has no random_state; drop it silently if left at the default.
        params.pop("random_state", None)
        return Ridge(**params)

    def fit(self, df: pd.DataFrame) -> "MLVolatilityModel":
        """Fit the pipeline on a supervised frame containing features + target."""
        X, y = self._xy(df, require_target=True)
        self.pipeline.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict future realized volatility, indexed like ``df``."""
        if not self.fitted_:
            raise RuntimeError("Model is not fitted; call fit() first.")
        X, _ = self._xy(df, require_target=False)
        preds = self.pipeline.predict(X)
        return pd.Series(preds, index=df.index, name="ml_forecast")

    @property
    def feature_importances_(self) -> pd.Series:
        """Feature importances (RF) or absolute coefficients (Ridge)."""
        if not self.fitted_:
            raise RuntimeError("Model is not fitted; call fit() first.")
        model = self.pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        else:  # Ridge: use standardized coefficients' magnitude
            values = np.abs(model.coef_)
        return pd.Series(values, index=self.features, name="importance").sort_values(
            ascending=False
        )

    def _xy(
        self, df: pd.DataFrame, require_target: bool
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        needed = list(self.features) + ([self.target] if require_target else [])
        _require_columns(df, needed, context=f"{type(self).__name__}")
        X = df.loc[:, self.features].astype(float)
        if X.isna().any().any():
            raise ValueError(
                "Feature matrix contains NaNs; run DataProcessor.process() first."
            )
        y = df[self.target].astype(float) if require_target else None
        return X, y


# ====================================================================== #
# Evaluation
# ====================================================================== #
class ModelEvaluator:
    """Compute and tabulate regression error metrics across models.

    Accumulates results so several models can be compared side by side on the
    same held-out target.

    Examples
    --------
    >>> ev = ModelEvaluator()
    >>> ev.evaluate("GARCH(1,1)", y_true, garch_pred)      # doctest: +SKIP
    >>> ev.evaluate("RandomForest", y_true, ml_pred)       # doctest: +SKIP
    >>> ev.summary()                                       # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._results: dict[str, dict[str, float]] = {}

    @staticmethod
    def metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
        """Return MSE, RMSE, MAE and R² over aligned, finite observations.

        Non-finite pairs (NaN forecasts or targets) are dropped so a single bad
        row does not poison the comparison; the surviving sample size is
        reported as ``n``.
        """
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        if yt.shape != yp.shape:
            raise ValueError(
                f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}."
            )
        mask = np.isfinite(yt) & np.isfinite(yp)
        if mask.sum() == 0:
            raise ValueError("No finite (y_true, y_pred) pairs to evaluate.")
        yt, yp = yt[mask], yp[mask]
        mse = float(mean_squared_error(yt, yp))
        return {
            "n": int(mask.sum()),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "MAE": float(mean_absolute_error(yt, yp)),
            "R2": float(r2_score(yt, yp)),
        }

    def evaluate(
        self,
        name: str,
        y_true: Sequence[float],
        y_pred: Sequence[float],
    ) -> dict[str, float]:
        """Compute metrics for ``name`` and store them for later comparison."""
        result = self.metrics(y_true, y_pred)
        self._results[name] = result
        return result

    def summary(self) -> pd.DataFrame:
        """Return all accumulated results as a tidy DataFrame (one row/model)."""
        if not self._results:
            raise RuntimeError("No results recorded; call evaluate() first.")
        return (
            pd.DataFrame.from_dict(self._results, orient="index")
            .rename_axis("model")
            .loc[:, ["n", "MSE", "RMSE", "MAE", "R2"]]
        )


# ====================================================================== #
# Internal helpers
# ====================================================================== #
def _require_columns(
    df: pd.DataFrame, columns: Sequence[str], context: str
) -> None:
    """Raise a clear ``ValueError`` if any required column is absent."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] missing required columns: {missing}.")
