"""Options-chain data processing and feature engineering.

This module exposes :class:`DataProcessor`, which ingests a raw options-chain
DataFrame, backs out Black-Scholes implied volatility via a vectorized
Newton-Raphson solver, and engineers a clean, model-ready feature matrix.

The design follows three principles enforced across the project:

* **Vectorized** — all pricing, root-finding, and feature math operate on
  NumPy arrays / pandas Series; no per-row Python loops.
* **Defensive** — market data is assumed dirty. Inputs are validated and
  coerced, non-economic rows (non-positive prices, strikes, spots, or
  sub-intrinsic quotes) are removed, and solver failures degrade to ``NaN``
  rather than raising.
* **Self-documenting** — every public method carries a docstring and full
  type hints.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from scipy.optimize import newton
from scipy.stats import norm

__all__ = ["DataProcessor"]


class DataProcessor:
    """Ingest an options chain and engineer volatility-modeling features.

    Parameters
    ----------
    df:
        Raw options-chain data. Must contain the columns listed in
        :attr:`REQUIRED_COLUMNS`.
    trading_days:
        Number of trading days per year, used to annualize realized
        volatility and to express time-to-expiry as a year fraction.
        Defaults to ``252``.
    rv_window:
        Look-back window (in trading days) for the rolling realized-volatility
        estimate. Defaults to ``21`` (~one calendar month).

    Notes
    -----
    ``option_type`` values are normalized case-insensitively; any value whose
    first character is ``c`` is treated as a call, everything else as a put.
    """

    REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
        "date",
        "strike",
        "spot_price",
        "option_type",
        "expiration_date",
        "option_price",
        "risk_free_rate",
    )

    # Solver / sanity bounds.
    _MIN_IV: Final[float] = 1e-4       # 0.01% vol floor
    _MAX_IV: Final[float] = 5.0        # 500% vol ceiling
    _PRICE_TOL: Final[float] = 1e-4    # max acceptable pricing error at solution
    _SIGMA_FLOOR: Final[float] = 1e-8  # numerical guard against sigma -> 0

    def __init__(
        self,
        df: pd.DataFrame,
        trading_days: int = 252,
        rv_window: int = 21,
    ) -> None:
        self.trading_days: int = int(trading_days)
        self.rv_window: int = int(rv_window)
        self.raw: pd.DataFrame = self._validate_and_coerce(df)
        # Populated by :meth:`process`.
        self.processed: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Validation / cleansing
    # ------------------------------------------------------------------ #
    def _validate_and_coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate schema and coerce raw columns to usable dtypes.

        Missing required columns raise ``ValueError``. Dates are parsed and
        numeric fields coerced; unparseable entries become ``NaT``/``NaN`` and
        are handled downstream during cleansing.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df)!r}.")

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing columns: {missing}.")

        out = df.loc[:, list(self.REQUIRED_COLUMNS)].copy()

        # Dates -> datetime64 (invalid -> NaT).
        for col in ("date", "expiration_date"):
            out[col] = pd.to_datetime(out[col], errors="coerce")

        # Numerics -> float (invalid -> NaN).
        numeric_cols = ("strike", "spot_price", "option_price", "risk_free_rate")
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        # Normalize option type to a boolean is_call flag.
        out["is_call"] = (
            out["option_type"].astype(str).str.strip().str.lower().str.startswith("c")
        )

        return out

    @staticmethod
    def _drop_non_economic(df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows that cannot represent a valid option quote.

        Filters non-positive strikes/spots/prices and rows missing the fields
        required to price the option. Sub-intrinsic quotes are handled later,
        once time-to-expiry is known.
        """
        mask = (
            df["strike"].gt(0)
            & df["spot_price"].gt(0)
            & df["option_price"].gt(0)
            & df["risk_free_rate"].notna()
            & df["date"].notna()
            & df["expiration_date"].notna()
        )
        return df.loc[mask].copy()

    # ------------------------------------------------------------------ #
    # Black-Scholes primitives (vectorized)
    # ------------------------------------------------------------------ #
    @classmethod
    def _d1_d2(
        cls,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        sigma: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the Black-Scholes ``d1`` and ``d2`` terms elementwise."""
        sigma = np.maximum(sigma, cls._SIGMA_FLOOR)
        sqrt_T = np.sqrt(np.maximum(T, cls._SIGMA_FLOOR))
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return d1, d2

    @classmethod
    def _bs_price(
        cls,
        sigma: np.ndarray,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        is_call: np.ndarray,
    ) -> np.ndarray:
        """Black-Scholes price for European calls/puts (vectorized).

        Signature is ordered ``(sigma, ...)`` so the function can be handed
        directly to :func:`scipy.optimize.newton` as the objective's price leg.
        """
        d1, d2 = cls._d1_d2(S, K, T, r, sigma)
        disc = np.exp(-r * T)
        call = S * norm.cdf(d1) - K * disc * norm.cdf(d2)
        put = K * disc * norm.cdf(-d2) - S * norm.cdf(-d1)
        return np.where(is_call, call, put)

    @classmethod
    def _bs_vega(
        cls,
        sigma: np.ndarray,
        S: np.ndarray,
        K: np.ndarray,
        T: np.ndarray,
        r: np.ndarray,
        is_call: np.ndarray,  # unused; kept so newton can share one args tuple
        market: np.ndarray = 0.0,  # unused; newton passes the objective's args
    ) -> np.ndarray:
        """Black-Scholes vega (∂price/∂sigma), identical for calls and puts."""
        d1, _ = cls._d1_d2(S, K, T, r, sigma)
        return S * np.sqrt(np.maximum(T, cls._SIGMA_FLOOR)) * norm.pdf(d1)

    # ------------------------------------------------------------------ #
    # Implied volatility
    # ------------------------------------------------------------------ #
    def compute_implied_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Back out Black-Scholes implied volatility per row.

        Uses a vectorized Newton-Raphson iteration (:func:`scipy.optimize.newton`
        with analytic vega as the derivative). Rows that fail to converge, land
        outside ``[_MIN_IV, _MAX_IV]``, or fail to reprice the market quote
        within ``_PRICE_TOL`` are returned as ``NaN``.

        Parameters
        ----------
        df:
            Frame containing ``spot_price``, ``strike``, ``time_to_expiry``,
            ``risk_free_rate``, ``option_price`` and ``is_call`` columns.

        Returns
        -------
        pandas.Series
            Implied volatility aligned to ``df.index``.
        """
        S = df["spot_price"].to_numpy(dtype=float)
        K = df["strike"].to_numpy(dtype=float)
        T = df["time_to_expiry"].to_numpy(dtype=float)
        r = df["risk_free_rate"].to_numpy(dtype=float)
        market = df["option_price"].to_numpy(dtype=float)
        is_call = df["is_call"].to_numpy(dtype=bool)

        # Brenner-Subrahmanyam ATM approximation as a warm start, floored to a
        # sane default where it is undefined.
        with np.errstate(divide="ignore", invalid="ignore"):
            guess = np.sqrt(2.0 * np.pi / np.where(T > 0, T, np.nan)) * (market / S)
        guess = np.where(np.isfinite(guess) & (guess > 0), guess, 0.2)
        guess = np.clip(guess, self._MIN_IV, self._MAX_IV)

        def objective(
            sigma: np.ndarray,
            S: np.ndarray,
            K: np.ndarray,
            T: np.ndarray,
            r: np.ndarray,
            is_call: np.ndarray,
            market: np.ndarray,
        ) -> np.ndarray:
            return self._bs_price(sigma, S, K, T, r, is_call) - market

        # scipy.optimize.newton operates elementwise on array inputs. Failed
        # entries are returned (with a RuntimeWarning) rather than raising; we
        # discard them via the repricing check below.
        with np.errstate(all="ignore"):
            iv = newton(
                objective,
                x0=guess,
                fprime=self._bs_vega,
                args=(S, K, T, r, is_call, market),
                maxiter=100,
                tol=1e-8,
            )
            iv = np.asarray(iv, dtype=float)

            # Reject out-of-bounds or non-converged solutions by repricing.
            reprice_err = np.abs(
                self._bs_price(iv, S, K, T, r, is_call) - market
            )
            valid = (
                np.isfinite(iv)
                & (iv >= self._MIN_IV)
                & (iv <= self._MAX_IV)
                & (reprice_err <= self._PRICE_TOL)
            )

        iv = np.where(valid, iv, np.nan)
        return pd.Series(iv, index=df.index, name="implied_volatility")

    # ------------------------------------------------------------------ #
    # Feature engineering
    # ------------------------------------------------------------------ #
    def _compute_time_to_expiry(self, df: pd.DataFrame) -> pd.Series:
        """Trading-day time-to-expiry as a fraction of a year.

        Counts business days between ``date`` and ``expiration_date`` (a
        holiday-agnostic proxy for trading days) and divides by
        ``self.trading_days``.
        """
        start = df["date"].to_numpy(dtype="datetime64[D]")
        end = df["expiration_date"].to_numpy(dtype="datetime64[D]")
        busdays = np.busday_count(start, end).astype(float)
        return pd.Series(
            busdays / self.trading_days, index=df.index, name="time_to_expiry"
        )

    def _compute_realized_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Annualized rolling realized volatility of the underlying spot.

        Realized volatility is a property of the underlying time series, not of
        an individual option, so it is computed once per trading date from the
        daily spot series and then broadcast back to every option on that date.

        The estimate is the rolling standard deviation of daily log returns over
        ``self.rv_window`` observations, annualized by ``sqrt(trading_days)``.
        """
        spot_by_date = df.groupby("date")["spot_price"].first().sort_index()
        log_ret = np.log(spot_by_date / spot_by_date.shift(1))
        rv = log_ret.rolling(window=self.rv_window).std() * np.sqrt(self.trading_days)
        # Broadcast the per-date value back onto every option row.
        return df["date"].map(rv).rename("realized_volatility")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach moneyness, time-to-expiry, realized vol, and implied vol.

        Adds the following columns and drops sub-intrinsic option quotes (which
        admit no positive implied volatility) once ``time_to_expiry`` is known:

        * ``moneyness`` — spot / strike (S/K)
        * ``time_to_expiry`` — trading-day year fraction
        * ``realized_volatility`` — annualized 21-day rolling RV
        * ``implied_volatility`` — Black-Scholes IV
        """
        out = df.copy()
        out["moneyness"] = out["spot_price"] / out["strike"]
        out["time_to_expiry"] = self._compute_time_to_expiry(out)
        out["realized_volatility"] = self._compute_realized_volatility(out)

        # Keep only live options (positive time to expiry) whose quote is at
        # least the discounted intrinsic value; otherwise IV is undefined.
        disc = np.exp(-out["risk_free_rate"] * out["time_to_expiry"])
        intrinsic = np.where(
            out["is_call"],
            np.maximum(out["spot_price"] - out["strike"] * disc, 0.0),
            np.maximum(out["strike"] * disc - out["spot_price"], 0.0),
        )
        live = out["time_to_expiry"].gt(0) & (out["option_price"] >= intrinsic)
        out = out.loc[live].copy()

        out["implied_volatility"] = self.compute_implied_volatility(out)
        return out

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def process(self, feature_only: bool = False) -> pd.DataFrame:
        """Run the full pipeline and return a clean, model-ready DataFrame.

        Steps: cleanse non-economic rows → engineer features → drop any row
        with a missing engineered feature so that Black-Scholes IV and realized
        volatility are cleanly aligned.

        Parameters
        ----------
        feature_only:
            If ``True``, return only the modeling columns (identifiers plus
            engineered features). If ``False`` (default), return the full
            enriched frame.

        Returns
        -------
        pandas.DataFrame
            Cleansed data indexed ``0..n-1`` with no NaNs in the feature set.
        """
        cleansed = self._drop_non_economic(self.raw)
        featured = self.engineer_features(cleansed)

        feature_cols = [
            "moneyness",
            "time_to_expiry",
            "realized_volatility",
            "implied_volatility",
        ]
        featured = featured.dropna(subset=feature_cols).reset_index(drop=True)

        if feature_only:
            id_cols = ["date", "strike", "option_type", "expiration_date"]
            featured = featured.loc[:, id_cols + feature_cols]

        self.processed = featured
        return featured
