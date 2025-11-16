#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 15:09:58 2025

@author: eoinmoran
"""

"""
delta_hedging.py

Delta-hedging simulation for a short ATM call, using either:
- implied volatility (per day or fixed), or
- rolling historical volatility (e.g. 22-day)

Data-loading is left as placeholders – you just need to supply a pandas
DataFrame with at least: ['date', 'spot'] and either vol series or a constant vol.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
# from typing import Optional
import warnings
warnings.filterwarnings(
    "ignore",
    message="Parsing dates in .*%Y-%m-%d.*dayfirst=True.*"
)


# =========================
# Black–Scholes utilities
# =========================

def bs_call_price(spot: float, strike: float, tau: float, r: float, sigma: float) -> float:
    """
    Black–Scholes call price for non-dividend-paying stock.
    """
    if tau <= 0 or sigma <= 0:
        return max(spot - strike, 0.0)

    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    from scipy.stats import norm
    return spot * norm.cdf(d1) - strike * np.exp(-r * tau) * norm.cdf(d2)


def bs_call_delta(spot: float, strike: float, tau: float, r: float, sigma: float) -> float:
    """
    Black–Scholes delta of a call (∂C/∂S).
    """
    if tau <= 0 or sigma <= 0:
        # At expiry, delta is 1 if in the money, 0 otherwise (ignoring boundary)
        return 1.0 if spot > strike else 0.0

    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    from scipy.stats import norm
    return norm.cdf(d1)


# =========================
# Data model
# =========================

@dataclass
class HedgeResult:
    df: pd.DataFrame          # per-step details
    final_pnl: float          # total P&L at expiry
    premium_received: float   # initial option premium
    payoff: float             # option payoff at expiry


# =========================
# Core delta-hedging engine
# =========================

def simulate_delta_hedge(
    df: pd.DataFrame,
    strike: float,
    r_annual: float,
    vol_series: pd.Series,
    position_short_calls: float = 1.0,
    steps_per_year: int = 252,
    premium_override: float = 0.0,
) -> HedgeResult:
    """
    Simulate delta-hedging of a short call option.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least:
            'date' : datetime-like
            'spot' : underlying close price
        One row per hedging date, from trade date (t=0) to expiry (t=T).

    strike : float
        Option strike K.

    r_annual : float
        Annual risk-free rate (e.g. 0.02 for 2%).

    vol_series : pd.Series
        Volatility to use at each step (same index as df).
        This can be:
            - implied vol per day, or
            - rolling realized vol estimate, etc.

    position_short_calls : float
        Number of calls sold (positive for short; default = 1.0).

    steps_per_year : int
        Trading days per year, used for dt.

    Returns
    -------
    HedgeResult
        Contains P&L path, final P&L, premium, payoff.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    if n < 2:
        raise ValueError("Need at least two rows: trade date and expiry.")

    dt = 1.0 / steps_per_year
    r_daily = r_annual  # if you prefer exp(r*dt), we use it below with tau

    # Time to expiry τ_t in years (T at final row)
    # τ at row i = (n-1 - i) * dt
    df["tau"] = (n - 1 - df.index) * dt

    # Attach vol series
    df["sigma"] = vol_series.values

    # Compute deltas for each day
    deltas = []
    for i, row in df.iterrows():
        S_t = row["spot"]
        tau_t = row["tau"]
        sigma_t = row["sigma"]
        delta_t = bs_call_delta(S_t, strike, tau_t, r_annual, sigma_t)
        deltas.append(delta_t)
    df["delta"] = deltas

    # Initial option premium (mark-to-model using day 0 sigma)
    S0 = df.loc[0, "spot"]
    tau0 = df.loc[0, "tau"]
    sigma0 = df.loc[0, "sigma"]
    if premium_override == 0:
        premium = bs_call_price(S0, strike, tau0, r_annual, sigma0)
    else:
        premium = premium_override
        

    # Position: short 'position_short_calls' calls
    short_calls = position_short_calls

    # Hedge: we hold N_t shares of underlying.
    # For a short call, delta of the *option* is delta_t (positive).
    # To hedge, we go LONG delta_t * short_calls shares.
    df["shares"] = 0.0
    df["cash"] = 0.0

    # ---------- t = 0 (trade date) ----------
    delta_0 = df.loc[0, "delta"]
    N_0 = short_calls * delta_0  # underlying shares
    df.loc[0, "shares"] = N_0

    # Cash: receive premium from selling calls, pay for shares
    cash_0 = premium * short_calls - N_0 * S0
    df.loc[0, "cash"] = cash_0

    # Track portfolio value
    df["portfolio_value"] = 0.0
    df.loc[0, "portfolio_value"] = df.loc[0, "cash"] + df.loc[0, "shares"] * S0

    # ---------- Re-hedging from t=1 to t=T-1 ----------
    for t in range(1, n):
        S_t = df.loc[t, "spot"]
        S_prev = df.loc[t - 1, "spot"]

        cash_prev = df.loc[t - 1, "cash"]
        N_prev = df.loc[t - 1, "shares"]

        # Accrue risk-free interest on cash (continuously compounded)
        # If you want simple interest instead, use: cash_prev * (1 + r_annual*dt)
        cash_t = cash_prev * np.exp(r_annual * dt)

        # Recompute target delta (already in df["delta"])
        delta_t = df.loc[t, "delta"]
        N_target = short_calls * delta_t

        # Trade needed in underlying
        dN = N_target - N_prev

        # Buying dN shares costs dN * S_t (if dN>0), selling adds cash
        cash_t -= dN * S_t

        df.loc[t, "shares"] = N_target
        df.loc[t, "cash"] = cash_t

        df.loc[t, "portfolio_value"] = cash_t + N_target * S_t

    # ---------- Expiry settlement (last row) ----------
    S_T = df.loc[n - 1, "spot"]
    payoff = max(S_T - strike, 0.0) * short_calls  # we are short, so we pay this

    # We pay option payoff out of cash
    df.loc[n - 1, "cash"] -= payoff
    # We then liquidate underlying position at S_T
    final_shares = df.loc[n - 1, "shares"]
    df.loc[n - 1, "cash"] += final_shares * S_T
    df.loc[n - 1, "shares"] = 0.0

    # Final portfolio value is just cash now
    df.loc[n - 1, "portfolio_value"] = df.loc[n - 1, "cash"]

    final_pnl = df.loc[n - 1, "portfolio_value"]

    return HedgeResult(
        df=df,
        final_pnl=final_pnl,
        premium_received=premium * short_calls,
        payoff=payoff,
    )


# =========================
# PLACEHOLDER: wiring data
# =========================

def load_data_placeholder() -> pd.DataFrame:
    """
    Placeholder function.
    Replace this with real data loading from Excel / CSV.

    Expected output format:
        df with columns:
            'date' : datetime
            'spot' : float (underlying close price)
        One row per day from trade date (index 0) to expiry (last index).
    """
    # EXAMPLE DUMMY STRUCTURE (remove in real code):
    dates = pd.date_range("2025-09-19", periods=21, freq="B")
    spots = np.linspace(100, 110, len(dates))  # fake upward trend
    df = pd.DataFrame({"date": dates, "spot": spots})
    return df


def compute_vol_placeholder(df: pd.DataFrame) -> pd.Series:
    """
    Placeholder vol series.
    Replace with:
        - implied vol per date, OR
        - rolling 22-day realized vol, etc.

    For now, just returns a flat 29%, gotten from bbg.
    """
    return pd.Series(0.2917772293, index=df.index)

# =========================
# DATA WIRE: wiring actual data
# =========================
def load_underlying_from_excel(path: str) -> pd.DataFrame:
    # Your screenshot suggests:
    # rows 0–5 = metadata, row 6 = header ("Date", "PX_ADJ_CLOSE")
    df = pd.read_excel(path, skiprows=6, sheet_name="Stock Prices")   # adjust if header row changes

    # Keep the two useful columns
    df = df[["Date", "PX_ADJ_CLOSE"]].dropna(subset=["Date"])

    # Rename to what your hedging code expects
    df = df.rename(columns={
        "Date": "date",
        "PX_ADJ_CLOSE": "spot"
    })

    # Parse dates (day-first, since you have 17/10/2025)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    # Sort oldest → newest, reindex
    df = df.sort_values("date").reset_index(drop=True)

    return df


def compute_rolling_realized_vol(df: pd.DataFrame, window: int = 22) -> pd.Series:
    """
    Realised volatility from log returns, rolling window, annualised.

    df needs a 'spot' column.
    window = number of trading days in the lookback (e.g. 22 ≈ 1 month).
    """
    # log returns
    log_ret = np.log(df["spot"]).diff()

    # rolling std of daily returns
    daily_vol = log_ret.rolling(window=window).std()

    # annualise (assuming 252 trading days)
    ann_vol = daily_vol * np.sqrt(252)

    # Optional: forward-fill initial NaNs once you have your first estimate
    ann_vol = ann_vol.ffill()

    return ann_vol


# =========================
# Formatter
# =========================
     
    
def run_hedge_case(name, df_prices, vol_series, STRIKE, R_ANNUAL, SHORT_CALLS):
    result = simulate_delta_hedge(
        df=df_prices,
        strike=STRIKE,
        r_annual=R_ANNUAL,
        vol_series=vol_series,
        position_short_calls=SHORT_CALLS,
        steps_per_year=252,
    )

    final_net_pnl = result.final_pnl - result.payoff  # net result vs payoff

    print(f"\n=== {name} ===")
    print(f"Premium received:      {result.premium_received: .4f}")
    print(f"Option payoff:         {result.payoff: .4f}")
    print(f"Hedging P&L:           {result.final_pnl: .4f}")
    print(f"Net P&L:               {final_net_pnl: .4f}")

    return result    


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    # 1. Load your underlying price data
    # df_prices = load_data_placeholder()   # <-- REPLACE with real Excel/CSV load
    df_prices = load_underlying_from_excel(path="/Users/eoinmoran/Library/CloudStorage/GoogleDrive-eoin.moran3@ucdconnect.ie/My Drive/DS Group Project/Derivative Data MS.xlsx")   # <-- REPLACE with real Excel/CSV load
    df_prices = df_prices.set_index("date")
    rolling_vol_series = compute_rolling_realized_vol(df_prices, window=22)
    implied_vol_series = compute_vol_placeholder(df_prices)  # <-- REPLACE with real vols

    
    start_date = pd.to_datetime("2025-09-19", dayfirst=True)
    df_prices = df_prices.loc[df_prices.index >= start_date]
    rolling_vol_series = rolling_vol_series.loc[rolling_vol_series.index >= start_date]
    implied_vol_series = implied_vol_series.loc[implied_vol_series.index >= start_date]
    # 2. Build your vol series (either implied vol or rolling vol)
    vol_series = compute_vol_placeholder(df_prices)  # <-- REPLACE with real vols

    # 3. Set your parameters
    STRIKE = 160.0        # TODO: set this to your actual strike
    R_ANNUAL = 0.04       # 4% risk-free rate from 
    SHORT_CALLS = 1.0     # short 1 call
    PREMIUM = 5.60

    
    # 4. Run the hedge simulation (implied) 
    result = simulate_delta_hedge(
        df=df_prices,
        strike=STRIKE,
        r_annual=R_ANNUAL,
        vol_series=implied_vol_series,
        position_short_calls=SHORT_CALLS,
        premium_override=PREMIUM,
        steps_per_year=252,
    )
    
    print("=== Delta Hedging Result (IMPLIED)===")
    print(f"Premium received: {result.premium_received:.4f}")
    print(f"Option payoff at expiry: {result.payoff:.4f}")
    print(f"Final hedging P&L: {result.final_pnl:.4f}")

    # 4. Run the hedge simulation (rolling)
    result = simulate_delta_hedge(
        df=df_prices,
        strike=STRIKE,
        r_annual=R_ANNUAL,
        vol_series=rolling_vol_series,
        position_short_calls=SHORT_CALLS,
        premium_override=PREMIUM,
        steps_per_year=252,
    )

    print("=== Delta Hedging Result (ROLLING) ===")
    print(f"Premium received: {result.premium_received:.4f}")
    print(f"Option payoff at expiry: {result.payoff:.4f}")
    print(f"Final hedging P&L: {result.final_pnl:.4f}")
    

    # If you want to inspect the path:
    # print(result.df.head())
    # print(result.df.tail())


    