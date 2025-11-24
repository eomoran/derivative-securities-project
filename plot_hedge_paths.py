#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:18:24 2025

@author: eoinmoran
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ---------- CONFIG: EDIT THESE ----------
IMPLIED_CSV = "delta_hedge_path_implied_vol.csv"
ROLLING_CSV = "delta_hedge_path_rolling_vol.csv"   # same format as implied

PREMIUM_RECEIVED = 5.6        # what you sold the call for
STRIKE = 160.0                # <-- put your actual strike here
DATE_COL = "date"
PORTFOLIO_COL = "portfolio_value"
SPOT_COL = "spot"
# ----------------------------------------


def load_with_total_pnl(csv_path: str) -> pd.DataFrame:
    """
    Load a hedge path CSV and compute:
      - hedge_pnl_t = portfolio_value_t - portfolio_value_0
      - option_payoff_t (0 until expiry, then max(spot_T - K, 0))
      - total_pnl_t = premium + hedge_pnl_t - option_payoff_t
      - total_pnl_from_start = total_pnl_t - total_pnl_0  (starts at 0)
    """
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Hedge PnL from the underlying trading
    df["hedge_pnl"] = df[PORTFOLIO_COL] - df[PORTFOLIO_COL].iloc[0]

    # Option payoff: zero until expiry, then max(S_T - K, 0)
    df["option_payoff"] = 0.0
    payoff_T = max(df[SPOT_COL].iloc[-1] - STRIKE, 0.0)
    df.loc[df.index[-1], "option_payoff"] = payoff_T

    # Total PnL of the short call + hedge
    df["total_pnl"] = PREMIUM_RECEIVED + df["hedge_pnl"] - df["option_payoff"]

    # For plotting: PnL since trade date (start at 0, end at final profit)
    df["total_pnl_from_start"] = df["total_pnl"] #- df["total_pnl"].iloc[0]

    return df


def plot_implied_only(df_imp: pd.DataFrame, save_path: str | None = None) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df_imp[DATE_COL], df_imp["total_pnl_from_start"],
             marker="o", label="Total PnL (implied vol)")
    plt.axhline(0, linewidth=1)
    plt.title("Total PnL Path: Short Call + Delta Hedge (Implied Vol)")
    plt.xlabel("Date")
    plt.ylabel("Total PnL since trade date")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=30)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))  # ~6 ticks
    plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    plt.axhline(0, linewidth=1, color='black', alpha=0.6)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.title("Delta Hedge Total P&L Comparison\nImplied Vol")
    plt.xlabel("Date")
    plt.ylabel("Total P&L ($)")
    plt.legend(frameon=True, loc="upper right")
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()

    print("Implied vol final total PnL:",
          df_imp["total_pnl_from_start"].iloc[-1])


def plot_implied_vs_rolling(df_imp: pd.DataFrame,
                            df_roll: pd.DataFrame,
                            save_path: str | None = None) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df_imp[DATE_COL], df_imp["total_pnl_from_start"],
             marker="o", label="Total PnL (implied vol)")
    plt.plot(df_roll[DATE_COL], df_roll["total_pnl_from_start"],
             marker="o", linestyle="--",
             label="Total PnL (rolling 22d vol)")

    plt.axhline(0, linewidth=1)
    plt.title("Total PnL Path: Implied Vol vs Rolling 22-Day Vol")
    plt.xlabel("Date")
    plt.ylabel("Total PnL since trade date")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=30)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(6))  # ~6 ticks
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.1f}'))
    plt.axhline(0, linewidth=1, color='black', alpha=0.6)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.title("Delta Hedge Total P&L Comparison\nImplied Vol vs. 22-Day Historical Vol")
    plt.xlabel("Date")
    plt.ylabel("Total P&L ($)")
    plt.legend(frameon=True, loc="upper right")
    plt.annotate(f"${df_imp['total_pnl'].iloc[-1]:.2f}",
             xy=(df_imp[DATE_COL].iloc[-1],
                 df_imp["total_pnl"].iloc[-1]),
             xytext=(-5, 10), textcoords="offset points")
    plt.annotate(f"${df_roll['total_pnl'].iloc[-1]:.2f}",
             xy=(df_roll[DATE_COL].iloc[-1],
                 df_roll["total_pnl"].iloc[-1]),
             xytext=(-5, 10), textcoords="offset points")
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()

    print("Implied vol final total PnL :",
          df_imp["total_pnl_from_start"].iloc[-1])
    print("Rolling 22d vol final total PnL:",
          df_roll["total_pnl_from_start"].iloc[-1])


def main():
    # Implied-vol hedge
    df_imp = load_with_total_pnl(IMPLIED_CSV)
    plot_implied_only(df_imp, save_path="total_pnl_implied.svg")

    # Rolling-vol hedge
    df_roll = load_with_total_pnl(ROLLING_CSV)

    # Plot comparison
    plot_implied_vs_rolling(df_imp, df_roll,
                            save_path="total_pnl_implied_vs_rolling.svg")


if __name__ == "__main__":
    main()