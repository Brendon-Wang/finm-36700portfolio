"""Helpers for Sections 4-5 of the Betting Against Beta project (adds to bab_core; edits nothing in it).

- load_panel / make_books: the exact recipe for the two books carried into Sections 6-7
  (the default book and the Section 4 NYSE-20th-percentile size-screened book).
- drawdown_episodes / episode_decomposition: drawdown peak, trough, recovery and months under
  water on the compounded path of an excess return, and where an episode's loss came from.
- rolling_beta_series, tail_risk, up_down_beta: the Section 5.3 and 5.4 statistics.

Conventions follow bab_core: series are indexed by holding month; excess returns throughout;
plain OLS t-statistics.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import bab_core as bc

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_panel(data_dir=DATA_DIR):
    """bab_core.load_panel pointed at Project_Beta/data."""
    return bc.load_panel(data_dir)


def size_screen(panel, pct=20):
    """True where market cap at t is at or above the NYSE `pct` percentile of market cap at t."""
    return panel.cap.ge(bc.nyse_breakpoint(panel, pct), axis=0)


def make_books(panel, return_weights=True):
    """The two books used in Sections 4-7, built exactly as in sections4_5.ipynb.

    default  : 60-month beta (min 36), combined 0.6 * beta + 0.4, rank weights, no screen
    screened : the same, after dropping stocks below the NYSE 20th market-cap percentile at t

    Returns {name: (book, weights)} with return_weights=True, else {name: book}.
    """
    combined = bc.shrink(bc.rolling_beta(panel.ex, panel.mkt, 60, 36), 0.6)
    screens = {"default": None, "screened": size_screen(panel, 20)}
    return {name: bc.build_book(combined, panel, screen=s, return_weights=return_weights)
            for name, s in screens.items()}


def wealth_path(excess):
    """Compounded path of an excess return, starting at 1 the month before the first return."""
    wealth = (1 + excess).cumprod()
    start = excess.index[0] - pd.offsets.MonthEnd(1)
    return pd.concat([pd.Series([1.0], index=[start]), wealth])


def drawdown_episodes(excess, mkt):
    """Every drawdown episode of the compounded excess-return path, deepest first.

    peak     : last month at the running maximum before the trough
    trough   : month of the lowest wealth before recovery
    recovery : first later month in which wealth regains the peak level (NaT if not recovered)
    months_under_water : months from the peak to the recovery month, or to the end of the
                         data if not recovered
    mkt_peak_to_trough : product of (1 + Mkt-RF) over the months after the peak up to and
                         including the trough, minus 1
    """
    wealth = wealth_path(excess.dropna())
    under = (wealth / wealth.cummax() - 1).to_numpy() < 0
    dates = wealth.index
    rows, i = [], 1
    while i < len(wealth):
        if not under[i]:
            i += 1
            continue
        start = i                                   # first month below the peak
        while i < len(wealth) and under[i]:
            i += 1
        peak_i = start - 1
        trough_i = start + int(np.argmin(wealth.iloc[start:i].to_numpy()))
        recovered = i < len(wealth)
        end_i = i if recovered else len(wealth) - 1
        m = mkt.reindex(dates[peak_i + 1:trough_i + 1])
        rows.append({
            "peak": dates[peak_i], "trough": dates[trough_i],
            "depth": wealth.iloc[trough_i] / wealth.iloc[peak_i] - 1,
            "recovery": dates[i] if recovered else pd.NaT,
            "recovered": recovered,
            "months_under_water": end_i - peak_i,
            "mkt_peak_to_trough": (1 + m).prod() - 1,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("depth").reset_index(drop=True)


def episode_decomposition(book, mkt, peak, trough):
    """Split a book's loss over the months after `peak` up to `trough` into its two legs.

    Each month the book's excess return is lev_L * r_L - lev_H * r_H exactly, so the sums of the
    monthly contributions add up to the sum of the book's monthly returns. Compounded figures
    are given alongside for the book, each unlevered leg and the market.
    """
    b = book.loc[peak:trough].iloc[1:]             # months after the peak through the trough
    long_c = b["lev_L"] * b["r_L"]
    short_c = -b["lev_H"] * b["r_H"]
    m = mkt.reindex(b.index)
    return pd.Series({
        "months": len(b),
        "avg lev_L": b["lev_L"].mean(), "avg lev_H": b["lev_H"].mean(),
        "avg beta_L (ex-ante)": b["beta_L"].mean(), "avg beta_H (ex-ante)": b["beta_H"].mean(),
        "sum long contribution L_L*r_L": long_c.sum(),
        "sum short contribution -L_H*r_H": short_c.sum(),
        "sum book return": b["book"].sum(),
        "compounded book": (1 + b["book"]).prod() - 1,
        "compounded r_L (unlevered)": (1 + b["r_L"]).prod() - 1,
        "compounded r_H (unlevered)": (1 + b["r_H"]).prod() - 1,
        "compounded Mkt-RF": (1 + m).prod() - 1,
    })


def rolling_beta_series(excess, mkt, window=36):
    """OLS slope of excess on Mkt-RF over the trailing `window` months (full window required)."""
    x = mkt.reindex(excess.index)
    return excess.rolling(window).cov(x) / x.rolling(window).var()


def tail_risk(excess, level=0.05, z=-1.645):
    """Monthly historical VaR and CVaR, normal VaR, skewness and excess kurtosis.

    VaR and CVaR are returned as returns (negative numbers are losses), not as positive losses.
    Historical VaR is the `level` quantile (numpy's linear interpolation); CVaR is the mean of
    the returns at or below it; normal VaR is mean + z * standard deviation.
    """
    x = excess.dropna()
    var = x.quantile(level)
    return pd.Series({
        "hist_VaR_5%": var,
        "hist_CVaR_5%": x[x <= var].mean(),
        "normal_VaR_5%": x.mean() + z * x.std(),
        "skew": x.skew(),
        "excess_kurtosis": x.kurt(),
        "months": len(x),
    })


def up_down_beta(excess, mkt):
    """OLS betas (with intercept) over months with Mkt-RF > 0 and months with Mkt-RF < 0."""
    x = mkt.reindex(excess.index)
    rows = {}
    for name, mask in [("up (Mkt-RF > 0)", x > 0), ("down (Mkt-RF < 0)", x < 0)]:
        alpha, beta, t_alpha, t_beta, _, r2 = bc.ols(excess[mask], x[mask])
        rows[name] = {"beta": beta, "t_beta": t_beta, "alpha_ann": 12 * alpha,
                      "t_alpha": t_alpha, "R2": r2, "months": int(mask.sum())}
    return pd.DataFrame(rows).T
