"""Helpers for Section 6 of the Betting Against Beta project (adds to bab_core and bab_extra;
edits nothing in them).

- nyse_quintile: each name's NYSE size quintile at a month end.
- traded_by_quintile: each month's traded weight |w_t - w_drift_{t-1}|, summed within NYSE size
  quintiles at the formation date, rebuilt from the signed levered weights that build_book returns.
- trading_cost, financing_drag, break_even: the 6.1 quantities.

Conventions follow bab_core: series are indexed by holding month; a trade made at formation date
t is charged to the return of holding month t+1.
"""

import numpy as np
import pandas as pd

import bab_core as bc

QUINTILES = [1, 2, 3, 4, 5]                       # 1 = smallest
HIGH = pd.Series([0.0100, 0.0050, 0.0030, 0.0020, 0.0010], index=QUINTILES, name="high schedule")


def nyse_quintile(panel):
    """NYSE size quintile (1-5) of every name at each month end; NaN without a positive cap.

    Breakpoints are the 20th, 40th, 60th and 80th percentiles of market cap among NYSE names with
    positive cap at t (bc.nyse_breakpoint). Every name, whatever its exchange, is placed by them;
    a name at a breakpoint goes to the upper quintile, as in the Section 4 size screen.
    """
    cuts = pd.concat([bc.nyse_breakpoint(panel, p) for p in (20, 40, 60, 80)], axis=1).to_numpy()
    cap = panel.cap.to_numpy()
    q = 1 + sum((cap >= cuts[:, [k]]).astype(int) for k in range(4))
    return pd.DataFrame(np.where(cap > 0, q, np.nan), index=panel.cap.index, columns=panel.cap.columns)


def traded_by_quintile(weights, panel, book=None):
    """Traded weight per holding month, by NYSE size quintile at the formation date.

    weights : long frame (date = formation date, permno, weight) from build_book(return_weights=True)
    book    : optional build_book frame; if given, the rebuilt total is checked against its turnover

    The trade at formation date t is |w_t - w_drift_{t-1}| for every name, with w_drift as in
    build_book: last month's weight times (1 + its total return in month t, zero if missing),
    rescaled to last month's gross exposure. The first formation date is the initial purchase
    (w_drift = 0). A name sold at t with no positive cap at t (e.g. it delisted) takes its
    quintile from t-1, the last date it was held. Returns (traded, info): traded has columns 1-5
    and "total"; info counts the trades that needed that fallback.
    """
    dates, permnos = panel.ret.index, panel.ret.columns
    col = pd.Series(np.arange(len(permnos)), index=permnos)
    ret_all = np.nan_to_num(panel.ret.to_numpy(), nan=0.0)
    quint = nyse_quintile(panel).to_numpy()

    form_dates = pd.DatetimeIndex(sorted(weights["date"].unique()))
    pos = dates.get_indexer(form_dates)
    assert (pos >= 0).all() and (np.diff(pos) == 1).all(), "formation dates must be consecutive months"
    grouped = weights.groupby("date")

    rows, prev_w, n_fallback, w_fallback = [], None, 0, 0.0
    for t, i in zip(form_dates, pos):
        g = grouped.get_group(t)
        w = np.zeros(len(permnos))
        w[col[g["permno"]].to_numpy()] = g["weight"].to_numpy()
        if prev_w is None:
            drift = np.zeros_like(w)
        else:
            drift = prev_w * (1 + ret_all[i])
            drift *= np.abs(prev_w).sum() / np.abs(drift).sum()
        trade = np.abs(w - drift)

        q = quint[i].copy()
        need = (trade > 0) & np.isnan(q)
        if need.any():
            q[need] = quint[i - 1][need]
            n_fallback += int(need.sum())
            w_fallback += trade[need].sum()
        assert not np.isnan(q[trade > 0]).any(), f"traded name without a size quintile at {t:%Y-%m}"

        row = {k: trade[q == k].sum() for k in QUINTILES}
        row["total"] = trade.sum()
        row["date"] = dates[i + 1]                  # charged to the holding month
        rows.append(row)
        prev_w = w

    traded = pd.DataFrame(rows).set_index("date")
    if book is not None:
        gap = (traded["total"] - book["turnover"]).iloc[1:].abs().max()
        assert gap < 1e-10, f"rebuilt turnover differs from build_book by {gap:.2e}"
    info = pd.Series({"trades taking the t-1 quintile": n_fallback,
                      "their share of all traded weight": w_fallback / traded["total"].sum()})
    return traded, info


def trading_cost(traded, schedule=HIGH):
    """Monthly cost as a fraction of capital: traded weight in each quintile times its one-way cost."""
    return traded[QUINTILES] @ schedule.reindex(QUINTILES)


def financing_drag(book, spread=0.005):
    """Monthly drag of an annual spread on the financing notional (L_L - 1)^+ + L_H."""
    notional = (book["lev_L"] - 1).clip(lower=0) + book["lev_H"]
    return spread / 12 * notional


def break_even(book, traded, schedule=HIGH):
    """Uniform one-way cost per unit traded, and multiple of the schedule, that set the mean net
    return to zero (trading costs only; the mean is over the same months as the book)."""
    gross = book["book"].mean()
    return pd.Series({
        "mean gross excess (ann.)": 12 * gross,
        "traded weight / month": traded["total"].mean(),
        "schedule cost per unit traded": trading_cost(traded, schedule).mean() / traded["total"].mean(),
        "break-even uniform cost per unit traded": gross / traded["total"].mean(),
        "break-even multiple of schedule": gross / trading_cost(traded, schedule).mean(),
    })
