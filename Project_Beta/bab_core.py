"""Shared building blocks for the Betting Against Beta project.

Conventions (from the brief):
- Returns are the supplied delisting-adjusted CRSP returns (`ret`); excess returns subtract FF `RF`.
- The market is FF `Mkt-RF`.
- A portfolio formed at month end t uses data through t, earns the return of t+1, and is
  weighted with market cap at t.
- Membership at t requires a beta, positive market cap, a price, and no delisting code at t.
- A held stock with a missing return in t+1 earns total return zero (excess -RF).
- Betas omit months with a missing return.
- Every series is labelled by its holding month; full-sample exhibits run 1966-01 to 2026-09.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
FIRST_FORMATION = pd.Timestamp("1965-12-31")


@dataclass
class Panel:
    ret: pd.DataFrame        # total return, month x permno
    ex: pd.DataFrame         # excess return
    cap: pd.DataFrame        # market cap, $m
    prc: pd.DataFrame        # price
    exch: pd.DataFrame       # exchange code
    delisted: pd.DataFrame   # True in a month with a delisting code
    mkt: pd.Series           # Mkt-RF
    rf: pd.Series            # RF

    @property
    def eligible(self):
        """Stocks that can enter a portfolio at month end t (before requiring a beta)."""
        return (self.cap > 0) & self.prc.notna() & ~self.delisted


def load_panel(data_dir=DATA_DIR):
    monthly = pd.read_parquet(
        Path(data_dir) / "crsp_monthly_1963_2026.parquet",
        columns=["permno", "date", "ret", "prc", "mktcap", "exchcd", "dlstcd"],
    )
    ff = pd.read_parquet(Path(data_dir) / "ff_factors_monthly.parquet")
    ff.index = pd.to_datetime(ff.index) + pd.offsets.MonthEnd(0)
    months = ff.index

    def wide(col):
        return monthly.pivot(index="date", columns="permno", values=col).reindex(months)

    ret = wide("ret").astype(float)
    return Panel(
        ret=ret,
        ex=ret.sub(ff["RF"], axis=0),
        cap=wide("mktcap").astype(float),
        prc=wide("prc").astype(float),
        exch=wide("exchcd").astype(float),
        delisted=wide("dlstcd").notna(),
        mkt=ff["Mkt-RF"],
        rf=ff["RF"],
    )


def rolling_beta(ex, mkt, window, min_obs):
    """Trailing OLS beta over the last `window` months in which the stock has a return."""
    y = ex.to_numpy()
    has = ~np.isnan(y)
    x = np.where(has, mkt.reindex(ex.index).to_numpy()[:, None], 0.0)
    y = np.where(has, y, 0.0)

    def rolling_sum(a):
        c = np.cumsum(a, axis=0, dtype=float)
        c[window:] = c[window:] - c[:-window]
        return c

    n = rolling_sum(has)
    sx, sy = rolling_sum(x), rolling_sum(y)
    sxy, sxx = rolling_sum(x * y), rolling_sum(x * x)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (sxy - sx * sy / n) / (sxx - sx * sx / n)
    beta[n < min_obs] = np.nan
    return pd.DataFrame(beta, index=ex.index, columns=ex.columns)


def shrink(beta, w):
    """Convex combination with one: w * beta + (1 - w)."""
    return w * beta + (1 - w)


def _formation_index(panel, first_formation=FIRST_FORMATION):
    """Row positions of formation dates t that have a following holding month."""
    dates = panel.ret.index
    return np.flatnonzero((dates >= first_formation) & (np.arange(len(dates)) < len(dates) - 1))


def _ranks(values):
    """Ranks 1..N, ties broken by order."""
    r = np.empty(len(values), dtype=int)
    r[np.argsort(values, kind="stable")] = np.arange(1, len(values) + 1)
    return r


def decile_sort(beta, panel, n_groups=10, first_formation=FIRST_FORMATION):
    """Value-weighted beta-sorted portfolios held one month.

    Returns (excess returns, pre-ranking betas, stock counts), each indexed by holding month
    with columns 1..n_groups (1 = lowest beta). A stock at a breakpoint goes to the lower group.
    """
    dates = panel.ret.index
    ok_all = (panel.eligible & beta.notna()).to_numpy()
    b_all, cap_all = beta.to_numpy(), panel.cap.to_numpy()
    ret_all, rf = panel.ret.to_numpy(), panel.rf.to_numpy()

    rows = _formation_index(panel, first_formation)
    ex = np.full((len(rows), n_groups), np.nan)
    pre, count = ex.copy(), ex.copy()
    for k, i in enumerate(rows):
        ok = ok_all[i]
        if ok.sum() < n_groups:              # no estimates yet (e.g. long windows early on)
            continue
        b, c = b_all[i, ok], cap_all[i, ok]
        r = np.nan_to_num(ret_all[i + 1, ok], nan=0.0)
        group = np.ceil(_ranks(b) * n_groups / len(b)).astype(int)
        for g in range(1, n_groups + 1):
            m = group == g
            w = c[m] / c[m].sum()
            ex[k, g - 1] = w @ r[m] - rf[i + 1]
            pre[k, g - 1] = w @ b[m]
            count[k, g - 1] = m.sum()

    idx = dates[rows + 1]
    cols = pd.Index(range(1, n_groups + 1), name="decile")
    keep = ~np.isnan(count[:, 0])
    return (pd.DataFrame(ex, idx, cols)[keep], pd.DataFrame(pre, idx, cols)[keep],
            pd.DataFrame(count, idx, cols)[keep].astype(int))


def nyse_breakpoint(panel, pct):
    """Percentile of market cap among NYSE stocks at each month end."""
    cap = panel.cap.where((panel.exch == 1) & (panel.cap > 0))
    return cap.quantile(pct / 100, axis=1)


def build_book(sort_beta, panel, lev_beta=None, weighting="rank", screen=None,
               first_formation=FIRST_FORMATION, return_weights=False):
    """Zero-beta BAB book: long the low half, short the high half, each leg scaled to ex-ante beta one.

    sort_beta : beta frame used to rank stocks and split the legs
    lev_beta  : beta frame whose leg-weighted average sets each leg's leverage (default: sort_beta)
    weighting : "rank" (|z - zbar|), "value" (cap at t) or "equal"
    screen    : optional boolean frame; False removes a stock at that formation date

    Returns a DataFrame indexed by holding month with leg betas, leverages, cash, leg counts,
    unlevered leg excess returns, the book's excess return, turnover and leg composition.
    With return_weights=True also returns the signed levered weights in long format
    (formation date, permno, weight).
    """
    lev_beta = sort_beta if lev_beta is None else lev_beta
    ok_frame = panel.eligible & sort_beta.notna() & lev_beta.notna()
    if screen is not None:
        ok_frame &= screen.reindex_like(ok_frame).fillna(False).astype(bool)

    dates, permnos = panel.ret.index, panel.ret.columns
    ok_all = ok_frame.to_numpy()
    sb_all, lb_all = sort_beta.to_numpy(), lev_beta.to_numpy()
    cap_all, prc_all = panel.cap.to_numpy(), panel.prc.to_numpy()
    ret_all, rf = panel.ret.to_numpy(), panel.rf.to_numpy()
    small_cut = nyse_breakpoint(panel, 20).to_numpy()

    rows = _formation_index(panel, first_formation)
    records, weight_frames = [], []
    prev_w = prev_r = None
    for i in rows:
        idx = np.flatnonzero(ok_all[i])
        n = len(idx)
        if n < 2:                             # no estimates yet (e.g. long windows early on)
            continue
        rank = _ranks(sb_all[i, idx])
        long = rank <= (n + 1) // 2           # odd count: the middle stock goes long

        if weighting == "rank":
            raw = np.abs(rank - (n + 1) / 2)
        elif weighting == "value":
            raw = cap_all[i, idx]
        elif weighting == "equal":
            raw = np.ones(n)
        else:
            raise ValueError(f"unknown weighting {weighting!r}")
        w_l = raw[long] / raw[long].sum()
        w_h = raw[~long] / raw[~long].sum()

        beta_l = w_l @ lb_all[i, idx[long]]
        beta_h = w_h @ lb_all[i, idx[~long]]
        lev_l, lev_h = 1 / beta_l, 1 / beta_h

        r_next = np.nan_to_num(ret_all[i + 1], nan=0.0)
        r_l = w_l @ r_next[idx[long]] - rf[i + 1]
        r_h = w_h @ r_next[idx[~long]] - rf[i + 1]

        w = np.zeros(len(permnos))
        w[idx[long]] = lev_l * w_l
        w[idx[~long]] = -lev_h * w_h

        turnover = np.nan
        if prev_w is not None:
            drift = prev_w * (1 + prev_r)
            drift *= np.abs(prev_w).sum() / np.abs(drift).sum()
            turnover = np.abs(w - drift).sum()
        prev_w, prev_r = w, r_next

        small = cap_all[i, idx] < small_cut[i]
        penny = prc_all[i, idx] < 5
        records.append({
            "date": dates[i + 1],
            "beta_L": beta_l, "beta_H": beta_h, "lev_L": lev_l, "lev_H": lev_h,
            "cash": 1 - lev_l + lev_h, "n_L": int(long.sum()), "n_H": int((~long).sum()),
            "r_L": r_l, "r_H": r_h, "book": lev_l * r_l - lev_h * r_h, "rf": rf[i + 1],
            "turnover": turnover,
            "small_L": w_l @ small[long], "small_H": w_h @ small[~long],
            "penny_L": w_l @ penny[long], "penny_H": w_h @ penny[~long],
        })
        if return_weights:
            nz = np.flatnonzero(w)
            weight_frames.append(pd.DataFrame({"date": dates[i], "permno": permnos[nz], "weight": w[nz]}))

    book = pd.DataFrame(records).set_index("date")
    book["failed"] = book["book"] + book["rf"] < -1     # lost more than the book's capital
    if return_weights:
        return book, pd.concat(weight_frames, ignore_index=True)
    return book


def ols(y, x):
    """OLS of y on a constant and x. Returns (alpha, beta, t_alpha, t_beta, residuals, r2)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return coef[0], coef[1], coef[0] / se[0], coef[1] / se[1], resid, r2


def drawdown(excess):
    """Max drawdown on the compounded path of an excess return, with peak and trough dates."""
    wealth = (1 + excess).cumprod()
    start = excess.index[0] - pd.offsets.MonthEnd(1)
    wealth = pd.concat([pd.Series([1.0], index=[start]), wealth])
    dd = wealth / wealth.cummax() - 1
    trough = dd.idxmin()
    peak = wealth.loc[:trough].idxmax()
    return dd.min(), peak, trough


def performance(excess, mkt):
    """Annualized summary of a monthly excess-return series (brief conventions)."""
    excess = excess.dropna()
    mean, vol = 12 * excess.mean(), np.sqrt(12) * excess.std()
    sr = mean / vol
    years = len(excess) / 12
    alpha, beta, t_alpha, _, _, _ = ols(excess, mkt.reindex(excess.index))
    mdd, peak, trough = drawdown(excess)
    return pd.Series({
        "mean": mean, "vol": vol, "sharpe": sr,
        "sharpe_se": np.sqrt((1 + sr**2 / 24) / years),
        "skew": excess.skew(), "worst_month": excess.min(),
        "max_dd": mdd, "dd_peak": peak.strftime("%Y-%m"), "dd_trough": trough.strftime("%Y-%m"),
        "alpha": 12 * alpha, "t_alpha": t_alpha, "beta": beta, "months": len(excess),
    })
