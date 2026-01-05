import time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist


# ----------------------------
# Config / Helpers
# ----------------------------

def normalize_stooq_us(sym: str) -> str:
    sym = (sym or "").strip().upper()
    if not sym:
        return sym
    if "." in sym:
        return sym
    return f"{sym}.US"


def stooq_csv_url(sym: str) -> str:
    # Stooq CSV endpoint prefers lowercase.
    return f"https://stooq.com/q/d/l/?s={sym.lower()}&i=d"


def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff().dropna()


def align(y: pd.Series, x: pd.Series) -> Tuple[pd.Series, pd.Series]:
    df = pd.concat([y, x], axis=1).dropna()
    return df.iloc[:, 0], df.iloc[:, 1]


def fit_linear(y: pd.Series, x: pd.Series, robust: bool):
    X = sm.add_constant(x.values)
    if robust:
        m = sm.RLM(y.values, X, M=HuberT()).fit()
        return float(m.params[0]), float(m.params[1]), np.nan
    m = sm.OLS(y.values, X).fit()
    return float(m.params[0]), float(m.params[1]), float(m.rsquared)


def beta_standard(y: pd.Series, x: pd.Series, robust: bool) -> dict:
    y, x = align(y, x)
    a, b, r2 = fit_linear(y, x, robust)
    return {"alpha": a, "beta": b, "r2": r2, "n": int(len(y))}


def beta_conditional(y: pd.Series, x: pd.Series, robust: bool) -> dict:
    y, x = align(y, x)
    up = x > 0
    down = x < 0

    beta_up = np.nan
    beta_down = np.nan

    if up.sum() >= 25:
        _, beta_up, _ = fit_linear(y[up], x[up], robust)
    if down.sum() >= 25:
        _, beta_down, _ = fit_linear(y[down], x[down], robust)

    return {"beta_up": beta_up, "beta_down": beta_down, "n_up": int(up.sum()), "n_down": int(down.sum())}


def beta_tail(y: pd.Series, x: pd.Series, tail_q: float, robust: bool) -> dict:
    y, x = align(y, x)
    cutoff = x.quantile(tail_q)
    tail = x <= cutoff
    if tail.sum() < 25:
        return {"beta_tail": np.nan, "n_tail": int(tail.sum()), "tail_q": float(tail_q)}
    _, b, _ = fit_linear(y[tail], x[tail], robust)
    return {"beta_tail": b, "n_tail": int(tail.sum()), "tail_q": float(tail_q)}


def beta_quadratic(y: pd.Series, x: pd.Series, robust: bool) -> dict:
    y, x = align(y, x)
    X = np.column_stack([np.ones(len(x)), x.values, x.values**2])

    if robust:
        m = sm.RLM(y.values, X, M=HuberT()).fit()
        r2 = np.nan
    else:
        m = sm.OLS(y.values, X).fit()
        r2 = float(m.rsquared)

    return {
        "beta_linear_q": float(m.params[1]),
        "beta_convexity_q": float(m.params[2]),
        "r2_q": r2,
        "n_q": int(len(y)),
    }


def beta_piecewise(y: pd.Series, x: pd.Series, q_low: float, q_high: float, robust: bool) -> dict:
    y, x = align(y, x)
    k1 = x.quantile(q_low)
    k2 = x.quantile(q_high)

    xp1 = np.maximum(x.values - k1, 0.0)
    xp2 = np.maximum(x.values - k2, 0.0)
    X = np.column_stack([np.ones(len(x)), x.values, xp1, xp2])

    if robust:
        m = sm.RLM(y.values, X, M=HuberT()).fit()
        r2 = np.nan
    else:
        m = sm.OLS(y.values, X).fit()
        r2 = float(m.rsquared)

    b, c, d = float(m.params[1]), float(m.params[2]), float(m.params[3])

    return {
        "beta_low_pw": b,
        "beta_mid_pw": b + c,
        "beta_high_pw": b + c + d,
        "k1_pw": float(k1),
        "k2_pw": float(k2),
        "r2_pw": r2,
        "n_pw": int(len(y)),
    }


def zscore(series: pd.Series) -> pd.Series:
    s = series.copy()
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


# ----------------------------
# Simple in-memory TTL cache
# ----------------------------

@dataclass
class CacheItem:
    ts: float
    df: pd.DataFrame

_PRICE_CACHE: Dict[Tuple[str, int], CacheItem] = {}
CACHE_TTL_SECONDS = 60 * 30  # 30 minutes


def get_prices_cached(sym: str, years: int) -> pd.DataFrame:
    key = (sym, years)
    now = time.time()
    item = _PRICE_CACHE.get(key)
    if item and (now - item.ts) < CACHE_TTL_SECONDS:
        return item.df.copy()

    url = stooq_csv_url(sym)
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed fetching {sym} from Stooq: {repr(e)}")

    # Stooq CSV: Date,Open,High,Low,Close,Volume
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        raise HTTPException(status_code=502, detail=f"Unexpected data format for {sym}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    df = df[df["Date"] >= cutoff]

    out = df[["Date", "Close"]].rename(columns={"Close": sym}).set_index("Date")
    if len(out) < 50:
        raise HTTPException(status_code=422, detail=f"Too few data points for {sym} (got {len(out)}). Increase years.")
    _PRICE_CACHE[key] = CacheItem(ts=now, df=out.copy())
    return out


# ----------------------------
# API models
# ----------------------------

class AnalyzeRequest(BaseModel):
    stocks: conlist(str, min_length=1, max_length=10) = Field(..., description="Up to 10 tickers")
    market: str = Field("SPY", description="Benchmark ticker")
    years: int = Field(2, ge=1, le=20)
    window_bars: int = Field(252, ge=80, le=5000)
    tail_q: float = Field(0.05, ge=0.01, le=0.2)
    q_low: float = Field(0.30, ge=0.1, le=0.49)
    q_high: float = Field(0.70, ge=0.51, le=0.9)
    robust: bool = Field(True)


# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI(title="Nonlinear Beta API", version="1.0.0")

# Allow your frontend domain(s). In production, replace ["*"] with your Vercel domain.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https://.*\.vercel\.app$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    stocks = [normalize_stooq_us(s) for s in req.stocks]
    market = normalize_stooq_us(req.market)

    # Fetch closes
    tickers = list(dict.fromkeys(stocks + [market]))
    closes = []
    for t in tickers:
        closes.append(get_prices_cached(t, req.years))

    px = pd.concat(closes, axis=1).dropna(how="all")

    # returns
    rets = log_returns(px)

    if market not in rets.columns:
        raise HTTPException(status_code=422, detail=f"Benchmark {market} missing from returns.")
    mkt = rets[market].dropna()

    rows = []
    for s in stocks:
        if s not in rets.columns:
            rows.append({"stock": s, "error": "No return series (missing data)."})
            continue

        y = rets[s].dropna()
        y, x = align(y, mkt)

        if len(y) < max(req.window_bars, 80):
            rows.append({"stock": s, "error": f"Not enough bars ({len(y)}). Increase years or lower window_bars."})
            continue

        y = y.iloc[-req.window_bars:]
        x = x.iloc[-req.window_bars:]

        base = beta_standard(y, x, req.robust)
        cond = beta_conditional(y, x, req.robust)
        tail = beta_tail(y, x, req.tail_q, req.robust)
        quad = beta_quadratic(y, x, req.robust)
        pw = beta_piecewise(y, x, req.q_low, req.q_high, req.robust)

        rows.append({
            "as_of": str(y.index[-1].date()),
            "stock": s,
            "market": market,
            "interval": "1d",
            "period": f"{req.years}y",
            "window_bars": int(req.window_bars),

            "beta": base["beta"],
            "beta_down": cond["beta_down"],
            "beta_up": cond["beta_up"],
            "beta_tail": tail["beta_tail"],

            "beta_convexity_q": quad["beta_convexity_q"],

            "beta_low_pw": pw["beta_low_pw"],
            "beta_mid_pw": pw["beta_mid_pw"],
            "beta_high_pw": pw["beta_high_pw"],

            "error": None
        })

    df = pd.DataFrame(rows)
    if "error" not in df.columns:
        df["error"] = np.nan

    ok = df["error"].isna()
    if ok.any():
        sub = df.loc[ok, ["beta_down", "beta_tail", "beta_convexity_q"]].copy()
        sub = sub.apply(lambda c: c.fillna(c.median()))
        z = sub.apply(zscore)
        df.loc[ok, "DefensiveScore"] = (
            0.45*z["beta_down"] + 0.45*z["beta_tail"] + 0.10*(-z["beta_convexity_q"])
        )
    else:
        df["DefensiveScore"] = np.nan

    as_of = None
    try:
        as_of = str(mkt.index[-1].date())
    except Exception:
        pass

    return {
        "as_of": as_of,
        "rows": df.to_dict(orient="records"),
    }
