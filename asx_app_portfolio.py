# asx_app_portfolio.py — ASX ML v7 (Strong-signal gating + CS Walk-Forward, daily holdings/prices)
import io
import zipfile
import datetime as dt
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

# Optional: older sklearns needed this import; harmless if not required
try:
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
except Exception:
    pass

APP_VERSION = "ASX-ML v7 (Strong-signal gating, CS Walk-Forward, daily holdings/prices)"

# --------------------------- Page & Sidebar ---------------------------
st.set_page_config(page_title="ASX Ben's Machine Learning App", layout="wide")
st.title("ASX Ben's Machine Learning App")
st.caption(f"Running: {APP_VERSION}")

with st.sidebar:
    st.header("Universe / Data")
    default_stocks = [
        "CBA.AX","BHP.AX","RIO.AX","WBC.AX","NAB.AX",
        "CSL.AX","WES.AX","ANZ.AX","MQG.AX","GMG.AX"
    ]
    tickers = st.multiselect("ASX Stocks (Yahoo tickers)", default_stocks, default=default_stocks, key="u_tickers")
    years = st.slider("Lookback (years)", 1, 10, 5, 1, key="u_years")
    horizon = st.slider("Prediction horizon (trading days ahead)", 1, 10, 5, 1, key="u_horizon")

    st.markdown("---")
    st.header("Label Filter (training)")
    band = st.slider("Ignore |forward return| ≤ band (reduces noise)", 0.0, 0.05, 0.01, 0.001, format="%.3f", key="u_band")

    st.markdown("---")
    st.header("Signal / Sizing (single-asset view)")
    exp_up_thr = st.number_input("Go Long if E[ret] ≥", value=0.002, step=0.001, format="%.3f", key="u_thr_up")
    exp_dn_thr = st.number_input("Go Short/Flat if E[ret] ≤", value=-0.002, step=0.001, format="%.3f", key="u_thr_dn")
    sizing_mode = st.selectbox("Position sizing mode", ["Binary (±1 or 0)", "Proportional (clip to [-1, 1])"], key="u_size_mode")
    # Default changed to True
    allow_short = st.toggle("Allow shorting", value=True, key="u_allow_short")

    st.markdown("---")
    st.header("Backtest (single-asset)")
    test_frac = st.slider("Test fraction (time-based split)", 0.1, 0.5, 0.2, 0.05, key="u_test_frac")
    costs_bps_default = 5
    costs_bps_single = st.number_input("Transaction costs (bps per unit turnover)", 0, 200, costs_bps_default, 1, key="u_costs_single")
    debug = st.toggle("Debug shapes", value=False, key="u_debug")

# --------------------------- Utilities ---------------------------
def compute_rsi(series, period=14):
    series = pd.to_numeric(series, errors="coerce")
    d = series.diff()
    up = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    roll_dn = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _to_1d_series(obj, index, name):
    if isinstance(obj, pd.DataFrame):
        obj = obj.squeeze("columns")
    arr = np.asarray(obj)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return pd.Series(pd.to_numeric(arr, errors="coerce"), index=index, name=name)

def _ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}
    if "Close" not in df.columns:
        if "adj close" in cols_lower:
            df["Close"] = df[cols_lower["adj close"]]
        else:
            maybe = [c for c in df.columns if "close" in c.lower()]
            if maybe:
                df["Close"] = df[maybe[0]]
            else:
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if num_cols:
                    df["Close"] = df[num_cols[0]]
                else:
                    raise KeyError("No numeric column available to use as Close.")
    if "High" not in df.columns:
        df["High"] = df["Close"]
    if "Low" not in df.columns:
        df["Low"] = df["Close"]
    for c in ("Close", "High", "Low"):
        df[c] = _to_1d_series(df[c], df.index, c)
    return df

def fetch_equity(ticker, years):
    end = dt.date.today()
    start = end - dt.timedelta(days=int(years*365.25))
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.rename(columns=str.title)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = _ensure_price_columns(df)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = _to_1d_series(df["Close"], df.index, "Close")
    high  = _to_1d_series(df["High"],  df.index, "High")
    low   = _to_1d_series(df["Low"],   df.index, "Low")

    df["sma_10"] = close.rolling(10).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_ratio"] = df["sma_10"] / df["sma_50"]
    df["rsi_14"] = compute_rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = macd - signal

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2*std20
    lower = sma20 - 2*std20
    df["bb_width"] = (upper - lower) / close

    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)
    df["vol_20"] = df["ret_1"].rolling(20).std() * np.sqrt(252)

    df["mom_5"] = close.pct_change(5)
    df["hl_range"] = (high - low) / close
    df["to_sma20"] = close / sma20 - 1
    df["Close"] = close
    return df

INDICATOR_COLS = [
    "sma_10","sma_50","sma_ratio","rsi_14","macd","macd_signal","macd_hist",
    "bb_width","ret_1","ret_5","vol_20","mom_5","hl_range","to_sma20"
]

# --------------------------- Single-Asset REGRESSION ---------------------------
def make_features_and_labels(df, horizon=5, band=0.01):
    """Regression target = forward k-day simple return. Train on |fwd_ret| > band."""
    close = _to_1d_series(df["Close"], df.index, "Close")
    fwd_ret_vals = (close.shift(-horizon).to_numpy().reshape(-1) / close.to_numpy().reshape(-1)) - 1.0
    fwd_ret = pd.Series(fwd_ret_vals, index=df.index, name="fwd_ret")

    mask = (fwd_ret.abs() > band).fillna(False) if band > 0 else pd.Series(True, index=df.index)
    X = df[INDICATOR_COLS].astype("float64").replace([np.inf, -np.inf], np.nan)
    out = pd.concat([X, fwd_ret], axis=1)
    out = out.loc[mask].dropna(how="any")
    return out, INDICATOR_COLS

def train_and_signal_reg(df, horizon, band, test_frac, exp_up_thr, exp_dn_thr, sizing_mode, allow_short):
    df_feat = compute_indicators(df.copy())
    data, feat_cols = make_features_and_labels(df_feat, horizon=horizon, band=band)
    if len(data) < 100:
        return None, "Not enough data after feature/label prep.", None, None, None, None, None

    n = len(data)
    n_test = int(max(30, np.floor(n * test_frac)))
    n_train = n - n_test
    train = data.iloc[:n_train].copy()
    test  = data.iloc[n_train:].copy()

    pipe = Pipeline([("reg", GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9))])
    pipe.fit(train[feat_cols], train["fwd_ret"])

    test_pred = pipe.predict(test[feat_cols])
    mae = float(np.mean(np.abs(test_pred - test["fwd_ret"])))
    rmse = float(np.sqrt(np.mean((test_pred - test["fwd_ret"])**2)))

    last_row = data.iloc[[-1]]
    e_ret = float(pipe.predict(last_row[feat_cols])[0])

    if sizing_mode.startswith("Binary"):
        if e_ret >= exp_up_thr:
            signal = "BUY"
        elif e_ret <= exp_dn_thr:
            signal = "SELL" if allow_short else "HOLD"
        else:
            signal = "HOLD"
    else:
        scale = max(abs(exp_up_thr), abs(exp_dn_thr), 1e-6)
        raw_pos = e_ret / scale
        pos = float(np.clip(raw_pos, -1.0, 1.0))
        if not allow_short:
            pos = max(pos, 0.0)
        signal = ("BUY" if pos > 0.1 else ("SELL" if pos < -0.1 else "HOLD"))

    return pipe, signal, e_ret, mae, rmse, data, feat_cols

def compute_metrics_daily(returns, freq=252):
    returns = returns.fillna(0.0)
    equity = (1 + returns).cumprod()
    total_years = max((len(returns) / freq), 1e-9)
    cagr = float(equity.iloc[-1] ** (1/total_years) - 1)
    avg = float(returns.mean())
    vol = float(returns.std())
    sharpe = float((avg / vol * np.sqrt(freq)) if vol > 0 else np.nan)
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    max_dd = float(drawdown.min())
    hit_rate = float((returns > 0).mean())
    return {"CAGR": cagr, "Sharpe": sharpe, "Max Drawdown": max_dd, "Hit Rate": hit_rate}, equity, drawdown

def backtest_test_period_reg(
    df_feat_labels, model, feat_cols, df_raw,
    exp_up_thr=0.002, exp_dn_thr=-0.002,
    sizing_mode="Binary (±1 or 0)", allow_short=True,  # default changed to True
    costs_bps=5, test_frac=0.2
):
    n = len(df_feat_labels)
    n_test = int(max(30, np.floor(n * test_frac)))
    n_train = n - n_test
    test = df_feat_labels.iloc[n_train:].copy()

    e_ret = model.predict(test[feat_cols])
    test["E_ret"] = e_ret

    if sizing_mode.startswith("Binary"):
        if allow_short:
            test["position"] = np.where(test["E_ret"] >= exp_up_thr, 1, np.where(test["E_ret"] <= exp_dn_thr, -1, 0))
        else:
            test["position"] = np.where(test["E_ret"] >= exp_up_thr, 1, 0)
    else:
        scale = max(abs(exp_up_thr), abs(exp_dn_thr), 1e-6)
        pos = np.clip(test["E_ret"] / scale, -1.0, 1.0)
        if not allow_short:
            pos = np.maximum(pos, 0.0)
        test["position"] = pos

    daily_ret = _to_1d_series(df_raw["Close"], df_raw.index, "Close").pct_change(1)
    test["ret_1"] = daily_ret.reindex(test.index)

    test["position_lag"] = test["position"].shift(1).fillna(0.0)

    cost_per_unit = costs_bps / 1e4
    turnover = (test["position_lag"] - test["position_lag"].shift(1).fillna(0.0)).abs()
    test["costs"] = turnover * cost_per_unit

    test["strat_ret"] = (test["position_lag"] * test["ret_1"]).fillna(0.0) - test["costs"].fillna(0.0)

    bh_ret = test["ret_1"].fillna(0.0)
    bh_metrics, bh_equity, bh_dd = compute_metrics_daily(bh_ret)

    strat_metrics, strat_equity, strat_dd = compute_metrics_daily(test["strat_ret"])

    out = {
        "frame": test,
        "strat_metrics": strat_metrics,
        "bh_metrics": bh_metrics,
        "strat_equity": strat_equity,
        "bh_equity": bh_equity,
        "strat_drawdown": strat_dd,
        "bh_drawdown": bh_dd,
    }
    return out

def plot_price(raw_df: pd.DataFrame, title: str):
    df = raw_df.copy()
    df["SMA10"] = df["Close"].rolling(10).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df_plot = df.tail(365)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"], name="Close", mode="lines"))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["SMA10"], name="SMA 10", mode="lines"))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["SMA50"], name="SMA 50", mode="lines"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price (AUD)", height=400, legend=dict(orientation="h"))
    return fig

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode("utf-8")

def _present_metrics(m):
    def pct(x):  return "—" if x is None or not np.isfinite(x) else f"{x:.2%}"
    def num(x):  return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"
    return {"CAGR": pct(m.get("CAGR", np.nan)), "Sharpe": num(m.get("Sharpe", np.nan)),
            "Max Drawdown": pct(m.get("Max Drawdown", np.nan)), "Hit Rate": pct(m.get("Hit Rate", np.nan))}

# --------------------------- Cross-Sectional (Pooled) Walk-Forward (Aligned Dates + daily holdings/prices) ---------------------------
def _fetch_many(tickers, years):
    raw = {}
    for t in tickers:
        df = fetch_equity(t, years)
        if df is not None and not df.empty:
            raw[t] = df
    return raw

def _prep_one(df, horizon):
    df_feat = compute_indicators(df.copy())
    close = _to_1d_series(df_feat["Close"], df_feat.index, "Close")
    fwd_ret = (close.shift(-horizon) / close) - 1.0
    df_feat["fwd_ret"] = fwd_ret
    return df_feat

def make_cs_dataset(raw_map, horizon=5, band=0.01):
    frames = []
    for tkr, df in raw_map.items():
        x = _prep_one(df, horizon)
        keep = x[INDICATOR_COLS + ["fwd_ret"]].astype("float64").replace([np.inf, -np.inf], np.nan)
        keep = keep.dropna(how="any")
        if band > 0:
            keep = keep.loc[keep["fwd_ret"].abs() > band]
        if keep.empty:
            continue
        keep = keep.copy()
        keep["Ticker"] = tkr
        frames.append(keep)
    if not frames:
        return pd.DataFrame(), []
    data = pd.concat(frames).sort_index()
    tick_dummies = pd.get_dummies(data["Ticker"], prefix="T")
    data = pd.concat([data.drop(columns=["Ticker"]), tick_dummies], axis=1)
    feat_cols = [c for c in data.columns if c != "fwd_ret"]
    return data, feat_cols

def _metrics_periodic(series, freq=52):
    r = series.fillna(0.0)
    eq = (1 + r).cumprod()
    years = max(len(r)/freq, 1e-9)
    cagr = float(eq.iloc[-1]**(1/years) - 1)
    vol = float(r.std()*np.sqrt(freq))
    sharpe = float(r.mean()/r.std()*np.sqrt(freq)) if r.std() > 0 else np.nan
    dd = (eq/eq.cummax() - 1).min()
    hit = float((r > 0).mean())
    return {"CAGR": cagr, "Sharpe": sharpe, "Ann.Vol": vol, "MaxDD": float(dd), "HitRate": hit}

def walkforward_portfolio_backtest(
    data, feat_cols, raw_map,
    horizon=5,
    test_frac=0.3,
    rebalance="W-FRI",      # "W-FRI", "W-WED", "M", etc.
    top_k=3,
    allow_short=True,
    costs_bps=5,
    target_vol=None,        # e.g., 0.10 for 10% annualized; None = off
    long_floor=0.0,         # only long if E_ret ≥ long_floor
    short_floor=0.0,        # only short if E_ret ≤ −short_floor
):
    """
    Walk-forward CS portfolio with strong-signal gating:
      - Train pooled model up to each rebalance date d
      - Build per-ticker features at each ticker's last trading day ≤ d
      - Only take positions passing thresholds; else stay in cash
      - Equal-weight among selected names; market-neutral if both sides exist
      - Realize forward return from that ticker's feature date
      - Turnover-based costs; optional vol targeting
      - Returns both rebalance-only results and DAILY holdings+prices
    """
    if data.empty:
        return {"error": "No data after feature/label prep."}

    # Time split from pooled calendar
    all_dates = pd.DatetimeIndex(sorted(set(data.index)))
    if len(all_dates) < 120:
        return {"error": "Not enough pooled history for walk-forward."}
    n_test = int(max(60, np.floor(len(all_dates) * test_frac)))
    cutoff = all_dates[-n_test]

    # Rebalance dates
    s = pd.Series(1, index=all_dates[all_dates >= cutoff])
    rebal_dates = s.resample(rebalance).first().index
    if len(rebal_dates) == 0:
        return {"error": "No rebalance dates in test region. Try different test_frac/frequency."}

    # Precompute realized forward returns & closes
    realized_map = {t: _prep_one(df.copy(), horizon=horizon)[["fwd_ret"]] for t, df in raw_map.items()}
    close_map = {t: _ensure_price_columns(df.copy())["Close"].copy() for t, df in raw_map.items()}

    indicator_cols = [c for c in feat_cols if not c.startswith("T_")]
    tick_cols = [c for c in feat_cols if c.startswith("T_")]

    equity = [1.0]
    eq_dates = []
    prev_weights = {t: 0.0 for t in raw_map.keys()}
    rows = []
    weights_history = {}

    def _fit_model(X_tr, y_tr):
        mdl = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, random_state=42
        )
        mdl.fit(X_tr, y_tr)
        return mdl

    # sanitize floors
    long_floor = max(0.0, float(long_floor))
    short_floor = max(0.0, float(short_floor))

    for d in rebal_dates:
        # Train set strictly before d
        train_mask = (data.index < d)
        if train_mask.sum() < 200:
            continue
        X_train = data.loc[train_mask, feat_cols]
        y_train = data.loc[train_mask, "fwd_ret"]
        model = _fit_model(X_train, y_train)

        # Build per-ticker candidate at each ticker's last trading day ≤ d
        candidates = []
        for tkr, df in raw_map.items():
            df_sub = df.loc[:d]
            if df_sub.empty:
                continue
            feat_date = df_sub.index[-1]
            base = compute_indicators(df_sub.copy()).iloc[[-1]]
            feat_ind = base.reindex(columns=indicator_cols)
            if feat_ind.isna().any(axis=1).iloc[0]:
                continue

            dummies = pd.DataFrame([[0.0]*len(tick_cols)], index=feat_ind.index, columns=tick_cols)
            col_name = f"T_{tkr}"
            if col_name in dummies.columns:
                dummies[col_name] = 1.0

            feat_row = pd.concat([feat_ind, dummies], axis=1).reindex(columns=feat_cols, fill_value=0.0)
            feat_row["Ticker"] = tkr
            feat_row["_feat_date"] = feat_date
            candidates.append(feat_row)

        if not candidates:
            continue

        C = pd.concat(candidates)
        C["E_ret"] = model.predict(C[feat_cols])
        C = C.sort_values("E_ret", ascending=False)

        # --- Threshold-gated selection (strong signals only) ---
        C_pos = C[C["E_ret"] >= long_floor].copy()
        C_neg = C[C["E_ret"] <= -short_floor].copy()

        kL = int(min(top_k, len(C_pos)))
        kS = int(min(top_k, len(C_neg))) if allow_short else 0

        longs  = C_pos.head(kL)
        shorts = C_neg.sort_values("E_ret").head(kS) if kS > 0 else pd.DataFrame(columns=C.columns)

        weights = {}
        if allow_short and kL > 0 and kS > 0:
            # market-neutral: 50% long basket, 50% short basket
            wL =  0.5 / kL
            wS = -0.5 / kS
            for t in longs["Ticker"]:
                weights[t] = wL
            for t in shorts["Ticker"]:
                weights[t] = weights.get(t, 0.0) + wS
        elif kL > 0:
            # long-only (no shorts or none pass threshold)
            w = 1.0 / kL
            for t in longs["Ticker"]:
                weights[t] = w
        else:
            # no strong opportunities -> go to CASH
            weights = {}

        # Realize forward return at that ticker's feature date
        port_ret = 0.0
        turnover = 0.0
        for t in raw_map.keys():
            w_new = weights.get(t, 0.0)
            w_old = prev_weights.get(t, 0.0)
            turnover += abs(w_new - w_old)
            prev_weights[t] = w_new

            if w_new == 0.0:
                continue
            sel = C.loc[C["Ticker"] == t, "_feat_date"]
            if len(sel) == 0:
                continue
            feat_date_t = pd.to_datetime(sel.iloc[0])
            fr = realized_map[t].reindex(index=[feat_date_t])["fwd_ret"]
            r = float(fr.iloc[0]) if len(fr) and pd.notna(fr.iloc[0]) else 0.0
            port_ret += w_new * r

        cost = turnover * (costs_bps / 1e4)
        net_ret = port_ret - cost

        # Optional vol targeting using trailing 60 rebalances
        if target_vol is not None and target_vol > 0:
            recent = pd.Series([row["net_ret"] for row in rows[-60:]])
            if len(recent) > 10:
                per_vol = float(recent.std())
                ann_vol = per_vol * np.sqrt(52)
                if np.isfinite(ann_vol) and ann_vol > 1e-8:
                    lev = np.clip(target_vol / ann_vol, 0.0, 3.0)
                    net_ret *= lev

        rows.append({"date": d, "gross_ret": port_ret, "cost": cost, "net_ret": net_ret})
        equity.append(equity[-1] * (1.0 + net_ret))
        eq_dates.append(d)
        weights_history[pd.to_datetime(d)] = weights.copy()

    if not rows:
        return {"error": "Walk-forward produced no trades (likely due to thresholds too tight or data alignment)."}

    # ---- Rebalance-only result
    res = pd.DataFrame(rows).set_index("date")
    eq = pd.Series(equity[1:], index=eq_dates, name="equity")
    m = _metrics_periodic(res["net_ret"], freq=52)

    # ---- DAILY holdings + prices frame
    first_reb = min(weights_history.keys())
    last_date = max([ser.index.max() for ser in close_map.values()])
    union_dates = sorted(set().union(*[ser.loc[first_reb:last_date].index for ser in close_map.values()]))
    daily_idx = pd.DatetimeIndex(union_dates)

    daily = pd.DataFrame(index=daily_idx)
    daily = daily.join(res[["gross_ret", "cost", "net_ret"]], how="left")
    daily["rebalance"] = daily.index.isin(res.index)

    for t in raw_map.keys():
        w_series = pd.Series({d: weights_history.get(d, {}).get(t, 0.0) for d in sorted(weights_history.keys())})
        w_series = w_series.reindex(daily_idx).ffill().fillna(0.0)
        daily[f"w_{t}"] = w_series

    for t, px in close_map.items():
        daily[f"px_{t}"] = px.reindex(daily_idx)

    return {
        "frame": res,                 # rebalance-only
        "daily_frame": daily,         # daily holdings + prices
        "equity": eq,
        "metrics": m,
        "rebalance_dates": rebal_dates
    }

# --------------------------- App Body: Single-Asset Loop ---------------------------
if len(tickers) == 0:
    st.warning("Select at least one ticker.")
    st.stop()

summary_rows = []
csv_bundle = {}

for tkr in tickers:
    st.subheader(tkr)
    df = fetch_equity(tkr, years)
    if df is None or df.empty:
        st.warning(f"No data for {tkr}.")
        continue

    if debug:
        st.write("dtypes:", dict(df.dtypes))
        st.write("columns:", list(df.columns))
        st.write("Close shape:", np.asarray(df["Close"]).shape)

    csv_bytes = df_to_csv_bytes(df)
    csv_name = f"{tkr}_{df.index.min().date()}_{df.index.max().date()}.csv".replace(":", "")
    st.download_button(f"Download {tkr} OHLCV (CSV)", csv_bytes, csv_name, "text/csv", key=f"dl_{tkr}")
    csv_bundle[csv_name] = csv_bytes

    model, signal, e_ret, mae, rmse, data, feat_cols = train_and_signal_reg(
        df, horizon, band, test_frac, exp_up_thr, exp_dn_thr, sizing_mode, allow_short
    )
    if model is None:
        st.warning(f"{tkr}: {signal}")
        continue

    latest_close = float(df["Close"].iloc[-1])

    # --- Display metrics (MAE/RMSE removed) ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Close (AUD)", f"{latest_close:.2f}")
    c2.metric(f"E[ret] next {horizon}d", f"{e_ret:.3%}")
    c3.metric("Signal", signal)

    # --- Collect for Signals Summary ---
    summary_rows.append({
        "Ticker": tkr,
        "Signal": signal,
        "E[ret]": e_ret,
        "Close": latest_close
    })

    st.plotly_chart(plot_price(df, f"{tkr} — Close with SMAs"), use_container_width=True)

    # ---- Single-asset Backtest (unique keys) ----
    with st.expander("Backtest (single asset, test period only)"):
        colb1, colb2, colb3 = st.columns(3)
        bt_costs_bps = colb1.number_input("Transaction cost (bps per unit turnover)", 0, 200, costs_bps_default, 1, key=f"bt_costs_{tkr}")
        bt_sizing_mode = colb2.selectbox("Sizing mode", ["Binary (±1 or 0)", "Proportional (clip to [-1, 1])"],
                                         index=0 if sizing_mode.startswith("Binary") else 1, key=f"bt_size_{tkr}")
        bt_allow_short = colb3.toggle("Allow shorting", value=allow_short, key=f"bt_short_{tkr}")

        colb4, colb5 = st.columns(2)
        bt_up_thr = colb4.number_input("Long if E[ret] ≥", value=float(exp_up_thr), step=0.001, format="%.3f", key=f"bt_up_{tkr}")
        bt_dn_thr = colb5.number_input("Short/Flat if E[ret] ≤", value=float(exp_dn_thr), step=0.001, format="%.3f", key=f"bt_dn_{tkr}")

        if st.button(f"Run backtest: {tkr}", key=f"bt_run_{tkr}"):
            with st.spinner("Running single-asset backtest..."):
                bt = backtest_test_period_reg(
                    df_feat_labels=data, model=model, feat_cols=feat_cols, df_raw=df,
                    exp_up_thr=bt_up_thr, exp_dn_thr=bt_dn_thr,
                    sizing_mode=bt_sizing_mode, allow_short=bt_allow_short,
                    costs_bps=bt_costs_bps, test_frac=test_frac
                )

            m_strat = _present_metrics(bt["strat_metrics"])
            m_bh = _present_metrics(bt["bh_metrics"])
            metrics_df = pd.DataFrame({"Strategy": m_strat, "Buy & Hold": m_bh})
            st.write("**Performance (test period)**")
            st.table(metrics_df)

            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=bt["strat_equity"].index, y=bt["strat_equity"].values, name="Strategy"))
            fig_eq.add_trace(go.Scatter(x=bt["bh_equity"].index, y=bt["bh_equity"].values, name="Buy & Hold"))
            fig_eq.update_layout(title="Equity Curve (normalised to 1.0)", xaxis_title="Date", yaxis_title="Equity", height=400, legend=dict(orientation="h"))
            st.plotly_chart(fig_eq, use_container_width=True)

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=bt["strat_drawdown"].index, y=bt["strat_drawdown"].values, name="Strategy DD"))
            fig_dd.update_layout(title="Strategy Drawdown", xaxis_title="Date", yaxis_title="Drawdown", height=300, legend=dict(orientation="h"))
            st.plotly_chart(fig_dd, use_container_width=True)

            show_cols = ["E_ret", "position", "position_lag", "ret_1", "strat_ret"]
            keep_cols = [c for c in show_cols if c in bt["frame"].columns]
            st.dataframe(bt["frame"][keep_cols].tail(20))

            bt_csv = bt["frame"].to_csv(index=True).encode("utf-8")
            st.download_button(f"Download backtest frame (CSV) — {tkr}", bt_csv, f"{tkr}_backtest_reg.csv", "text/csv", key=f"bt_dl_{tkr}")

# --------------------------- Zip download for all raw CSVs ---------------------------
if csv_bundle:
    memzip = io.BytesIO()
    with zipfile.ZipFile(memzip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, b in csv_bundle.items():
            zf.writestr(fname, b)
    st.download_button("Download all OHLCV CSVs (ZIP)", memzip.getvalue(), "asx_data.zip", "application/zip")

# --------------------------- Cross-Sectional Portfolio Backtest ---------------------------
st.markdown("---")
st.header("Cross-Sectional Walk-Forward Portfolio Backtest")

colp1, colp2, colp3, colp4 = st.columns(4)
cs_topk = colp1.number_input("Top K (max per side)", 1, 5, 3, 1, key="p_topk")
cs_short = colp2.toggle("Allow shorting (Bottom K)", value=True, key="p_short")
cs_costs = colp3.number_input("Costs (bps per unit turnover)", 0, 200, costs_bps_default, 1, key="p_costs")
cs_rebal = colp4.selectbox("Rebalance frequency", ["W-FRI","W-WED","M"], index=0, key="p_rebal")

colp5, colp6 = st.columns(2)
cs_test_frac = colp5.slider("Test fraction (pooled time split)", 0.1, 0.6, 0.3, 0.05, key="p_test_frac")
cs_target_vol = colp6.number_input("Target annual vol (0=off)", 0.00, 1.00, 0.00, 0.01, format="%.2f", key="p_tgtvol")

# strong-signal gates
colp7, colp8 = st.columns(2)
cs_long_thr  = colp7.number_input("Min E[ret] to go LONG", 0.000, 0.050, 0.002, 0.001, format="%.3f", key="p_long_thr")
cs_short_thr = colp8.number_input("Min |E[ret]| to go SHORT", 0.000, 0.050, 0.002, 0.001, format="%.3f", key="p_short_thr")

st.caption("Tip: set horizon ≈ rebalance cadence (e.g., horizon=5 for weekly). If no names pass thresholds, portfolio holds cash.")

if st.button("Run portfolio backtest", key="p_run"):
    with st.spinner("Running portfolio backtest..."):
        raw_map = _fetch_many(tickers, years)
        data_cs, feat_cols_cs = make_cs_dataset(raw_map, horizon=horizon, band=band)
        if data_cs.empty:
            st.warning("No pooled data after filtering; try lowering the band or increasing lookback.")
        else:
            res = walkforward_portfolio_backtest(
                data_cs, feat_cols_cs, raw_map,
                horizon=horizon, test_frac=cs_test_frac,
                rebalance=cs_rebal, top_k=cs_topk,
                allow_short=cs_short, costs_bps=cs_costs,
                target_vol=(None if cs_target_vol == 0.0 else cs_target_vol),
                long_floor=cs_long_thr, short_floor=cs_short_thr
            )
            if "error" in res:
                st.warning(res["error"])
            else:
                m = res["metrics"]
                st.write("**Portfolio (walk-forward) performance**")
                st.table(pd.DataFrame(m, index=["Portfolio"]).T.style.format({
                    "CAGR": "{:.2%}", "Sharpe": "{:.2f}", "Ann.Vol": "{:.2%}", "MaxDD": "{:.2%}", "HitRate": "{:.2%}"
                }))

                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"].values, name="Portfolio"))
                fig_eq.update_layout(title="Portfolio Equity (walk-forward, normalised to 1.0)",
                                     xaxis_title="Date", yaxis_title="Equity", height=420, legend=dict(orientation="h"))
                st.plotly_chart(fig_eq, use_container_width=True)

                st.write("**Rebalance rows (last 20)**")
                st.dataframe(res["frame"].tail(20))

                # Downloads
                st.download_button(
                    "Download portfolio (DAILY holdings + prices) — portfolio_walkforward.csv",
                    res["daily_frame"].to_csv().encode("utf-8"),
                    "portfolio_walkforward.csv", "text/csv"
                )
                st.download_button(
                    "Download portfolio (REBALANCE-ONLY) — portfolio_rebalance.csv",
                    res["frame"].to_csv().encode("utf-8"),
                    "portfolio_rebalance.csv", "text/csv"
                )

# --------------------------- Signals Summary (Bottom) ---------------------------
if summary_rows:
    st.markdown("---")
    st.header("Signals Summary (all tickers)")

    sum_df = pd.DataFrame(summary_rows)

    # Normalize + order by signal (BUY → HOLD → SELL), then by expected return
    sum_df["Signal"] = sum_df["Signal"].str.upper()
    cat = CategoricalDtype(["BUY", "HOLD", "SELL"], ordered=True)
    sum_df["Signal"] = sum_df["Signal"].astype(cat)
    sum_df = sum_df.sort_values(["Signal", "E[ret]"], ascending=[True, False])

    st.dataframe(
        sum_df.set_index("Ticker")
              .style.format({"Close": "{:.2f}", "E[ret]": "{:.2%}"}),
        use_container_width=True
    )

    # Quick counts by class
    counts = sum_df["Signal"].value_counts().reindex(["BUY", "HOLD", "SELL"]).fillna(0).astype(int)
    c1, c2, c3 = st.columns(3)
    c1.metric("BUY",  int(counts.get("BUY", 0)))
    c2.metric("HOLD", int(counts.get("HOLD", 0)))
    c3.metric("SELL", int(counts.get("SELL", 0)))

    # Optional: download
    st.download_button(
        "Download signals (CSV)",
        sum_df.to_csv(index=False).encode("utf-8"),
        "signals_summary.csv",
        "text/csv"
    )
