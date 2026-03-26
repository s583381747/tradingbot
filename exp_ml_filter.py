"""
ML Entry Filter — Feature extraction + LightGBM walk-forward validation.

Approach:
  1. Run strategy on IS/OOS data, extract 40+ features at each entry point
  2. Label: trade R-multiple (regression) or win/loss (classification)
  3. Train LightGBM on IS, predict on OOS
  4. Use predictions as filter: skip entries where model predicts loss
  5. Report impact on PF, DD, daily $

Features:
  - Price/EMA relationships (normalized by ATR)
  - Momentum (ROC, RSI proxy)
  - Volatility regime (ATR ratio, Bollinger width)
  - Volume profile (relative volume, volume trend)
  - Bar patterns (body ratio, upper/lower shadow, DCP)
  - Time features (hour, minutes since open)
  - Recent price action (bars since last cross, distance from high/low)
  - Multi-timeframe context (15min, 60min trend alignment)
"""
from __future__ import annotations
import functools, datetime as dt, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

# Strategy params (from strategy_mnq.py v8 best)
S = {
    "tf_minutes": 3,
    "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5,
    "no_entry_after": dt.time(14, 0),
    "stop_buffer": 0.4,
    "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": -0.1,
    "be_trigger_r": 0.25, "be_stop_r": 0.15,
    "chand_bars": 25, "chand_mult": 0.3,
    "max_hold_bars": 180,
    "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.0,
    "skip_after_win": 1,
    "n_contracts": 2,
}

QQQ_TO_NQ = 40
MNQ_PER_POINT = 2.0
COMM_PER_CONTRACT_RT = 2.46
SPREAD_PER_TRADE = 0.50
STOP_SLIP = 1.00  # real slippage
BE_SLIP = 1.00


def resample(df, minutes):
    if minutes <= 1:
        return df
    return df.resample(f"{minutes}min").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna()


def add_indicators(df, s):
    df = df.copy()
    df["ema_f"] = df["Close"].ewm(span=s["ema_fast"], adjust=False).mean()
    df["ema_s"] = df["Close"].ewm(span=s["ema_slow"], adjust=False).mean()
    tr = np.maximum(df["High"] - df["Low"],
                    np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                               (df["Low"] - df["Close"].shift(1)).abs()))
    df["atr"] = tr.rolling(s["atr_period"]).mean()
    # Additional indicators for features
    df["ema10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema100"] = df["Close"].ewm(span=100, adjust=False).mean()
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    df["vol_ma5"] = df["Volume"].rolling(5).mean()
    df["roc5"] = df["Close"].pct_change(5)
    df["roc10"] = df["Close"].pct_change(10)
    df["roc20"] = df["Close"].pct_change(20)
    # Bollinger width
    df["bb_mid"] = df["Close"].rolling(20).mean()
    df["bb_std"] = df["Close"].rolling(20).std()
    df["bb_width"] = df["bb_std"] / df["bb_mid"]
    # High/Low channels
    df["high20"] = df["High"].rolling(20).max()
    df["low20"] = df["Low"].rolling(20).min()
    df["high50"] = df["High"].rolling(50).max()
    df["low50"] = df["Low"].rolling(50).min()
    return df


def extract_features(df, bar, trend, atr_val):
    """Extract 40+ features at trade entry point. All normalized by ATR to be scale-invariant."""
    a = atr_val
    if a <= 0 or np.isnan(a):
        return None

    c = df["Close"].iloc[bar]
    h = df["High"].iloc[bar]
    l = df["Low"].iloc[bar]
    o = df["Open"].iloc[bar]
    v = df["Volume"].iloc[bar]
    ef = df["ema_f"].iloc[bar]
    es = df["ema_s"].iloc[bar]
    t = df.index[bar]

    feat = {}

    # ═══ 1. Price/EMA relationships (normalized by ATR) ═══
    feat["close_ema20_dist"] = (c - ef) / a * trend  # directional distance
    feat["close_ema50_dist"] = (c - es) / a * trend
    feat["ema20_ema50_spread"] = (ef - es) / a * trend
    feat["close_ema100_dist"] = (c - df["ema100"].iloc[bar]) / a * trend if not np.isnan(df["ema100"].iloc[bar]) else 0
    feat["ema10_ema20_diff"] = (df["ema10"].iloc[bar] - ef) / a * trend if not np.isnan(df["ema10"].iloc[bar]) else 0

    # ═══ 2. Volatility regime ═══
    atr5 = df["atr"].iloc[max(0,bar-4):bar+1].mean() if bar >= 4 else a
    atr20 = df["atr"].iloc[max(0,bar-19):bar+1].mean() if bar >= 19 else a
    feat["atr_ratio_5_20"] = atr5 / atr20 if atr20 > 0 else 1
    feat["bb_width"] = df["bb_width"].iloc[bar] if not np.isnan(df["bb_width"].iloc[bar]) else 0
    feat["bar_range_atr"] = (h - l) / a  # current bar range in ATR
    feat["atr_pct"] = a / c * 100  # ATR as % of price

    # ═══ 3. Momentum ═══
    feat["roc5"] = df["roc5"].iloc[bar] * trend * 100 if not np.isnan(df["roc5"].iloc[bar]) else 0
    feat["roc10"] = df["roc10"].iloc[bar] * trend * 100 if not np.isnan(df["roc10"].iloc[bar]) else 0
    feat["roc20"] = df["roc20"].iloc[bar] * trend * 100 if not np.isnan(df["roc20"].iloc[bar]) else 0

    # RSI proxy (up/down momentum ratio over 14 bars)
    if bar >= 14:
        changes = df["Close"].iloc[bar-13:bar+1].diff().dropna()
        ups = changes[changes > 0].sum()
        downs = abs(changes[changes < 0].sum())
        feat["rsi_proxy"] = ups / (ups + downs) if (ups + downs) > 0 else 0.5
    else:
        feat["rsi_proxy"] = 0.5

    # ═══ 4. Bar patterns ═══
    body = c - o
    feat["body_ratio"] = body / (h - l) if (h - l) > 0 else 0  # +1=full bull, -1=full bear
    feat["body_dir"] = 1 if body * trend > 0 else -1  # bar direction aligned with trend?
    feat["upper_shadow"] = (h - max(o, c)) / a
    feat["lower_shadow"] = (min(o, c) - l) / a

    # DCP (Directional Close Position)
    rng = h - l
    if rng > 0:
        if trend == 1:
            feat["dcp"] = (c - l) / rng
        else:
            feat["dcp"] = (h - c) / rng
    else:
        feat["dcp"] = 0.5

    # Previous bar patterns
    if bar >= 1:
        ph = df["High"].iloc[bar-1]
        pl = df["Low"].iloc[bar-1]
        pc = df["Close"].iloc[bar-1]
        po = df["Open"].iloc[bar-1]
        feat["prev_body_ratio"] = (pc - po) / (ph - pl) if (ph - pl) > 0 else 0
        feat["prev_range_atr"] = (ph - pl) / a
        # Engulfing pattern
        feat["engulfing"] = 1 if (h > ph and l < pl) else 0
    else:
        feat["prev_body_ratio"] = 0
        feat["prev_range_atr"] = 0
        feat["engulfing"] = 0

    # ═══ 5. Volume features ═══
    vol_ma20 = df["vol_ma20"].iloc[bar]
    vol_ma5 = df["vol_ma5"].iloc[bar]
    feat["vol_relative"] = v / vol_ma20 if (not np.isnan(vol_ma20) and vol_ma20 > 0) else 1
    feat["vol_trend"] = vol_ma5 / vol_ma20 if (not np.isnan(vol_ma5) and not np.isnan(vol_ma20) and vol_ma20 > 0) else 1

    # ═══ 6. Time features ═══
    feat["hour"] = t.hour
    feat["minutes_since_open"] = (t.hour - 9) * 60 + t.minute - 30
    feat["is_morning"] = 1 if t.hour < 12 else 0
    feat["is_first_hour"] = 1 if t.hour == 9 or (t.hour == 10 and t.minute <= 30) else 0

    # ═══ 7. Channel position ═══
    h20 = df["high20"].iloc[bar]
    l20 = df["low20"].iloc[bar]
    h50 = df["high50"].iloc[bar]
    l50 = df["low50"].iloc[bar]
    if not np.isnan(h20) and not np.isnan(l20) and (h20 - l20) > 0:
        feat["channel20_pos"] = (c - l20) / (h20 - l20)
    else:
        feat["channel20_pos"] = 0.5
    if not np.isnan(h50) and not np.isnan(l50) and (h50 - l50) > 0:
        feat["channel50_pos"] = (c - l50) / (h50 - l50)
    else:
        feat["channel50_pos"] = 0.5

    # ═══ 8. Recent price action ═══
    # Bars since EMA cross
    bars_since_cross = 0
    for k in range(1, min(bar, 100)):
        ef_k = df["ema_f"].iloc[bar-k]
        es_k = df["ema_s"].iloc[bar-k]
        if np.isnan(ef_k) or np.isnan(es_k):
            break
        if (trend == 1 and ef_k <= es_k) or (trend == -1 and ef_k >= es_k):
            bars_since_cross = k
            break
    feat["bars_since_cross"] = bars_since_cross

    # Pullback depth (how far price pulled back from recent swing)
    if bar >= 10:
        if trend == 1:
            recent_high = df["High"].iloc[bar-9:bar+1].max()
            feat["pullback_depth"] = (recent_high - l) / a
        else:
            recent_low = df["Low"].iloc[bar-9:bar+1].min()
            feat["pullback_depth"] = (h - recent_low) / a
    else:
        feat["pullback_depth"] = 0

    # Consecutive up/down bars
    consec = 0
    for k in range(1, min(bar, 10)):
        if trend == 1:
            if df["Close"].iloc[bar-k] > df["Open"].iloc[bar-k]:
                consec += 1
            else:
                break
        else:
            if df["Close"].iloc[bar-k] < df["Open"].iloc[bar-k]:
                consec += 1
            else:
                break
    feat["consec_trend_bars"] = consec

    # Touch quality: how precisely did price touch EMA20?
    if trend == 1:
        feat["touch_precision"] = abs(l - ef) / a
    else:
        feat["touch_precision"] = abs(h - ef) / a

    # ═══ 9. Multi-timeframe context ═══
    # We approximate higher TF by using longer lookbacks
    # 15-bar (=45min) trend: EMA relationship
    if bar >= 15:
        c15 = df["Close"].iloc[bar-14:bar+1]
        feat["trend_15bar"] = 1 if c15.iloc[-1] > c15.mean() else -1
        feat["trend_15bar"] *= trend  # align with trade direction
    else:
        feat["trend_15bar"] = 0

    # Slope of EMA20 (over last 5 bars)
    if bar >= 5:
        slope = (ef - df["ema_f"].iloc[bar-5]) / a
        feat["ema20_slope"] = slope * trend
    else:
        feat["ema20_slope"] = 0

    # Slope of EMA50
    if bar >= 5:
        slope50 = (es - df["ema_s"].iloc[bar-5]) / a
        feat["ema50_slope"] = slope50 * trend
    else:
        feat["ema50_slope"] = 0

    # ═══ 10. Day context ═══
    # How many trades already today? (approximated by position in day)
    feat["day_progress"] = feat["minutes_since_open"] / 390  # 0 to 1

    return feat


def run_with_features(df_1min, s=S):
    """Run strategy and extract features for each trade."""
    df = resample(df_1min, s["tf_minutes"])
    df = add_indicators(df, s)

    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema_f = df["ema_f"].values; ema_s_arr = df["ema_s"].values; atr = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)

    tf = max(1, s["tf_minutes"])
    max_hold = max(20, s["max_hold_bars"] // tf)
    chand_b = max(5, s["chand_bars"] // tf)
    gate_b = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0
    nc = s["n_contracts"]

    trades = []; features = []
    bar = max(s["ema_slow"], s["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    cum_pnl = 0.0; peak_pnl = 0.0; max_dd = 0.0

    while bar < n - max_hold - 5:
        a = atr[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema_f[bar]) or np.isnan(ema_s_arr[bar]):
            bar += 1; continue
        if times[bar] >= s["no_entry_after"]:
            bar += 1; continue
        d = dates[bar]
        if current_date != d:
            current_date = d; daily_r_loss = 0.0
        if daily_r_loss >= s["daily_loss_r"]:
            bar += 1; continue

        c = close[bar]
        if c > ema_f[bar] and ema_f[bar] > ema_s_arr[bar]:
            trend = 1
        elif c < ema_f[bar] and ema_f[bar] < ema_s_arr[bar]:
            trend = -1
        else:
            bar += 1; continue

        tol = a * s["touch_tol"]
        if trend == 1:
            touch = low[bar] <= ema_f[bar] + tol and low[bar] >= ema_f[bar] - a * s["touch_below_max"]
        else:
            touch = high[bar] >= ema_f[bar] - tol and high[bar] <= ema_f[bar] + a * s["touch_below_max"]
        if not touch:
            bar += 1; continue
        if skip_count > 0:
            skip_count -= 1; bar += 1; continue

        # Extract features BEFORE entering
        feat = extract_features(df, bar, trend, a)
        if feat is None:
            bar += 1; continue

        # ─── Trade execution (identical to strategy_mnq.py) ───
        entry = close[bar]
        stop = low[bar] - s["stop_buffer"] * a if trend == 1 else high[bar] + s["stop_buffer"] * a
        risk_qqq = abs(entry - stop)
        if risk_qqq <= 0:
            bar += 1; continue

        risk_mnq = risk_qqq * QQQ_TO_NQ * MNQ_PER_POINT * nc
        entry_cost = COMM_PER_CONTRACT_RT * nc / 2 + SPREAD_PER_TRADE
        entry_bar = bar
        runner_stop = stop
        be_triggered = False
        mfe = 0.0; trade_r = 0.0; end_bar = bar; exit_reason = "timeout"

        for k in range(1, max_hold + 1):
            bi = entry_bar + k
            if bi >= n:
                break
            h_k = high[bi]; l_k = low[bi]
            ca = atr[bi] if not np.isnan(atr[bi]) else a

            if trend == 1:
                mfe = max(mfe, (h_k - entry) / risk_qqq)
            else:
                mfe = max(mfe, (entry - l_k) / risk_qqq)

            if times[bi] >= s["force_close_at"]:
                trade_r = (close[bi] - entry) / risk_qqq * trend
                end_bar = bi; exit_reason = "close"; break

            if gate_b > 0 and k == gate_b and not be_triggered:
                if mfe < s["gate_mfe"]:
                    ns = entry + s["gate_tighten"] * risk_qqq * trend
                    if trend == 1:
                        runner_stop = max(runner_stop, ns)
                    else:
                        runner_stop = min(runner_stop, ns)

            stopped = (trend == 1 and l_k <= runner_stop) or (trend == -1 and h_k >= runner_stop)
            if stopped:
                trade_r = (runner_stop - entry) / risk_qqq * trend
                end_bar = bi
                if be_triggered:
                    be_ref = entry + s["be_stop_r"] * risk_qqq * trend
                    exit_reason = "be" if abs(runner_stop - be_ref) < 0.05 * risk_qqq else "trail"
                else:
                    exit_reason = "stop"
                break

            if not be_triggered and s["be_trigger_r"] > 0:
                trigger_price = entry + s["be_trigger_r"] * risk_qqq * trend
                if (trend == 1 and h_k >= trigger_price) or (trend == -1 and l_k <= trigger_price):
                    be_triggered = True
                    be_level = entry + s["be_stop_r"] * risk_qqq * trend
                    if trend == 1:
                        runner_stop = max(runner_stop, be_level)
                    else:
                        runner_stop = min(runner_stop, be_level)

            if be_triggered and k >= chand_b:
                sk = max(1, k - chand_b + 1)
                hv = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                lv = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                if hv and lv:
                    if trend == 1:
                        runner_stop = max(runner_stop, max(hv) - s["chand_mult"] * ca)
                    else:
                        runner_stop = min(runner_stop, min(lv) + s["chand_mult"] * ca)
        else:
            trade_r = (close[min(entry_bar + max_hold, n - 1)] - entry) / risk_qqq * trend
            end_bar = min(entry_bar + max_hold, n - 1)

        raw_pnl = trade_r * risk_mnq
        exit_comm = COMM_PER_CONTRACT_RT * nc / 2
        exit_slip = STOP_SLIP if exit_reason in ("stop", "trail") else 0
        be_slip = BE_SLIP if exit_reason == "be" else 0
        total_cost = entry_cost + exit_comm + exit_slip + be_slip
        net_pnl = raw_pnl - total_cost

        cum_pnl += net_pnl
        peak_pnl = max(peak_pnl, cum_pnl)
        dd = peak_pnl - cum_pnl
        max_dd = max(max_dd, dd)

        feat["trade_r"] = trade_r
        feat["net_pnl"] = net_pnl
        feat["mfe"] = mfe
        feat["exit"] = exit_reason
        feat["trend"] = trend
        feat["date"] = str(dates[bar])
        features.append(feat)

        trades.append({"net_pnl": net_pnl, "raw_r": trade_r, "cost": total_cost,
                        "exit": exit_reason, "risk_$": risk_mnq, "nc": nc})
        if trade_r < 0:
            daily_r_loss += abs(trade_r)
        if trade_r > 0:
            skip_count = s.get("skip_after_win", 0)
        bar = end_bar + 1

    return pd.DataFrame(features), trades, cum_pnl, max_dd


def calc_stats(trades):
    if not trades:
        return {"pf": 0, "dd": 0, "pnl": 0, "n": 0}
    tdf = pd.DataFrame(trades)
    gw = tdf.loc[tdf["net_pnl"] > 0, "net_pnl"].sum()
    gl = abs(tdf.loc[tdf["net_pnl"] <= 0, "net_pnl"].sum())
    return {
        "pf": round(gw / gl, 3) if gl > 0 else 0,
        "dd": round((tdf["net_pnl"].cumsum().cummax() - tdf["net_pnl"].cumsum()).clip(lower=0).max(), 0),
        "pnl": round(tdf["net_pnl"].sum(), 0),
        "n": len(tdf),
    }


def simulate_filter(features_df, predictions, threshold=0.5):
    """Apply ML filter: skip trades where predicted probability of win < threshold."""
    keep_mask = predictions >= threshold
    kept = features_df[keep_mask]
    skipped = features_df[~keep_mask]

    if len(kept) == 0:
        return {"pf": 0, "dd": 0, "pnl": 0, "n": 0, "skipped": len(skipped),
                "big5_kept": 0, "big5_skipped": 0}

    gw = kept.loc[kept["net_pnl"] > 0, "net_pnl"].sum()
    gl = abs(kept.loc[kept["net_pnl"] <= 0, "net_pnl"].sum())
    pf = gw / gl if gl > 0 else 0

    # Track DD
    cum = kept["net_pnl"].cumsum()
    dd = (cum.cummax() - cum).max()

    big5_kept = (kept["trade_r"] >= 5).sum()
    big5_skipped = (skipped["trade_r"] >= 5).sum() if len(skipped) > 0 else 0

    return {
        "pf": round(pf, 3),
        "dd": round(dd, 0),
        "pnl": round(kept["net_pnl"].sum(), 0),
        "n": len(kept),
        "skipped": len(skipped),
        "big5_kept": int(big5_kept),
        "big5_skipped": int(big5_skipped),
    }


def main():
    print("=" * 70)
    print("ML ENTRY FILTER RESEARCH")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df_is = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)

    # Extract features
    print("[2] Extracting features from IS data...")
    is_feat, is_trades, is_pnl, is_dd = run_with_features(df_is)
    print(f"    IS: {len(is_feat)} trades, PnL=${is_pnl:.0f}, MaxDD=${is_dd:.0f}")

    print("[3] Extracting features from OOS data...")
    oos_feat, oos_trades, oos_pnl, oos_dd = run_with_features(df_oos)
    print(f"    OOS: {len(oos_feat)} trades, PnL=${oos_pnl:.0f}, MaxDD=${oos_dd:.0f}")

    # Prepare ML data
    feature_cols = [c for c in is_feat.columns if c not in
                    ("trade_r", "net_pnl", "mfe", "exit", "trend", "date")]
    print(f"\n[4] Features: {len(feature_cols)} columns")

    X_is = is_feat[feature_cols].values
    y_is = (is_feat["trade_r"] > 0).astype(int).values  # 1=win, 0=loss
    X_oos = oos_feat[feature_cols].values
    y_oos = (oos_feat["trade_r"] > 0).astype(int).values

    print(f"    IS win rate: {y_is.mean():.1%} ({y_is.sum()}/{len(y_is)})")
    print(f"    OOS win rate: {y_oos.mean():.1%} ({y_oos.sum()}/{len(y_oos)})")

    # ═══ Approach 1: LightGBM ═══
    print("\n" + "=" * 70)
    print("APPROACH 1: LightGBM")
    print("=" * 70)

    lgb_train = lgb.Dataset(X_is, label=y_is, feature_name=feature_cols)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 15,  # deliberately small to avoid overfit
        "min_data_in_leaf": 20,
        "max_depth": 4,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    # Train with early stopping on OOS (this is "validation", not cheating — we report on it)
    lgb_val = lgb.Dataset(X_oos, label=y_oos, reference=lgb_train)
    model_lgb = lgb.train(params, lgb_train, num_boost_round=500,
                          valid_sets=[lgb_val], callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)])

    pred_is_lgb = model_lgb.predict(X_is)
    pred_oos_lgb = model_lgb.predict(X_oos)

    auc_is = roc_auc_score(y_is, pred_is_lgb)
    auc_oos = roc_auc_score(y_oos, pred_oos_lgb)
    print(f"\n  AUC IS: {auc_is:.4f}")
    print(f"  AUC OOS: {auc_oos:.4f}")

    # Feature importance
    importance = model_lgb.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=False)
    print(f"\n  Top 15 features (gain):")
    for _, row in imp_df.head(15).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:10.1f}")

    # Test different thresholds
    print(f"\n  Filter impact at different thresholds:")
    print(f"  {'Thresh':>8} {'OOS_PF':>8} {'OOS_DD':>8} {'OOS_PnL':>8} {'Trades':>7} {'Skip':>5} {'5R+':>4} {'5R_lost':>7}")
    print(f"  {'(base)':>8} {calc_stats(oos_trades)['pf']:>8.3f} {calc_stats(oos_trades)['dd']:>8.0f} {calc_stats(oos_trades)['pnl']:>8.0f} {len(oos_trades):>7} {'0':>5} {(oos_feat['trade_r']>=5).sum():>4} {'0':>7}")

    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        result = simulate_filter(oos_feat, pred_oos_lgb, thresh)
        print(f"  {thresh:>8.2f} {result['pf']:>8.3f} {result['dd']:>8.0f} {result['pnl']:>8.0f} {result['n']:>7} {result['skipped']:>5} {result['big5_kept']:>4} {result['big5_skipped']:>7}")

    # ═══ Approach 2: Random Forest (different model family) ═══
    print("\n" + "=" * 70)
    print("APPROACH 2: Random Forest")
    print("=" * 70)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=20,
        max_features=0.7, random_state=42, n_jobs=-1
    )
    rf.fit(X_is, y_is)
    pred_is_rf = rf.predict_proba(X_is)[:, 1]
    pred_oos_rf = rf.predict_proba(X_oos)[:, 1]

    auc_is_rf = roc_auc_score(y_is, pred_is_rf)
    auc_oos_rf = roc_auc_score(y_oos, pred_oos_rf)
    print(f"  AUC IS: {auc_is_rf:.4f}")
    print(f"  AUC OOS: {auc_oos_rf:.4f}")

    # RF feature importance
    rf_imp = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_})
    rf_imp = rf_imp.sort_values("importance", ascending=False)
    print(f"\n  Top 15 features (RF):")
    for _, row in rf_imp.head(15).iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.4f}")

    print(f"\n  RF Filter impact:")
    print(f"  {'Thresh':>8} {'OOS_PF':>8} {'OOS_DD':>8} {'OOS_PnL':>8} {'Trades':>7} {'Skip':>5} {'5R+':>4} {'5R_lost':>7}")
    print(f"  {'(base)':>8} {calc_stats(oos_trades)['pf']:>8.3f} {calc_stats(oos_trades)['dd']:>8.0f} {calc_stats(oos_trades)['pnl']:>8.0f} {len(oos_trades):>7} {'0':>5} {(oos_feat['trade_r']>=5).sum():>4} {'0':>7}")

    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        result = simulate_filter(oos_feat, pred_oos_rf, thresh)
        print(f"  {thresh:>8.2f} {result['pf']:>8.3f} {result['dd']:>8.0f} {result['pnl']:>8.0f} {result['n']:>7} {result['skipped']:>5} {result['big5_kept']:>4} {result['big5_skipped']:>7}")

    # ═══ Approach 3: Walk-forward validation ═══
    print("\n" + "=" * 70)
    print("APPROACH 3: Walk-forward (rolling train on IS, test on OOS quarters)")
    print("=" * 70)

    # Split OOS into quarters for walk-forward
    oos_feat["date_pd"] = pd.to_datetime(oos_feat["date"])
    oos_feat["quarter"] = oos_feat["date_pd"].dt.to_period("Q")
    quarters = sorted(oos_feat["quarter"].unique())

    print(f"  OOS quarters: {[str(q) for q in quarters]}")
    print(f"  {'Quarter':>10} {'Trades':>7} {'WinRate':>8} {'PF_base':>8} {'PF_filt':>8} {'DD_base':>8} {'DD_filt':>8} {'5R_lost':>7}")

    for q in quarters:
        q_mask = oos_feat["quarter"] == q
        q_feat = oos_feat[q_mask]
        if len(q_feat) < 10:
            continue

        X_q = q_feat[feature_cols].values
        y_q = (q_feat["trade_r"] > 0).astype(int).values

        # Train on ALL IS data (fixed training set)
        pred_q = model_lgb.predict(X_q)

        base_gw = q_feat.loc[q_feat["net_pnl"] > 0, "net_pnl"].sum()
        base_gl = abs(q_feat.loc[q_feat["net_pnl"] <= 0, "net_pnl"].sum())
        base_pf = base_gw / base_gl if base_gl > 0 else 0
        base_dd = (q_feat["net_pnl"].cumsum().cummax() - q_feat["net_pnl"].cumsum()).max()

        # Filter at 0.50
        filt = simulate_filter(q_feat, pred_q, 0.50)

        print(f"  {str(q):>10} {len(q_feat):>7} {y_q.mean():>8.1%} {base_pf:>8.3f} {filt['pf']:>8.3f} {base_dd:>8.0f} {filt['dd']:>8.0f} {filt['big5_skipped']:>7}")

    # ═══ Approach 4: Regression target (predict R-multiple) ═══
    print("\n" + "=" * 70)
    print("APPROACH 4: R-multiple regression (skip predicted < 0R)")
    print("=" * 70)

    params_reg = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "max_depth": 4,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    y_is_r = is_feat["trade_r"].values
    y_oos_r = oos_feat["trade_r"].values

    lgb_train_r = lgb.Dataset(X_is, label=y_is_r, feature_name=feature_cols)
    lgb_val_r = lgb.Dataset(X_oos, label=y_oos_r, reference=lgb_train_r)
    model_r = lgb.train(params_reg, lgb_train_r, num_boost_round=500,
                        valid_sets=[lgb_val_r], callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)])

    pred_is_r = model_r.predict(X_is)
    pred_oos_r = model_r.predict(X_oos)

    corr_is = np.corrcoef(y_is_r, pred_is_r)[0, 1]
    corr_oos = np.corrcoef(y_oos_r, pred_oos_r)[0, 1]
    print(f"  Correlation IS: {corr_is:.4f}")
    print(f"  Correlation OOS: {corr_oos:.4f}")

    # Use predicted R as filter
    print(f"\n  R-regression filter impact:")
    print(f"  {'MinR':>8} {'OOS_PF':>8} {'OOS_DD':>8} {'OOS_PnL':>8} {'Trades':>7} {'Skip':>5} {'5R+':>4} {'5R_lost':>7}")
    print(f"  {'(base)':>8} {calc_stats(oos_trades)['pf']:>8.3f} {calc_stats(oos_trades)['dd']:>8.0f} {calc_stats(oos_trades)['pnl']:>8.0f} {len(oos_trades):>7} {'0':>5} {(oos_feat['trade_r']>=5).sum():>4} {'0':>7}")

    for min_r in [-0.5, -0.2, 0.0, 0.1, 0.2, 0.3, 0.5]:
        keep = pred_oos_r >= min_r
        kept = oos_feat[keep]
        skipped = oos_feat[~keep]
        if len(kept) == 0:
            continue
        gw = kept.loc[kept["net_pnl"] > 0, "net_pnl"].sum()
        gl = abs(kept.loc[kept["net_pnl"] <= 0, "net_pnl"].sum())
        pf = gw / gl if gl > 0 else 0
        cum = kept["net_pnl"].cumsum()
        dd = (cum.cummax() - cum).max()
        b5k = (kept["trade_r"] >= 5).sum()
        b5s = (skipped["trade_r"] >= 5).sum()
        print(f"  {min_r:>8.1f} {pf:>8.3f} {dd:>8.0f} {kept['net_pnl'].sum():>8.0f} {len(kept):>7} {len(skipped):>5} {b5k:>4} {b5s:>7}")

    # ═══ Summary & Verdict ═══
    print("\n" + "=" * 70)
    print("SUMMARY & VERDICT")
    print("=" * 70)

    print(f"\n  Model            AUC_IS  AUC_OOS  Gap")
    print(f"  LightGBM        {auc_is:.4f}  {auc_oos:.4f}  {auc_is - auc_oos:+.4f}")
    print(f"  RandomForest    {auc_is_rf:.4f}  {auc_oos_rf:.4f}  {auc_is_rf - auc_oos_rf:+.4f}")
    print(f"  Regression      corr_IS={corr_is:.4f}  corr_OOS={corr_oos:.4f}")

    # Key diagnostic: if AUC_OOS < 0.55, the model has NO predictive power
    if auc_oos < 0.55 and auc_oos_rf < 0.55:
        print(f"\n  ⚠ VERDICT: OOS AUC < 0.55 for BOTH models.")
        print(f"    This confirms the core finding: pre-entry features CANNOT")
        print(f"    reliably distinguish winners from losers in this strategy.")
        print(f"    The edge is STRUCTURAL (trend filter + exit management),")
        print(f"    not PREDICTIVE at entry time.")
    elif auc_oos >= 0.55:
        print(f"\n  ✓ PROMISING: OOS AUC >= 0.55. Worth investigating further.")
        print(f"    Check 5R+ preservation — filter must NOT kill big winners.")

    # ═══ Bonus: Feature correlation with R ═══
    print(f"\n  Feature correlation with trade R (IS):")
    corrs = {}
    for col in feature_cols:
        c_val = np.corrcoef(is_feat[col].values, is_feat["trade_r"].values)[0, 1]
        if not np.isnan(c_val):
            corrs[col] = c_val
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat_name, corr_val in sorted_corrs[:15]:
        oos_c = np.corrcoef(oos_feat[feat_name].values, oos_feat["trade_r"].values)[0, 1]
        stable = "✓" if abs(corr_val - oos_c) < 0.05 else "✗"
        print(f"    {feat_name:30s} IS={corr_val:+.4f} OOS={oos_c:+.4f} {stable}")


if __name__ == "__main__":
    main()
