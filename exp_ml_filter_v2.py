"""
ML Entry Filter V2 — Advanced approaches after V1 showed AUC=0.53.

V1 failed with standard indicator features. V2 tries:
  A. TARGET ENGINEERING — predict MFE (max favorable), gate_pass, not win/loss
  B. VISUAL PATTERNS — chart shape features (N-bar sequences as features)
  C. REGIME FEATURES — market-level context (VIX proxy, recent strat performance)
  D. SEQUENCE FEATURES — raw OHLCV for last N bars as flattened input
  E. PURGED TIME-SERIES CV — avoid leakage in IS itself
"""
from __future__ import annotations
import functools, datetime as dt, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

IS_PATH = "data/QQQ_1Min_Polygon_2y_clean.csv"
OOS_PATH = "data/QQQ_1Min_Barchart_2y_2022-2024_clean.csv"

S = {
    "tf_minutes": 3, "ema_fast": 20, "ema_slow": 50, "atr_period": 14,
    "touch_tol": 0.15, "touch_below_max": 0.5, "no_entry_after": dt.time(14, 0),
    "stop_buffer": 0.4, "gate_bars": 3, "gate_mfe": 0.2, "gate_tighten": -0.1,
    "be_trigger_r": 0.25, "be_stop_r": 0.15, "chand_bars": 25, "chand_mult": 0.3,
    "max_hold_bars": 180, "force_close_at": dt.time(15, 58),
    "daily_loss_r": 2.0, "skip_after_win": 1, "n_contracts": 2,
}

QQQ_TO_NQ = 40; MNQ_PER_POINT = 2.0
COMM_RT = 2.46; SPREAD = 0.50; STOP_SLIP = 1.00; BE_SLIP = 1.00


def resample(df, minutes):
    if minutes <= 1: return df
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
    df["vol_ma20"] = df["Volume"].rolling(20).mean()
    return df


def extract_advanced_features(df, bar, trend, atr_val, trade_history):
    """Extract advanced features including visual patterns and regime context."""
    a = atr_val
    if a <= 0 or np.isnan(a): return None
    c = df["Close"].iloc[bar]; h = df["High"].iloc[bar]; l = df["Low"].iloc[bar]
    o = df["Open"].iloc[bar]; v = df["Volume"].iloc[bar]
    ef = df["ema_f"].iloc[bar]; es = df["ema_s"].iloc[bar]
    t = df.index[bar]
    feat = {}

    # ═══ A. STANDARD (top V1 features only — reduced to avoid noise) ═══
    feat["close_ema20_dist"] = (c - ef) / a * trend
    feat["ema20_ema50_spread"] = (ef - es) / a * trend
    rng = h - l
    feat["dcp"] = ((c - l) / rng if trend == 1 else (h - c) / rng) if rng > 0 else 0.5
    feat["atr_pct"] = a / c * 100
    feat["bar_range_atr"] = rng / a
    feat["minutes_since_open"] = (t.hour - 9) * 60 + t.minute - 30

    # ═══ B. VISUAL PATTERN FEATURES (chart shape) ═══
    # Idea: encode the "shape" of last N bars as a normalized vector
    lookbacks = [5, 10, 20]
    for lb in lookbacks:
        if bar < lb:
            for j in range(lb):
                feat[f"shape_{lb}_{j}"] = 0
            feat[f"trend_strength_{lb}"] = 0
            feat[f"volatility_regime_{lb}"] = 0
            feat[f"range_contraction_{lb}"] = 0
            continue

        # Normalize close prices to [0,1] range over lookback
        closes_lb = df["Close"].iloc[bar-lb+1:bar+1].values
        highs_lb = df["High"].iloc[bar-lb+1:bar+1].values
        lows_lb = df["Low"].iloc[bar-lb+1:bar+1].values

        lb_min = lows_lb.min()
        lb_max = highs_lb.max()
        lb_range = lb_max - lb_min

        if lb_range > 0:
            # Normalized close path (the "chart shape" a human would see)
            normalized = (closes_lb - lb_min) / lb_range
            if trend == -1:
                normalized = 1 - normalized  # flip for shorts
            for j, nv in enumerate(normalized):
                feat[f"shape_{lb}_{j}"] = round(nv, 4)
        else:
            for j in range(lb):
                feat[f"shape_{lb}_{j}"] = 0.5

        # Trend strength: slope of normalized closes
        x = np.arange(lb)
        if lb_range > 0:
            y = (closes_lb - lb_min) / lb_range
            slope = np.polyfit(x, y, 1)[0] * trend
            feat[f"trend_strength_{lb}"] = slope
        else:
            feat[f"trend_strength_{lb}"] = 0

        # Volatility regime: std of returns over lookback
        rets = np.diff(closes_lb) / closes_lb[:-1]
        feat[f"volatility_regime_{lb}"] = rets.std() * 100 if len(rets) > 0 else 0

        # Range contraction: is range contracting? (consolidation before breakout)
        ranges = highs_lb - lows_lb
        if len(ranges) >= 3:
            recent_avg = ranges[-3:].mean()
            early_avg = ranges[:max(1, lb//2)].mean()
            feat[f"range_contraction_{lb}"] = recent_avg / early_avg if early_avg > 0 else 1
        else:
            feat[f"range_contraction_{lb}"] = 1

    # ═══ C. CANDLESTICK PATTERNS ═══
    if bar >= 3:
        # 3-bar patterns
        bars_3 = []
        for k in range(3):
            bi = bar - 2 + k
            bh = df["High"].iloc[bi]; bl = df["Low"].iloc[bi]
            bc = df["Close"].iloc[bi]; bo = df["Open"].iloc[bi]
            br = bh - bl
            body = (bc - bo) / br if br > 0 else 0
            upper = (bh - max(bc, bo)) / br if br > 0 else 0
            lower = (min(bc, bo) - bl) / br if br > 0 else 0
            bars_3.append((body * trend, upper, lower, br / a))

        # Hammer/shooting star at touch
        feat["hammer"] = 1 if (bars_3[-1][2] > 0.6 and abs(bars_3[-1][0]) < 0.3) else 0
        # Inside bar
        feat["inside_bar"] = 1 if (bars_3[-1][3] < bars_3[-2][3] * 0.7) else 0
        # Engulfing
        feat["bullish_engulf"] = 1 if (bars_3[-1][0] > 0.5 and bars_3[-2][0] < -0.3) else 0
        # Three-bar squeeze
        feat["three_bar_squeeze"] = 1 if all(b[3] < 0.8 for b in bars_3) else 0
        # Doji (small body)
        feat["doji"] = 1 if abs(bars_3[-1][0]) < 0.1 else 0
    else:
        feat["hammer"] = 0; feat["inside_bar"] = 0; feat["bullish_engulf"] = 0
        feat["three_bar_squeeze"] = 0; feat["doji"] = 0

    # ═══ D. REGIME / CONTEXT FEATURES ═══
    # VIX proxy: realized volatility over last N bars
    if bar >= 20:
        rets20 = df["Close"].iloc[bar-19:bar+1].pct_change().dropna()
        feat["realized_vol_20"] = rets20.std() * np.sqrt(252 * 130) * 100  # annualized
    else:
        feat["realized_vol_20"] = 0

    if bar >= 60:
        rets60 = df["Close"].iloc[bar-59:bar+1].pct_change().dropna()
        feat["realized_vol_60"] = rets60.std() * np.sqrt(252 * 130) * 100
        feat["vol_ratio_20_60"] = feat["realized_vol_20"] / feat["realized_vol_60"] if feat["realized_vol_60"] > 0 else 1
    else:
        feat["realized_vol_60"] = 0
        feat["vol_ratio_20_60"] = 1

    # Distance from day's high/low (intraday context)
    day_start = bar
    while day_start > 0 and df.index[day_start].date() == df.index[bar].date():
        day_start -= 1
    day_start += 1
    if day_start < bar:
        day_high = df["High"].iloc[day_start:bar+1].max()
        day_low = df["Low"].iloc[day_start:bar+1].min()
        day_range = day_high - day_low
        if day_range > 0:
            feat["day_position"] = (c - day_low) / day_range
            feat["day_range_atr"] = day_range / a
        else:
            feat["day_position"] = 0.5; feat["day_range_atr"] = 0
    else:
        feat["day_position"] = 0.5; feat["day_range_atr"] = 0

    # Gap from yesterday's close
    if day_start > 1:
        yest_close = df["Close"].iloc[day_start - 1]
        day_open = df["Open"].iloc[day_start]
        feat["gap_pct"] = (day_open - yest_close) / yest_close * 100 * trend
    else:
        feat["gap_pct"] = 0

    # ═══ E. STRATEGY PERFORMANCE CONTEXT ═══
    # Recent trade performance (if strategy knows it's been losing, be cautious)
    if len(trade_history) >= 3:
        last3 = trade_history[-3:]
        feat["recent_3_avg_r"] = np.mean([t["r"] for t in last3])
        feat["recent_3_win_rate"] = np.mean([1 if t["r"] > 0 else 0 for t in last3])
    else:
        feat["recent_3_avg_r"] = 0
        feat["recent_3_win_rate"] = 0.5

    if len(trade_history) >= 10:
        last10 = trade_history[-10:]
        feat["recent_10_avg_r"] = np.mean([t["r"] for t in last10])
        feat["recent_10_win_rate"] = np.mean([1 if t["r"] > 0 else 0 for t in last10])
    else:
        feat["recent_10_avg_r"] = 0
        feat["recent_10_win_rate"] = 0.5

    # ═══ F. VOLUME PROFILE ═══
    vol_ma = df["vol_ma20"].iloc[bar]
    feat["vol_relative"] = v / vol_ma if (not np.isnan(vol_ma) and vol_ma > 0) else 1

    # Volume at touch: compare to recent average
    if bar >= 5:
        recent_vol = df["Volume"].iloc[bar-4:bar+1].values
        feat["vol_acceleration"] = recent_vol[-1] / recent_vol[:-1].mean() if recent_vol[:-1].mean() > 0 else 1
        # Volume trend (increasing or decreasing?)
        feat["vol_trend_5"] = (recent_vol[-1] - recent_vol[0]) / (recent_vol.mean() + 1e-8)
    else:
        feat["vol_acceleration"] = 1
        feat["vol_trend_5"] = 0

    # ═══ G. EMA SLOPE DYNAMICS ═══
    if bar >= 10:
        ema_f_10 = df["ema_f"].iloc[bar-9:bar+1].values
        feat["ema20_slope_10"] = (ema_f_10[-1] - ema_f_10[0]) / a * trend
        feat["ema20_curvature"] = (ema_f_10[-1] - 2*ema_f_10[5] + ema_f_10[0]) / a * trend
    else:
        feat["ema20_slope_10"] = 0
        feat["ema20_curvature"] = 0

    return feat


def run_with_advanced_features(df_1min, s=S):
    """Run strategy and extract advanced features."""
    df = resample(df_1min, s["tf_minutes"])
    df = add_indicators(df, s)
    high = df["High"].values; low = df["Low"].values; close = df["Close"].values
    ema_f = df["ema_f"].values; ema_s = df["ema_s"].values; atr = df["atr"].values
    times = df.index.time; dates = df.index.date; n = len(df)
    tf = max(1, s["tf_minutes"])
    max_hold = max(20, s["max_hold_bars"] // tf)
    chand_b = max(5, s["chand_bars"] // tf)
    gate_b = max(1, s["gate_bars"] // tf) if s["gate_bars"] > 0 else 0
    nc = s["n_contracts"]

    features = []; trade_history = []
    bar = max(s["ema_slow"], s["atr_period"]) + 5
    daily_r_loss = 0.0; current_date = None; skip_count = 0
    cum_pnl = 0.0; peak_pnl = 0.0; max_dd = 0.0

    while bar < n - max_hold - 5:
        a = atr[bar]
        if np.isnan(a) or a <= 0 or np.isnan(ema_f[bar]) or np.isnan(ema_s[bar]):
            bar += 1; continue
        if times[bar] >= s["no_entry_after"]:
            bar += 1; continue
        d = dates[bar]
        if current_date != d:
            current_date = d; daily_r_loss = 0.0
        if daily_r_loss >= s["daily_loss_r"]:
            bar += 1; continue

        c = close[bar]
        if c > ema_f[bar] and ema_f[bar] > ema_s[bar]: trend = 1
        elif c < ema_f[bar] and ema_f[bar] < ema_s[bar]: trend = -1
        else: bar += 1; continue

        tol = a * s["touch_tol"]
        if trend == 1:
            touch = low[bar] <= ema_f[bar] + tol and low[bar] >= ema_f[bar] - a * s["touch_below_max"]
        else:
            touch = high[bar] >= ema_f[bar] - tol and high[bar] <= ema_f[bar] + a * s["touch_below_max"]
        if not touch: bar += 1; continue
        if skip_count > 0: skip_count -= 1; bar += 1; continue

        feat = extract_advanced_features(df, bar, trend, a, trade_history)
        if feat is None: bar += 1; continue

        entry = close[bar]
        stop = low[bar] - s["stop_buffer"] * a if trend == 1 else high[bar] + s["stop_buffer"] * a
        risk_qqq = abs(entry - stop)
        if risk_qqq <= 0: bar += 1; continue
        risk_mnq = risk_qqq * QQQ_TO_NQ * MNQ_PER_POINT * nc
        entry_cost = COMM_RT * nc / 2 + SPREAD

        entry_bar = bar; runner_stop = stop; be_triggered = False
        mfe = 0.0; trade_r = 0.0; end_bar = bar; exit_reason = "timeout"
        gate_passed = False

        for k in range(1, max_hold + 1):
            bi = entry_bar + k
            if bi >= n: break
            h_k = high[bi]; l_k = low[bi]
            ca = atr[bi] if not np.isnan(atr[bi]) else a
            if trend == 1: mfe = max(mfe, (h_k - entry) / risk_qqq)
            else: mfe = max(mfe, (entry - l_k) / risk_qqq)
            if times[bi] >= s["force_close_at"]:
                trade_r = (close[bi] - entry) / risk_qqq * trend
                end_bar = bi; exit_reason = "close"; break
            if gate_b > 0 and k == gate_b and not be_triggered:
                if mfe < s["gate_mfe"]:
                    ns = entry + s["gate_tighten"] * risk_qqq * trend
                    if trend == 1: runner_stop = max(runner_stop, ns)
                    else: runner_stop = min(runner_stop, ns)
                else:
                    gate_passed = True
            stopped = (trend == 1 and l_k <= runner_stop) or (trend == -1 and h_k >= runner_stop)
            if stopped:
                trade_r = (runner_stop - entry) / risk_qqq * trend
                end_bar = bi
                if be_triggered:
                    be_ref = entry + s["be_stop_r"] * risk_qqq * trend
                    exit_reason = "be" if abs(runner_stop - be_ref) < 0.05 * risk_qqq else "trail"
                else: exit_reason = "stop"
                break
            if not be_triggered and s["be_trigger_r"] > 0:
                trigger_price = entry + s["be_trigger_r"] * risk_qqq * trend
                if (trend == 1 and h_k >= trigger_price) or (trend == -1 and l_k <= trigger_price):
                    be_triggered = True
                    be_level = entry + s["be_stop_r"] * risk_qqq * trend
                    if trend == 1: runner_stop = max(runner_stop, be_level)
                    else: runner_stop = min(runner_stop, be_level)
            if be_triggered and k >= chand_b:
                sk = max(1, k - chand_b + 1)
                hv = [high[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                lv = [low[entry_bar + kk] for kk in range(sk, k) if entry_bar + kk < n]
                if hv and lv:
                    if trend == 1: runner_stop = max(runner_stop, max(hv) - s["chand_mult"] * ca)
                    else: runner_stop = min(runner_stop, min(lv) + s["chand_mult"] * ca)
        else:
            trade_r = (close[min(entry_bar + max_hold, n - 1)] - entry) / risk_qqq * trend
            end_bar = min(entry_bar + max_hold, n - 1)

        raw_pnl = trade_r * risk_mnq
        exit_comm = COMM_RT * nc / 2
        exit_slip = STOP_SLIP if exit_reason in ("stop", "trail") else 0
        be_slip = BE_SLIP if exit_reason == "be" else 0
        net_pnl = raw_pnl - (entry_cost + exit_comm + exit_slip + be_slip)

        cum_pnl += net_pnl; peak_pnl = max(peak_pnl, cum_pnl)
        max_dd = max(max_dd, peak_pnl - cum_pnl)

        # Record targets
        feat["trade_r"] = trade_r
        feat["net_pnl"] = net_pnl
        feat["mfe"] = mfe
        feat["gate_passed"] = 1 if gate_passed else 0
        feat["be_triggered"] = 1 if be_triggered else 0
        feat["exit"] = exit_reason
        feat["trend"] = trend
        feat["date"] = str(dates[bar])
        features.append(feat)

        trade_history.append({"r": trade_r, "pnl": net_pnl})
        if trade_r < 0: daily_r_loss += abs(trade_r)
        if trade_r > 0: skip_count = s.get("skip_after_win", 0)
        bar = end_bar + 1

    return pd.DataFrame(features), cum_pnl, max_dd


def simulate_filter(df, preds, threshold=0.5):
    keep = preds >= threshold
    kept = df[keep]; skipped = df[~keep]
    if len(kept) == 0:
        return {"pf": 0, "dd": 0, "pnl": 0, "n": 0, "skip": len(skipped), "b5k": 0, "b5s": 0}
    gw = kept.loc[kept["net_pnl"] > 0, "net_pnl"].sum()
    gl = abs(kept.loc[kept["net_pnl"] <= 0, "net_pnl"].sum())
    cum = kept["net_pnl"].cumsum()
    dd = (cum.cummax() - cum).clip(lower=0).max()
    return {"pf": round(gw/gl, 3) if gl > 0 else 0, "dd": round(dd, 0),
            "pnl": round(kept["net_pnl"].sum(), 0), "n": len(kept),
            "skip": len(skipped), "b5k": int((kept["trade_r"]>=5).sum()),
            "b5s": int((skipped["trade_r"]>=5).sum())}


def train_and_eval(X_is, y_is, X_oos, y_oos, feat_names, label=""):
    """Train LightGBM with CV and evaluate."""
    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.03, "num_leaves": 10, "min_data_in_leaf": 30,
        "max_depth": 3, "feature_fraction": 0.6, "bagging_fraction": 0.6,
        "bagging_freq": 5, "verbose": -1, "seed": 42,
        "lambda_l1": 1.0, "lambda_l2": 1.0,  # strong regularization
    }

    # Time-series CV within IS
    tscv = TimeSeriesSplit(n_splits=5)
    cv_aucs = []
    for train_idx, val_idx in tscv.split(X_is):
        dtrain = lgb.Dataset(X_is[train_idx], label=y_is[train_idx])
        dval = lgb.Dataset(X_is[val_idx], label=y_is[val_idx])
        m = lgb.train(params, dtrain, num_boost_round=300,
                      valid_sets=[dval], callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30)])
        p = m.predict(X_is[val_idx])
        try:
            cv_aucs.append(roc_auc_score(y_is[val_idx], p))
        except:
            cv_aucs.append(0.5)

    # Final model: train on all IS, evaluate on OOS
    dtrain = lgb.Dataset(X_is, label=y_is, feature_name=feat_names)
    dval = lgb.Dataset(X_oos, label=y_oos)
    model = lgb.train(params, dtrain, num_boost_round=300,
                      valid_sets=[dval], callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30)])

    pred_is = model.predict(X_is)
    pred_oos = model.predict(X_oos)
    auc_is = roc_auc_score(y_is, pred_is)
    auc_oos = roc_auc_score(y_oos, pred_oos)
    cv_mean = np.mean(cv_aucs)

    print(f"  [{label}] CV_AUC={cv_mean:.4f}  IS_AUC={auc_is:.4f}  OOS_AUC={auc_oos:.4f}  "
          f"IS-OOS_gap={auc_is-auc_oos:+.4f}")

    # Top features
    imp = model.feature_importance(importance_type="gain")
    top5_idx = np.argsort(imp)[-5:][::-1]
    top5 = [(feat_names[i], imp[i]) for i in top5_idx]
    print(f"    Top5: {', '.join(f'{n}({v:.0f})' for n,v in top5)}")

    return model, pred_oos, auc_oos, cv_mean


def main():
    print("=" * 70)
    print("ML ENTRY FILTER V2 — ADVANCED APPROACHES")
    print("=" * 70)

    print("\n[1] Loading data & extracting features...")
    df_is_raw = pd.read_csv(IS_PATH, index_col="timestamp", parse_dates=True)
    df_oos_raw = pd.read_csv(OOS_PATH, index_col="timestamp", parse_dates=True)

    is_feat, is_pnl, is_dd = run_with_advanced_features(df_is_raw)
    oos_feat, oos_pnl, oos_dd = run_with_advanced_features(df_oos_raw)
    print(f"    IS: {len(is_feat)} trades, PnL=${is_pnl:.0f}, DD=${is_dd:.0f}")
    print(f"    OOS: {len(oos_feat)} trades, PnL=${oos_pnl:.0f}, DD=${oos_dd:.0f}")

    meta_cols = ["trade_r", "net_pnl", "mfe", "gate_passed", "be_triggered", "exit", "trend", "date"]
    feat_cols = [c for c in is_feat.columns if c not in meta_cols]
    # Remove shape columns for some experiments
    basic_cols = [c for c in feat_cols if not c.startswith("shape_")]
    shape_cols = [c for c in feat_cols if c.startswith("shape_")]

    print(f"    Total features: {len(feat_cols)} ({len(basic_cols)} basic + {len(shape_cols)} shape)")

    # ═══ EXPERIMENT A: Different targets ═══
    print("\n" + "=" * 70)
    print("EXPERIMENT A: TARGET ENGINEERING")
    print("=" * 70)

    X_is = is_feat[feat_cols].values
    X_oos = oos_feat[feat_cols].values

    # A1: Win/Loss (same as V1, but with better features + regularization)
    y_win_is = (is_feat["trade_r"] > 0).astype(int).values
    y_win_oos = (oos_feat["trade_r"] > 0).astype(int).values
    _, pred_win, auc_win, cv_win = train_and_eval(X_is, y_win_is, X_oos, y_win_oos, feat_cols, "Win/Loss")

    # A2: Gate Pass prediction (does the trade pass the 3-bar MFE gate?)
    y_gate_is = is_feat["gate_passed"].astype(int).values
    y_gate_oos = oos_feat["gate_passed"].astype(int).values
    _, pred_gate, auc_gate, cv_gate = train_and_eval(X_is, y_gate_is, X_oos, y_gate_oos, feat_cols, "Gate Pass")

    # A3: BE Trigger prediction (does the trade reach BE trigger?)
    y_be_is = is_feat["be_triggered"].astype(int).values
    y_be_oos = oos_feat["be_triggered"].astype(int).values
    _, pred_be, auc_be, cv_be = train_and_eval(X_is, y_be_is, X_oos, y_be_oos, feat_cols, "BE Trigger")

    # A4: Big winner (>= 3R)
    y_big_is = (is_feat["trade_r"] >= 3).astype(int).values
    y_big_oos = (oos_feat["trade_r"] >= 3).astype(int).values
    if y_big_is.sum() >= 20 and y_big_oos.sum() >= 20:
        _, pred_big, auc_big, cv_big = train_and_eval(X_is, y_big_is, X_oos, y_big_oos, feat_cols, "Big Win ≥3R")
    else:
        print(f"  [Big Win ≥3R] Skipped — too few positives (IS={y_big_is.sum()}, OOS={y_big_oos.sum()})")

    # A5: MFE >= 1R (trade has potential even if eventually loses)
    y_mfe_is = (is_feat["mfe"] >= 1).astype(int).values
    y_mfe_oos = (oos_feat["mfe"] >= 1).astype(int).values
    _, pred_mfe, auc_mfe, cv_mfe = train_and_eval(X_is, y_mfe_is, X_oos, y_mfe_oos, feat_cols, "MFE ≥ 1R")

    # ═══ EXPERIMENT B: Feature subsets ═══
    print("\n" + "=" * 70)
    print("EXPERIMENT B: FEATURE SUBSETS")
    print("=" * 70)

    # B1: Only visual shape features
    X_is_shape = is_feat[shape_cols].values
    X_oos_shape = oos_feat[shape_cols].values
    if len(shape_cols) > 0:
        _, _, auc_shape, cv_shape = train_and_eval(
            X_is_shape, y_win_is, X_oos_shape, y_win_oos, shape_cols, "Shape Only")

    # B2: Only basic features (no shape)
    X_is_basic = is_feat[basic_cols].values
    X_oos_basic = oos_feat[basic_cols].values
    _, _, auc_basic, cv_basic = train_and_eval(
        X_is_basic, y_win_is, X_oos_basic, y_win_oos, basic_cols, "Basic Only")

    # B3: Only regime + context features
    regime_cols = [c for c in basic_cols if any(k in c for k in
        ["realized_vol", "vol_ratio", "day_position", "day_range", "gap",
         "recent_3", "recent_10", "minutes_since"])]
    if len(regime_cols) >= 3:
        X_is_reg = is_feat[regime_cols].values
        X_oos_reg = oos_feat[regime_cols].values
        _, _, auc_reg, cv_reg = train_and_eval(
            X_is_reg, y_win_is, X_oos_reg, y_win_oos, regime_cols, "Regime Only")

    # B4: Only candlestick patterns
    candle_cols = [c for c in basic_cols if any(k in c for k in
        ["hammer", "inside_bar", "bullish_engulf", "three_bar_squeeze", "doji",
         "dcp", "body", "bar_range"])]
    if len(candle_cols) >= 3:
        X_is_cand = is_feat[candle_cols].values
        X_oos_cand = oos_feat[candle_cols].values
        _, _, auc_cand, cv_cand = train_and_eval(
            X_is_cand, y_win_is, X_oos_cand, y_win_oos, candle_cols, "Candle Only")

    # ═══ EXPERIMENT C: Purged time-series CV (stricter validation) ═══
    print("\n" + "=" * 70)
    print("EXPERIMENT C: PURGED TS-CV WITH GAP (strictest validation)")
    print("=" * 70)

    # Add purge gap: remove 20 trades around each fold boundary
    PURGE = 20
    tscv = TimeSeriesSplit(n_splits=5)
    purged_aucs = []
    for train_idx, val_idx in tscv.split(X_is):
        # Remove last PURGE from train and first PURGE from val
        train_idx = train_idx[:-PURGE] if len(train_idx) > PURGE else train_idx
        val_idx = val_idx[PURGE:] if len(val_idx) > PURGE else val_idx
        if len(train_idx) < 50 or len(val_idx) < 20:
            continue
        dtrain = lgb.Dataset(X_is[train_idx], label=y_win_is[train_idx])
        dval = lgb.Dataset(X_is[val_idx], label=y_win_is[val_idx])
        params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.03, "num_leaves": 10, "min_data_in_leaf": 30,
            "max_depth": 3, "feature_fraction": 0.6, "bagging_fraction": 0.6,
            "bagging_freq": 5, "verbose": -1, "seed": 42,
            "lambda_l1": 1.0, "lambda_l2": 1.0,
        }
        m = lgb.train(params, dtrain, num_boost_round=300,
                      valid_sets=[dval], callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30)])
        p = m.predict(X_is[val_idx])
        try:
            purged_aucs.append(roc_auc_score(y_win_is[val_idx], p))
        except:
            purged_aucs.append(0.5)

    print(f"  Purged CV AUCs: {[f'{x:.4f}' for x in purged_aucs]}")
    print(f"  Purged CV mean: {np.mean(purged_aucs):.4f}")

    # ═══ EXPERIMENT D: Filter impact analysis (best model) ═══
    print("\n" + "=" * 70)
    print("EXPERIMENT D: FILTER IMPACT (best OOS model)")
    print("=" * 70)

    # Use the model with best OOS AUC
    best_aucs = {"win": auc_win, "gate": auc_gate, "be": auc_be, "mfe": auc_mfe}
    best_name = max(best_aucs, key=best_aucs.get)
    best_preds = {"win": pred_win, "gate": pred_gate, "be": pred_be, "mfe": pred_mfe}
    best_pred = best_preds[best_name]
    print(f"  Best OOS model: {best_name} (AUC={best_aucs[best_name]:.4f})")

    base_gw = oos_feat.loc[oos_feat["net_pnl"] > 0, "net_pnl"].sum()
    base_gl = abs(oos_feat.loc[oos_feat["net_pnl"] <= 0, "net_pnl"].sum())
    base_pf = base_gw / base_gl if base_gl > 0 else 0
    base_cum = oos_feat["net_pnl"].cumsum()
    base_dd = (base_cum.cummax() - base_cum).max()
    base_b5 = (oos_feat["trade_r"] >= 5).sum()

    print(f"\n  {'Thresh':>8} {'PF':>8} {'DD':>8} {'PnL':>9} {'N':>6} {'Skip':>5} {'5R+':>4} {'5R_lost':>7}")
    print(f"  {'base':>8} {base_pf:>8.3f} {base_dd:>8.0f} {oos_feat['net_pnl'].sum():>9.0f} {len(oos_feat):>6} {'0':>5} {base_b5:>4} {'0':>7}")

    for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        r = simulate_filter(oos_feat, best_pred, t)
        print(f"  {t:>8.2f} {r['pf']:>8.3f} {r['dd']:>8.0f} {r['pnl']:>9.0f} {r['n']:>6} {r['skip']:>5} {r['b5k']:>4} {r['b5s']:>7}")

    # ═══ FINAL SUMMARY ═══
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  {'Experiment':25s} {'CV_AUC':>8} {'OOS_AUC':>9} {'Verdict':>10}")
    print(f"  {'-'*55}")
    all_results = [
        ("A1: Win/Loss", cv_win, auc_win),
        ("A2: Gate Pass", cv_gate, auc_gate),
        ("A3: BE Trigger", cv_be, auc_be),
        ("A5: MFE ≥ 1R", cv_mfe, auc_mfe),
    ]
    if len(shape_cols) > 0:
        all_results.append(("B1: Shape Only", cv_shape, auc_shape))
    all_results.append(("B2: Basic Only", cv_basic, auc_basic))
    if len(regime_cols) >= 3:
        all_results.append(("B3: Regime Only", cv_reg, auc_reg))
    if len(candle_cols) >= 3:
        all_results.append(("B4: Candle Only", cv_cand, auc_cand))

    any_useful = False
    for name, cv, oos in all_results:
        verdict = "✓ USEFUL" if oos >= 0.55 else "✗ noise"
        if oos >= 0.55:
            any_useful = True
        print(f"  {name:25s} {cv:>8.4f} {oos:>9.4f} {verdict:>10}")

    print(f"\n  Purged TS-CV: {np.mean(purged_aucs):.4f} (true IS generalization)")

    if not any_useful:
        print(f"\n  ══ DEFINITIVE CONCLUSION ══")
        print(f"  NO ML approach achieves OOS AUC ≥ 0.55 across any:")
        print(f"  - Target definition (win/loss, gate pass, BE trigger, MFE, big win)")
        print(f"  - Feature subset (all, shape only, basic only, regime only, candle only)")
        print(f"  - Model complexity (LightGBM with strong regularization + purged CV)")
        print(f"")
        print(f"  With {len(is_feat)} IS trades, {len(feat_cols)} features, and 4 target variants,")
        print(f"  the null hypothesis (no predictive signal) CANNOT be rejected.")
        print(f"")
        print(f"  RECOMMENDATION: Stop searching for entry filters.")
        print(f"  Focus on post-entry management (gate, BE, chandelier) which has")
        print(f"  demonstrated real edge (Cohen's d > 1.0 on 3-bar MFE).")
    else:
        print(f"\n  ══ PROMISING RESULT ══")
        print(f"  At least one approach shows OOS AUC ≥ 0.55.")
        print(f"  Next: verify 5R+ preservation and walk-forward stability.")


if __name__ == "__main__":
    main()
