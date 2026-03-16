# -*- coding: utf-8 -*-
"""
preprocess_vehicle_v14.py
=========================
车辆数据预处理（完整整合版）
- 保留所有原始列 + t_s + LTR_est
- 启动阶段剔除：固定5s + 自动稳定点（std<0.002 & |mean|<0.01，窗口~1s，扫描前10s）
- 事件检测：
    * roll 迟滞（0.08/0.14/退出0.06）→ critical/extreme
    * LTR 迟滞（同阈值）→ critical/extreme
    * sub-critical（低阈长期波动）：std阈0.012，窗口1.0s，持续≥1.5s，且峰值<0.08
    * dynamic-only（放宽阈值）：roll_rate>0.05, yaw_rate>0.30, steer_rate>0.60, ay>1.3，满足≥2条
- 统一事件合并：gap≤0.5s 合并，优先级 dynamic-only < sub-critical < critical < extreme
- 分段可视化：左轴 roll(蓝)、右轴 LTR(绿)；红=extreme、橙=critical、青=sub-critical、灰=dynamic-only
- 输出：
    * 每被试：*_fixed_200Hz_v14.csv、*_roll_ltr_events_v14_seg*.png、roll_events.csv
    * 全局：process_log_v14.csv（数量、时长、占比等）
"""

import math, traceback
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ===================== 参数区（根据需要修改） =====================
ROOT_IN  = Path(r"F:\数据集处理\datasetprocess\数据预处理\车辆数据处理\初步处理数据")
ROOT_OUT = Path(r"F:\数据集处理\datasetprocess\数据预处理\车辆数据处理\处理后车辆数据_v14")

TRACK_M   = 2.176
H_CG_M    = 1.2
G         = 9.80665

TARGET_FS   = 200.0
USE_SAVGOL  = True
SAVGOL_POLY = 2
SG_WIN_S    = 0.05

ROLL_ENTER_LOW   = 0.08
ROLL_ENTER_HIGH  = 0.14
ROLL_EXIT_TH     = 0.06
MIN_DUR_S        = 0.20
MERGE_GAP_S      = 0.50

SUB_STD_TH       = 0.012
SUB_MIN_DUR_S    = 1.50
SUB_WIN_S        = 1.00

DYN_ROLL_RATE_TH  = 0.05
DYN_YAW_RATE_TH   = 0.30
DYN_STEER_RATE_TH = 0.60
DYN_AY_TH         = 1.30
DYN_MIN_DUR_S     = 0.30
DYN_PREPAD_S      = 0.20
DYN_POSTPAD_S     = 0.40

STARTUP_ENABLED     = True
STARTUP_FIXED_S     = 5.0
AUTO_STABLE_SCAN_S  = 10.0
ROLL_STABLE_STD     = 0.002
ROLL_STABLE_MEAN    = 0.010

PLOT_MAX_SECONDS = 60.0
FIG_SIZE = (12, 3.6)

def _pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None

def _to_rad_per_s(series):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    q = np.nanquantile(np.abs(s.dropna()), 0.99) if np.isfinite(s).any() else 0.0
    if q > 6.0:
        return s * (np.pi/180.0)
    return s

def _to_rad(series):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    q = np.nanquantile(np.abs(s.dropna()), 0.99) if np.isfinite(s).any() else 0.0
    if q > 3.2:
        return s * (np.pi/180.0)
    return s

def ensure_datetime_series(s):
    if np.issubdtype(s.dtype, np.datetime64):
        return s
    return pd.to_datetime(s, errors="coerce")

def to_relative_seconds(dt):
    t0 = dt.iloc[0]
    return (dt - t0).dt.total_seconds()

def maybe_savgol(x):
    if not USE_SAVGOL:
        return x
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return x
    if np.isnan(x).all():
        return x
    if np.isnan(x).any():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    win = max(5, int(round(TARGET_FS * SG_WIN_S)) | 1)
    if win >= len(x):
        win = len(x) - 1 if len(x) % 2 == 0 else len(x)
    if win < 5:
        return x
    try:
        y = savgol_filter(x, window_length=win, polyorder=SAVGOL_POLY, mode="interp")
    except Exception:
        y = x
    return y

def resample_to_target(df, time_col):
    if df.empty:
        return df.copy()
    t0 = pd.Timestamp("1970-01-01 00:00:00")
    pseudo_dt = t0 + pd.to_timedelta(df[time_col], unit="s")
    df2 = df.copy(); df2.index = pseudo_dt
    period_ms = int(round(1000.0 / TARGET_FS))
    rule = f"{period_ms}ms"
    out_index = pd.date_range(start=pseudo_dt.iloc[0], end=pseudo_dt.iloc[-1], freq=rule)
    out = pd.DataFrame(index=out_index)
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        out[c] = df2[c].reindex(out.index).interpolate(method="time")
    other_cols = [c for c in df2.columns if c not in num_cols and c != time_col]
    for c in other_cols:
        out[c] = df2[c].reindex(out.index).ffill()
    out[time_col] = (out.index - t0).total_seconds()
    out.reset_index(drop=True, inplace=True)
    return out

def estimate_ltr(ay):
    if ay is None:
        return None
    return (2.0 * H_CG_M / TRACK_M) * (ay / G)

def detect_hysteresis_events(t, x,
                             enter_low=ROLL_ENTER_LOW, enter_high=ROLL_ENTER_HIGH,
                             exit_th=ROLL_EXIT_TH, min_dur=MIN_DUR_S,
                             label_low="critical", label_high="extreme"):
    events = []; state="none"; start_idx=None; peak_val=None; peak_time=None
    abs_x = np.abs(x)
    def close_event(end_idx):
        nonlocal start_idx, peak_val, peak_time, state
        if start_idx is None: return
        dur = t[end_idx] - t[start_idx]
        if dur >= min_dur:
            etype = label_high if state=="high" else label_low
            events.append({"start_s": float(t[start_idx]), "end_s": float(t[end_idx]),
                           "duration_s": float(dur), "peak_abs_roll": float(peak_val if peak_val is not None else np.nan),
                           "peak_time_s": float(peak_time if peak_time is not None else np.nan),
                           "event_type": etype})
        start_idx=None; peak_val=None; peak_time=None; state="none"
    for i,val in enumerate(abs_x):
        if state=="none":
            if val>=enter_high: state="high"; start_idx=i; peak_val=val; peak_time=t[i]
            elif val>=enter_low: state="low"; start_idx=i; peak_val=val; peak_time=t[i]
        elif state=="low":
            if val>=enter_high: state="high"
            if peak_val is None or val>peak_val: peak_val=val; peak_time=t[i]
            if val<exit_th: close_event(i)
        elif state=="high":
            if peak_val is None or val>peak_val: peak_val=val; peak_time=t[i]
            if val<exit_th: close_event(i)
    if start_idx is not None: close_event(len(t)-1)
    if not events: return events
    merged=[events[0]]; rank={label_low:1,label_high:2}
    for ev in events[1:]:
        if ev["start_s"]-merged[-1]["end_s"]<=MERGE_GAP_S:
            merged[-1]["end_s"]=ev["end_s"]
            merged[-1]["duration_s"]=merged[-1]["end_s"]-merged[-1]["start_s"]
            if (not np.isnan(ev["peak_abs_roll"])) and ev["peak_abs_roll"]>merged[-1]["peak_abs_roll"]:
                merged[-1]["peak_abs_roll"]=ev["peak_abs_roll"]; merged[-1]["peak_time_s"]=ev["peak_time_s"]
            if rank.get(ev["event_type"],0)>rank.get(merged[-1]["event_type"],0):
                merged[-1]["event_type"]=ev["event_type"]
        else: merged.append(ev)
    return merged

def detect_dynamic_events(df, t_col="t_s", roll_col="zx|roll"):
    if t_col not in df.columns or roll_col not in df.columns: return []
    t=df[t_col].to_numpy(); roll=pd.to_numeric(df[roll_col],errors="coerce").astype(float).to_numpy()
    with np.errstate(invalid="ignore"): roll_rate=np.gradient(roll,t)
    yaw_col=_pick(df, ["zx|vyaw","zx|yawrate","zx|yawRate"])
    yaw_rate=_to_rad_per_s(df[yaw_col]).to_numpy() if yaw_col else np.zeros_like(roll)
    steer_col=_pick(df, ["zx|SteeringWheel","zx1|SteeringWheel","zx|steer","zx1|steer"])
    if steer_col:
        steer=_to_rad(df[steer_col]).to_numpy()
        with np.errstate(invalid="ignore"): steer_rate=np.gradient(steer,t)
    else: steer_rate=np.zeros_like(roll)
    ay_col=_pick(df, ["zx|ay","zx1|ay","ay"]); ay=pd.to_numeric(df[ay_col],errors="coerce").astype(float).to_numpy() if ay_col else np.zeros_like(roll)
    c1=np.abs(roll_rate)>DYN_ROLL_RATE_TH; c2=np.abs(yaw_rate)>DYN_YAW_RATE_TH; c3=np.abs(steer_rate)>DYN_STEER_RATE_TH; c4=np.abs(ay)>DYN_AY_TH
    mask=(c1.astype(int)+c2.astype(int)+c3.astype(int)+c4.astype(int))>=2
    events=[]; active=False; s=None
    for i,on in enumerate(mask):
        if on and not active: active=True; s=t[i]
        elif not on and active:
            e=t[i]; s_pad=max(t[0],s-DYN_PREPAD_S); e_pad=min(t[-1],e+DYN_POSTPAD_S)
            if (e_pad-s_pad)>=DYN_MIN_DUR_S:
                seg_mask=(t>=s_pad)&(t<=e_pad); roll_seg=np.abs(roll[seg_mask])
                if roll_seg.size>0 and np.isfinite(roll_seg).any():
                    peak_idx=np.nanargmax(roll_seg); peak_time=t[seg_mask][peak_idx]; peak_val=float(roll_seg[peak_idx])
                else: peak_time=np.nan; peak_val=np.nan
                events.append({"start_s":float(s_pad),"end_s":float(e_pad),"duration_s":float(e_pad-s_pad),
                               "peak_abs_roll":peak_val,"peak_time_s":float(peak_time),"event_type":"dynamic-only"})
            active=False
    if active:
        e=t[-1]; s_pad=max(t[0],s-DYN_PREPAD_S); e_pad=min(t[-1],e+DYN_POSTPAD_S)
        if (e_pad-s_pad)>=DYN_MIN_DUR_S:
            seg_mask=(t>=s_pad)&(t<=e_pad); roll_seg=np.abs(roll[seg_mask])
            if roll_seg.size>0 and np.isfinite(roll_seg).any():
                peak_idx=np.nanargmax(roll_seg); peak_time=t[seg_mask][peak_idx]; peak_val=float(roll_seg[peak_idx])
            else: peak_time=np.nan; peak_val=np.nan
            events.append({"start_s":float(s_pad),"end_s":float(e_pad),"duration_s":float(e_pad-s_pad),
                           "peak_abs_roll":peak_val,"peak_time_s":float(peak_time),"event_type":"dynamic-only"})
    return events

def detect_subcritical_oscillation_events(t, roll, std_th=SUB_STD_TH, min_dur=SUB_MIN_DUR_S, win_s=SUB_WIN_S):
    if len(t)<3: return []
    dt=np.diff(t); med=np.median(dt[dt>0]) if np.any(dt>0) else (1.0/TARGET_FS); fs=1.0/med if med>0 else TARGET_FS
    win_n=max(3,int(round(win_s*fs)))
    roll_abs=np.abs(roll); rolling_std=pd.Series(roll_abs).rolling(win_n,center=True).std().to_numpy()
    mask=rolling_std>std_th; events=[]; active=False; s=None
    for i,on in enumerate(mask):
        if on and not active: active=True; s=t[i]
        elif not on and active:
            e=t[i]
            if (e-s)>=min_dur:
                seg_mask=(t>=s)&(t<=e); seg=roll_abs[seg_mask]
                if seg.size>0 and np.isfinite(seg).any():
                    peak_val=float(np.nanmax(seg))
                    if peak_val<ROLL_ENTER_LOW:
                        peak_idx=np.nanargmax(seg); peak_time=float(t[seg_mask][peak_idx])
                        events.append({"start_s":float(s),"end_s":float(e),"duration_s":float(e-s),
                                       "peak_abs_roll":peak_val,"peak_time_s":peak_time,"event_type":"sub-critical"})
            active=False
    return events

def merge_events(events):
    if not events: return []
    events=sorted(events,key=lambda e:e["start_s"])
    merged=[events[0]]; priority={"dynamic-only":1,"sub-critical":2,"critical":3,"extreme":4}
    for e in events[1:]:
        prev=merged[-1]
        if e["start_s"]-prev["end_s"]<=MERGE_GAP_S:
            prev["end_s"]=max(prev["end_s"],e["end_s"]); prev["duration_s"]=prev["end_s"]-prev["start_s"]
            if e.get("peak_abs_roll",np.nan)>prev.get("peak_abs_roll",np.nan):
                prev["peak_abs_roll"]=e["peak_abs_roll"]; prev["peak_time_s"]=e["peak_time_s"]
            if priority.get(e["event_type"],0)>priority.get(prev["event_type"],0):
                prev["event_type"]=e["event_type"]
        else: merged.append(e)
    return merged

def remove_startup_artifacts(df, t_col, roll_col):
    if df.empty or not STARTUP_ENABLED: return df.copy(), 0.0, "disabled_or_empty"
    if t_col not in df.columns or roll_col not in df.columns: return df.copy(), 0.0, "no_time_or_roll"
    t=df[t_col].to_numpy(); roll=pd.to_numeric(df[roll_col],errors="coerce").astype(float).to_numpy()
    if len(t)<5: return df.copy(), 0.0, "too_short"
    fixed_start=STARTUP_FIXED_S
    scan_end=min(AUTO_STABLE_SCAN_S, t[-1]-t[0])
    median_dt=np.median(np.diff(t)[np.diff(t)>0]) if len(t)>2 else 1.0/TARGET_FS
    win_n=max(3,int(round(1.0/median_dt)))
    stable_start=None
    for i in range(win_n,len(t)):
        if t[i]>scan_end: break
        seg=roll[i-win_n:i]
        if np.isnan(seg).all(): continue
        s_std=np.nanstd(seg); s_mean=np.nanmean(seg)
        if s_std<ROLL_STABLE_STD and abs(s_mean)<ROLL_STABLE_MEAN:
            stable_start=t[i]; break
    if stable_start is not None and stable_start>fixed_start:
        valid_start=stable_start; reason=f"auto_stable({stable_start:.2f}s)"
    else:
        valid_start=fixed_start; reason=f"fixed({fixed_start:.2f}s)"
    mask=t>=valid_start
    if not mask.any(): return df.iloc[0:0].copy(), float(t[-1]-t[0]), f"{reason}_all_skipped"
    df2=df.loc[mask].reset_index(drop=True).copy()
    t0=df2[t_col].iloc[0]; df2[t_col]=df2[t_col]-t0
    return df2, float(valid_start), reason

def plot_roll_ltr_with_events(df,t_col,roll_col,out_png,span_events):
    if df.empty or t_col not in df.columns or roll_col not in df.columns or "LTR_est" not in df.columns: return
    t=df[t_col].to_numpy(); roll=pd.to_numeric(df[roll_col],errors="coerce").astype(float).to_numpy(); ltr=pd.to_numeric(df["LTR_est"],errors="coerce").astype(float).to_numpy()
    total_T=t[-1]-t[0] if len(t)>1 else 0.0; n_seg=max(1,int(math.ceil(total_T/PLOT_MAX_SECONDS)))
    for s in range(n_seg):
        ts=t[0]+s*PLOT_MAX_SECONDS; te=min(t[0]+(s+1)*PLOT_MAX_SECONDS,t[-1]); mask=(t>=ts)&(t<=te)
        if mask.sum()<5: continue
        fig,ax1=plt.subplots(figsize=FIG_SIZE); ax2=ax1.twinx()
        ax1.plot(t[mask],roll[mask],color="tab:blue",lw=1.2,label="roll (rad)"); ax1.set_ylabel("roll (rad)",color="tab:blue")
        ax2.plot(t[mask],ltr[mask],color="tab:green",lw=1.2,ls="--",label="LTR_est"); ax2.set_ylabel("LTR_est",color="tab:green")
        for y,c in [(ROLL_ENTER_LOW,"tab:orange"),(-ROLL_ENTER_LOW,"tab:orange"),(ROLL_ENTER_HIGH,"tab:red"),(-ROLL_ENTER_HIGH,"tab:red")]:
            ax1.axhline(y,color=c,ls=":",lw=1)
        for ev in span_events:
            if ev["end_s"]<ts or ev["start_s"]>te: continue
            xs=max(ev["start_s"],ts); xe=min(ev["end_s"],te)
            color_map={"extreme":"tab:red","critical":"tab:orange","sub-critical":"tab:cyan","dynamic-only":"gray"}
            ax1.axvspan(xs,xe,color=color_map.get(ev["event_type"],"gray"),alpha=0.18)
        ax1.grid(True,alpha=0.3,ls="--",color="gray"); ax1.set_xlabel("time (s)"); ax1.set_title(f"roll + LTR with events (segment {s+1}/{n_seg})")
        lines,labels=[],[]
        for ax in [ax1,ax2]: lns,lbs=ax.get_legend_handles_labels(); lines+=lns; labels+=lbs
        ax1.legend(lines,labels,loc="upper right",ncol=3)
        fig.tight_layout(); fig.savefig(out_png.with_name(out_png.stem+f"_seg{s+1}.png"),dpi=160); plt.close(fig)

def process_one_file(csv_path, out_root):
    try:
        df=pd.read_csv(csv_path)
        try: subject=csv_path.relative_to(ROOT_IN).parts[0]
        except Exception: subject=csv_path.parent.name
        time_col=_pick(df, ["StorageTime","storageTime","time","Time"])
        if time_col is None:
            print(f"⚠️ 跳过（无时间列）：{csv_path}"); return None, [], subject, None
        dt=ensure_datetime_series(df[time_col]); df=df.loc[~dt.isna(), :].copy(); dt=ensure_datetime_series(df[time_col])
        order=np.argsort(dt.values); df=df.iloc[order].reset_index(drop=True); dt=ensure_datetime_series(df[time_col])
        dup_mask=dt.duplicated()
        if dup_mask.any(): df=df.loc[~dup_mask].reset_index(drop=True); dt=ensure_datetime_series(df[time_col])
        df["t_s"]=to_relative_seconds(dt)
        if len(df)>=3:
            dt_s=np.diff(df["t_s"].to_numpy()); med=np.median(dt_s[dt_s>0]); est_fs=1.0/med if med>0 else TARGET_FS
        else: est_fs=TARGET_FS
        roll_col=_pick(df, ["zx|roll","roll","zx|phi","phi"])
        if roll_col is None: print(f"⚠️ 跳过（无 roll 列）：{csv_path}"); return None, [], subject, None
        df, removed_s, remove_reason = remove_startup_artifacts(df,"t_s",roll_col)
        ay_col=_pick(df, ["zx|ay","ay","zx1|ay"]); ay_series=df[ay_col] if ay_col is not None else None
        df[roll_col]=maybe_savgol(df[roll_col].to_numpy(dtype=float))
        if ay_series is not None: df[ay_col]=maybe_savgol(df[ay_col].to_numpy(dtype=float))
        df_rs=resample_to_target(df,"t_s")
        if ay_col is not None and ay_col in df_rs.columns: df_rs["LTR_est"]=estimate_ltr(df_rs[ay_col])
        else: df_rs["LTR_est"]=np.nan
        t=df_rs["t_s"].to_numpy(); roll=pd.to_numeric(df_rs[roll_col],errors="coerce").astype(float).to_numpy()
        events_roll=detect_hysteresis_events(t,roll,label_low="critical",label_high="extreme")
        if ay_col is not None and ay_col in df_rs.columns:
            ltr=pd.to_numeric(df_rs["LTR_est"],errors="coerce").astype(float).to_numpy()
            events_ltr=detect_hysteresis_events(t,ltr,label_low="critical",label_high="extreme")
        else: events_ltr=[]
        events_dyn=detect_dynamic_events(df_rs,"t_s",roll_col)
        events_sub=detect_subcritical_oscillation_events(t,roll)
        all_events=merge_events(events_roll+events_ltr+events_dyn+events_sub)
        subj_out_dir=out_root/subject; subj_out_dir.mkdir(parents=True,exist_ok=True)
        out_csv=subj_out_dir/(csv_path.stem+"_fixed_200Hz_v14.csv"); df_rs.to_csv(out_csv,index=False,encoding="utf-8-sig")
        out_png=subj_out_dir/(csv_path.stem+"_roll_ltr_events_v14.png")
        plot_roll_ltr_with_events(df_rs,"t_s",roll_col,out_png,all_events)
        total_T=float(df_rs["t_s"].iloc[-1]) if len(df_rs)>0 else 0.0
        def _sum_dur(kind): return float(np.sum([e["duration_s"] for e in all_events if e["event_type"]==kind])) if all_events else 0.0
        rec={"subject":subject,"file":csv_path.name,"fs_est":round(est_fs,3),
             "removed_startup_s":round(removed_s,3),"remove_reason":remove_reason,"duration_after_s":round(total_T,3),
             "n_total":len(all_events),"n_extreme":sum(e["event_type"]=="extreme" for e in all_events),
             "n_critical":sum(e["event_type"]=="critical" for e in all_events),
             "n_subcritical":sum(e["event_type"]=="sub-critical" for e in all_events),
             "n_dynamic":sum(e["event_type"]=="dynamic-only" for e in all_events),
             "dur_extreme_s":_sum_dur("extreme"),"dur_critical_s":_sum_dur("critical"),
             "dur_subcritical_s":_sum_dur("sub-critical"),"dur_dynamic_s":_sum_dur("dynamic-only"),
             "coverage_%":round(100.0*(sum(e["duration_s"] for e in all_events)/total_T) if total_T>0 else 0.0,2),
             "roll_std":float(np.nanstd(roll)) if len(roll)>0 else np.nan,
             "roll_max_abs":float(np.nanmax(np.abs(roll))) if len(roll)>0 else np.nan}
        return df_rs, all_events, subject, rec
    except Exception:
        print(f"❌ 处理失败：{csv_path}\n{traceback.format_exc()}"); return None, [], csv_path.parent.name, None

def main():
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    subject_events={}; records=[]
    files=list(ROOT_IN.rglob("*_vehicle*.csv"))
    if not files:
        print(f"⚠️ 未找到 *_vehicle.csv，检查输入目录：{ROOT_IN}"); return
    for f in files:
        df_200, evs, subj, rec = process_one_file(f, ROOT_OUT)
        if df_200 is None: continue
        subject_events.setdefault(subj, []).extend([{**e, "file": f.name} for e in evs])
        if rec is not None: records.append(rec)
        print(f"✅ 处理完成：{f.name}")
    for subj, evs in subject_events.items():
        out_csv = ROOT_OUT / subj / "roll_events.csv"
        if evs:
            pd.DataFrame(evs).sort_values("start_s").to_csv(out_csv, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame(columns=["start_s","end_s","duration_s","peak_abs_roll","peak_time_s","event_type","file"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"📝 事件表导出：{subj} → {out_csv}")
    if records:
        pd.DataFrame(records).to_csv(ROOT_OUT/"process_log_v14.csv", index=False, encoding="utf-8-sig")
        print(f"📒 全局处理日志：{ROOT_OUT/'process_log_v14.csv'}")
    print("\n🎯 所有车辆数据处理完成（v14 完整版）。")

if __name__ == "__main__":
    main()
