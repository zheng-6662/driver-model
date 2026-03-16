# -*- coding: utf-8 -*-
"""
Step C-1: Build a FIXED road-type template from ONE reference vehicle CSV.

What it does
- Reads reference run vehicle CSV
- Uses lane curvature column (e.g., zx1|lanecurvatureXY) to label road type with hysteresis
- Builds a reference polyline (x_ref,y_ref) and cumulative distance s_ref
- Saves an NPZ template: x_ref, y_ref, s_ref, kappa_smooth, road_type_ref (0=straight,1=curve)

Outputs
- <out_dir>/road_template.npz
- <out_dir>/road_template_segments.csv
- <out_dir>/figures/template_xy_roadtype.png
- <out_dir>/figures/template_kappa_vs_s.png

Usage (Windows examples)
python build_road_template.py --ref_csv "F:\\...\\Entity_Recording_..._vehicle_aligned_cleaned.csv" --out_dir "F:\\...\\road_template"
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-12

def robust_read_csv(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, encoding_errors="ignore")

def first_existing_col(df: pd.DataFrame, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def cumulative_distance(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    return np.cumsum(np.sqrt(dx*dx + dy*dy))

def rolling_smooth(x, win=25, kind="median"):
    s = pd.Series(np.asarray(x, dtype=np.float64))
    if win <= 1:
        return s.to_numpy()
    if kind == "mean":
        return s.rolling(win, center=True, min_periods=max(1, win//3)).mean().to_numpy()
    return s.rolling(win, center=True, min_periods=max(1, win//3)).median().to_numpy()

def hysteresis_label(kappa_abs, enter_thr, exit_thr):
    kappa_abs = np.asarray(kappa_abs, dtype=np.float64)
    y = np.zeros_like(kappa_abs, dtype=np.int8)
    state = 0
    for i, ka in enumerate(kappa_abs):
        if state == 0 and ka >= enter_thr:
            state = 1
        elif state == 1 and ka <= exit_thr:
            state = 0
        y[i] = state
    return y

def segments_from_labels(labels):
    labels = np.asarray(labels, dtype=np.int8)
    if labels.size == 0:
        return []
    segs = []
    s = 0
    cur = int(labels[0])
    for i in range(1, len(labels)):
        if int(labels[i]) != cur:
            segs.append((s, i-1, cur))
            s = i
            cur = int(labels[i])
    segs.append((s, len(labels)-1, cur))
    return segs

def merge_short_segments(labels, s_ref, min_seg_m=30.0, max_iter=10):
    """Merge segments shorter than min_seg_m into neighboring longer segments."""
    labels = labels.copy().astype(np.int8)
    s_ref = np.asarray(s_ref, dtype=np.float64)

    def seg_len(a,b):
        return float(s_ref[b] - s_ref[a])

    for _ in range(max_iter):
        segs = segments_from_labels(labels)
        changed = False
        for j,(a,b,lab) in enumerate(segs):
            L = seg_len(a,b)
            if L >= min_seg_m:
                continue
            left = segs[j-1] if j-1 >= 0 else None
            right = segs[j+1] if j+1 < len(segs) else None
            if left is None and right is None:
                continue
            if left is None:
                new_lab = right[2]
            elif right is None:
                new_lab = left[2]
            else:
                new_lab = left[2] if seg_len(left[0], left[1]) >= seg_len(right[0], right[1]) else right[2]
            if int(new_lab) != int(lab):
                labels[a:b+1] = int(new_lab)
                changed = True
        if not changed:
            break
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ref_csv",
        default="F:\\数据集处理\\data_process\\datasetprocess\\多模态数据\\被试数据集合\\zx\\vehicle\\Entity_Recording_2025_09_27_17_45_11_vehicle_aligned_cleaned.csv",
        help="Reference vehicle CSV",
    )
    ap.add_argument(
        "--out_dir",
        default="F:\\数据集处理\\data_process\\datasetprocess\\多模态数据\\被试数据集合\\zx",
        help="Output directory",
    )
    ap.add_argument("--x_col", default=None)
    ap.add_argument("--y_col", default=None)
    ap.add_argument("--kappa_col", default=None)
    ap.add_argument("--time_col", default=None)

    ap.add_argument("--kappa_clip_to_zero", type=float, default=1e-7,
                    help="abs(kappa) below this will be set to 0")
    ap.add_argument("--smooth_win", type=int, default=25)
    ap.add_argument("--smooth_kind", choices=["median","mean"], default="median")

    ap.add_argument("--enter_curve", type=float, default=5e-5)
    ap.add_argument("--exit_curve", type=float, default=2e-5)
    ap.add_argument("--min_seg_m", type=float, default=30.0)

    ap.add_argument("--plot_step", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = robust_read_csv(args.ref_csv)

    x_col = args.x_col or first_existing_col(df, ["zx|x","x","X","pos_x"])
    y_col = args.y_col or first_existing_col(df, ["zx|y","y","Y","pos_y"])
    kappa_col = args.kappa_col or first_existing_col(df, ["zx1|lanecurvatureXY","lanecurvatureXY","kappa","curvature"])
    t_col = args.time_col or first_existing_col(df, ["t_s","StorageTime","time","Time"])

    if x_col is None or y_col is None or kappa_col is None:
        raise ValueError(f"Missing required columns. Found x={x_col}, y={y_col}, kappa={kappa_col}")

    x = df[x_col].to_numpy(dtype=np.float64)
    y = df[y_col].to_numpy(dtype=np.float64)
    kappa = df[kappa_col].to_numpy(dtype=np.float64)
    s_ref = cumulative_distance(x, y)

    # clean + smooth
    kappa = np.where(np.abs(kappa) < args.kappa_clip_to_zero, 0.0, kappa)
    kappa_s = rolling_smooth(kappa, win=args.smooth_win, kind=args.smooth_kind)
    k_abs = np.abs(kappa_s)

    labels = hysteresis_label(k_abs, args.enter_curve, args.exit_curve)
    labels = merge_short_segments(labels, s_ref, min_seg_m=args.min_seg_m)

    # save template
    npz_path = os.path.join(args.out_dir, "road_template.npz")
    np.savez_compressed(npz_path,
                        x_ref=x.astype(np.float32),
                        y_ref=y.astype(np.float32),
                        s_ref=s_ref.astype(np.float32),
                        kappa_smooth=kappa_s.astype(np.float32),
                        road_type_ref=labels.astype(np.int8))

    # segments csv
    segs = segments_from_labels(labels)
    rows = []
    for sid,(a,b,lab) in enumerate(segs):
        rows.append({
            "seg_id": sid,
            "label": int(lab),
            "label_str": "straight" if lab==0 else "curve",
            "start_idx": int(a),
            "end_idx": int(b),
            "start_s_m": float(s_ref[a]),
            "end_s_m": float(s_ref[b]),
            "len_m": float(s_ref[b]-s_ref[a]),
            "kappa_abs_mean": float(np.nanmean(np.abs(kappa_s[a:b+1]))),
            "kappa_abs_max": float(np.nanmax(np.abs(kappa_s[a:b+1]))),
        })
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "road_template_segments.csv"),
                              index=False, encoding="utf-8-sig")

    # plots
    step = max(1, int(args.plot_step))
    idx = np.arange(0, len(x), step)

    plt.figure(figsize=(6.2, 6.2))
    plt.scatter(x[idx], y[idx], c=labels[idx], s=3)
    plt.axis("equal")
    plt.xlabel(x_col); plt.ylabel(y_col)
    plt.title("Road template (reference): XY colored by road type (0=straight,1=curve)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "template_xy_roadtype.png"), dpi=220)
    plt.close()

    plt.figure(figsize=(9.0, 3.2))
    plt.plot(s_ref[idx], kappa_s[idx], lw=1)
    in_curve = False
    start = None
    for i in range(len(labels)):
        if labels[i]==1 and not in_curve:
            start = s_ref[i]; in_curve=True
        if labels[i]==0 and in_curve:
            plt.axvspan(start, s_ref[i], alpha=0.18)
            in_curve=False
    if in_curve:
        plt.axvspan(start, s_ref[-1], alpha=0.18)
    plt.xlabel("s_ref (m)")
    plt.ylabel("kappa_smooth (1/m)")
    plt.title("Road template: curvature vs distance (shaded=curve)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "template_kappa_vs_s.png"), dpi=220)
    plt.close()

    straight_ratio = float(np.mean(labels==0))
    curve_ratio = float(np.mean(labels==1))
    print("[OK] road template saved")
    print("  npz:", npz_path)
    print("  segments:", os.path.join(args.out_dir, "road_template_segments.csv"))
    print("  figs:", fig_dir)
    print(f"  ratios: straight={straight_ratio:.3f}, curve={curve_ratio:.3f}")
    print(f"  cols: x={x_col}, y={y_col}, kappa={kappa_col}, time={t_col}")

if __name__ == "__main__":
    main()
