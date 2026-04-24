# -*- coding: utf-8 -*-
"""
Step C-2: Apply a FIXED road-type template to ANY vehicle CSVs by projecting (x,y) onto the reference path.

Method
- Load road_template.npz (x_ref, y_ref, s_ref, road_type_ref)
- For each sample point (x,y) in target CSV:
    find nearest point on reference polyline (nearest neighbor in (x_ref,y_ref))
    assign s_ref_near and road_type_ref_near as fixed road labels

Outputs per input CSV
- *_roadtype_fixed.csv : original + road_s_ref_m + road_type_fixed + road_type_fixed_str + ref_nn_dist_m + ref_nn_ok
- figures: *_xy_fixed.png (colored by fixed label)

Usage
python apply_road_template.py --template "F:\\...\\road_template.npz" --input "F:\\...\\vehicle_folder" --out_dir "F:\\...\\labeled_out"
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def list_inputs(p, pattern):
    if os.path.isfile(p):
        return [p]
    if os.path.isdir(p):
        files = []
        for root, _, names in os.walk(p):
            for name in names:
                if not name.lower().endswith(".csv"):
                    continue
                if pattern and (pattern not in name):
                    continue
                files.append(os.path.join(root, name))
        return sorted(files)
    return sorted(glob.glob(p))

def build_nn_index(x_ref, y_ref):
    """Try fast NN index. Fallback to brute-force if SciPy unavailable."""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.c_[x_ref, y_ref])
        return ("ckdtree", tree)
    except Exception:
        return ("bruteforce", None)

def query_nn(index, x_ref, y_ref, xq, yq, chunk=5000):
    mode, tree = index
    xq = np.asarray(xq, dtype=np.float64)
    yq = np.asarray(yq, dtype=np.float64)
    if mode == "ckdtree":
        dist, ind = tree.query(np.c_[xq, yq], k=1, workers=-1)
        return ind.astype(np.int64), dist.astype(np.float32)

    # brute-force chunked
    n = len(xq)
    ind_out = np.zeros(n, dtype=np.int64)
    dist_out = np.zeros(n, dtype=np.float32)
    ref = np.c_[x_ref.astype(np.float64), y_ref.astype(np.float64)]
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        q = np.c_[xq[i0:i1], yq[i0:i1]]  # (m,2)
        q2 = (q*q).sum(axis=1, keepdims=True)            # (m,1)
        r2 = (ref*ref).sum(axis=1, keepdims=True).T      # (1,Nref)
        d2 = q2 + r2 - 2.0 * (q @ ref.T)                 # (m,Nref)
        j = np.argmin(d2, axis=1)
        ind_out[i0:i1] = j
        dist_out[i0:i1] = np.sqrt(np.maximum(d2[np.arange(i1-i0), j], 0.0)).astype(np.float32)
    return ind_out, dist_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--template",
        default="F:\\数据集处理\\data_process\\datasetprocess\\多模态数据\\被试数据集合\\zx\\road_template.npz",
        help="road_template.npz",
    )
    ap.add_argument(
        "--input",
        default="F:\\数据集处理\\data_process\\datasetprocess\\多模态数据\\被试数据集合",
        help="vehicle csv / folder / glob",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="output dir for figures (default: alongside each CSV)",
    )
    ap.add_argument("--x_col", default=None)
    ap.add_argument("--y_col", default=None)
    ap.add_argument("--plot_step", type=int, default=5)
    ap.add_argument("--max_nn_dist_m", type=float, default=20.0)
    ap.add_argument(
        "--name_contains",
        default="vehicle_aligned_cleaned",
        help="only process CSVs whose filename contains this string (case-sensitive)",
    )
    args = ap.parse_args()

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        fig_dir_root = os.path.join(args.out_dir, "figures")
        os.makedirs(fig_dir_root, exist_ok=True)
    else:
        fig_dir_root = None

    tpl = np.load(args.template, allow_pickle=True)
    x_ref = tpl["x_ref"].astype(np.float64)
    y_ref = tpl["y_ref"].astype(np.float64)
    s_ref = tpl["s_ref"].astype(np.float64)
    road_type_ref = tpl["road_type_ref"].astype(np.int8)

    nn_index = build_nn_index(x_ref, y_ref)

    files = list_inputs(args.input, args.name_contains)
    if not files:
        raise FileNotFoundError(f"No CSV found: {args.input}")

    for fp in files:
        try:
            df = robust_read_csv(fp)
            x_col = args.x_col or first_existing_col(df, ["zx|x","x","X","pos_x"])
            y_col = args.y_col or first_existing_col(df, ["zx|y","y","Y","pos_y"])
            if x_col is None or y_col is None:
                raise ValueError(f"Missing x/y columns. Found x={x_col}, y={y_col}")

            x = df[x_col].to_numpy(dtype=np.float64)
            y = df[y_col].to_numpy(dtype=np.float64)

            ind, dist = query_nn(nn_index, x_ref, y_ref, x, y)
            s_near = s_ref[ind]
            lab = road_type_ref[ind].astype(np.int8)

            out = df.copy()
            out["road_s_ref_m"] = s_near.astype(np.float32)
            out["road_type_fixed"] = lab.astype(np.int8)
            out["road_type_fixed_str"] = np.where(lab==0, "straight", "curve")
            out["ref_nn_dist_m"] = dist.astype(np.float32)
            out["ref_nn_ok"] = (dist <= float(args.max_nn_dist_m)).astype(np.int8)

            base = os.path.splitext(os.path.basename(fp))[0]
            # overwrite original file to add labels in-place
            out_csv = fp
            out.to_csv(out_csv, index=False, encoding="utf-8-sig")

            # plot
            step = max(1, int(args.plot_step))
            idx = np.arange(0, len(x), step)
            plt.figure(figsize=(6.2, 6.2))
            plt.scatter(x[idx], y[idx], c=lab[idx], s=3)
            plt.axis("equal")
            plt.xlabel(x_col); plt.ylabel(y_col)
            plt.title(f"Fixed road type by template (0=straight,1=curve)\\n{base}")
            plt.tight_layout()
            fig_dir = fig_dir_root or os.path.join(os.path.dirname(fp), "figures")
            os.makedirs(fig_dir, exist_ok=True)
            plt.savefig(os.path.join(fig_dir, f"{base}_xy_fixed.png"), dpi=220)
            plt.close()

            ok_ratio = float(np.mean(out["ref_nn_ok"].to_numpy()==1))
            print(f"[OK] {base} -> {out_csv} | nn_ok_ratio={ok_ratio:.3f}")
        except Exception as e:
            print(f"[FAIL] {fp}: {e}")

    print("[DONE]")

if __name__ == "__main__":
    main()
