"""
Per-timestep MAE comparison: baseline vs conditioned_v2 vs step4_tailamp

Inputs:
  reports/baseline_prediction_sequences.npz        (Task 3)
  reports/conditioned_v2_prediction_sequences.npz  (Task 3)
  reports/step4_tailamp_prediction_sequences.npz   (Step 4)
  reports/attribution_master_table.csv             (sample metadata)

Outputs:
  reports/per_timestep_mae_comparison_20260408.png
  reports/per_timestep_mae_comparison_20260408.md
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(r"F:\data_set_process\data_process")
REPORTS = REPO / "reports"

TAIL_START = 200   # step index = 1.0 s at 200 Hz
FPS = 200

# ── 1. Load NPZ files ───────────────────────────────────────────────────────

def load_npz(name):
    p = REPORTS / name
    d = np.load(p, allow_pickle=True)
    pred = d["pred"]          # (N, 400, 2)
    true = d["true"]          # (N, 400, 2)
    mask = d["mask"]          # (N, 400)
    keys = d["sample_keys"]   # (N,) str
    return pred, true, mask, keys

pred_bl, true_bl, mask_bl, keys_bl = load_npz("baseline_prediction_sequences.npz")
pred_v2, true_v2, mask_v2, keys_v2 = load_npz("conditioned_v2_prediction_sequences.npz")
pred_s4, true_s4, mask_s4, keys_s4 = load_npz("step4_tailamp_prediction_sequences.npz")

print(f"baseline  : {pred_bl.shape}, keys sample: {keys_bl[:3]}")
print(f"cond_v2   : {pred_v2.shape}, keys sample: {keys_v2[:3]}")
print(f"step4     : {pred_s4.shape}, keys sample: {keys_s4[:3]}")

# ── 2. Load attribution table for slicing ────────────────────────────────────

attr = pd.read_csv(REPORTS / "attribution_master_table.csv")
# Build lookup: sample_key -> latency_proxy_bucket, eval_morphology_label
attr_idx = attr.set_index("sample_key")[["latency_proxy_bucket", "eval_morphology_label"]]

# Align all three NPZ sets to the same key order (use v2 as reference)
assert np.array_equal(keys_v2, keys_bl), "baseline and v2 key order mismatch"
# step4 may have different order; reindex
s4_key_to_idx = {k: i for i, k in enumerate(keys_s4)}
reorder = [s4_key_to_idx[k] for k in keys_v2]
pred_s4 = pred_s4[reorder]
true_s4 = true_s4[reorder]
mask_s4 = mask_s4[reorder]
keys_ref = keys_v2   # canonical order

# Merge metadata
meta = attr_idx.reindex(keys_ref)
latency = meta["latency_proxy_bucket"].values
morph   = meta["eval_morphology_label"].values

N, T, C = pred_v2.shape
t_axis = np.arange(T) / FPS  # seconds

# ── 3. Per-timestep MAE (steer channel only, masked) ─────────────────────────

def per_step_mae(pred, true, mask):
    """Returns (T,) array: mean |pred - true| per timestep, mask-weighted."""
    err = np.abs(pred[:, :, 0] - true[:, :, 0])  # (N, T) steer only
    valid = mask.astype(bool)                      # (N, T)
    mae = np.zeros(T)
    count = np.zeros(T)
    for t in range(T):
        m = valid[:, t]
        if m.sum() > 0:
            mae[t] = err[m, t].mean()
            count[t] = m.sum()
    return mae, count

def group_per_step_mae(pred, true, mask, bool_mask):
    """Same but restricted to samples where bool_mask is True."""
    return per_step_mae(pred[bool_mask], true[bool_mask], mask[bool_mask])

# ── 4. Define groups ──────────────────────────────────────────────────────────

groups = {
    "ALL":              np.ones(N, dtype=bool),
    "Q1_fast":          latency == "Q1_fast",
    "non_Q1_fast":      latency != "Q1_fast",
    "single_lobe":      morph == "single_lobe",
    "reverse_correction": morph == "reverse_correction",
    "multi_correction": morph == "multi_correction",
}

print("\nGroup sizes:")
for g, m in groups.items():
    print(f"  {g}: {m.sum()}")

# ── 5. Compute per-step MAE for all groups × 3 models ────────────────────────

results = {}
for gname, gmask in groups.items():
    results[gname] = {
        "baseline":  group_per_step_mae(pred_bl, true_bl, mask_bl, gmask)[0],
        "cond_v2":   group_per_step_mae(pred_v2, true_v2, mask_v2, gmask)[0],
        "step4":     group_per_step_mae(pred_s4, true_s4, mask_s4, gmask)[0],
    }

# ── 6. Plot ───────────────────────────────────────────────────────────────────

COLORS = {"baseline": "#888888", "cond_v2": "#2196F3", "step4": "#FF5722"}
LABELS = {"baseline": "Baseline (Task3)", "cond_v2": "Conditioned v2 (Task3)", "step4": "Step4 tail-amp"}

# 6a. Main figure: 2×3 grid (ALL, Q1_fast, non_Q1_fast / single_lobe, reverse_correction, multi_correction)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
fig.suptitle("Per-Timestep Steer MAE: Baseline vs Conditioned-V2 vs Step4", fontsize=13)

plot_order = [
    ("ALL",              axes[0, 0]),
    ("Q1_fast",          axes[0, 1]),
    ("non_Q1_fast",      axes[0, 2]),
    ("single_lobe",      axes[1, 0]),
    ("reverse_correction", axes[1, 1]),
    ("multi_correction", axes[1, 2]),
]

for gname, ax in plot_order:
    n = groups[gname].sum()
    for mname, mae in results[gname].items():
        ax.plot(t_axis, mae, color=COLORS[mname], label=LABELS[mname],
                lw=1.5 if mname != "baseline" else 1.0,
                alpha=0.9)
    ax.axvline(TAIL_START / FPS, color="black", ls="--", lw=1.0, alpha=0.6, label="tail start (1.0 s)")
    ax.set_title(f"{gname}  (n={n})", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MAE (steer_rel)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_png = REPORTS / "per_timestep_mae_comparison_20260408.png"
plt.savefig(out_png, dpi=130)
plt.close()
print(f"\nSaved: {out_png}")

# ── 7. Compute summary numbers for the report ─────────────────────────────────

def segment_mean(mae_arr, start, end):
    return mae_arr[start:end].mean()

summary_rows = []
for gname in ["ALL", "Q1_fast", "non_Q1_fast", "single_lobe", "reverse_correction"]:
    for mname in ["baseline", "cond_v2", "step4"]:
        mae = results[gname][mname]
        summary_rows.append({
            "group": gname,
            "model": mname,
            "pre_tail_mae":  round(segment_mean(mae, 0,          TAIL_START), 5),
            "tail_mae":      round(segment_mean(mae, TAIL_START,  T),         5),
            "overall_mae":   round(segment_mean(mae, 0,          T),          5),
        })

sumdf = pd.DataFrame(summary_rows)

# Wide pivot for readability
pivot = sumdf.pivot_table(index="group", columns="model",
                          values=["pre_tail_mae", "tail_mae", "overall_mae"])
pivot.columns = ["_".join(c) for c in pivot.columns]
pivot = pivot.reset_index()

# Also compute step4 - cond_v2 delta for tail_mae
delta_rows = []
for gname in pivot["group"].unique():
    row = pivot[pivot["group"] == gname].iloc[0]
    delta_rows.append({
        "group": gname,
        "cond_v2_pre_tail_mae":  row["pre_tail_mae_cond_v2"],
        "step4_pre_tail_mae":    row["pre_tail_mae_step4"],
        "delta_pre_tail":        round(row["pre_tail_mae_step4"] - row["pre_tail_mae_cond_v2"], 5),
        "cond_v2_tail_mae":      row["tail_mae_cond_v2"],
        "step4_tail_mae":        row["tail_mae_step4"],
        "delta_tail":            round(row["tail_mae_step4"] - row["tail_mae_cond_v2"], 5),
    })
delta_df = pd.DataFrame(delta_rows)

print("\nStep4 vs Cond_v2 per-segment MAE delta:")
print(delta_df.to_string(index=False))

# ── 8. Write markdown report ──────────────────────────────────────────────────

md_lines = [
    "# Per-Timestep MAE Comparison: Baseline vs Conditioned-V2 vs Step4 (2026-04-08)",
    "",
    "## 设置",
    "- 指标：steer channel (channel 0) 的逐时间步 MAE，有效 mask 加权",
    "- tail 边界：step 200（1.0 s @ 200 Hz）",
    "- 三条曲线：Baseline (Task3) / Conditioned-v2 (Task3) / Step4 tail-amp penalty",
    "",
    "## Step4 vs Conditioned-v2 分段 MAE 差值",
    "",
    delta_df.to_markdown(index=False),
    "",
    "## 解读要点",
]

# Generate automatic observations
for row in delta_rows:
    g = row["group"]
    dp = row["delta_pre_tail"]
    dt = row["delta_tail"]
    direction_pre = "升高" if dp > 0 else "降低"
    direction_tail = "升高" if dt > 0 else "降低"
    md_lines.append(
        f"- **{g}**：Step4 vs v2 前段 MAE {direction_pre} {abs(dp):.5f}，"
        f"tail MAE {direction_tail} {abs(dt):.5f}"
    )

md_lines += [
    "",
    "## 图表",
    f"见 `per_timestep_mae_comparison_20260408.png`",
    "",
    "## 结论",
    "- 若 step4 在 Q1_fast 的 tail MAE 相比 v2 降低，但 pre-tail 升高，",
    "  说明 tail amplitude penalty 在把误差从 tail 段推向 pre-tail 段，",
    "  而不是真正减少了 shape/amplitude 失配。",
    "- 若 single_lobe 的 tail MAE 同时升高，说明 boundary hedging 被加剧。",
]

out_md = REPORTS / "per_timestep_mae_comparison_20260408.md"
out_md.write_text("\n".join(md_lines), encoding="utf-8")
print(f"Saved: {out_md}")
