# -*- coding: utf-8 -*-
"""
基于当前主路线中心线，生成多个横向偏移候选道路中心线，
并与所有实际轨迹叠加对比。

用途：
1) 读取你已经生成好的 road_centerline_main_route.csv
2) 读取所有被试 vehicle/*.csv 里的实际轨迹
3) 自动筛选较完整的大文件轨迹
4) 对齐到当前中心线坐标系
5) 生成多条候选偏移线：
      当前线
      左偏 2.5 m
      左偏 4.25 m
      左偏 6.0 m
      右偏 2.5 m
      右偏 4.25 m
      右偏 6.0 m
6) 画图用于人工判断哪条最像“整条道路几何中心”

依赖：
pip install numpy pandas matplotlib
"""

from pathlib import Path
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1) 配置区
# =========================================================
ROOT_DIR = Path(r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合")

# 你上一版聚类得到的主路线中心线
CENTERLINE_CSV = ROOT_DIR / "道路中心线融合结果_聚类版" / "road_centerline_main_route.csv"

OUT_DIR = ROOT_DIR / "道路中心线偏移验证结果"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FIG = OUT_DIR / "offset_candidates_vs_trajectories.png"
OUT_CENTERLINES_CSV = OUT_DIR / "offset_centerline_candidates.csv"

# 轨迹列名
X_COL = "zx|x"
Y_COL = "zx|y"

# 文件筛选
MIN_FILE_SIZE_MB = 20
MIN_POINTS = 5000
MIN_TRAJ_LENGTH_M = 8000

# 重采样点数（和中心线统一）
N_RESAMPLE = 3000

# 你当前怀疑的几个偏移量（单位 m）
OFFSETS = [0.0, 2.5, 4.25, 6.0, -2.5, -4.25, -6.0]


# =========================================================
# 2) 基础函数
# =========================================================
def compute_arclength(xy: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(d)])


def traj_length(xy: np.ndarray) -> float:
    return float(compute_arclength(xy)[-1])


def clean_xy(xy: np.ndarray) -> np.ndarray:
    xy = xy[~np.isnan(xy).any(axis=1)]
    if len(xy) < 2:
        return xy

    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = d > 1e-6
    xy = xy[keep]

    if len(xy) < 2:
        return xy

    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    med = np.median(d) if len(d) > 0 else 0.0
    if med > 0:
        keep = np.ones(len(xy), dtype=bool)
        keep[1:] = d < max(20.0, 20 * med)
        xy = xy[keep]

    return xy


def resample_by_arclength(xy: np.ndarray, n_samples: int) -> np.ndarray:
    s = compute_arclength(xy)
    if len(s) < 2 or s[-1] < 1e-9:
        return xy.copy()
    s_new = np.linspace(0.0, s[-1], n_samples)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    return np.column_stack([x_new, y_new])


def rotate_points(xy: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return xy @ R.T


def get_main_direction_angle(xy: np.ndarray) -> float:
    v = xy[-1] - xy[0]
    return float(np.arctan2(v[1], v[0]))


def align_trajectory(xy: np.ndarray, ref_angle: float = 0.0) -> np.ndarray:
    xy2 = xy - xy[0]
    ang = get_main_direction_angle(xy2)
    return rotate_points(xy2, ref_angle - ang)


def maybe_flip_trajectory(xy: np.ndarray, ref_xy: np.ndarray) -> np.ndarray:
    a = np.mean(np.linalg.norm(xy - ref_xy, axis=1))
    b = np.mean(np.linalg.norm(xy[::-1] - ref_xy, axis=1))
    return xy if a <= b else xy[::-1]


def compute_tangent_and_normal(xy: np.ndarray):
    """
    对每个点计算单位切向量、单位法向量
    法向定义为左法向：( -ty, tx )
    """
    s = compute_arclength(xy)
    dx = np.gradient(xy[:, 0], s, edge_order=1)
    dy = np.gradient(xy[:, 1], s, edge_order=1)

    t = np.column_stack([dx, dy])
    norm_t = np.linalg.norm(t, axis=1, keepdims=True)
    norm_t[norm_t < 1e-9] = 1.0
    t = t / norm_t

    n = np.column_stack([-t[:, 1], t[:, 0]])
    return t, n


def offset_centerline(xy: np.ndarray, offset: float) -> np.ndarray:
    _, n = compute_tangent_and_normal(xy)
    return xy + offset * n


# =========================================================
# 3) 扫描轨迹
# =========================================================
def scan_vehicle_csvs(root_dir: Path):
    files = []
    for subject_dir in root_dir.iterdir():
        if not subject_dir.is_dir():
            continue
        vehicle_dir = subject_dir / "vehicle"
        if not vehicle_dir.exists():
            continue
        for f in vehicle_dir.glob("*.csv"):
            files.append((subject_dir.name, f))
    return files


def read_xy_from_csv(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if X_COL not in df.columns or Y_COL not in df.columns:
        raise ValueError(f"缺少列 {X_COL}/{Y_COL}")
    xy = df[[X_COL, Y_COL]].to_numpy(dtype=float)
    return clean_xy(xy)


# =========================================================
# 4) 主流程
# =========================================================
def main():
    # ---------- 读取中心线 ----------
    center_df = pd.read_csv(CENTERLINE_CSV)
    center_xy = center_df[["x", "y"]].to_numpy(dtype=float)
    center_xy = resample_by_arclength(center_xy, N_RESAMPLE)

    # ---------- 扫描实际轨迹 ----------
    all_files = scan_vehicle_csvs(ROOT_DIR)
    print(f"扫描到 csv 数量: {len(all_files)}")

    candidates = []
    for subject, f in all_files:
        try:
            size_mb = f.stat().st_size / (1024 * 1024)
            if size_mb < MIN_FILE_SIZE_MB:
                continue

            xy = read_xy_from_csv(f)
            if len(xy) < MIN_POINTS:
                continue

            length_m = traj_length(xy)
            if length_m < MIN_TRAJ_LENGTH_M:
                continue

            # 对齐到与中心线相同坐标系
            xy_aligned = align_trajectory(xy)
            xy_rs = resample_by_arclength(xy_aligned, N_RESAMPLE)
            xy_rs = maybe_flip_trajectory(xy_rs, center_xy)

            candidates.append({
                "subject": subject,
                "file": str(f),
                "size_mb": size_mb,
                "traj_length_m": length_m,
                "xy_aligned": xy_rs,
            })

        except Exception:
            continue

    print(f"用于叠加验证的轨迹数: {len(candidates)}")
    if len(candidates) == 0:
        print("没有符合条件的轨迹。")
        return

    # ---------- 生成偏移候选 ----------
    candidate_lines = {}
    for off in OFFSETS:
        candidate_lines[off] = offset_centerline(center_xy, off)

    # ---------- 导出所有候选线到一个 CSV ----------
    rows = []
    for off, xy in candidate_lines.items():
        s = compute_arclength(xy)
        for i in range(len(xy)):
            rows.append([off, s[i], xy[i, 0], xy[i, 1]])
    out_df = pd.DataFrame(rows, columns=["offset_m", "s", "x", "y"])
    out_df.to_csv(OUT_CENTERLINES_CSV, index=False, encoding="utf-8-sig")

    # ---------- 绘图 ----------
    plt.figure(figsize=(11, 9))

    # 实际轨迹
    for item in candidates:
        xy = item["xy_aligned"]
        plt.plot(xy[:, 0], xy[:, 1], linewidth=0.7, alpha=0.35)

    # 候选线
    for off in OFFSETS:
        xy = candidate_lines[off]
        if abs(off) < 1e-9:
            plt.plot(xy[:, 0], xy[:, 1], linewidth=2.6, label=f"offset {off:.2f} m")
        else:
            plt.plot(xy[:, 0], xy[:, 1], linewidth=1.8, label=f"offset {off:.2f} m")

    plt.axis("equal")
    plt.xlabel("Aligned X [m]")
    plt.ylabel("Aligned Y [m]")
    plt.title("Offset Centerline Candidates vs Actual Trajectories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=220)
    plt.close()

    print("=" * 80)
    print("完成")
    print(f"候选线输出: {OUT_CENTERLINES_CSV}")
    print(f"叠加图输出: {OUT_FIG}")
    print("偏移量列表:", OFFSETS)
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("程序运行失败：")
        print(e)
        traceback.print_exc()