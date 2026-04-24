
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1) 路径配置：按需修改
# =========================
CENTERLINE_CSV = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\道路信息\full_centerline_layout.csv"

# 这里改成你的实际车辆轨迹 CSV
ACTUAL_TRAJ_CSV = r"F:\数据集处理\data_process\road_centerline_generated.csv"

# 输出图
OUT_FIG = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\道路信息"

# =========================
# 2) 可能的轨迹列名候选
#    你如果知道自己 CSV 的精确列名，直接改成唯一列名更稳
# =========================
X_CANDIDATES = [
    "x", "X", "pos_x", "global_x", "world_x",
    "zx1|INS_Position_X", "zx|INS_Position_X",
    "zx1|X", "zx|X"
]
Y_CANDIDATES = [
    "y", "Y", "pos_y", "global_y", "world_y",
    "zx1|INS_Position_Y", "zx|INS_Position_Y",
    "zx1|Y", "zx|Y"
]


def find_first_existing(cols, candidates):
    colset = set(cols)
    for c in candidates:
        if c in colset:
            return c
    return None


def infer_xy_columns(df):
    x_col = find_first_existing(df.columns, X_CANDIDATES)
    y_col = find_first_existing(df.columns, Y_CANDIDATES)

    if x_col is not None and y_col is not None:
        return x_col, y_col

    # 兜底：找看起来像 x/y 的列
    lower_map = {c.lower(): c for c in df.columns}
    possible_x = [c for c in df.columns if c.lower().endswith("x") or "_x" in c.lower()]
    possible_y = [c for c in df.columns if c.lower().endswith("y") or "_y" in c.lower()]

    if possible_x and possible_y:
        return possible_x[0], possible_y[0]

    raise ValueError(
        "未能自动识别实际轨迹的 x/y 列名。\n"
        f"当前列名为：\n{list(df.columns)}\n"
        "请把脚本里的 X_CANDIDATES / Y_CANDIDATES 改成你的真实列名。"
    )


def rigid_align_2d(source_xy, target_xy):
    """
    用 2D 刚体配准（旋转+平移，不缩放）
    把 source 对齐到 target
    """
    src = np.asarray(source_xy, dtype=float)
    tgt = np.asarray(target_xy, dtype=float)

    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)

    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 防止反射
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_mean - (R @ src_mean)
    src_aligned = (R @ src.T).T + t

    return src_aligned, R, t


def resample_polyline(xy, n_samples=3000):
    """
    按弧长重采样，避免两条曲线点数差太大
    """
    xy = np.asarray(xy, dtype=float)
    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total <= 1e-9:
        return xy.copy()

    s_new = np.linspace(0.0, total, n_samples)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])
    return np.column_stack([x_new, y_new])


def rmse(a, b):
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))


def main():
    center_df = pd.read_csv(CENTERLINE_CSV)
    if not os.path.exists(ACTUAL_TRAJ_CSV):
        raise FileNotFoundError(
            f"找不到实际轨迹文件：{ACTUAL_TRAJ_CSV}\n"
            "请把 ACTUAL_TRAJ_CSV 改成你本机真实 CSV 路径后再运行。"
        )

    actual_df = pd.read_csv(ACTUAL_TRAJ_CSV)

    if "x" not in center_df.columns or "y" not in center_df.columns:
        raise ValueError(f"中心线文件缺少 x/y 列，当前列名：{list(center_df.columns)}")

    x_col, y_col = infer_xy_columns(actual_df)

    center_xy = center_df[["x", "y"]].dropna().to_numpy(dtype=float)
    actual_xy = actual_df[[x_col, y_col]].dropna().to_numpy(dtype=float)

    # 为了更稳，先重采样到相近点数
    n = min(3000, len(center_xy), len(actual_xy))
    center_rs = resample_polyline(center_xy, n_samples=n)
    actual_rs = resample_polyline(actual_xy, n_samples=n)

    # 把中心线刚体对齐到实际轨迹
    center_aligned, R, t = rigid_align_2d(center_rs, actual_rs)

    # 评价误差
    err_rmse = rmse(center_aligned, actual_rs)

    # 画图
    plt.figure(figsize=(10, 8))
    plt.plot(actual_rs[:, 0], actual_rs[:, 1], linewidth=1.5, label=f"Actual trajectory ({x_col}, {y_col})")
    plt.plot(center_aligned[:, 0], center_aligned[:, 1], linewidth=1.5, label="Centerline (rigid aligned)")
    plt.axis("equal")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title(f"Centerline vs Actual Trajectory Overlay\nRMSE = {err_rmse:.3f} m")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.show()

    print("=" * 60)
    print(f"识别到实际轨迹列名: x = {x_col}, y = {y_col}")
    print(f"RMSE = {err_rmse:.3f} m")
    print(f"图像已保存到: {OUT_FIG}")
    print("旋转矩阵 R =")
    print(R)
    print("平移向量 t =")
    print(t)
    print("=" * 60)


if __name__ == "__main__":
    main()
