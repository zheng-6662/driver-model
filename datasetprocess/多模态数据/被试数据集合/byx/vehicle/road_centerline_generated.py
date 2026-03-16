# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# def reconstruct_road_from_kappa(
#     csv_file,
#     output_dir,
#     s_col="road_s_m",
#     kappa_col="kappa_smooth",
#     fallback_kappa_col="kappa_used",
# ):
#     """
#     从 road_s_m + kappa_smooth 重建道路中心线坐标

#     输入:
#         csv_file: 原始 roadtype_labeled.csv 文件
#         output_dir: 输出文件夹
#         s_col: 弧长列名
#         kappa_col: 优先使用的曲率列
#         fallback_kappa_col: 备用曲率列

#     输出:
#         road_centerline_generated.csv
#         road_centerline_plot.png
#         road_kappa_plot.png
#     """

#     os.makedirs(output_dir, exist_ok=True)

#     print("=" * 70)
#     print("开始重建道路中心线")
#     print("输入文件:", csv_file)
#     print("输出目录:", output_dir)
#     print("=" * 70)

#     # -----------------------------
#     # 1. 读取数据
#     # -----------------------------
#     df = pd.read_csv(csv_file)
#     print(f"[INFO] 原始数据行数: {len(df)}")

#     if s_col not in df.columns:
#         raise ValueError(f"找不到列: {s_col}")

#     if kappa_col not in df.columns:
#         print(f"[WARN] 找不到 {kappa_col}，尝试使用 {fallback_kappa_col}")
#         if fallback_kappa_col not in df.columns:
#             raise ValueError(f"找不到曲率列: {kappa_col} / {fallback_kappa_col}")
#         kappa_col = fallback_kappa_col

#     # -----------------------------
#     # 2. 提取并清洗
#     # -----------------------------
#     use_df = df[[s_col, kappa_col]].copy()
#     use_df.columns = ["s", "kappa"]

#     # 去掉 NaN / inf
#     use_df = use_df.replace([np.inf, -np.inf], np.nan).dropna()
#     print(f"[INFO] 去除 NaN/inf 后行数: {len(use_df)}")

#     if len(use_df) < 2:
#         raise ValueError("有效数据太少，无法重建道路")

#     # 按 s 排序
#     use_df = use_df.sort_values("s").reset_index(drop=True)

#     # 去重（同一个 s 可能有重复）
#     use_df = use_df.groupby("s", as_index=False)["kappa"].mean()
#     print(f"[INFO] s 去重后行数: {len(use_df)}")

#     s = use_df["s"].to_numpy(dtype=float)
#     kappa = use_df["kappa"].to_numpy(dtype=float)

#     # -----------------------------
#     # 3. 处理异常 ds
#     # -----------------------------
#     ds = np.diff(s)

#     # 只保留严格递增的 s
#     valid = np.ones_like(s, dtype=bool)
#     valid[1:] = ds > 1e-9

#     s = s[valid]
#     kappa = kappa[valid]

#     if len(s) < 2:
#         raise ValueError("去除非递增 s 后有效数据不足")

#     ds = np.diff(s)

#     print(f"[INFO] s 范围: {s[0]:.3f} ~ {s[-1]:.3f} m")
#     print(f"[INFO] 道路长度: {s[-1] - s[0]:.3f} m")
#     print(f"[INFO] ds: min={ds.min():.6f}, max={ds.max():.6f}, mean={ds.mean():.6f}")

#     # -----------------------------
#     # 4. 积分重建道路
#     #    theta(s)=∫kappa ds
#     #    x(s)=∫cos(theta) ds
#     #    y(s)=∫sin(theta) ds
#     # -----------------------------
#     n = len(s)
#     theta = np.zeros(n, dtype=float)
#     x = np.zeros(n, dtype=float)
#     y = np.zeros(n, dtype=float)

#     # 初始条件
#     theta[0] = 0.0
#     x[0] = 0.0
#     y[0] = 0.0

#     for i in range(1, n):
#         dsi = s[i] - s[i - 1]

#         # 用梯形法积分 theta
#         theta[i] = theta[i - 1] + 0.5 * (kappa[i - 1] + kappa[i]) * dsi

#         # 用上一时刻和当前时刻航向的平均值积分位置
#         theta_mid = 0.5 * (theta[i - 1] + theta[i])
#         x[i] = x[i - 1] + np.cos(theta_mid) * dsi
#         y[i] = y[i - 1] + np.sin(theta_mid) * dsi

#     # -----------------------------
#     # 5. 输出 csv
#     # -----------------------------
#     out_df = pd.DataFrame({
#         "s_m": s,
#         "kappa_1pm": kappa,
#         "theta_rad": theta,
#         "x_m": x,
#         "y_m": y,
#     })

#     csv_out = os.path.join(output_dir, "road_centerline_generated.csv")
#     out_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
#     print(f"[OK] 已保存道路坐标: {csv_out}")

#     # -----------------------------
#     # 6. 画道路图
#     # -----------------------------
#     plt.figure(figsize=(10, 6))
#     plt.plot(x, y, linewidth=1.5)
#     plt.xlabel("x (m)")
#     plt.ylabel("y (m)")
#     plt.title("Reconstructed Road Centerline")
#     plt.axis("equal")
#     plt.grid(True)

#     road_fig = os.path.join(output_dir, "road_centerline_plot.png")
#     plt.tight_layout()
#     plt.savefig(road_fig, dpi=200)
#     plt.close()
#     print(f"[OK] 已保存道路图: {road_fig}")

#     # -----------------------------
#     # 7. 画曲率图
#     # -----------------------------
#     plt.figure(figsize=(10, 5))
#     plt.plot(s, kappa, linewidth=1.2)
#     plt.xlabel("s (m)")
#     plt.ylabel("kappa (1/m)")
#     plt.title("Road Curvature")
#     plt.grid(True)

#     kappa_fig = os.path.join(output_dir, "road_kappa_plot.png")
#     plt.tight_layout()
#     plt.savefig(kappa_fig, dpi=200)
#     plt.close()
#     print(f"[OK] 已保存曲率图: {kappa_fig}")

#     # -----------------------------
#     # 8. 打印摘要
#     # -----------------------------
#     print("=" * 70)
#     print("重建完成")
#     print(f"输出 CSV : {csv_out}")
#     print(f"输出道路图: {road_fig}")
#     print(f"输出曲率图: {kappa_fig}")
#     print("=" * 70)

#     return out_df


# if __name__ == "__main__":
#     csv_file = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\byx\vehicle\Entity_Recording_2025_09_28_17_05_51_vehicle_aligned_cleaned_roadtype_labeled.csv"

#     output_dir = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\byx\vehicle\road_reconstruction_output"

#     reconstruct_road_from_kappa(
#         csv_file=csv_file,
#         output_dir=output_dir,
#         s_col="road_s_m",
#         kappa_col="kappa_smooth",
#         fallback_kappa_col="kappa_used",
#     )



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_road_trajectory(csv_file, output_dir=None):
    """
    读取包含 s, kappa, x, y 的道路/轨迹 csv，
    绘制：
    1) x-y 轨迹图
    2) kappa-s 曲率图
    """

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_file), "trajectory_plot_output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("输入文件:", csv_file)
    print("输出目录:", output_dir)
    print("=" * 70)

    # 读取数据
    df = pd.read_csv(csv_file)
    print("[INFO] 原始列名:", list(df.columns))

    # 自动兼容列名
    col_map = {}

    for c in df.columns:
        cl = c.strip().lower()
        if cl == "s" or cl == "s_m":
            col_map["s"] = c
        elif cl == "kappa" or cl == "kappa_1pm":
            col_map["kappa"] = c
        elif cl == "x" or cl == "x_m":
            col_map["x"] = c
        elif cl == "y" or cl == "y_m":
            col_map["y"] = c

    required = ["s", "kappa", "x", "y"]
    missing = [k for k in required if k not in col_map]
    if missing:
        raise ValueError(f"缺少必要列: {missing}，当前识别到: {col_map}")

    use_df = df[[col_map["s"], col_map["kappa"], col_map["x"], col_map["y"]]].copy()
    use_df.columns = ["s", "kappa", "x", "y"]

    # 清洗
    use_df = use_df.replace([np.inf, -np.inf], np.nan).dropna()
    use_df = use_df.sort_values("s").reset_index(drop=True)

    print(f"[INFO] 有效数据行数: {len(use_df)}")
    print(f"[INFO] s范围: {use_df['s'].min():.6f} ~ {use_df['s'].max():.6f}")
    print(f"[INFO] x范围: {use_df['x'].min():.6f} ~ {use_df['x'].max():.6f}")
    print(f"[INFO] y范围: {use_df['y'].min():.6f} ~ {use_df['y'].max():.6f}")
    print(f"[INFO] kappa范围: {use_df['kappa'].min():.6f} ~ {use_df['kappa'].max():.6f}")

    s = use_df["s"].to_numpy()
    kappa = use_df["kappa"].to_numpy()
    x = use_df["x"].to_numpy()
    y = use_df["y"].to_numpy()

    # -------------------------
    # 1) 轨迹图 x-y
    # -------------------------
    plt.figure(figsize=(9, 6))
    plt.plot(x, y, linewidth=1.5)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajectory / Road Shape (x-y)")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    xy_fig = os.path.join(output_dir, "trajectory_xy.png")
    plt.savefig(xy_fig, dpi=200)
    plt.show()
    plt.close()

    # -------------------------
    # 2) 曲率图 kappa-s
    # -------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(s, kappa, linewidth=1.2)
    plt.xlabel("s (m)")
    plt.ylabel("kappa (1/m)")
    plt.title("Curvature vs Arc Length")
    plt.grid(True)
    plt.tight_layout()

    kappa_fig = os.path.join(output_dir, "kappa_vs_s.png")
    plt.savefig(kappa_fig, dpi=200)
    plt.show()
    plt.close()

    # -------------------------
    # 3) x / y 随 s 变化
    # -------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(s, x, label="x")
    plt.plot(s, y, label="y")
    plt.xlabel("s (m)")
    plt.ylabel("value (m)")
    plt.title("x(s) and y(s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    xsys_fig = os.path.join(output_dir, "x_y_vs_s.png")
    plt.savefig(xsys_fig, dpi=200)
    plt.show()
    plt.close()

    print("=" * 70)
    print("[OK] 已保存:")
    print("轨迹图:", xy_fig)
    print("曲率图:", kappa_fig)
    print("x/y图 :", xsys_fig)
    print("=" * 70)

    # 给一个简单判断
    if np.nanmax(np.abs(kappa)) < 1e-8 and np.nanmax(np.abs(y)) < 1e-8:
        print("[判断] 当前这份数据基本是一段直线。")
    else:
        print("[判断] 当前这份数据包含曲率变化，可能存在弯道。")


if __name__ == "__main__":
    csv_file = r"F:\数据集处理\data_process\road_centerline_generated.csv"
    plot_road_trajectory(csv_file)