import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ========= 1. 读取数据 =========
csv_path = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\道路信息\proving_ground_full_path_v3_tangent.csv"  # 改成你的实际路径
df = pd.read_csv(csv_path)

print("数据列名：")
print(df.columns.tolist())
print("\n前5行：")
print(df.head())
print("\n数据规模：", df.shape)
print("\n模块名：")
if "module_name" in df.columns:
    print(df["module_name"].unique())


# ========= 2. 基本检查 =========
required_cols = ["s", "x", "y"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"缺少必要列: {col}")

if "z" not in df.columns:
    df["z"] = 0.0
if "yaw" not in df.columns:
    df["yaw"] = np.nan
if "curvature" not in df.columns:
    df["curvature"] = np.nan


# ========= 3. 整体平面轨迹 =========
plt.figure(figsize=(10, 8))
plt.plot(df["x"], df["y"], linewidth=1.0)
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Proving Ground Full Path (XY)")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()


# ========= 4. 按模块分段显示 =========
if "module_name" in df.columns:
    plt.figure(figsize=(12, 9))
    for mod_name, g in df.groupby("module_name", sort=False):
        plt.plot(g["x"], g["y"], linewidth=1.2, label=str(mod_name))
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Full Path by Module")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ========= 5. 航向角 yaw 随 s 变化 =========
if df["yaw"].notna().any():
    plt.figure(figsize=(12, 4))
    plt.plot(df["s"], df["yaw"], linewidth=1.0)
    plt.xlabel("s (m)")
    plt.ylabel("yaw (rad)")
    plt.title("Yaw vs s")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ========= 6. 曲率 curvature 随 s 变化 =========
if df["curvature"].notna().any():
    plt.figure(figsize=(12, 4))
    plt.plot(df["s"], df["curvature"], linewidth=1.0)
    plt.xlabel("s (m)")
    plt.ylabel("curvature (1/m)")
    plt.title("Curvature vs s")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ========= 7. 查看每个模块长度 =========
if "module_name" in df.columns:
    print("\n各模块长度统计：")
    module_stats = []
    for mod_name, g in df.groupby("module_name", sort=False):
        s_len = g["s"].iloc[-1] - g["s"].iloc[0]
        module_stats.append([mod_name, len(g), s_len])
    stats_df = pd.DataFrame(module_stats, columns=["module_name", "num_points", "length_m"])
    print(stats_df)


# ========= 8. 3D轨迹查看（如果你想看 z） =========
if "z" in df.columns:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(df["x"], df["y"], df["z"], linewidth=0.8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Path")
    plt.tight_layout()
    plt.show()