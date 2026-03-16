# # -*- coding: utf-8 -*-
# """
# 基于 main route 参考线恢复道路几何（固定路径版）

# 功能：
# 1. 直接读取顶部指定的 road_centerline_main_route.csv
# 2. 读取 s/x/y（至少需要 x,y）
# 3. 计算切向、法向
# 4. 按 offset 偏移得到道路几何中心线
# 5. 从道路中心线扩展左右边界
# 6. 绘图检查
# 7. 导出 CSV

# 使用方式：
# 1. 修改 INPUT_FILE 为你的实际文件路径
# 2. 运行脚本
# 3. 查看输出目录中的 csv 和 png

# 作者：ChatGPT
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# # =========================================================
# # 1. 输入文件路径（你主要改这里）
# # =========================================================
# INPUT_FILE = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\道路中心线融合结果_聚类版\road_centerline_main_route.csv"


# # =========================================================
# # 2. 参数区（你主要改这里）
# # =========================================================

# # 从 main route 到道路几何中心线的偏移（单位：m）
# # 先用 -4.5，若图上方向反了，改成 +4.5
# OFFSET_TO_ROAD_CENTER = -4.5

# # 道路结构（单位：m）
# LANE_WIDTH = 3.5
# MEDIAN_WIDTH = 1.5
# EDGE_WIDTH = 1.5

# # 车道数：双向四车道 -> 每侧2车道
# LANES_PER_SIDE = 2

# # 是否绘图
# SHOW_PLOT = True

# # 是否保存预览图
# SAVE_FIG = True

# # 图像 DPI
# FIG_DPI = 200


# # =========================================================
# # 3. 工具函数
# # =========================================================
# def safe_read_csv(file_path):
#     """尽量稳健地读取 CSV"""
#     encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
#     last_err = None
#     for enc in encodings:
#         try:
#             return pd.read_csv(file_path, encoding=enc)
#         except Exception as e:
#             last_err = e
#     raise last_err


# def normalize_vectors(vx, vy, eps=1e-12):
#     """向量归一化"""
#     norm = np.sqrt(vx**2 + vy**2)
#     norm = np.where(norm < eps, 1.0, norm)
#     return vx / norm, vy / norm


# def compute_tangent_normal(x, y):
#     """
#     根据轨迹点计算单位切向和单位法向

#     切向:
#         t = [dx, dy] 归一化

#     左法向:
#         n = [-ty, tx]
#     """
#     dx = np.gradient(x)
#     dy = np.gradient(y)

#     tx, ty = normalize_vectors(dx, dy)

#     # 左法向
#     nx = -ty
#     ny = tx

#     nx, ny = normalize_vectors(nx, ny)
#     return tx, ty, nx, ny


# def offset_polyline(x, y, nx, ny, offset):
#     """沿法向偏移折线"""
#     xo = x + offset * nx
#     yo = y + offset * ny
#     return xo, yo


# def compute_arclength(x, y):
#     """根据 x/y 重新计算弧长 s"""
#     ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
#     s = np.zeros(len(x))
#     s[1:] = np.cumsum(ds)
#     return s


# def compute_heading(x, y):
#     """计算航向角 yaw（弧度）"""
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     yaw = np.arctan2(dy, dx)
#     return yaw


# def compute_curvature(x, y):
#     """
#     根据参数形式计算曲率
#     kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
#     """
#     dx = np.gradient(x)
#     dy = np.gradient(y)
#     ddx = np.gradient(dx)
#     ddy = np.gradient(dy)

#     denom = (dx**2 + dy**2)**1.5
#     denom = np.where(denom < 1e-12, 1e-12, denom)

#     kappa = (dx * ddy - dy * ddx) / denom
#     return kappa


# def build_road_boundaries(center_x, center_y, nx, ny):
#     """
#     基于道路几何中心线扩展边界

#     约定：
#     沿 +n 方向为左侧
#     沿 -n 方向为右侧
#     """
#     half_median = MEDIAN_WIDTH / 2.0
#     side_lane_total = LANES_PER_SIDE * LANE_WIDTH
#     outer_total = half_median + side_lane_total + EDGE_WIDTH

#     offsets = {
#         "centerline": 0.0,

#         # 中央带边界
#         "median_left_edge": +half_median,
#         "median_right_edge": -half_median,

#         # 左侧两车道外边界
#         "left_lane1_outer": +(half_median + LANE_WIDTH),
#         "left_lane2_outer": +(half_median + 2.0 * LANE_WIDTH),

#         # 右侧两车道外边界
#         "right_lane1_outer": -(half_median + LANE_WIDTH),
#         "right_lane2_outer": -(half_median + 2.0 * LANE_WIDTH),

#         # 最外侧道路边缘
#         "left_road_edge": +outer_total,
#         "right_road_edge": -outer_total,
#     }

#     lines = {}
#     for name, off in offsets.items():
#         xo, yo = offset_polyline(center_x, center_y, nx, ny, off)
#         lines[name] = {
#             "x": xo,
#             "y": yo,
#             "offset": off,
#         }

#     return lines


# def export_line_csv(out_dir, file_stem, x, y):
#     """导出单条线"""
#     s = compute_arclength(x, y)
#     yaw = compute_heading(x, y)
#     kappa = compute_curvature(x, y)

#     df = pd.DataFrame({
#         "s": s,
#         "x": x,
#         "y": y,
#         "yaw_rad": yaw,
#         "curvature": kappa
#     })

#     out_path = os.path.join(out_dir, f"{file_stem}.csv")
#     df.to_csv(out_path, index=False, encoding="utf-8-sig")
#     return out_path


# def validate_input_path(file_path):
#     """检查输入路径"""
#     if not file_path:
#         raise ValueError("INPUT_FILE 为空，请在代码顶部填写 CSV 文件路径。")

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"输入文件不存在：{file_path}")

#     if not os.path.isfile(file_path):
#         raise FileNotFoundError(f"输入路径不是文件：{file_path}")


# # =========================================================
# # 4. 主流程
# # =========================================================
# def main():
#     file_path = INPUT_FILE
#     validate_input_path(file_path)

#     print("=" * 80)
#     print("输入文件：")
#     print(file_path)
#     print("=" * 80)

#     # 读取 CSV
#     df = safe_read_csv(file_path)

#     print("读取成功，列名如下：")
#     print(list(df.columns))

#     # 至少需要 x, y
#     required_cols = ["x", "y"]
#     for c in required_cols:
#         if c not in df.columns:
#             raise ValueError(
#                 f"缺少必要列：{c}\n"
#                 f"当前列名为：{list(df.columns)}"
#             )

#     x = pd.to_numeric(df["x"], errors="coerce").to_numpy()
#     y = pd.to_numeric(df["y"], errors="coerce").to_numpy()

#     valid = np.isfinite(x) & np.isfinite(y)
#     x = x[valid]
#     y = y[valid]

#     if len(x) < 10:
#         raise ValueError("有效点数过少，无法建路。")

#     print(f"有效轨迹点数：{len(x)}")

#     # 输入线的切向/法向
#     tx, ty, nx, ny = compute_tangent_normal(x, y)

#     # 按偏移得到道路几何中心线
#     road_cx, road_cy = offset_polyline(x, y, nx, ny, OFFSET_TO_ROAD_CENTER)

#     # 对偏移后的中心线重新计算切向/法向，更稳
#     tx2, ty2, nx2, ny2 = compute_tangent_normal(road_cx, road_cy)

#     # 基于道路几何中心线扩展道路边界
#     road_lines = build_road_boundaries(road_cx, road_cy, nx2, ny2)

#     # 输出目录
#     in_dir = os.path.dirname(file_path)
#     in_name = os.path.splitext(os.path.basename(file_path))[0]
#     out_dir = os.path.join(in_dir, f"{in_name}_road_rebuild")
#     os.makedirs(out_dir, exist_ok=True)

#     # 导出输入主路线与道路几何中心线
#     export_line_csv(out_dir, "00_main_route_input", x, y)
#     export_line_csv(out_dir, "01_road_geometric_centerline", road_cx, road_cy)

#     # 导出各边界线
#     for name, data in road_lines.items():
#         export_line_csv(out_dir, f"road_{name}", data["x"], data["y"])

#     # 导出汇总表
#     summary_df = pd.DataFrame({
#         "name": ["main_route_input", "road_geometric_centerline"] + list(road_lines.keys()),
#         "offset_m": [0.0, OFFSET_TO_ROAD_CENTER] + [road_lines[k]["offset"] for k in road_lines.keys()]
#     })
#     summary_path = os.path.join(out_dir, "road_lines_summary.csv")
#     summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

#     # 画图
#     if SHOW_PLOT:
#         plt.figure(figsize=(10, 8))

#         # 输入主路线
#         plt.plot(x, y, linewidth=1.2, label="Input Main Route")

#         # 道路几何中心线
#         plt.plot(road_cx, road_cy, linewidth=1.8, label="Road Geometric Centerline")

#         # 中央带边界
#         plt.plot(
#             road_lines["median_left_edge"]["x"],
#             road_lines["median_left_edge"]["y"],
#             "--", linewidth=1.0, label="Median Left Edge"
#         )
#         plt.plot(
#             road_lines["median_right_edge"]["x"],
#             road_lines["median_right_edge"]["y"],
#             "--", linewidth=1.0, label="Median Right Edge"
#         )

#         # 左右车道外边界
#         plt.plot(
#             road_lines["left_lane1_outer"]["x"],
#             road_lines["left_lane1_outer"]["y"],
#             ":", linewidth=1.0, label="Left Lane 1 Outer"
#         )
#         plt.plot(
#             road_lines["left_lane2_outer"]["x"],
#             road_lines["left_lane2_outer"]["y"],
#             ":", linewidth=1.0, label="Left Lane 2 Outer"
#         )
#         plt.plot(
#             road_lines["right_lane1_outer"]["x"],
#             road_lines["right_lane1_outer"]["y"],
#             ":", linewidth=1.0, label="Right Lane 1 Outer"
#         )
#         plt.plot(
#             road_lines["right_lane2_outer"]["x"],
#             road_lines["right_lane2_outer"]["y"],
#             ":", linewidth=1.0, label="Right Lane 2 Outer"
#         )

#         # 最外侧边缘
#         plt.plot(
#             road_lines["left_road_edge"]["x"],
#             road_lines["left_road_edge"]["y"],
#             linewidth=1.3, label="Left Road Edge"
#         )
#         plt.plot(
#             road_lines["right_road_edge"]["x"],
#             road_lines["right_road_edge"]["y"],
#             linewidth=1.3, label="Right Road Edge"
#         )

#         # 起点
#         plt.scatter(x[0], y[0], s=40, label="Input Start")
#         plt.scatter(road_cx[0], road_cy[0], s=40, label="Road Center Start")

#         plt.axis("equal")
#         plt.grid(True, alpha=0.3)
#         plt.xlabel("X (m)")
#         plt.ylabel("Y (m)")
#         plt.title("Road Rebuild from Main Route")
#         plt.legend(fontsize=8)
#         plt.tight_layout()

#         if SAVE_FIG:
#             fig_path = os.path.join(out_dir, "road_rebuild_preview.png")
#             plt.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
#             print(f"预览图已保存：{fig_path}")

#         plt.show()

#     print("=" * 80)
#     print("道路恢复完成。")
#     print(f"输出目录：{out_dir}")
#     print(f"汇总文件：{summary_path}")
#     print("=" * 80)


# if __name__ == "__main__":
#     main()
# -*- coding: utf-8 -*-
"""
基于 main route 生成最终四车道完整道路（1m重采样版）

功能：
1. 读取固定路径的 main route csv
2. 按法向偏移得到道路几何中心线
3. 对道路中心线做 1m 等弧长重采样
4. 按四车道结构扩展出完整道路
5. 导出所有关键线
6. 绘制最终道路预览图

输入文件至少需要列：
x, y
如果有 s 也可以，但程序会重新计算

作者：ChatGPT
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 输入文件路径（只改这里）
# =========================================================
INPUT_FILE = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\道路中心线融合结果_聚类版\road_centerline_main_route.csv"


# =========================================================
# 2. 参数区
# =========================================================

# main route 到道路几何中心线的偏移
OFFSET_TO_ROAD_CENTER = -4.5

# 重采样间距
RESAMPLE_DS = 1.0   # 1 m

# 道路结构
LANE_WIDTH = 3.5
MEDIAN_WIDTH = 1.5
EDGE_WIDTH = 1.5
LANES_PER_SIDE = 2  # 双向四车道

# 绘图与保存
SHOW_PLOT = True
SAVE_FIG = True
FIG_DPI = 220


# =========================================================
# 3. 基础函数
# =========================================================
def safe_read_csv(file_path):
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def validate_input_path(file_path):
    if not file_path:
        raise ValueError("INPUT_FILE 为空，请填写输入路径。")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"输入文件不存在：{file_path}")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"输入路径不是文件：{file_path}")


def normalize_vectors(vx, vy, eps=1e-12):
    norm = np.sqrt(vx**2 + vy**2)
    norm = np.where(norm < eps, 1.0, norm)
    return vx / norm, vy / norm


def compute_arclength(x, y):
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.zeros(len(x))
    s[1:] = np.cumsum(ds)
    return s


def compute_tangent_normal(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    tx, ty = normalize_vectors(dx, dy)

    # 左法向
    nx = -ty
    ny = tx
    nx, ny = normalize_vectors(nx, ny)
    return tx, ty, nx, ny


def offset_polyline(x, y, nx, ny, offset):
    xo = x + offset * nx
    yo = y + offset * ny
    return xo, yo


def compute_heading(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.arctan2(dy, dx)
    return yaw


def unwrap_angle(angle):
    return np.unwrap(angle)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_curvature_from_s(x, y):
    s = compute_arclength(x, y)
    yaw = compute_heading(x, y)
    yaw_unwrap = unwrap_angle(yaw)

    ds = np.gradient(s)
    ds = np.where(np.abs(ds) < 1e-12, 1e-12, ds)

    kappa = np.gradient(yaw_unwrap) / ds
    return kappa


def remove_duplicate_points(x, y, min_dist=1e-6):
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dist = np.sqrt(dx**2 + dy**2)

    keep = np.ones(len(x), dtype=bool)
    keep[1:] = dist[1:] > min_dist
    return x[keep], y[keep]


def resample_polyline_equal_ds(x, y, ds=1.0):
    """
    沿弧长等间距重采样
    使用线性插值，稳定、依赖少，适合当前工程任务
    """
    x, y = remove_duplicate_points(x, y)
    s = compute_arclength(x, y)

    total_len = s[-1]
    s_new = np.arange(0.0, total_len, ds)
    if total_len - s_new[-1] > 0.3 * ds:
        s_new = np.append(s_new, total_len)

    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)

    return s_new, x_new, y_new


def export_line_csv(out_dir, file_stem, x, y):
    s = compute_arclength(x, y)
    yaw = compute_heading(x, y)
    yaw = wrap_to_pi(yaw)
    kappa = compute_curvature_from_s(x, y)

    df = pd.DataFrame({
        "s": s,
        "x": x,
        "y": y,
        "yaw_rad": yaw,
        "curvature": kappa
    })

    out_path = os.path.join(out_dir, f"{file_stem}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def build_road_boundaries(center_x, center_y, nx, ny):
    """
    基于道路几何中心线扩展四车道完整道路
    +n 方向为左侧
    -n 方向为右侧
    """
    half_median = MEDIAN_WIDTH / 2.0
    side_lane_total = LANES_PER_SIDE * LANE_WIDTH
    outer_total = half_median + side_lane_total + EDGE_WIDTH

    offsets = {
        "road_centerline": 0.0,

        "median_left_edge": +half_median,
        "median_right_edge": -half_median,

        "left_lane1_outer": +(half_median + 1.0 * LANE_WIDTH),
        "left_lane2_outer": +(half_median + 2.0 * LANE_WIDTH),

        "right_lane1_outer": -(half_median + 1.0 * LANE_WIDTH),
        "right_lane2_outer": -(half_median + 2.0 * LANE_WIDTH),

        "left_road_edge": +outer_total,
        "right_road_edge": -outer_total,
    }

    lines = {}
    for name, off in offsets.items():
        xo, yo = offset_polyline(center_x, center_y, nx, ny, off)
        lines[name] = {
            "offset_m": off,
            "x": xo,
            "y": yo
        }
    return lines


# =========================================================
# 4. 主流程
# =========================================================
def main():
    validate_input_path(INPUT_FILE)

    print("=" * 90)
    print("输入文件：")
    print(INPUT_FILE)
    print("=" * 90)

    df = safe_read_csv(INPUT_FILE)
    print("读取成功，列名：")
    print(list(df.columns))

    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"输入文件缺少 x/y 列，当前列名：{list(df.columns)}")

    x = pd.to_numeric(df["x"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["y"], errors="coerce").to_numpy()

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 10:
        raise ValueError("有效点数过少，无法处理。")

    # 原始主路线法向
    _, _, nx, ny = compute_tangent_normal(x, y)

    # 偏移得到道路几何中心线
    road_cx_raw, road_cy_raw = offset_polyline(x, y, nx, ny, OFFSET_TO_ROAD_CENTER)

    # 对道路几何中心线做等弧长重采样
    s_center, road_cx, road_cy = resample_polyline_equal_ds(
        road_cx_raw, road_cy_raw, ds=RESAMPLE_DS
    )

    # 以重采样后的中心线重新求法向
    _, _, nx2, ny2 = compute_tangent_normal(road_cx, road_cy)

    # 生成完整四车道道路
    road_lines = build_road_boundaries(road_cx, road_cy, nx2, ny2)

    # 输出目录
    in_dir = os.path.dirname(INPUT_FILE)
    in_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    out_dir = os.path.join(in_dir, f"{in_name}_FINAL_4LANE_ROAD_1m")
    os.makedirs(out_dir, exist_ok=True)

    # 导出原始输入与偏移前后中心线
    export_line_csv(out_dir, "00_main_route_input", x, y)
    export_line_csv(out_dir, "01_road_geometric_centerline_raw", road_cx_raw, road_cy_raw)
    export_line_csv(out_dir, "02_road_geometric_centerline_resampled_1m", road_cx, road_cy)

    # 导出完整道路各条线
    exported_paths = {}
    for name, data in road_lines.items():
        exported_paths[name] = export_line_csv(out_dir, f"final_{name}", data["x"], data["y"])

    # 生成汇总表
    summary_rows = [
        {"name": "main_route_input", "offset_m": 0.0, "file": "00_main_route_input.csv"},
        {"name": "road_geometric_centerline_raw", "offset_m": OFFSET_TO_ROAD_CENTER, "file": "01_road_geometric_centerline_raw.csv"},
        {"name": "road_geometric_centerline_resampled_1m", "offset_m": OFFSET_TO_ROAD_CENTER, "file": "02_road_geometric_centerline_resampled_1m.csv"},
    ]
    for name, data in road_lines.items():
        summary_rows.append({
            "name": name,
            "offset_m": data["offset_m"],
            "file": f"final_{name}.csv"
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "final_4lane_road_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 生成简化的 CarSim 候选中心线文件
    carsim_df = pd.DataFrame({
        "s": s_center,
        "x": road_cx,
        "y": road_cy,
        "yaw_rad": wrap_to_pi(compute_heading(road_cx, road_cy)),
        "curvature": compute_curvature_from_s(road_cx, road_cy)
    })
    carsim_path = os.path.join(out_dir, "carsim_reference_centerline_1m.csv")
    carsim_df.to_csv(carsim_path, index=False, encoding="utf-8-sig")

    # 画图
    if SHOW_PLOT:
        plt.figure(figsize=(11, 8))

        # 原始主路线
        plt.plot(x, y, linewidth=1.1, label="Input Main Route")

        # 最终道路中心线
        plt.plot(road_cx, road_cy, linewidth=1.8, label="Final Road Centerline (1m)")

        # 四车道结构线
        show_names = [
            "median_left_edge", "median_right_edge",
            "left_lane1_outer", "left_lane2_outer",
            "right_lane1_outer", "right_lane2_outer",
            "left_road_edge", "right_road_edge"
        ]

        for name in show_names:
            plt.plot(
                road_lines[name]["x"],
                road_lines[name]["y"],
                linewidth=1.0,
                label=name
            )

        plt.scatter(x[0], y[0], s=35, label="Input Start")
        plt.scatter(road_cx[0], road_cy[0], s=35, label="Road Center Start")

        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title("Final 4-Lane Road (1m Resampled)")
        plt.legend(fontsize=8)
        plt.tight_layout()

        if SAVE_FIG:
            fig_path = os.path.join(out_dir, "final_4lane_road_preview.png")
            plt.savefig(fig_path, dpi=FIG_DPI, bbox_inches="tight")
            print(f"预览图已保存：{fig_path}")

        plt.show()

    print("=" * 90)
    print("最终四车道完整道路已生成。")
    print(f"输出目录：{out_dir}")
    print(f"CarSim候选中心线：{carsim_path}")
    print(f"汇总表：{summary_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()