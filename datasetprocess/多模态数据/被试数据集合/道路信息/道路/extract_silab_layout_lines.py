import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0) 基础配置
# ============================================================
AED_DIR = r"F:\数据集处理\data_process\datasetprocess\多模态数据\被试数据集合\道路信息\道路"

OUT_CSV = os.path.join(AED_DIR, "silab_route_centerline.csv")
OUT_NPY = os.path.join(AED_DIR, "silab_route_centerline.npy")
OUT_FIG = os.path.join(AED_DIR, "silab_route_centerline.png")
OUT_CURV_FIG = os.path.join(AED_DIR, "silab_route_curvature.png")

# AED内部几何离散步长
DS_SEG = 1.0

# 最终centerline重采样步长
DS_RESAMPLE = 1.0

# 模块内部几何段拼接容差
MODULE_CONNECT_TOL = 30.0

# 模块之间拼接容差
ROUTE_CONNECT_TOL = 120.0


# ============================================================
# 1) 模块顺序（已按你确认的信息修改）
#    实验从 section1.Port1 开始
# ============================================================
ROUTE_ORDER = [
    "section1",
    "longstraight",
    "section2",
    "curve1",
    "section3",
    "fix_road",
    "section4",
    "curve2",
    "section5",
    "stop",
    "section6",
    "mu1",
    "section7",
    "curve3",
    "section8",
    "zd",
]


# ============================================================
# 2) 模块名 -> 文件名映射
#    按你目前给出的文件名补全
# ============================================================
FILE_MAP = {
    "section1": "middle_section.autosave.1.autosave.1.aed",
    "longstraight": "longsrtaight.autosave.1.aed",
    "section2": "middle_section.autosave.1.autosave.1.aed",
    "curve1": "curve1.autosave.2.aed",
    "section3": "middle_section.autosave.1.autosave.1.aed",
    "fix_road": "fix_road.autosave.1.aed",
    "section4": "middle_section.autosave.1.autosave.1.aed",
    "curve2": "curve2.autosave.1.aed",
    "section5": "middle_section.autosave.1.autosave.1.aed",
    "stop": "stop.autosave.1.aed",
    "section6": "middle_section.autosave.1.autosave.1.aed",
    "mu1": "differentmu_road.aed",
    "section7": "middle_section.autosave.1.autosave.1.aed",
    "curve3": "curve3.autosave.2.aed",
    "section8": "middle_section.autosave.1.autosave.1.aed",
    "zd": "zd.autosave.1.autosave.2.aed",
}


# ============================================================
# 3) 正则：Line / Circle(Arc)
# ============================================================
PAT_LINE = re.compile(
    r'#\s*AR2FLayoutLine.*?x0\s*=\s*([-\d\.eE]+).*?y0\s*=\s*([-\d\.eE]+).*?x1\s*=\s*([-\d\.eE]+).*?y1\s*=\s*([-\d\.eE]+)',
    re.S
)

PAT_CIRCLE = re.compile(
    r'#\s*AR2FLayoutCircle.*?x0\s*=\s*([-\d\.eE]+).*?y0\s*=\s*([-\d\.eE]+).*?r\s*=\s*([-\d\.eE]+).*?IsArc\s*=\s*([-\d\.eE]+).*?phi0\s*=\s*([-\d\.eE]+).*?phi1\s*=\s*([-\d\.eE]+).*?invert\s*=\s*([-\d\.eE]+)',
    re.S
)


# ============================================================
# 4) 基础函数
# ============================================================
def safe_read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def remove_duplicate_neighbors(pts, tol=1e-9):
    if len(pts) <= 1:
        return pts.copy()
    out = [pts[0]]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - out[-1]) > tol:
            out.append(pts[i])
    return np.asarray(out, dtype=float)


def polyline_length(pts):
    if len(pts) < 2:
        return 0.0
    ds = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    return float(np.sum(ds))


def polyline_arclength(pts):
    if len(pts) < 2:
        return np.array([0.0])
    ds = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s


def sample_line(x0, y0, x1, y1, ds=1.0):
    L = float(np.hypot(x1 - x0, y1 - y0))
    n = max(2, int(np.ceil(L / ds)) + 1)
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    return np.column_stack([xs, ys])


def sample_arc(cx, cy, r, phi0, phi1, invert, ds=1.0):
    """
    基于 AED 中 Circle + IsArc = 1 的圆弧离散
    """
    phi0 = float(phi0)
    phi1 = float(phi1)
    r = float(r)

    dphi = phi1 - phi0

    # 尝试按 invert 调整方向
    if int(invert) != 0:
        phi0, phi1 = phi1, phi0
        dphi = phi1 - phi0

    arc_len = abs(dphi * r)
    n = max(8, int(np.ceil(arc_len / ds)) + 1)
    phis = np.linspace(phi0, phi1, n)

    xs = cx + r * np.cos(phis)
    ys = cy + r * np.sin(phis)
    return np.column_stack([xs, ys])


def resample_polyline(pts, ds_new=1.0):
    s = polyline_arclength(pts)
    if len(s) == 1 or s[-1] < 1e-9:
        return pts.copy(), s.copy()

    s_new = np.arange(0.0, s[-1] + ds_new, ds_new)
    x_new = np.interp(s_new, s, pts[:, 0])
    y_new = np.interp(s_new, s, pts[:, 1])
    pts_new = np.column_stack([x_new, y_new])
    return pts_new, s_new


def compute_heading_curvature(pts, s):
    x = pts[:, 0]
    y = pts[:, 1]

    if len(pts) < 3:
        heading = np.zeros(len(pts))
        curvature = np.zeros(len(pts))
        return heading, curvature

    dx = np.gradient(x, s, edge_order=2)
    dy = np.gradient(y, s, edge_order=2)
    heading = np.arctan2(dy, dx)

    ddx = np.gradient(dx, s, edge_order=2)
    ddy = np.gradient(dy, s, edge_order=2)

    denom = (dx * dx + dy * dy) ** 1.5
    denom[denom < 1e-9] = 1e-9
    curvature = (dx * ddy - dy * ddx) / denom

    return heading, curvature


# ============================================================
# 5) AED 几何提取
# ============================================================
def parse_aed_geometries(aed_path, ds=1.0):
    text = safe_read_text(aed_path)

    pieces = []

    # ---- 线段 ----
    line_matches = PAT_LINE.findall(text)
    for m in line_matches:
        x0, y0, x1, y1 = map(float, m)
        pts = sample_line(x0, y0, x1, y1, ds=ds)
        pieces.append({
            "type": "line",
            "pts": pts,
            "start": pts[0].copy(),
            "end": pts[-1].copy(),
            "length": polyline_length(pts),
        })

    # ---- 圆弧 ----
    circle_matches = PAT_CIRCLE.findall(text)
    for m in circle_matches:
        cx, cy, r, is_arc, phi0, phi1, invert = m
        cx = float(cx)
        cy = float(cy)
        r = float(r)
        is_arc = int(float(is_arc))
        phi0 = float(phi0)
        phi1 = float(phi1)
        invert = int(float(invert))

        if is_arc == 1 and abs(r) > 1e-9:
            pts = sample_arc(cx, cy, r, phi0, phi1, invert, ds=ds)
            pieces.append({
                "type": "arc",
                "pts": pts,
                "start": pts[0].copy(),
                "end": pts[-1].copy(),
                "length": polyline_length(pts),
            })

    return pieces


# ============================================================
# 6) 模块内部拼接
# ============================================================
def build_module_polyline(pieces, connect_tol=30.0):
    """
    一个模块可能由多段 line/arc 组成。
    这里基于端点最近原则，在模块内部串起来。
    """
    if len(pieces) == 0:
        raise RuntimeError("模块内没有解析到任何几何段")

    if len(pieces) == 1:
        return remove_duplicate_neighbors(pieces[0]["pts"])

    degrees = {i: 0 for i in range(len(pieces))}
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            pi = pieces[i]
            pj = pieces[j]
            dlist = [
                np.linalg.norm(pi["start"] - pj["start"]),
                np.linalg.norm(pi["start"] - pj["end"]),
                np.linalg.norm(pi["end"] - pj["start"]),
                np.linalg.norm(pi["end"] - pj["end"]),
            ]
            if min(dlist) < connect_tol:
                degrees[i] += 1
                degrees[j] += 1

    candidates = [i for i, d in degrees.items() if d <= 1]
    if len(candidates) == 0:
        start_idx = int(np.argmax([p["length"] for p in pieces]))
    else:
        start_idx = candidates[int(np.argmax([pieces[i]["length"] for i in candidates]))]

    used = set([start_idx])
    current = pieces[start_idx]["pts"].copy()

    while len(used) < len(pieces):
        head = current[0]
        tail = current[-1]

        best_j = None
        best_mode = None
        best_dist = None

        for j in range(len(pieces)):
            if j in used:
                continue

            p = pieces[j]
            s = p["start"]
            e = p["end"]

            modes = [
                ("tail_start", np.linalg.norm(tail - s)),
                ("tail_end", np.linalg.norm(tail - e)),
                ("head_start", np.linalg.norm(head - s)),
                ("head_end", np.linalg.norm(head - e)),
            ]
            mode, dist = min(modes, key=lambda x: x[1])

            if (best_dist is None) or (dist < best_dist):
                best_dist = dist
                best_j = j
                best_mode = mode

        if best_j is None or best_dist is None or best_dist > connect_tol:
            break

        add_pts = pieces[best_j]["pts"].copy()

        if best_mode == "tail_start":
            current = np.vstack([current, add_pts[1:]])
        elif best_mode == "tail_end":
            add_pts = add_pts[::-1]
            current = np.vstack([current, add_pts[1:]])
        elif best_mode == "head_start":
            add_pts = add_pts[::-1]
            current = np.vstack([add_pts[:-1], current])
        elif best_mode == "head_end":
            current = np.vstack([add_pts[:-1], current])

        used.add(best_j)

    current = remove_duplicate_neighbors(current)
    return current


# ============================================================
# 7) 模块间方向判定与拼接
# ============================================================
def orient_first_module_by_x(pts):
    """
    你已知从 section1.Port1 开始，但我们没有直接解析到 Port1/Port2 坐标。
    先采用一个稳定策略：
    默认让第一段整体更偏“从左往右”。
    如果后续发现不对，可再调整。
    """
    if len(pts) < 2:
        return pts
    if pts[0, 0] <= pts[-1, 0]:
        return pts
    return pts[::-1]


def orient_polyline_to_connect(prev_tail, pts):
    d_forward = np.linalg.norm(prev_tail - pts[0])
    d_reverse = np.linalg.norm(prev_tail - pts[-1])

    if d_reverse < d_forward:
        return pts[::-1], d_reverse, "reverse"
    else:
        return pts, d_forward, "forward"


def stitch_route(module_polylines, route_order, connect_tol=120.0):
    route_pts = None
    gap_info = []

    for i, name in enumerate(route_order):
        pts = module_polylines[name].copy()

        if i == 0:
            pts = orient_first_module_by_x(pts)
            route_pts = pts
            gap_info.append((name, None, 0.0, "first"))
            continue

        pts, gap, orient_mode = orient_polyline_to_connect(route_pts[-1], pts)
        gap_info.append((name, route_order[i - 1], gap, orient_mode))

        if gap < connect_tol:
            if np.linalg.norm(route_pts[-1] - pts[0]) < connect_tol:
                route_pts = np.vstack([route_pts, pts[1:]])
            else:
                route_pts = np.vstack([route_pts, pts])
        else:
            # 即使 gap 大也先拼，便于整体看图排查
            route_pts = np.vstack([route_pts, pts])

    route_pts = remove_duplicate_neighbors(route_pts)
    return route_pts, gap_info


# ============================================================
# 8) 主流程
# ============================================================
def main():
    print("=" * 80)
    print("SILAB AED -> proving ground centerline reconstruction")
    print("=" * 80)

    for name in ROUTE_ORDER:
        if name not in FILE_MAP:
            raise KeyError(f"FILE_MAP 缺少模块名: {name}")

    module_polylines = {}

    print("\n[1/4] 解析各模块 AED ...")
    for name in ROUTE_ORDER:
        fname = FILE_MAP[name]
        path = os.path.join(AED_DIR, fname)

        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 AED 文件: {path}")

        pieces = parse_aed_geometries(path, ds=DS_SEG)

        if len(pieces) == 0:
            raise RuntimeError(f"{name} 未解析到任何 LayoutLine / LayoutCircle: {path}")

        poly = build_module_polyline(pieces, connect_tol=MODULE_CONNECT_TOL)
        module_polylines[name] = poly

        n_line = sum(1 for p in pieces if p["type"] == "line")
        n_arc = sum(1 for p in pieces if p["type"] == "arc")
        print(
            f"{name:>12s} | file = {fname:<40s} | "
            f"line = {n_line:2d} | arc = {n_arc:2d} | pts = {len(poly):4d} | "
            f"L = {polyline_length(poly):8.3f} m"
        )

    print("\n[2/4] 按主路线顺序拼接模块 ...")
    route_pts_raw, gap_info = stitch_route(
        module_polylines,
        ROUTE_ORDER,
        connect_tol=ROUTE_CONNECT_TOL
    )

    print("\n模块拼接信息：")
    for cur_name, prev_name, gap, mode in gap_info:
        if prev_name is None:
            print(f"{cur_name:>12s} | start module | mode = {mode}")
        else:
            flag = "OK" if gap < ROUTE_CONNECT_TOL else "LARGE_GAP"
            print(
                f"{prev_name:>12s} -> {cur_name:<12s} | "
                f"gap = {gap:8.3f} m | mode = {mode:<7s} | {flag}"
            )

    print("\n[3/4] 重采样 + 计算 heading / curvature ...")
    route_pts, s = resample_polyline(route_pts_raw, ds_new=DS_RESAMPLE)
    heading, curvature = compute_heading_curvature(route_pts, s)
    z = np.zeros_like(s)

    df = pd.DataFrame({
        "s": s,
        "x": route_pts[:, 0],
        "y": route_pts[:, 1],
        "z": z,
        "heading_rad": heading,
        "curvature_1pm": curvature,
    })

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    np.save(OUT_NPY, route_pts)

    print(f"已导出 CSV : {OUT_CSV}")
    print(f"已导出 NPY : {OUT_NPY}")

    print("\n[4/4] 画图 ...")

    plt.figure(figsize=(13, 9))

    # 先画各模块虚线
    for name in ROUTE_ORDER:
        pts = module_polylines[name]
        plt.plot(pts[:, 0], pts[:, 1], "--", linewidth=1, alpha=0.8)
        mid = pts[len(pts) // 2]
        plt.text(mid[0], mid[1], name, fontsize=8)

    # 再画最终路线
    plt.plot(route_pts[:, 0], route_pts[:, 1], linewidth=2)
    plt.scatter(route_pts[0, 0], route_pts[0, 1], s=60, marker='o')
    plt.scatter(route_pts[-1, 0], route_pts[-1, 1], s=60, marker='x')

    plt.axis("equal")
    plt.grid(True)
    plt.title("Topology-based SILAB Route Centerline")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.show()

    plt.figure(figsize=(13, 4))
    plt.plot(s, curvature, linewidth=1.5)
    plt.grid(True)
    plt.xlabel("s [m]")
    plt.ylabel("curvature [1/m]")
    plt.title("Route Curvature")
    plt.tight_layout()
    plt.savefig(OUT_CURV_FIG, dpi=200)
    plt.show()

    print(f"已保存路线图 : {OUT_FIG}")
    print(f"已保存曲率图 : {OUT_CURV_FIG}")

    print("\n完成。")


if __name__ == "__main__":
    main()