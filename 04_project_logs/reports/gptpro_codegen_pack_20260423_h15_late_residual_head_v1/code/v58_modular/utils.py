from .paths import *

class TeeStdout:
    """同时写控制台与日志文件（捕获所有 print）"""

    def __init__(self, log_path, console_stream=None):
        self.console = console_stream if console_stream is not None else sys.__stdout__
        self.f = open(str(log_path), 'w', encoding='utf-8')

    def write(self, s):
        try:
            self.console.write(s)
        except Exception:
            pass
        try:
            self.f.write(s)
        except Exception:
            pass
        self.flush()

    def flush(self):
        try:
            self.console.flush()
        except Exception:
            pass
        try:
            self.f.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def save_json(path, obj):
    with open(str(path), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(str(path), 'r', encoding='utf-8') as f:
        return json.load(f)


def env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception as exc:
        raise ValueError(f"Invalid float for {name}={raw!r}") from exc


def env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except Exception as exc:
        raise ValueError(f"Invalid int for {name}={raw!r}") from exc


def env_choice(name, default, choices):
    raw = os.environ.get(name, default)
    value = str(raw).strip().lower()
    if value not in choices:
        raise ValueError(f"Unsupported {name}={raw!r}; expected one of {sorted(choices)}")
    return value


STEER_SOURCE_UNIT = "rad"
STEER_ANGLE_UNIT = os.environ.get("DRIVER_MODEL_STEER_ANGLE_UNIT", "rad").strip().lower()
if STEER_ANGLE_UNIT not in {"rad", "deg"}:
    raise ValueError(f"Unsupported DRIVER_MODEL_STEER_ANGLE_UNIT={STEER_ANGLE_UNIT!r}; expected 'rad' or 'deg'")
STEER_PLOT_UNIT = os.environ.get("DRIVER_MODEL_STEER_PLOT_UNIT", "deg").strip().lower()
if STEER_PLOT_UNIT not in {"rad", "deg"}:
    raise ValueError(f"Unsupported DRIVER_MODEL_STEER_PLOT_UNIT={STEER_PLOT_UNIT!r}; expected 'rad' or 'deg'")
STEER_ANGLE_SCALE = float(180.0 / np.pi) if STEER_ANGLE_UNIT == "deg" else 1.0
STEER_PLOT_SCALE = float(180.0 / np.pi) if STEER_PLOT_UNIT == "deg" else 1.0
STEER_PLOT_FROM_TARGET_SCALE = float(STEER_PLOT_SCALE / STEER_ANGLE_SCALE)
STEER_PLOT_LABEL = f"steer angle ({STEER_PLOT_UNIT})"
STEER_PEAK_PLOT_LABEL = f"peak|steer angle| (GT, {STEER_PLOT_UNIT})"


def steer_value_from_rad(value: float) -> float:
    return float(value) * STEER_ANGLE_SCALE


def steer_array_from_rad(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * np.float32(STEER_ANGLE_SCALE)


def steer_value_for_plot(value: float) -> float:
    return float(value) * STEER_PLOT_FROM_TARGET_SCALE


def steer_array_for_plot(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * np.float32(STEER_PLOT_FROM_TARGET_SCALE)



# =========================
# Road-type (curve/straight) utilities
# =========================
def find_feature_in_list(feature_names, keywords):
    """Return the first feature name in feature_names that contains any keyword (case-insensitive)."""
    lower = [f.lower() for f in feature_names]
    for kw in keywords:
        kwl = kw.lower()
        for i, f in enumerate(lower):
            if kwl in f:
                return feature_names[i], i
    return None, None


def otsu_threshold_log10(values, eps=1e-10, bins=256):
    """Otsu threshold on log10(values+eps). Returns threshold in original scale."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 100:
        return float(np.nanpercentile(v, 85)) if v.size else 0.0

    lv = np.log10(np.maximum(v, 0.0) + eps)
    hist, edges = np.histogram(lv, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]

    # avoid division by zero
    m1 = np.cumsum(hist * centers) / np.maximum(w1, 1)
    m2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(w2[::-1], 1))[::-1]

    between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    k = int(np.argmax(between)) if between.size else 0

    thr_log = float(centers[k])
    thr = float(10 ** thr_log - eps)
    return max(thr, 0.0)


def auto_curve_threshold(curve_scores, eps=1e-10):
    """Pick a robust threshold to split straight vs curve using ONLY history-window curvature stats.
    Strategy:
      1) Otsu on log-scale
      2) If split is too extreme, fallback to a percentile threshold.
    """
    cs = np.asarray(curve_scores, dtype=np.float64)
    cs = cs[np.isfinite(cs)]
    if cs.size == 0:
        return 0.0

    thr = otsu_threshold_log10(cs, eps=eps, bins=256)
    ratio = float(np.mean(cs > thr))

    # If Otsu yields an overly imbalanced split, fallback to a safer percentile.
    if ratio < 0.05:
        thr = float(np.nanpercentile(cs, 90))
    elif ratio > 0.95:
        thr = float(np.nanpercentile(cs, 10))

    # Avoid tiny numerical noise thresholds
    thr = max(thr, 1e-8)
    return thr

def try_copy_self(run_dir):
    """可选：复制当前脚本到输出目录，方便复现"""
    try:
        src = ENTRY_SCRIPT_PATH if ENTRY_SCRIPT_PATH.exists() else Path(__file__).resolve()
        dst = Path(run_dir) / src.name
        shutil.copy2(str(src), str(dst))
        pkg_dst = Path(run_dir) / MODULE_DIR.name
        if pkg_dst.exists():
            shutil.rmtree(str(pkg_dst))
        shutil.copytree(str(MODULE_DIR), str(pkg_dst))
    except Exception:
        pass


