from .utils import *

FS = 200
WIN_SEC = 3.0
WIN_LEN = int(WIN_SEC * FS)         # 600
DEFAULT_FUTURE_SEC = 2.0
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 40
DEFAULT_LR = 1e-3
DEFAULT_OPTIMIZER = "adam"
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_SCHEDULER = "none"
DEFAULT_WARMUP_EPOCHS = 0
DEFAULT_GRAD_CLIP_NORM = 0.0
DEFAULT_D_MODEL = 128
DEFAULT_N_HEAD = 2
DEFAULT_FFN_DIM = 256
DEFAULT_DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def refresh_runtime_training_config():
    global FUTURE_SEC, FUTURE_LEN
    global BATCH_SIZE, EPOCHS, LR
    global OPTIMIZER_NAME, WEIGHT_DECAY, SCHEDULER_NAME, WARMUP_EPOCHS, GRAD_CLIP_NORM
    global D_MODEL, N_HEAD, FFN_DIM, DROPOUT

    FUTURE_SEC = env_float("DRIVER_MODEL_FUTURE_SEC", DEFAULT_FUTURE_SEC)
    if FUTURE_SEC <= 0:
        raise ValueError(f"DRIVER_MODEL_FUTURE_SEC must be positive, got {FUTURE_SEC}")
    FUTURE_LEN = max(1, int(round(FUTURE_SEC * FS)))

    BATCH_SIZE = max(1, env_int("DRIVER_MODEL_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    EPOCHS = max(1, env_int("DRIVER_MODEL_EPOCHS", DEFAULT_EPOCHS))
    LR = env_float("DRIVER_MODEL_LR", DEFAULT_LR)
    if LR <= 0:
        raise ValueError(f"DRIVER_MODEL_LR must be positive, got {LR}")

    OPTIMIZER_NAME = env_choice("DRIVER_MODEL_OPTIMIZER", DEFAULT_OPTIMIZER, {"adam", "adamw"})
    WEIGHT_DECAY = max(0.0, env_float("DRIVER_MODEL_WEIGHT_DECAY", DEFAULT_WEIGHT_DECAY))
    SCHEDULER_NAME = env_choice("DRIVER_MODEL_SCHEDULER", DEFAULT_SCHEDULER, {"none", "cosine"})
    WARMUP_EPOCHS = max(0, env_int("DRIVER_MODEL_WARMUP_EPOCHS", DEFAULT_WARMUP_EPOCHS))
    GRAD_CLIP_NORM = max(0.0, env_float("DRIVER_MODEL_GRAD_CLIP_NORM", DEFAULT_GRAD_CLIP_NORM))

    D_MODEL = max(1, env_int("DRIVER_MODEL_D_MODEL", DEFAULT_D_MODEL))
    N_HEAD = max(1, env_int("DRIVER_MODEL_N_HEAD", DEFAULT_N_HEAD))
    FFN_DIM = max(1, env_int("DRIVER_MODEL_FFN_DIM", DEFAULT_FFN_DIM))
    DROPOUT = env_float("DRIVER_MODEL_DROPOUT", DEFAULT_DROPOUT)
    if not (0.0 <= DROPOUT < 1.0):
        raise ValueError(f"DRIVER_MODEL_DROPOUT must be in [0, 1), got {DROPOUT}")
    if D_MODEL % N_HEAD != 0:
        raise ValueError(f"D_MODEL={D_MODEL} must be divisible by N_HEAD={N_HEAD}")


refresh_runtime_training_config()

SMOKE_MODE = os.environ.get("DRIVER_MODEL_SMOKE", "0") == "1"
SMOKE_MAX_SAMPLES = int(os.environ.get("DRIVER_MODEL_SMOKE_MAX_SAMPLES", "256"))
SMOKE_EPOCHS = int(os.environ.get("DRIVER_MODEL_SMOKE_EPOCHS", "2"))
SMOKE_BATCH_SIZE = int(os.environ.get("DRIVER_MODEL_SMOKE_BATCH_SIZE", "32"))


def apply_smoke_overrides():
    global EPOCHS, BATCH_SIZE
    if SMOKE_MODE:
        EPOCHS = SMOKE_EPOCHS
        BATCH_SIZE = SMOKE_BATCH_SIZE
        print(f"[SMOKE] enabled | max_samples={SMOKE_MAX_SAMPLES} | epochs={EPOCHS} | batch_size={BATCH_SIZE}")


apply_smoke_overrides()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LTR_COEFF = 0.11243
STRONG_LABELS = ["medium_active", "strong_active", "extreme_active"]

# Transformer
NUM_LAYERS_ENC = 2
NUM_LAYERS_DEC = 2

# Multi-scale loss (encourage high-frequency details)
W_DIFF1 = 0.15   # first-derivative loss weight
W_DIFF2 = 0.05   # second-derivative loss weight

# Response-state-aware v1 switches
# v1 targets:
# - teacher-aligned latent state (legacy A/C or data-driven PCA latent)
# - reversal aux classification
# - peak timing aux alignment
# - peak intensity supervision (reuse amplitude loss on trajectory)
ENABLE_RESPONSE_STATE_V1 = True
ENABLE_STATE_DISTILL = ENABLE_RESPONSE_STATE_V1
ENABLE_REVERSAL_AUX = ENABLE_RESPONSE_STATE_V1
ENABLE_PEAKTIME_AUX = ENABLE_RESPONSE_STATE_V1
ENABLE_PEAKINTENSITY_AUX = ENABLE_RESPONSE_STATE_V1

# Teacher-state representation
TEACHER_STATE_MODE = "pca_latent"   # "old_ac" | "pca_latent"
TEACHER_STATE_DIM = 4

# Reversal / phase-aware auxiliary losses
W_REVSEQ = 0.0        # keep baseline default; 0.05 smoke worsened tail despite better late-peak recall
W_PEAKTIME = 0.05 if ENABLE_PEAKTIME_AUX else 0.0   # steer-rate peak timing alignment loss
REVSEQ_ALPHA_FRAC = 0.25  # alpha = frac * steer_std (soft sign for reversal)
PEAK_TEMP_FRAC = 0.35     # temp = frac * mean|steer_rate| (soft-argmax)

# Steer local-detail weighting (emphasize reversals + high steer-rate)
W_STEER_WT = 0.50       # weighted steer MSE added to task loss
W_STEER_RATE = 1.00     # baseline emphasis on high-|steer_rate| segments
W_STEER_REV = 0.35      # modest reversal emphasis to reduce tail flattening on correction events
STEER_WT_MAX = 4.0      # cap for stability

# Distillation / auxiliary heads
LAMBDA_STATE = 0.08 if ENABLE_STATE_DISTILL else 0.0
W_TASK_STEER = 1.50   # steer 主任务权重
W_TASK_YAW   = 1.00   # yawrate 主任务权重
W_TASK_AY    = 0.70   # ay 主任务权重
W_AMP        = 0.30 if ENABLE_PEAKINTENSITY_AUX else 0.0   # peak intensity supervision via trajectory amplitude loss
W_TREND      = 0.10   # coarse steer-trend alignment on the full 2s future window
TREND_POOL_KERNEL = 20
TREND_POOL_STRIDE = 20
TREND_SIGN_EPS = steer_value_from_rad(0.02)
TREND_LOSS_MODE = os.environ.get("DRIVER_MODEL_TREND_LOSS_MODE", "pooled_level_mse_v1")
TREND_LEVEL_WEIGHT = float(os.environ.get("DRIVER_MODEL_TREND_LEVEL_WEIGHT", "0.25"))
TREND_DELTA_WEIGHT = float(os.environ.get("DRIVER_MODEL_TREND_DELTA_WEIGHT", "0.50"))
TREND_DIR_WEIGHT = float(os.environ.get("DRIVER_MODEL_TREND_DIR_WEIGHT", "0.25"))
ENABLE_STEER_COARSE_FINE = os.environ.get("DRIVER_MODEL_STEER_COARSE_FINE", "0") == "1"
W_TREND_COARSE = float(os.environ.get("DRIVER_MODEL_W_TREND_COARSE", "0.10"))
W_FINE_DC = float(os.environ.get("DRIVER_MODEL_W_FINE_DC", "0.02"))
ENABLE_PHASE_ADAPTIVE_TREND = os.environ.get("DRIVER_MODEL_PHASE_ADAPTIVE_TREND", "0") == "1"
TREND_EARLY_BINS = int(os.environ.get("DRIVER_MODEL_TREND_EARLY_BINS", "12"))
TREND_LATE_STRAIGHT_DOWN = float(os.environ.get("DRIVER_MODEL_TREND_LATE_STRAIGHT_DOWN", "0.35"))
TREND_LATE_STRONGREV_DOWN = float(os.environ.get("DRIVER_MODEL_TREND_LATE_STRONGREV_DOWN", "0.45"))
ENABLE_LATE_REV_GATE = os.environ.get("DRIVER_MODEL_LATE_REV_GATE", "0") == "1"
LATE_REV_GATE_START_SEC = float(os.environ.get("DRIVER_MODEL_LATE_REV_GATE_START_SEC", "1.05"))
LATE_REV_GATE_SCALE = float(os.environ.get("DRIVER_MODEL_LATE_REV_GATE_SCALE", "0.60"))
LATE_REV_GATE_RAMP_POWER = float(os.environ.get("DRIVER_MODEL_LATE_REV_GATE_RAMP_POWER", "1.50"))
ENABLE_STRONG_POS_GATE = os.environ.get("DRIVER_MODEL_STRONG_POS_GATE", "0") == "1"
STRONG_POS_GATE_START_SEC = float(os.environ.get("DRIVER_MODEL_STRONG_POS_GATE_START_SEC", "1.20"))
STRONG_POS_GATE_SCALE = float(os.environ.get("DRIVER_MODEL_STRONG_POS_GATE_SCALE", "0.45"))
STRONG_POS_GATE_RAMP_POWER = float(os.environ.get("DRIVER_MODEL_STRONG_POS_GATE_RAMP_POWER", "1.75"))
STRONG_POS_GATE_PROB_CENTER = float(os.environ.get("DRIVER_MODEL_STRONG_POS_GATE_PROB_CENTER", "0.60"))
ENABLE_HARD_LATE_FINE = os.environ.get("DRIVER_MODEL_HARD_LATE_FINE", "0") == "1"
ENABLE_MANUAL_COARSE_UPSAMPLE = os.environ.get("DRIVER_MODEL_MANUAL_COARSE_UPSAMPLE", "0") == "1"
W_HARD_LATE_FINE = float(os.environ.get("DRIVER_MODEL_W_HARD_LATE_FINE", "0.06"))
HARD_LATE_START_SEC = float(os.environ.get("DRIVER_MODEL_HARD_LATE_START_SEC", "1.25"))
HARD_TAIL_START_SEC = float(os.environ.get("DRIVER_MODEL_HARD_TAIL_START_SEC", "1.50"))
HARD_PEAK_QUANTILE = float(os.environ.get("DRIVER_MODEL_HARD_PEAK_QUANTILE", "0.90"))
HARD_TAIL_QUANTILE = float(os.environ.get("DRIVER_MODEL_HARD_TAIL_QUANTILE", "0.80"))
INPUT_PIPELINE_VERSION = os.environ.get("DRIVER_MODEL_INPUT_PIPELINE_VERSION", "fixed_v20260421").strip().lower()
if INPUT_PIPELINE_VERSION not in {"legacy_v1", "fixed_v20260421"}:
    raise ValueError(
        f"Unsupported DRIVER_MODEL_INPUT_PIPELINE_VERSION={INPUT_PIPELINE_VERSION!r}; "
        "expected one of {'legacy_v1','fixed_v20260421'}"
    )
USE_PEDALS = os.environ.get("DRIVER_MODEL_USE_PEDALS", "0") == "1"
USE_VY = os.environ.get("DRIVER_MODEL_USE_VY", "0") == "1"
USE_VROLL = os.environ.get("DRIVER_MODEL_USE_VROLL", "0") == "1"
USE_MU = os.environ.get("DRIVER_MODEL_USE_MU", "0") == "1"
USE_Z = os.environ.get("DRIVER_MODEL_USE_Z", "1") == "1"
USE_IS_CURVE_CTX = os.environ.get("DRIVER_MODEL_USE_IS_CURVE_CTX", "0") == "1"
REV_SAMPLE_WEIGHT = 1.80   # 强反打样本整体 loss 加权（建议先用 1.5~2.0）
REV_ZERO_EPS      = 1e-4    # 过零检测小阈值，避免数值噪声误判
LAMBDA_REV  = 0.05 if ENABLE_REVERSAL_AUX else 0.0
LAMBDA_STRONG_POS_GATE = float(os.environ.get("DRIVER_MODEL_LAMBDA_STRONG_POS_GATE", "0.10"))
REV_SAMPLE_WEIGHT_MODE = os.environ.get("DRIVER_MODEL_REV_SAMPLE_WEIGHT_MODE", "strong").strip().lower()
if REV_SAMPLE_WEIGHT_MODE not in {"strong", "weak", "hybrid"}:
    raise ValueError(
        f"Unsupported DRIVER_MODEL_REV_SAMPLE_WEIGHT_MODE={REV_SAMPLE_WEIGHT_MODE!r}; "
        "expected one of {'strong','weak','hybrid'}"
    )
REV_AUX_TARGET = os.environ.get("DRIVER_MODEL_REV_AUX_TARGET", "strong").strip().lower()
if REV_AUX_TARGET not in {"strong", "weak"}:
    raise ValueError(
        f"Unsupported DRIVER_MODEL_REV_AUX_TARGET={REV_AUX_TARGET!r}; expected one of {'strong','weak'}"
    )
REV_EPS_WEAK    = 0.02   # 弱反转判定阈值（方向盘单位若已归一化，请相应缩放）
REV_EPS_STRONG  = 0.20   # 强反转判定阈值（更贴近紧急变道“明显反打”）
STRONG_PEAK_THR = 2.0    # 强反转附加条件：未来窗内 |steer| 峰值需超过该阈值（单位同 steer）
# Anchor selection (v5.6): 用于“同一套模型同时覆盖过弯 + 紧急变道(多次反打)”的对齐
# - 弯道/高侧倾：roll 峰值更稳定
# - 直道/紧急变道：steer_rate 的“最早主峰”更稳定（避免 anchor 落在后续反打）
CURVE_THR_FOR_ANCHOR = 1.0e-6   # 事件段内平均|curvature| 超过此阈值 => 认为是弯道（用于选 anchor）
STEER_RATE_PEAK_FRAC = 0.80     # 直道事件：把 |steer_rate| 达到 max 的 80% 的“最早时刻”作为 anchor
REV_HYBRID_WEAK_COEF = float(os.environ.get("DRIVER_MODEL_REV_HYBRID_WEAK_COEF", "0.60"))
REV_HYBRID_STRONG_COEF = float(os.environ.get("DRIVER_MODEL_REV_HYBRID_STRONG_COEF", "0.40"))
if REV_HYBRID_WEAK_COEF < 0.0 or REV_HYBRID_STRONG_COEF < 0.0:
    raise ValueError("Hybrid reversal sample-weight coefficients must be non-negative")
REV_BRIDGE_MODE = os.environ.get("DRIVER_MODEL_REV_BRIDGE_MODE", "static").strip().lower()
if REV_BRIDGE_MODE not in {"static", "b_to_a_linear"}:
    raise ValueError(
        f"Unsupported DRIVER_MODEL_REV_BRIDGE_MODE={REV_BRIDGE_MODE!r}; expected one of {'static','b_to_a_linear'}"
    )
USE_STRONG_REV_LOSS = (REV_AUX_TARGET == "strong")  # backward-compatible alias
W_FIRSTREV_LOCAL = float(os.environ.get("DRIVER_MODEL_W_FIRSTREV_LOCAL", "0.0"))
FIRSTREV_LOCAL_RADIUS = int(os.environ.get("DRIVER_MODEL_FIRSTREV_LOCAL_RADIUS", "16"))
REV_EPS = REV_EPS_WEAK   # backward-compatible alias

LANE_WIDTH_M     = 3.5   # 车道宽（用于 lateraldistance 解缠）
LANE_JUMP_THR_M  = 1.8   # lateraldistance 跳变检测阈值（约半个车道宽）
LANE_JUMP_MAX_MULTIPLES = 3
LANE_SIGNAL_ABS_CLIP_M = 20.0
LANE_RATE_ABS_CLIP_MPS = 15.0
LANE_ACC_ABS_CLIP_MPS2 = 50.0
EEG_HIST_SEC = 2      # 你现在提取 EEG 事件特征用的 hist2s 文件名后缀
EPS = 1e-6
REV_EPS_WEAK = steer_value_from_rad(0.02)
REV_EPS_STRONG = steer_value_from_rad(0.20)
STRONG_PEAK_THR = steer_value_from_rad(2.0)
STEER_ONSET_THR_ABS = steer_value_from_rad(0.02)
REV_EPS = REV_EPS_WEAK
ROAD_OK_RATIO_THR = 0.7  # use road_type_fixed when ref_nn_ok ratio >= this

SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def compute_epoch_lr(epoch: int, total_epochs: int, base_lr: float, scheduler_name: str, warmup_epochs: int) -> float:
    if scheduler_name == "none":
        return float(base_lr)
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler_name={scheduler_name!r}")
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        if warmup_epochs == 1:
            return float(base_lr)
        progress = float(epoch - 1) / float(max(1, warmup_epochs - 1))
        return float(base_lr) * (0.1 + 0.9 * progress)
    cosine_total = max(1, total_epochs - warmup_epochs)
    cosine_idx = max(0, epoch - warmup_epochs - 1)
    if cosine_total == 1:
        cosine_progress = 1.0
    else:
        cosine_progress = min(1.0, float(cosine_idx) / float(cosine_total - 1))
    eta_min = float(base_lr) * 0.1
    cosine_scale = 0.5 * (1.0 + np.cos(np.pi * cosine_progress))
    return float(eta_min + (float(base_lr) - eta_min) * cosine_scale)


def set_optimizer_lr(optim: torch.optim.Optimizer, lr_value: float) -> None:
    for group in optim.param_groups:
        group["lr"] = float(lr_value)


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    if OPTIMIZER_NAME == "adam":
        return torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if OPTIMIZER_NAME == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Unsupported optimizer {OPTIMIZER_NAME!r}")
