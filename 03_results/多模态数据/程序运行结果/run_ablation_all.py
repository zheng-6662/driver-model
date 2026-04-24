import subprocess
import sys
from pathlib import Path

# 改成你的 ablation 脚本文件名
ABLATION_SCRIPT = Path(__file__).with_name(
    "future_steer_event_rollpeak_transformer_v5_6_2_anchorlog_FIXED_v5_straightrev_ABLATION.py"
)

def run_one(mode: str, seed: int = 2026):
    cmd = [
        sys.executable,
        str(ABLATION_SCRIPT),
        "--ablation", mode,
        "--seed", str(seed),
    ]
    print("\n==============================")
    print("Running:", " ".join(cmd))
    print("==============================\n")
    subprocess.check_call(cmd)

if __name__ == "__main__":
    # 串行依次跑 A0~A3
    for m in ["A0", "A1", "A2", "A3"]:
        run_one(m, seed=2026)
