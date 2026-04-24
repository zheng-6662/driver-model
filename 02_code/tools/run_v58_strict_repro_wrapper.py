from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_rng_state_payload() -> dict[str, Any]:
    np_state = np.random.get_state()
    np_hasher = hashlib.sha256()
    np_hasher.update(str(np_state[0]).encode("utf-8"))
    np_hasher.update(np.asarray(np_state[1], dtype=np.uint32).tobytes())
    np_hasher.update(str(int(np_state[2])).encode("utf-8"))
    np_hasher.update(str(int(np_state[3])).encode("utf-8"))
    np_hasher.update(str(float(np_state[4])).encode("utf-8"))

    payload: dict[str, Any] = {
        "python_random_sha256": sha256_bytes(pickle.dumps(random.getstate())),
        "numpy_random_sha256": np_hasher.hexdigest(),
        "torch_cpu_rng_sha256": sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes()),
        "torch_initial_seed": int(torch.initial_seed()),
    }
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
        payload["torch_cuda_rng_sha256"] = [
            sha256_bytes(state.cpu().numpy().tobytes()) for state in cuda_states
        ]
        payload["torch_cuda_initial_seed"] = int(torch.cuda.initial_seed())
    else:
        payload["torch_cuda_rng_sha256"] = []
        payload["torch_cuda_initial_seed"] = None
    return payload


def model_state_digest(model: torch.nn.Module) -> dict[str, Any]:
    hasher = hashlib.sha256()
    tensor_count = 0
    numel = 0
    sum_sq = 0.0
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(value.dtype).encode("utf-8"))
        hasher.update(str(tuple(value.shape)).encode("utf-8"))
        hasher.update(value.numpy().tobytes())
        tensor_count += 1
        numel += int(value.numel())
        if value.is_floating_point():
            sum_sq += float(value.float().pow(2).sum().item())
    return {
        "sha256": hasher.hexdigest(),
        "tensor_count": tensor_count,
        "numel": numel,
        "l2_norm": math.sqrt(sum_sq),
    }


def parameter_list_digest(params: list[torch.Tensor]) -> dict[str, Any]:
    hasher = hashlib.sha256()
    tensor_count = 0
    numel = 0
    sum_sq = 0.0
    for idx, param in enumerate(params):
        value = param.detach().cpu().contiguous()
        hasher.update(str(idx).encode("utf-8"))
        hasher.update(str(value.dtype).encode("utf-8"))
        hasher.update(str(tuple(value.shape)).encode("utf-8"))
        hasher.update(value.numpy().tobytes())
        tensor_count += 1
        numel += int(value.numel())
        if value.is_floating_point():
            sum_sq += float(value.float().pow(2).sum().item())
    return {
        "sha256": hasher.hexdigest(),
        "tensor_count": tensor_count,
        "numel": numel,
        "l2_norm": math.sqrt(sum_sq),
    }


def append_audit_event(captured: dict[str, Path], event: str, payload: dict[str, Any]) -> None:
    run_dir = captured.get("run_dir")
    if run_dir is None:
        return
    append_jsonl(run_dir / "logs" / "strict_repro_audit.jsonl", {"event": event, **payload})


def summarize_history(history_csv: Path) -> dict[str, Any] | None:
    if not history_csv.exists():
        return None
    rows: list[dict[str, Any]] = []
    with history_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None

    def to_float(row: dict[str, Any], key: str) -> float:
        return float(row[key])

    best_row = min(rows, key=lambda row: to_float(row, "val_loss"))
    first_row = rows[0]
    last_row = rows[-1]
    return {
        "row_count": len(rows),
        "best_epoch": int(float(best_row["epoch"])),
        "best_val_loss": to_float(best_row, "val_loss"),
        "first_epoch": int(float(first_row["epoch"])),
        "first_train_loss": to_float(first_row, "train_loss"),
        "first_val_loss": to_float(first_row, "val_loss"),
        "last_epoch": int(float(last_row["epoch"])),
        "last_train_loss": to_float(last_row, "train_loss"),
        "last_val_loss": to_float(last_row, "val_loss"),
    }


def apply_env(args) -> None:
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = args.cublas_workspace_config
    os.environ["DRIVER_MODEL_RESULT_ROOT"] = str(Path(args.result_root).resolve())
    if args.protocol_config_path:
        os.environ["DRIVER_MODEL_PROTOCOL_CONFIG"] = str(Path(args.protocol_config_path).resolve())
    if args.frozen_split_path:
        os.environ["DRIVER_MODEL_FROZEN_SPLIT"] = str(Path(args.frozen_split_path).resolve())
    if args.protocol_split_summary_path:
        os.environ["DRIVER_MODEL_PROTOCOL_SPLIT_SUMMARY"] = str(Path(args.protocol_split_summary_path).resolve())
    if args.protocol_dir:
        os.environ["DRIVER_MODEL_PROTOCOL_DIR"] = str(Path(args.protocol_dir).resolve())
    elif args.protocol_config_path:
        os.environ["DRIVER_MODEL_PROTOCOL_DIR"] = str(Path(args.protocol_config_path).resolve().parent)
    if args.driver_root:
        os.environ["DRIVER_MODEL_ROOT"] = str(Path(args.driver_root).resolve())
    if args.style_csv:
        os.environ["DRIVER_MODEL_STYLE_CSV"] = str(Path(args.style_csv).resolve())
    if args.steer_angle_unit:
        os.environ["DRIVER_MODEL_STEER_ANGLE_UNIT"] = args.steer_angle_unit


def apply_determinism(warn_only: bool) -> dict[str, Any]:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    result: dict[str, Any] = {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms_enabled": None,
        "deterministic_algorithms_warn_only": warn_only,
    }
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            result["deterministic_algorithms_enabled"] = torch.are_deterministic_algorithms_enabled()
    return result


def apply_attention_backend(args) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requested_force_math_sdp": bool(args.force_math_sdp),
        "flash_enabled": None,
        "mem_efficient_enabled": None,
        "math_enabled": None,
    }
    cuda_backends = getattr(torch.backends, "cuda", None)
    if cuda_backends is None:
        return result

    if args.force_math_sdp:
        if hasattr(cuda_backends, "enable_flash_sdp"):
            cuda_backends.enable_flash_sdp(False)
        if hasattr(cuda_backends, "enable_mem_efficient_sdp"):
            cuda_backends.enable_mem_efficient_sdp(False)
        if hasattr(cuda_backends, "enable_math_sdp"):
            cuda_backends.enable_math_sdp(True)

    if hasattr(cuda_backends, "flash_sdp_enabled"):
        result["flash_enabled"] = bool(cuda_backends.flash_sdp_enabled())
    if hasattr(cuda_backends, "mem_efficient_sdp_enabled"):
        result["mem_efficient_enabled"] = bool(cuda_backends.mem_efficient_sdp_enabled())
    if hasattr(cuda_backends, "math_sdp_enabled"):
        result["math_enabled"] = bool(cuda_backends.math_sdp_enabled())
    return result


def manual_linear_upsample_1d_align_corners(input_tensor: torch.Tensor, size: int) -> torch.Tensor:
    if input_tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor for manual 1D upsample, got shape={tuple(input_tensor.shape)}")
    target = int(size)
    if target <= 0:
        raise ValueError(f"size must be positive, got {target}")
    length_in = int(input_tensor.shape[-1])
    if length_in == target:
        return input_tensor.clone()
    if target == 1:
        return input_tensor[..., :1].clone()
    if length_in == 1:
        return input_tensor.expand(*input_tensor.shape[:-1], target)

    device = input_tensor.device
    weight_dtype = input_tensor.dtype if input_tensor.is_floating_point() else torch.float32
    out_pos = torch.arange(target, device=device, dtype=weight_dtype)
    scale = (length_in - 1) / (target - 1)
    src_pos = out_pos * scale
    left_idx = torch.floor(src_pos).to(torch.long)
    right_idx = torch.clamp(left_idx + 1, max=length_in - 1)
    weight_right = (src_pos - left_idx.to(weight_dtype)).view(1, 1, target)
    weight_left = 1.0 - weight_right

    left_vals = input_tensor.index_select(-1, left_idx)
    right_vals = input_tensor.index_select(-1, right_idx)
    return left_vals * weight_left + right_vals * weight_right


def collect_env_payload(args, script_path: Path, determinism_info: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "wrapper_script": str(Path(__file__).resolve()),
        "target_script": str(script_path),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "determinism": determinism_info,
        "env": {
            key: os.environ.get(key)
            for key in [
                "PYTHONIOENCODING",
                "PYTHONUTF8",
                "PYTHONUNBUFFERED",
                "CUBLAS_WORKSPACE_CONFIG",
                "DRIVER_MODEL_RESULT_ROOT",
                "DRIVER_MODEL_PROTOCOL_DIR",
                "DRIVER_MODEL_PROTOCOL_CONFIG",
                "DRIVER_MODEL_FROZEN_SPLIT",
                "DRIVER_MODEL_PROTOCOL_SPLIT_SUMMARY",
                "DRIVER_MODEL_ROOT",
                "DRIVER_MODEL_STYLE_CSV",
                "DRIVER_MODEL_STEER_ANGLE_UNIT",
            ]
        },
        "wrapper_args": vars(args),
    }
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Run a v5.8 training script under stricter reproducibility controls.")
    parser.add_argument("--script-path", required=True, help="Training script to run.")
    parser.add_argument("--result-root", required=True, help="Output root passed through DRIVER_MODEL_RESULT_ROOT.")
    parser.add_argument("--protocol-config-path", default=None)
    parser.add_argument("--frozen-split-path", default=None)
    parser.add_argument("--protocol-split-summary-path", default=None)
    parser.add_argument("--protocol-dir", default=None)
    parser.add_argument("--driver-root", default=None)
    parser.add_argument("--style-csv", default=None)
    parser.add_argument("--steer-angle-unit", default=None, choices=["rad", "deg"])
    parser.add_argument("--cublas-workspace-config", default=":4096:8")
    parser.add_argument("--capture-train-batches", type=int, default=3, help="How many leading train batches to log per epoch.")
    parser.add_argument("--capture-optimizer-steps", type=int, default=0, help="How many leading optimizer steps to hash.")
    parser.add_argument("--override-epochs", type=int, default=None, help="Override module EPOCHS without changing the target script.")
    parser.add_argument("--force-device", default=None, choices=["cpu", "cuda"], help="Force the effective runtime device seen by the target script.")
    parser.add_argument("--force-math-sdp", action="store_true", help="Disable flash/memory-efficient SDPA and force the math backend when available.")
    parser.add_argument("--replace-linear-upsample", action="store_true", help="Audit-only: replace 1D linear align_corners interpolate with a manual implementation.")
    parser.add_argument("--warn-only-deterministic", action="store_true")
    return parser.parse_args()


def serialize_indices(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, int):
        return [int(value)]
    return [int(x) for x in value]


def main():
    args = parse_args()
    script_path = Path(args.script_path).resolve()
    orig_cuda_is_available = torch.cuda.is_available
    if args.force_device == "cpu":
        torch.cuda.is_available = lambda: False
    apply_env(args)
    determinism_info = apply_determinism(warn_only=args.warn_only_deterministic)
    attention_backend_info = apply_attention_backend(args)
    env_payload = collect_env_payload(args, script_path, determinism_info)
    env_payload["attention_backend"] = attention_backend_info
    env_payload["force_device"] = args.force_device
    module = load_module(script_path, f"strict_repro_target_{abs(hash(str(script_path)))}")
    if args.override_epochs is not None and hasattr(module, "EPOCHS"):
        module.EPOCHS = int(args.override_epochs)
        env_payload["override_epochs_applied"] = int(args.override_epochs)
    env_payload["replace_linear_upsample"] = bool(args.replace_linear_upsample)

    orig_interpolate = None
    if args.replace_linear_upsample and hasattr(module, "F") and hasattr(module.F, "interpolate"):
        orig_interpolate = module.F.interpolate

        def patched_interpolate(input_tensor, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None, antialias=False):
            if (
                mode == "linear"
                and align_corners is True
                and scale_factor is None
                and size is not None
                and recompute_scale_factor is None
                and not antialias
                and isinstance(input_tensor, torch.Tensor)
                and input_tensor.ndim == 3
            ):
                target = int(size[0]) if isinstance(size, (tuple, list)) else int(size)
                return manual_linear_upsample_1d_align_corners(input_tensor, target)
            return orig_interpolate(
                input_tensor,
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
                recompute_scale_factor=recompute_scale_factor,
                antialias=antialias,
            )

        module.F.interpolate = patched_interpolate

    captured: dict[str, Path] = {}
    audit_state = {"train_epoch": 0}
    if hasattr(module, "make_run_dir"):
        orig_make_run_dir = module.make_run_dir

        def wrapped_make_run_dir(prefix="TRAIN_V5_4_STATECOND_REV"):
            run_dir = Path(orig_make_run_dir(prefix))
            captured["run_dir"] = run_dir
            save_json(run_dir / "strict_repro_env.json", env_payload)
            append_audit_event(
                captured,
                "run_dir_created",
                {
                    "rng": collect_rng_state_payload(),
                },
            )
            return run_dir

        module.make_run_dir = wrapped_make_run_dir

    if hasattr(module, "Past2FutureMultiTaskRoadPreview"):
        orig_model_cls = module.Past2FutureMultiTaskRoadPreview

        class AuditModel(orig_model_cls):
            def __init__(self, *model_args, **model_kwargs):
                super().__init__(*model_args, **model_kwargs)
                append_audit_event(
                    captured,
                    "model_initialized",
                    {
                        "rng": collect_rng_state_payload(),
                        "model": model_state_digest(self),
                    },
                )

        module.Past2FutureMultiTaskRoadPreview = AuditModel

    if hasattr(module, "DataLoader"):
        orig_data_loader = module.DataLoader

        class AuditDataLoader(orig_data_loader):
            def __init__(self, *loader_args, **loader_kwargs):
                self._audit_capture = bool(loader_kwargs.get("shuffle", False)) and args.capture_train_batches > 0
                super().__init__(*loader_args, **loader_kwargs)

            def __iter__(self):
                iterator = super().__iter__()
                if not self._audit_capture:
                    for batch in iterator:
                        yield batch
                    return

                audit_state["train_epoch"] += 1
                epoch = int(audit_state["train_epoch"])
                captured_batches: list[dict[str, Any]] = []
                for batch_idx, batch in enumerate(iterator, start=1):
                    if batch_idx <= args.capture_train_batches and isinstance(batch, dict) and "idx" in batch:
                        captured_batches.append(
                            {
                                "batch": int(batch_idx),
                                "indices": serialize_indices(batch["idx"]),
                            }
                        )
                    yield batch

                run_dir = captured.get("run_dir")
                if run_dir is not None and captured_batches:
                    append_jsonl(
                        run_dir / "logs" / "train_batch_order_audit.jsonl",
                        {
                            "epoch": epoch,
                            "captured_batches": captured_batches,
                        },
                    )

        module.DataLoader = AuditDataLoader

    orig_adam = torch.optim.Adam

    class AuditAdam(orig_adam):
        def __init__(self, *optim_args, **optim_kwargs):
            super().__init__(*optim_args, **optim_kwargs)
            self._audit_step = 0
            append_audit_event(
                captured,
                "optimizer_initialized",
                {
                    "defaults": {
                        "lr": float(self.defaults.get("lr", 0.0)),
                        "betas": list(self.defaults.get("betas", ())),
                        "eps": float(self.defaults.get("eps", 0.0)),
                        "weight_decay": float(self.defaults.get("weight_decay", 0.0)),
                    },
                    "rng": collect_rng_state_payload(),
                },
            )

        def step(self, closure=None):
            if self._audit_step < args.capture_optimizer_steps:
                params = [
                    p for group in self.param_groups for p in group["params"]
                    if isinstance(p, torch.Tensor)
                ]
                pre_model = parameter_list_digest(params)
                pre_rng = collect_rng_state_payload()
            else:
                pre_model = None
                pre_rng = None

            out = super().step(closure)
            self._audit_step += 1

            if self._audit_step <= args.capture_optimizer_steps:
                params = [
                    p for group in self.param_groups for p in group["params"]
                    if isinstance(p, torch.Tensor)
                ]
                post_model = parameter_list_digest(params)
                append_audit_event(
                    captured,
                    "optimizer_step",
                    {
                        "step": int(self._audit_step),
                        "pre_step_rng": pre_rng,
                        "post_step_rng": collect_rng_state_payload(),
                        "pre_step_params": pre_model,
                        "post_step_params": post_model,
                    },
                )
            return out

    torch.optim.Adam = AuditAdam
    if hasattr(module, "torch") and hasattr(module.torch, "optim"):
        module.torch.optim.Adam = AuditAdam

    if not hasattr(module, "main"):
        raise RuntimeError(f"Target script {script_path} does not expose main()")

    try:
        module.main()
    finally:
        torch.optim.Adam = orig_adam
        torch.cuda.is_available = orig_cuda_is_available
        if orig_interpolate is not None and hasattr(module, "F"):
            module.F.interpolate = orig_interpolate
        if hasattr(module, "torch") and hasattr(module.torch, "optim"):
            module.torch.optim.Adam = orig_adam
        if hasattr(module, "torch") and hasattr(module.torch, "cuda"):
            module.torch.cuda.is_available = orig_cuda_is_available

    run_dir = captured.get("run_dir")
    if run_dir is not None:
        summary_payload: dict[str, Any] = {
            "run_dir": str(run_dir),
            "strict_repro_env": str(run_dir / "strict_repro_env.json"),
            "final_rng": collect_rng_state_payload(),
        }
        history_summary = summarize_history(run_dir / "loss_history.csv")
        if history_summary is not None:
            summary_payload["history"] = history_summary
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists():
            summary_payload["checkpoint_hashes"] = {
                path.name: file_sha256(path)
                for path in sorted(checkpoints_dir.glob("*.pth"))
            }
        save_json(run_dir / "strict_repro_audit_summary.json", summary_payload)
        print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"warning": "run_dir not captured; strict_repro_env.json may not have been written"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
