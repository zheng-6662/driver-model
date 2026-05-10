from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from baseline_eval_primary_aux import classify_eval_morphology
from conditioned_trajectory_head import ConditionedTrajectoryHead
from event_head import EVENT_SCHEMA_KEYS, EventHead
from event_targets import EventTargetConfig, sequence_to_event_targets
from future_steer_speed_subjectsplit_masked import FUTURE_LEN, WIN_LEN


RESPONSE_TYPE_KEYS = (
    "response_high_amp",
    "response_reverse",
    "response_multi",
    "response_tail_behavior",
    "response_late_peak",
    "response_hard_case",
)
RESPONSE_CANDIDATE_CLASS_KEY = "response_candidate_class"


def build_event_schema_targets(
    y_pool: np.ndarray,
    mask_pool: np.ndarray,
    future_len: int = FUTURE_LEN,
    event_bin_size: int = 20,
    turn_frac: float = 0.20,
    turn_min_amp: float = 0.015,
    reversal_frac: float = 0.30,
    reversal_min_rate: float = 0.002,
) -> dict[str, np.ndarray]:
    cfg = EventTargetConfig(
        future_len=int(future_len),
        bin_size=int(event_bin_size),
        turn_frac=float(turn_frac),
        turn_min_amp=float(turn_min_amp),
        reversal_frac=float(reversal_frac),
        reversal_min_rate=float(reversal_min_rate),
    )
    rows = [sequence_to_event_targets(y_pool[i, :, 0], int(mask_pool[i].sum()), config=cfg) for i in range(int(y_pool.shape[0]))]
    targets: dict[str, np.ndarray] = {}
    for key in EVENT_SCHEMA_KEYS:
        if key.endswith("_has"):
            targets[key] = np.asarray([row[key] for row in rows], dtype=np.float32)
        else:
            targets[key] = np.asarray([row[key] for row in rows], dtype=np.int64)
    return targets


def build_response_type_targets(
    y_pool: np.ndarray,
    mask_pool: np.ndarray,
    amp_threshold: float = 0.30,
    late_peak_threshold_s: float = 1.20,
    tail_start_frac: float = 0.70,
) -> dict[str, np.ndarray]:
    rows: list[dict[str, int]] = []
    dt = 2.0 / max(int(FUTURE_LEN), 1)
    for i in range(int(y_pool.shape[0])):
        valid_len = int(mask_pool[i].sum())
        valid_len = max(1, min(valid_len, int(y_pool.shape[1])))
        y = np.asarray(y_pool[i, :valid_len, 0], dtype=np.float64)
        peak_idx = int(np.argmax(np.abs(y)))
        peak = float(y[peak_idx])
        abs_peak = abs(peak)
        morphology = classify_eval_morphology(y, valid_len)
        tail_start = max(0, min(valid_len - 1, int(np.floor(float(tail_start_frac) * valid_len))))
        tail_mean = float(np.mean(y[tail_start:])) if tail_start < valid_len else float(y[-1])
        if abs(tail_mean) < max(0.06, 0.25 * abs_peak):
            tail_behavior = 0
        elif np.sign(tail_mean) == np.sign(peak):
            tail_behavior = 1
        else:
            tail_behavior = 2
        high_amp = int(abs_peak >= float(amp_threshold))
        reverse = int(morphology == "reverse_correction")
        multi = int(morphology == "multi_correction")
        late_peak = int((peak_idx * dt) >= float(late_peak_threshold_s))
        hard_case = int(high_amp and (reverse or multi or late_peak or tail_behavior != 0))
        if reverse:
            candidate_class = 2
        elif multi or tail_behavior == 2:
            candidate_class = 3
        elif high_amp or late_peak:
            candidate_class = 1
        else:
            candidate_class = 0
        rows.append(
            {
                "response_high_amp": high_amp,
                "response_reverse": reverse,
                "response_multi": multi,
                "response_tail_behavior": tail_behavior,
                "response_late_peak": late_peak,
                "response_hard_case": hard_case,
                RESPONSE_CANDIDATE_CLASS_KEY: candidate_class,
            }
        )
    return {
        "response_high_amp": np.asarray([row["response_high_amp"] for row in rows], dtype=np.float32),
        "response_reverse": np.asarray([row["response_reverse"] for row in rows], dtype=np.float32),
        "response_multi": np.asarray([row["response_multi"] for row in rows], dtype=np.float32),
        "response_tail_behavior": np.asarray([row["response_tail_behavior"] for row in rows], dtype=np.int64),
        "response_late_peak": np.asarray([row["response_late_peak"] for row in rows], dtype=np.float32),
        "response_hard_case": np.asarray([row["response_hard_case"] for row in rows], dtype=np.float32),
        RESPONSE_CANDIDATE_CLASS_KEY: np.asarray([row[RESPONSE_CANDIDATE_CLASS_KEY] for row in rows], dtype=np.int64),
    }


class EventConditionedDataset(Dataset):
    def __init__(
        self,
        X_norm: np.ndarray,
        y_pool: np.ndarray,
        curve_pool: np.ndarray,
        ctx_pool: np.ndarray,
        mask_pool: np.ndarray,
        norm_stats: dict[str, np.ndarray],
        event_targets: dict[str, np.ndarray],
        meta_df: pd.DataFrame | None = None,
        distill_target_pool: np.ndarray | None = None,
        distill_weight_pool: np.ndarray | None = None,
        response_targets: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.src = X_norm.astype(np.float32)
        self.y = y_pool.astype(np.float32)
        self.curve = curve_pool.astype(np.float32)
        self.ctx = ctx_pool.astype(np.float32)
        self.mask = mask_pool.astype(np.float32)
        self.event_targets = event_targets
        self.meta_df = None if meta_df is None else meta_df.reset_index(drop=True).copy()
        self.distill_target = None if distill_target_pool is None else distill_target_pool.astype(np.float32)
        self.distill_weight = None if distill_weight_pool is None else distill_weight_pool.astype(np.float32)
        self.response_targets = response_targets
        if self.distill_target is not None and int(self.distill_target.shape[0]) != int(self.src.shape[0]):
            raise ValueError(
                f"distill_target_pool first dimension {self.distill_target.shape[0]} "
                f"does not match dataset length {self.src.shape[0]}"
            )
        if self.distill_weight is not None and int(self.distill_weight.shape[0]) != int(self.src.shape[0]):
            raise ValueError(
                f"distill_weight_pool first dimension {self.distill_weight.shape[0]} "
                f"does not match dataset length {self.src.shape[0]}"
            )
        if self.response_targets is not None:
            for key in RESPONSE_TYPE_KEYS:
                if key not in self.response_targets:
                    raise ValueError(f"response_targets missing key: {key}")
                if int(self.response_targets[key].shape[0]) != int(self.src.shape[0]):
                    raise ValueError(
                        f"response_targets[{key}] first dimension {self.response_targets[key].shape[0]} "
                        f"does not match dataset length {self.src.shape[0]}"
                    )

        self.y_norm = ((self.y - norm_stats["y_mean"].reshape(1, 1, -1)) / norm_stats["y_std"].reshape(1, 1, -1)).astype(np.float32)
        self.curve_norm = (
            (self.curve - norm_stats["curve_mean"].reshape(1,)) / norm_stats["curve_std"].reshape(1,)
        ).astype(np.float32)
        self.ctx_norm = (
            (self.ctx - norm_stats["ctx_mean"].reshape(1, -1)) / norm_stats["ctx_std"].reshape(1, -1)
        ).astype(np.float32)

    def __len__(self) -> int:
        return int(self.src.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {
            "src": torch.from_numpy(self.src[idx]),
            "y_true": torch.from_numpy(self.y_norm[idx]),
            "curve_norm": torch.from_numpy(self.curve_norm[idx]),
            "ctx": torch.from_numpy(self.ctx_norm[idx]),
            "ctx_raw": torch.from_numpy(self.ctx[idx]),
            "event_mask": torch.from_numpy(self.mask[idx]),
        }
        for key in EVENT_SCHEMA_KEYS:
            arr = self.event_targets[key]
            if key.endswith("_has"):
                item[key] = torch.tensor(arr[idx], dtype=torch.float32)
            else:
                item[key] = torch.tensor(arr[idx], dtype=torch.long)
        if self.distill_target is not None:
            item["distill_y_soft"] = torch.from_numpy(self.distill_target[idx])
        if self.distill_weight is not None:
            item["distill_sample_weight"] = torch.tensor(self.distill_weight[idx], dtype=torch.float32)
        if self.response_targets is not None:
            for key in RESPONSE_TYPE_KEYS:
                arr = self.response_targets[key]
                if key == "response_tail_behavior":
                    item[key] = torch.tensor(arr[idx], dtype=torch.long)
                else:
                    item[key] = torch.tensor(arr[idx], dtype=torch.float32)
            if RESPONSE_CANDIDATE_CLASS_KEY in self.response_targets:
                item[RESPONSE_CANDIDATE_CLASS_KEY] = torch.tensor(
                    self.response_targets[RESPONSE_CANDIDATE_CLASS_KEY][idx],
                    dtype=torch.long,
                )
        return item


class ResponseTypeHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 96) -> None:
        super().__init__()
        hidden = max(int(hidden_dim), 16)
        self.shared = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.high_amp = nn.Linear(hidden, 1)
        self.reverse = nn.Linear(hidden, 1)
        self.multi = nn.Linear(hidden, 1)
        self.tail_behavior = nn.Linear(hidden, 3)
        self.late_peak = nn.Linear(hidden, 1)
        self.hard_case = nn.Linear(hidden, 1)
        self.summary_dim = 8

    def forward(self, pooled_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(pooled_latent)
        return {
            "response_high_amp_logit": self.high_amp(h).squeeze(-1),
            "response_reverse_logit": self.reverse(h).squeeze(-1),
            "response_multi_logit": self.multi(h).squeeze(-1),
            "response_tail_behavior_logits": self.tail_behavior(h),
            "response_late_peak_logit": self.late_peak(h).squeeze(-1),
            "response_hard_case_logit": self.hard_case(h).squeeze(-1),
        }

    def predicted_summary(self, logits: dict[str, torch.Tensor]) -> torch.Tensor:
        tail_probs = torch.softmax(logits["response_tail_behavior_logits"], dim=-1)
        return torch.cat(
            [
                torch.sigmoid(logits["response_high_amp_logit"]).unsqueeze(-1),
                torch.sigmoid(logits["response_reverse_logit"]).unsqueeze(-1),
                torch.sigmoid(logits["response_multi_logit"]).unsqueeze(-1),
                tail_probs,
                torch.sigmoid(logits["response_late_peak_logit"]).unsqueeze(-1),
                torch.sigmoid(logits["response_hard_case_logit"]).unsqueeze(-1),
            ],
            dim=-1,
        )


class SharedHistoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        enc_layers: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, WIN_LEN, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.pool_score = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(src) + self.pos[:, : src.shape[1], :]
        memory = self.encoder(self.dropout(h))
        score = self.pool_score(memory)
        alpha = torch.softmax(score, dim=1)
        pooled = torch.sum(alpha * memory, dim=1)
        return memory, pooled


class EventConditionedTrajectoryModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        future_len: int = FUTURE_LEN,
        event_bin_size: int = 20,
        d_model: int = 128,
        nhead: int = 2,
        enc_layers: int = 2,
        dec_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        event_embed_dim: int = 96,
        out_dim: int = 2,
        conditioning_mode: str = "baseline",
        structure_width: float = 0.065,
        gate_temperature: float = 0.040,
        event_residual_scale: float = 1.0,
        enable_response_type_head: bool = False,
        enable_response_type_condition: bool = False,
        response_type_hidden_dim: int = 96,
        num_trajectory_candidates: int = 1,
        candidate_delta_scale: float = 1.0,
        candidate_base_mode: str = "learned_delta",
        candidate_prototypes: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.encoder = SharedHistoryEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            enc_layers=enc_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.event_head = EventHead(
            d_model=d_model,
            future_len=future_len,
            event_bin_size=event_bin_size,
            event_embed_dim=event_embed_dim,
        )
        self.privileged_event_proj = nn.Linear(event_embed_dim, event_embed_dim)
        self.enable_response_type_head = bool(enable_response_type_head or enable_response_type_condition)
        self.enable_response_type_condition = bool(enable_response_type_condition)
        self.response_type_head = (
            ResponseTypeHead(d_model=d_model, hidden_dim=int(response_type_hidden_dim))
            if self.enable_response_type_head
            else None
        )
        self.traj_head = ConditionedTrajectoryHead(
            d_model=d_model,
            context_dim=context_dim,
            event_embed_dim=event_embed_dim,
            future_len=future_len,
            out_dim=out_dim,
            nhead=nhead,
            dec_layers=dec_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            conditioning_mode=conditioning_mode,
            structure_width=structure_width,
            gate_temperature=gate_temperature,
            event_residual_scale=event_residual_scale,
            response_summary_dim=(
                self.response_type_head.summary_dim
                if self.enable_response_type_condition and self.response_type_head is not None
                else 0
            ),
            num_trajectory_candidates=int(num_trajectory_candidates),
            candidate_delta_scale=float(candidate_delta_scale),
            candidate_base_mode=str(candidate_base_mode),
            candidate_prototypes=candidate_prototypes,
        )

    def forward(
        self,
        src: torch.Tensor,
        ctx: torch.Tensor,
        curve_norm: torch.Tensor,
        event_teacher: dict[str, torch.Tensor] | None = None,
        privileged_event_teacher: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        memory, pooled = self.encoder(src)
        event_logits = self.event_head(pooled)
        cond_emb, cond_meta = self.event_head.build_condition_embedding(event_logits, teacher_events=event_teacher)
        if privileged_event_teacher is not None:
            cond_emb = cond_emb + self.privileged_event_proj(privileged_event_teacher)
        response_logits = None
        response_summary = None
        if self.response_type_head is not None:
            response_logits = self.response_type_head(pooled)
            response_summary = self.response_type_head.predicted_summary(response_logits)
        y_hat, traj_extras = self.traj_head(
            memory=memory,
            pooled_latent=pooled,
            ctx=ctx,
            curve_norm=curve_norm,
            event_condition_emb=cond_emb,
            event_condition_summary=cond_meta["summary"],
            response_condition_summary=response_summary if self.enable_response_type_condition else None,
        )
        extras: dict[str, Any] = {
            "event_logits": event_logits,
            "event_condition_embedding": cond_emb,
            "event_condition_summary": cond_meta["summary"],
            "event_condition_predicted_summary": cond_meta["predicted_summary"],
            "event_condition_source": cond_meta["source"],
            "privileged_event_teacher_used": bool(privileged_event_teacher is not None),
            "response_type_logits": response_logits,
            "response_type_condition_used": bool(self.enable_response_type_condition and response_summary is not None),
        }
        extras.update(traj_extras)
        return y_hat, extras


def build_event_teacher_from_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: batch[key].to(device=device) for key in EVENT_SCHEMA_KEYS}


def masked_mse(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sq = (pred - true) ** 2
    denom = torch.clamp(mask.sum(), min=1.0)
    return (sq * mask).sum() / denom


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if valid_mask is None:
        return loss.mean()
    denom = torch.clamp(valid_mask.sum(), min=1.0)
    return (loss * valid_mask).sum() / denom


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    safe_target = target.clone()
    if valid_mask is not None:
        invalid = valid_mask <= 0
        if invalid.any():
            safe_target = safe_target.masked_fill(invalid, 0)
    loss = F.cross_entropy(logits, safe_target, reduction="none")
    if valid_mask is None:
        return loss.mean()
    denom = torch.clamp(valid_mask.sum(), min=1.0)
    return (loss * valid_mask).sum() / denom


@dataclass
class EventLossBreakdown:
    total: torch.Tensor
    turn_has: torch.Tensor
    turn_idx: torch.Tensor
    turn_bin: torch.Tensor
    turn_dir: torch.Tensor
    reversal_has: torch.Tensor
    reversal_idx: torch.Tensor
    reversal_bin: torch.Tensor
    peak_idx: torch.Tensor
    peak_bin: torch.Tensor
    peak_dir: torch.Tensor


def compute_event_loss(batch: dict[str, torch.Tensor], event_logits: dict[str, torch.Tensor]) -> EventLossBreakdown:
    turn_has_target = batch["first_major_turn_onset_has"]
    reversal_has_target = batch["first_reversal_has"]
    valid_peak = (batch["event_mask"].sum(dim=1) > 0).to(dtype=turn_has_target.dtype)

    turn_has = _masked_bce(event_logits["first_major_turn_onset_has_logit"], turn_has_target)
    turn_idx = _masked_ce(
        event_logits["first_major_turn_onset_idx_logits"],
        batch["first_major_turn_onset_idx"],
        valid_mask=turn_has_target,
    )
    turn_bin = _masked_ce(
        event_logits["first_major_turn_onset_bin_logits"],
        batch["first_major_turn_onset_bin"],
        valid_mask=turn_has_target,
    )
    turn_dir = _masked_ce(
        event_logits["first_major_turn_direction_logits"],
        batch["first_major_turn_direction"],
        valid_mask=turn_has_target,
    )

    reversal_has = _masked_bce(event_logits["first_reversal_has_logit"], reversal_has_target)
    reversal_idx = _masked_ce(
        event_logits["first_reversal_idx_logits"],
        batch["first_reversal_idx"],
        valid_mask=reversal_has_target,
    )
    reversal_bin = _masked_ce(
        event_logits["first_reversal_bin_logits"],
        batch["first_reversal_bin"],
        valid_mask=reversal_has_target,
    )

    peak_idx = _masked_ce(event_logits["main_peak_idx_logits"], batch["main_peak_idx"], valid_mask=valid_peak)
    peak_bin = _masked_ce(event_logits["main_peak_bin_logits"], batch["main_peak_bin"], valid_mask=valid_peak)
    peak_dir = _masked_ce(event_logits["main_peak_direction_logits"], batch["main_peak_direction"], valid_mask=valid_peak)

    total = (
        0.10 * turn_has
        + 0.14 * turn_idx
        + 0.10 * turn_bin
        + 0.10 * turn_dir
        + 0.10 * reversal_has
        + 0.12 * reversal_idx
        + 0.10 * reversal_bin
        + 0.10 * peak_idx
        + 0.07 * peak_bin
        + 0.07 * peak_dir
    )
    return EventLossBreakdown(
        total=total,
        turn_has=turn_has,
        turn_idx=turn_idx,
        turn_bin=turn_bin,
        turn_dir=turn_dir,
        reversal_has=reversal_has,
        reversal_idx=reversal_idx,
        reversal_bin=reversal_bin,
        peak_idx=peak_idx,
        peak_bin=peak_bin,
        peak_dir=peak_dir,
    )


def compute_response_type_loss(batch: dict[str, torch.Tensor], response_logits: dict[str, torch.Tensor] | None) -> torch.Tensor:
    if response_logits is None:
        return torch.zeros((), dtype=torch.float32, device=batch["src"].device)
    if not all(key in batch for key in RESPONSE_TYPE_KEYS):
        return torch.zeros((), dtype=torch.float32, device=batch["src"].device)
    binary_terms = [
        ("response_high_amp_logit", "response_high_amp"),
        ("response_reverse_logit", "response_reverse"),
        ("response_multi_logit", "response_multi"),
        ("response_late_peak_logit", "response_late_peak"),
        ("response_hard_case_logit", "response_hard_case"),
    ]
    losses = [
        F.binary_cross_entropy_with_logits(
            response_logits[logit_key],
            batch[target_key].to(dtype=response_logits[logit_key].dtype),
        )
        for logit_key, target_key in binary_terms
    ]
    losses.append(F.cross_entropy(response_logits["response_tail_behavior_logits"], batch["response_tail_behavior"].long()))
    return torch.stack(losses).mean()


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def subset_array_dict(arr_dict: dict[str, np.ndarray], indices: list[int]) -> dict[str, np.ndarray]:
    return {k: v[indices] for k, v in arr_dict.items()}
