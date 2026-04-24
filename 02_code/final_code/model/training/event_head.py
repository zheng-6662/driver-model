from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


EVENT_SCHEMA_KEYS = (
    "first_major_turn_onset_has",
    "first_major_turn_onset_idx",
    "first_major_turn_onset_bin",
    "first_major_turn_direction",
    "first_reversal_has",
    "first_reversal_idx",
    "first_reversal_bin",
    "main_peak_idx",
    "main_peak_bin",
    "main_peak_direction",
)


def _expected_normalized_index(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    grid = torch.linspace(0.0, 1.0, logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return (probs * grid.unsqueeze(0)).sum(dim=-1, keepdim=True)


def _expected_normalized_bin(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    grid = torch.linspace(0.0, 1.0, logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return (probs * grid.unsqueeze(0)).sum(dim=-1, keepdim=True)


def _direction_signed_prob(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return (probs[:, 1:2] - probs[:, 0:1]).contiguous()


class EventHead(nn.Module):
    """Predicts key structure events and builds a compact event condition embedding."""

    def __init__(
        self,
        d_model: int,
        future_len: int,
        event_bin_size: int = 20,
        event_embed_dim: int = 96,
    ) -> None:
        super().__init__()
        self.future_len = int(future_len)
        self.event_bin_size = int(event_bin_size)
        self.num_bins = max((self.future_len + self.event_bin_size - 1) // self.event_bin_size, 1)

        self.turn_has_head = nn.Linear(d_model, 1)
        self.turn_idx_head = nn.Linear(d_model, self.future_len)
        self.turn_bin_head = nn.Linear(d_model, self.num_bins)
        self.turn_dir_head = nn.Linear(d_model, 2)

        self.reversal_has_head = nn.Linear(d_model, 1)
        self.reversal_idx_head = nn.Linear(d_model, self.future_len)
        self.reversal_bin_head = nn.Linear(d_model, self.num_bins)

        self.peak_idx_head = nn.Linear(d_model, self.future_len)
        self.peak_bin_head = nn.Linear(d_model, self.num_bins)
        self.peak_dir_head = nn.Linear(d_model, 2)

        self.summary_dim = 10
        self.summary_mlp = nn.Sequential(
            nn.Linear(self.summary_dim, max(event_embed_dim, self.summary_dim)),
            nn.GELU(),
            nn.Linear(max(event_embed_dim, self.summary_dim), event_embed_dim),
        )

    def forward(self, pooled_latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "first_major_turn_onset_has_logit": self.turn_has_head(pooled_latent).squeeze(-1),
            "first_major_turn_onset_idx_logits": self.turn_idx_head(pooled_latent),
            "first_major_turn_onset_bin_logits": self.turn_bin_head(pooled_latent),
            "first_major_turn_direction_logits": self.turn_dir_head(pooled_latent),
            "first_reversal_has_logit": self.reversal_has_head(pooled_latent).squeeze(-1),
            "first_reversal_idx_logits": self.reversal_idx_head(pooled_latent),
            "first_reversal_bin_logits": self.reversal_bin_head(pooled_latent),
            "main_peak_idx_logits": self.peak_idx_head(pooled_latent),
            "main_peak_bin_logits": self.peak_bin_head(pooled_latent),
            "main_peak_direction_logits": self.peak_dir_head(pooled_latent),
        }

    def _predicted_summary(self, event_logits: dict[str, torch.Tensor]) -> torch.Tensor:
        turn_has = torch.sigmoid(event_logits["first_major_turn_onset_has_logit"]).unsqueeze(-1)
        turn_idx = _expected_normalized_index(event_logits["first_major_turn_onset_idx_logits"])
        turn_bin = _expected_normalized_bin(event_logits["first_major_turn_onset_bin_logits"])
        turn_dir = _direction_signed_prob(event_logits["first_major_turn_direction_logits"])

        reversal_has = torch.sigmoid(event_logits["first_reversal_has_logit"]).unsqueeze(-1)
        reversal_idx = _expected_normalized_index(event_logits["first_reversal_idx_logits"])
        reversal_bin = _expected_normalized_bin(event_logits["first_reversal_bin_logits"])

        peak_idx = _expected_normalized_index(event_logits["main_peak_idx_logits"])
        peak_bin = _expected_normalized_bin(event_logits["main_peak_bin_logits"])
        peak_dir = _direction_signed_prob(event_logits["main_peak_direction_logits"])
        return torch.cat(
            [
                turn_has,
                turn_idx,
                turn_bin,
                turn_dir,
                reversal_has,
                reversal_idx,
                reversal_bin,
                peak_idx,
                peak_bin,
                peak_dir,
            ],
            dim=-1,
        )

    def _teacher_summary(self, teacher_events: dict[str, torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        turn_has = teacher_events["first_major_turn_onset_has"].to(dtype=dtype).unsqueeze(-1)
        turn_idx = (teacher_events["first_major_turn_onset_idx"].to(dtype=dtype) / max(self.future_len - 1, 1)).unsqueeze(-1)
        turn_bin = (teacher_events["first_major_turn_onset_bin"].to(dtype=dtype) / max(self.num_bins - 1, 1)).unsqueeze(-1)
        turn_dir = (teacher_events["first_major_turn_direction"].to(dtype=dtype) * 2.0 - 1.0).unsqueeze(-1)

        reversal_has = teacher_events["first_reversal_has"].to(dtype=dtype).unsqueeze(-1)
        reversal_idx = (teacher_events["first_reversal_idx"].to(dtype=dtype) / max(self.future_len - 1, 1)).unsqueeze(-1)
        reversal_bin = (teacher_events["first_reversal_bin"].to(dtype=dtype) / max(self.num_bins - 1, 1)).unsqueeze(-1)

        peak_idx = (teacher_events["main_peak_idx"].to(dtype=dtype) / max(self.future_len - 1, 1)).unsqueeze(-1)
        peak_bin = (teacher_events["main_peak_bin"].to(dtype=dtype) / max(self.num_bins - 1, 1)).unsqueeze(-1)
        peak_dir = (teacher_events["main_peak_direction"].to(dtype=dtype) * 2.0 - 1.0).unsqueeze(-1)
        return torch.cat(
            [
                turn_has,
                turn_idx,
                turn_bin,
                turn_dir,
                reversal_has,
                reversal_idx,
                reversal_bin,
                peak_idx,
                peak_bin,
                peak_dir,
            ],
            dim=-1,
        )

    def build_condition_embedding(
        self,
        event_logits: dict[str, torch.Tensor],
        teacher_events: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        predicted_summary = self._predicted_summary(event_logits)
        if teacher_events is None:
            summary = predicted_summary
            source = "predicted"
        else:
            summary = self._teacher_summary(teacher_events, dtype=predicted_summary.dtype)
            source = "teacher"
        emb = self.summary_mlp(summary)
        return emb, {"summary": summary, "predicted_summary": predicted_summary, "source": source}
