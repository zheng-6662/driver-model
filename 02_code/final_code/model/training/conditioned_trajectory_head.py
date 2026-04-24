from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class ConditionedTrajectoryHead(nn.Module):
    """Deterministic trajectory decoder with explicit event-conditioned modulation."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        event_embed_dim: int,
        future_len: int,
        out_dim: int = 2,
        nhead: int = 2,
        dec_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        conditioning_mode: str = "baseline",
        structure_width: float = 0.065,
        gate_temperature: float = 0.040,
        event_residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.future_len = int(future_len)
        self.conditioning_mode = str(conditioning_mode)
        self.structure_width = float(structure_width)
        self.gate_temperature = float(gate_temperature)
        self.event_residual_scale = float(event_residual_scale)

        self.lat_proj = nn.Linear(d_model, d_model)
        self.ctx_proj = nn.Linear(context_dim, d_model)
        self.curve_proj = nn.Linear(1, d_model)
        self.event_to_tgt = nn.Linear(event_embed_dim, d_model)
        self.event_to_film = nn.Linear(event_embed_dim, d_model * 2)
        self.structure_track_dim = 10
        self.structure_to_tgt = nn.Linear(event_embed_dim + self.structure_track_dim, d_model)
        self.structure_to_film = nn.Linear(event_embed_dim + self.structure_track_dim, d_model * 2)
        self.structure_to_steer = nn.Linear(event_embed_dim + self.structure_track_dim, 1)
        self.dec_pos = nn.Parameter(torch.zeros(1, self.future_len, d_model))
        self.register_buffer(
            "time_grid",
            torch.linspace(0.0, 1.0, self.future_len, dtype=torch.float32).view(1, self.future_len, 1),
            persistent=False,
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, out_dim)

    def _gaussian_track(self, center: torch.Tensor, amplitude: torch.Tensor, width: float) -> torch.Tensor:
        time_grid = self.time_grid.to(device=center.device, dtype=center.dtype)
        denom = max(float(width), 1e-4)
        delta = (time_grid - center.unsqueeze(1)) / denom
        return amplitude.unsqueeze(1) * torch.exp(-0.5 * (delta**2))

    def _sigmoid_track(self, center: torch.Tensor, amplitude: torch.Tensor, temperature: float) -> torch.Tensor:
        time_grid = self.time_grid.to(device=center.device, dtype=center.dtype)
        tau = max(float(temperature), 1e-4)
        return amplitude.unsqueeze(1) * torch.sigmoid((time_grid - center.unsqueeze(1)) / tau)

    def _build_structure_tracks(self, event_condition_summary: torch.Tensor | None) -> torch.Tensor | None:
        if event_condition_summary is None:
            return None
        summary = event_condition_summary
        turn_has = summary[:, 0:1].clamp(0.0, 1.0)
        turn_idx = summary[:, 1:2].clamp(0.0, 1.0)
        turn_dir = summary[:, 3:4].clamp(-1.0, 1.0)
        reversal_has = summary[:, 4:5].clamp(0.0, 1.0)
        reversal_idx = summary[:, 5:6].clamp(0.0, 1.0)
        peak_idx = summary[:, 7:8].clamp(0.0, 1.0)
        peak_dir = summary[:, 9:10].clamp(-1.0, 1.0)

        turn_pulse = self._gaussian_track(turn_idx, turn_has, self.structure_width)
        post_turn = self._sigmoid_track(turn_idx, turn_has, self.gate_temperature)
        signed_turn = post_turn * turn_dir.unsqueeze(1)

        peak_pulse = self._gaussian_track(peak_idx, torch.ones_like(turn_has), self.structure_width * 1.15)
        post_peak = self._sigmoid_track(peak_idx, torch.ones_like(turn_has), self.gate_temperature)
        signed_peak = peak_pulse * peak_dir.unsqueeze(1)

        reversal_pulse = self._gaussian_track(reversal_idx, reversal_has, self.structure_width)
        post_reversal = self._sigmoid_track(reversal_idx, reversal_has, self.gate_temperature)
        signed_reversal = post_reversal * (-turn_dir.unsqueeze(1))

        tail_gate = torch.clamp(0.55 * post_turn + 0.45 * post_peak, 0.0, 1.0)
        return torch.cat(
            [
                turn_pulse,
                post_turn,
                signed_turn,
                peak_pulse,
                post_peak,
                signed_peak,
                reversal_pulse,
                post_reversal,
                signed_reversal,
                tail_gate,
            ],
            dim=-1,
        )

    def forward(
        self,
        memory: torch.Tensor,
        pooled_latent: torch.Tensor,
        ctx: torch.Tensor,
        curve_norm: torch.Tensor,
        event_condition_emb: torch.Tensor,
        event_condition_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        bsz = memory.shape[0]
        lat_token = self.lat_proj(pooled_latent).unsqueeze(1).expand(bsz, self.future_len, -1)
        ctx_token = self.ctx_proj(ctx).unsqueeze(1).expand(bsz, self.future_len, -1)
        curve_token = self.curve_proj(curve_norm.unsqueeze(-1))

        event_bias = self.event_to_tgt(event_condition_emb).unsqueeze(1).expand(bsz, self.future_len, -1)
        structure_tracks = self._build_structure_tracks(event_condition_summary)
        structure_bias = None
        if self.conditioning_mode == "structured_v2" and structure_tracks is not None:
            structure_inputs = torch.cat(
                [event_condition_emb.unsqueeze(1).expand(bsz, self.future_len, -1), structure_tracks],
                dim=-1,
            )
            structure_bias = self.structure_to_tgt(structure_inputs)
        else:
            structure_inputs = None
        tgt = self.dec_pos[:, : self.future_len, :] + lat_token + ctx_token + curve_token + event_bias
        if structure_bias is not None:
            tgt = tgt + structure_bias
        hidden = self.decoder(self.dropout(tgt), memory)

        # Explicitly modulate decoder hidden states with event-conditioned FiLM.
        gamma, beta = self.event_to_film(event_condition_emb).chunk(2, dim=-1)
        hidden = hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        if self.conditioning_mode == "structured_v2" and structure_inputs is not None:
            structure_gamma, structure_beta = self.structure_to_film(structure_inputs).chunk(2, dim=-1)
            hidden = hidden * (1.0 + structure_gamma) + structure_beta

        out = self.out_proj(hidden)
        steer_residual = None
        if self.conditioning_mode == "structured_v2" and structure_inputs is not None:
            steer_residual = self.event_residual_scale * self.structure_to_steer(structure_inputs)
            out[:, :, 0:1] = out[:, :, 0:1] + steer_residual
        return out, {
            "decoder_hidden": hidden,
            "event_bias": event_bias,
            "structure_bias": structure_bias,
            "structure_tracks": structure_tracks,
            "steer_residual": steer_residual,
        }
