from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        response_summary_dim: int = 0,
        num_trajectory_candidates: int = 1,
        candidate_delta_scale: float = 1.0,
        candidate_base_mode: str = "learned_delta",
        candidate_prototypes: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.future_len = int(future_len)
        self.conditioning_mode = str(conditioning_mode)
        allowed_modes = {
            "vehicle_direct",
            "vehicle_direct_coarse_fine",
            "baseline",
            "structured_v2",
            "structured_v2_coarse_fine",
        }
        if self.conditioning_mode not in allowed_modes:
            raise ValueError(f"Unsupported conditioning_mode={self.conditioning_mode!r}; expected one of {sorted(allowed_modes)}")
        self.structure_width = float(structure_width)
        self.gate_temperature = float(gate_temperature)
        self.event_residual_scale = float(event_residual_scale)
        self.response_summary_dim = int(response_summary_dim)
        self.num_trajectory_candidates = max(1, int(num_trajectory_candidates))
        self.candidate_delta_scale = float(candidate_delta_scale)
        self.candidate_base_mode = str(candidate_base_mode)
        if self.candidate_base_mode not in {"learned_delta", "response_prototype"}:
            raise ValueError(
                f"Unsupported candidate_base_mode={self.candidate_base_mode!r}; "
                "expected 'learned_delta' or 'response_prototype'"
            )

        self.lat_proj = nn.Linear(d_model, d_model)
        self.ctx_proj = nn.Linear(context_dim, d_model)
        self.curve_proj = nn.Linear(1, d_model)
        self.event_to_tgt = nn.Linear(event_embed_dim, d_model)
        self.event_to_film = nn.Linear(event_embed_dim, d_model * 2)
        self.response_to_tgt = nn.Linear(self.response_summary_dim, d_model) if self.response_summary_dim > 0 else None
        self.response_to_film = nn.Linear(self.response_summary_dim, d_model * 2) if self.response_summary_dim > 0 else None
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
        self.coarse_pool_kernel = 20
        self.coarse_pool_stride = 20
        self.coarse_out_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )
        self.fine_out_proj = nn.Linear(d_model, out_dim)
        if self.num_trajectory_candidates > 1:
            self.candidate_embed = nn.Parameter(torch.zeros(self.num_trajectory_candidates, d_model))
            nn.init.normal_(self.candidate_embed, mean=0.0, std=0.02)
            self.candidate_delta_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, out_dim),
            )
            self.candidate_selector = nn.Sequential(
                nn.Linear(d_model, max(32, d_model // 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(32, d_model // 2), self.num_trajectory_candidates),
            )
            if self.candidate_base_mode == "response_prototype":
                proto = torch.zeros(self.num_trajectory_candidates, self.future_len, out_dim, dtype=torch.float32)
                if candidate_prototypes is not None:
                    proto = torch.as_tensor(candidate_prototypes, dtype=torch.float32)
                    expected = (self.num_trajectory_candidates, self.future_len, out_dim)
                    if tuple(proto.shape) != expected:
                        raise ValueError(f"candidate_prototypes shape={tuple(proto.shape)} expected={expected}")
                self.register_buffer("candidate_prototypes", proto, persistent=False)
            else:
                self.register_buffer(
                    "candidate_prototypes",
                    torch.zeros(0, dtype=torch.float32),
                    persistent=False,
                )
        else:
            self.candidate_embed = None
            self.candidate_delta_proj = None
            self.candidate_selector = None
            self.register_buffer("candidate_prototypes", torch.zeros(0, dtype=torch.float32), persistent=False)

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
        response_condition_summary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        bsz = memory.shape[0]
        lat_token = self.lat_proj(pooled_latent).unsqueeze(1).expand(bsz, self.future_len, -1)
        ctx_token = self.ctx_proj(ctx).unsqueeze(1).expand(bsz, self.future_len, -1)
        curve_token = self.curve_proj(curve_norm.unsqueeze(-1))

        is_vehicle_direct = self.conditioning_mode in {"vehicle_direct", "vehicle_direct_coarse_fine"}
        is_structured = self.conditioning_mode in {"structured_v2", "structured_v2_coarse_fine"}
        is_coarse_fine = self.conditioning_mode in {"vehicle_direct_coarse_fine", "structured_v2_coarse_fine"}
        event_bias = None
        structure_tracks = None
        structure_bias = None
        structure_inputs = None
        tgt = self.dec_pos[:, : self.future_len, :] + lat_token + ctx_token + curve_token
        response_bias = None
        if self.response_to_tgt is not None and response_condition_summary is not None:
            response_bias = self.response_to_tgt(response_condition_summary).unsqueeze(1).expand(bsz, self.future_len, -1)
            tgt = tgt + response_bias
        if not is_vehicle_direct:
            event_bias = self.event_to_tgt(event_condition_emb).unsqueeze(1).expand(bsz, self.future_len, -1)
            structure_tracks = self._build_structure_tracks(event_condition_summary)
            if is_structured and structure_tracks is not None:
                structure_inputs = torch.cat(
                    [event_condition_emb.unsqueeze(1).expand(bsz, self.future_len, -1), structure_tracks],
                    dim=-1,
                )
                structure_bias = self.structure_to_tgt(structure_inputs)
            tgt = tgt + event_bias
        if structure_bias is not None:
            tgt = tgt + structure_bias
        hidden = self.decoder(self.dropout(tgt), memory)

        if self.response_to_film is not None and response_condition_summary is not None:
            response_gamma, response_beta = self.response_to_film(response_condition_summary).chunk(2, dim=-1)
            hidden = hidden * (1.0 + response_gamma.unsqueeze(1)) + response_beta.unsqueeze(1)

        if not is_vehicle_direct:
            # Explicitly modulate decoder hidden states with event-conditioned FiLM.
            gamma, beta = self.event_to_film(event_condition_emb).chunk(2, dim=-1)
            hidden = hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            if is_structured and structure_inputs is not None:
                structure_gamma, structure_beta = self.structure_to_film(structure_inputs).chunk(2, dim=-1)
                hidden = hidden * (1.0 + structure_gamma) + structure_beta

        coarse_out = None
        fine_out = None
        if is_coarse_fine:
            pool_k = max(1, min(int(self.coarse_pool_kernel), self.future_len))
            pool_s = max(1, int(self.coarse_pool_stride))
            pooled_hidden = F.avg_pool1d(hidden.transpose(1, 2), kernel_size=pool_k, stride=pool_s).transpose(1, 2)
            coarse_low = self.coarse_out_proj(pooled_hidden)
            coarse_out = F.interpolate(
                coarse_low.transpose(1, 2),
                size=self.future_len,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            fine_out = self.fine_out_proj(hidden)
            out = coarse_out + fine_out
        else:
            out = self.out_proj(hidden)
        steer_residual = None
        if is_structured and structure_inputs is not None:
            steer_residual = self.event_residual_scale * self.structure_to_steer(structure_inputs)
            out[:, :, 0:1] = out[:, :, 0:1] + steer_residual
        candidate_trajectories = None
        candidate_logits = None
        candidate_probs = None
        candidate_choice = None
        if self.num_trajectory_candidates > 1 and self.candidate_embed is not None:
            assert self.candidate_delta_proj is not None
            assert self.candidate_selector is not None
            candidate_hidden = hidden.unsqueeze(1) + self.candidate_embed.view(1, self.num_trajectory_candidates, 1, -1)
            candidate_delta = self.candidate_delta_proj(candidate_hidden)
            if self.candidate_base_mode == "response_prototype":
                proto = self.candidate_prototypes.to(device=out.device, dtype=out.dtype).unsqueeze(0)
                candidate_trajectories = proto + self.candidate_delta_scale * candidate_delta
            else:
                candidate_trajectories = out.unsqueeze(1) + self.candidate_delta_scale * candidate_delta
            candidate_logits = self.candidate_selector(pooled_latent)
            candidate_probs = torch.softmax(candidate_logits, dim=-1)
            if self.training:
                out = torch.sum(candidate_probs.view(bsz, self.num_trajectory_candidates, 1, 1) * candidate_trajectories, dim=1)
            else:
                candidate_choice = torch.argmax(candidate_logits, dim=-1)
                gather_idx = candidate_choice.view(bsz, 1, 1, 1).expand(-1, 1, self.future_len, candidate_trajectories.shape[-1])
                out = torch.gather(candidate_trajectories, dim=1, index=gather_idx).squeeze(1)
        return out, {
            "decoder_hidden": hidden,
            "event_bias": event_bias,
            "response_bias": response_bias,
            "structure_bias": structure_bias,
            "structure_tracks": structure_tracks,
            "steer_residual": steer_residual,
            "coarse_out": coarse_out,
            "fine_out": fine_out,
            "candidate_trajectories": candidate_trajectories,
            "candidate_logits": candidate_logits,
            "candidate_probs": candidate_probs,
            "candidate_choice": candidate_choice,
        }
