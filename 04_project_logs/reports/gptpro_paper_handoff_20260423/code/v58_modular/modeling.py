from .shared import *

# =========================
# Dataset
# =========================
class MultiTaskFutureWithCurveDataset(Dataset):
    def __init__(self, X_list, y_list, curve_list, ctx_list, z_phys_list, rev_gt_list, rev_gt_weak_list, rev_gt_strong_list,
                 y_mean, y_std, curve_mean, curve_std, ctx_mean, ctx_std,
                 curve_score_list=None, is_curve_list=None):
        self.X = X_list
        self.y = y_list
        self.curve = curve_list
        self.ctx = ctx_list
        self.z_phys = z_phys_list  # (N,2) or NaN (masked)
        self.rev_gt = rev_gt_list      # (N,) 0/1 (label used for rev_head)
        self.rev_gt_weak = rev_gt_weak_list if (rev_gt_weak_list is not None) else rev_gt_list
        self.rev_gt_strong = rev_gt_strong_list if (rev_gt_strong_list is not None) else rev_gt_list

        self.curve_score = curve_score_list
        self.is_curve = is_curve_list

        self.y_mean = y_mean.astype(np.float32)
        self.y_std  = y_std.astype(np.float32)
        self.curve_mean = float(curve_mean)
        self.curve_std  = float(curve_std)
        self.ctx_mean = ctx_mean.astype(np.float32)
        self.ctx_std  = ctx_std.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        curve_raw = self.curve[idx]
        ctx_raw = self.ctx[idx]
        z = self.z_phys[idx]  # (2,)

        y_norm = (y - self.y_mean) / self.y_std
        curve_norm = (curve_raw - self.curve_mean) / self.curve_std
        ctx_norm = (ctx_raw - self.ctx_mean) / self.ctx_std

        # z_phys 仅训练用：这里不再额外标准化（已经在 main 里做了）
        z_phys = z.astype(np.float32)
        z_mask = np.isfinite(z_phys).all().astype(np.float32)  # 1=有效, 0=缺失(生理缺)

        return {
            "src": x.astype(np.float32),
            "y_norm": y_norm.astype(np.float32),
            "curve_norm": curve_norm.astype(np.float32),
            "ctx": ctx_norm.astype(np.float32),
            "z_phys": z_phys,
            "z_mask": np.array([z_mask], dtype=np.float32),
            "rev_gt": np.array([self.rev_gt[idx]], dtype=np.float32),
            "rev_gt_weak": np.array([self.rev_gt_weak[idx]], dtype=np.float32),
            "rev_gt_strong": np.array([self.rev_gt_strong[idx]], dtype=np.float32),
            "idx": np.array([idx], dtype=np.int64),
            "curve_score": np.array([self.curve_score[idx]], dtype=np.float32) if self.curve_score is not None else np.array([np.nan], dtype=np.float32),
            "is_curve": np.array([self.is_curve[idx]], dtype=np.int64) if self.is_curve is not None else np.array([-1], dtype=np.int64),
        }


def _manual_linear_upsample_1d_align_corners(x: torch.Tensor, size: int) -> torch.Tensor:
    target = int(size)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape={tuple(x.shape)}")
    if target <= 0:
        raise ValueError(f"size must be positive, got {target}")
    length_in = int(x.shape[-1])
    if length_in == target:
        return x.clone()
    if target == 1:
        return x[..., :1].clone()
    if length_in == 1:
        return x.expand(*x.shape[:-1], target)

    weight_dtype = x.dtype if x.is_floating_point() else torch.float32
    out_pos = torch.arange(target, device=x.device, dtype=weight_dtype)
    src_pos = out_pos * ((length_in - 1) / (target - 1))
    left_idx = torch.floor(src_pos).to(torch.long)
    right_idx = torch.clamp(left_idx + 1, max=length_in - 1)
    w_right = (src_pos - left_idx.to(weight_dtype)).view(1, 1, target)
    w_left = 1.0 - w_right
    left_vals = x.index_select(-1, left_idx)
    right_vals = x.index_select(-1, right_idx)
    return left_vals * w_left + right_vals * w_right


# =========================
# Model (baseline + state head)
# =========================
class Past2FutureMultiTaskRoadPreview(nn.Module):
    """
    Output:
      y_hat_norm: (B, FUTURE_LEN, 3)
      z_veh:      (B, state_dim) from encoder memory pooling (train for distillation; inference optional)
    """
    def __init__(self, input_dim, context_dim, future_len, out_dim=3,
                 d_model=128, nhead=2,
                 num_layers_enc=2, num_layers_dec=2,
                 dim_feedforward=256, dropout=0.1,
                 max_len_enc=600, max_len_dec=400,
                 state_dim=2,
                 enable_steer_coarse_fine=False,
                 enable_manual_coarse_upsample=False,
                 trend_pool_kernel=20,
                 trend_pool_stride=20,
                 enable_late_reversal_gate=False,
                 late_rev_gate_start_sec=1.05,
                 late_rev_gate_scale=0.60,
                 late_rev_gate_ramp_power=1.50,
                 enable_strong_pos_gate=False,
                 strong_pos_gate_start_sec=1.20,
                 strong_pos_gate_scale=0.45,
                 strong_pos_gate_ramp_power=1.75,
                 strong_pos_gate_prob_center=0.60):
        super().__init__()
        self.d_model = d_model
        self.future_len = future_len
        self.out_dim = out_dim
        self.state_dim = state_dim
        self.enable_steer_coarse_fine = bool(enable_steer_coarse_fine)
        self.enable_manual_coarse_upsample = bool(enable_manual_coarse_upsample)
        self.trend_pool_kernel = int(trend_pool_kernel)
        self.trend_pool_stride = int(trend_pool_stride)
        self.enable_late_reversal_gate = bool(enable_late_reversal_gate)
        self.late_rev_gate_start_sec = float(late_rev_gate_start_sec)
        self.late_rev_gate_scale = float(late_rev_gate_scale)
        self.late_rev_gate_ramp_power = float(late_rev_gate_ramp_power)
        self.enable_strong_pos_gate = bool(enable_strong_pos_gate)
        self.strong_pos_gate_start_sec = float(strong_pos_gate_start_sec)
        self.strong_pos_gate_scale = float(strong_pos_gate_scale)
        self.strong_pos_gate_ramp_power = float(strong_pos_gate_ramp_power)
        self.strong_pos_gate_prob_center = float(strong_pos_gate_prob_center)

        # Encoder
        self.enc_input_proj = nn.Linear(input_dim, d_model)
        self.enc_pos_emb = nn.Parameter(torch.zeros(1, max_len_enc, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers_enc)

        # Decoder
        self.dec_pos_emb = nn.Parameter(torch.zeros(1, max_len_dec, d_model))
        self.ctx_proj    = nn.Linear(context_dim, d_model)
        self.curve_proj  = nn.Linear(1, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers_dec)

        self.out_proj = nn.Linear(d_model, out_dim)
        if self.enable_steer_coarse_fine:
            self.steer_fine_proj = nn.Linear(d_model, 1)
            self.steer_coarse_proj = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
            self.other_proj = nn.Linear(d_model, max(1, out_dim - 1))
        self.dropout = nn.Dropout(dropout)

        # ---- NEW: state head (encoder pooling) ----
        self.pool_score = nn.Linear(d_model, 1)
        self.state_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, state_dim)
        )

        # ---- NEW(v5.4): reversal classifier head (encoder pooling) ----
        self.rev_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        if self.enable_strong_pos_gate:
            self.strong_pos_gate_head = nn.Sequential(
                nn.Linear(d_model * 2, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )

    def forward(self, src, ctx, curve_norm):
        B, T_in, _ = src.shape
        T_out = self.future_len

        # Encoder
        h_src = self.enc_input_proj(src)
        h_src = h_src + self.enc_pos_emb[:, :T_in, :]
        memory = self.encoder(self.dropout(h_src))  # (B,T_in,d_model)

        # ---- encoder pooling -> z_veh ----
        scores = self.pool_score(memory)            # (B,T,1)
        alpha = torch.softmax(scores, dim=1)
        ctx_enc = torch.sum(alpha * memory, dim=1)  # (B,d_model)
        z_veh = self.state_head(ctx_enc)            # (B,state_dim)
        rev_logit = self.rev_head(ctx_enc).squeeze(-1)  # (B,)

        # Decoder input
        pos_tgt = self.dec_pos_emb[:, :T_out, :].expand(B, T_out, -1)
        ctx2 = torch.cat([ctx, z_veh], dim=1)  # (B, context_dim)
        ctx_emb = self.ctx_proj(ctx2).unsqueeze(1).expand(B, T_out, -1)

        curve_feat = curve_norm.unsqueeze(-1)   # (B,T_out,1)
        curve_emb = self.curve_proj(curve_feat) # (B,T_out,d_model)

        tgt = pos_tgt + ctx_emb + curve_emb
        out = self.decoder(tgt, memory)
        if not self.enable_steer_coarse_fine:
            y_hat_norm = self.out_proj(out)
            return y_hat_norm, z_veh, rev_logit

        steer_fine_norm = self.steer_fine_proj(out).squeeze(-1)
        pool_k = max(1, min(int(self.trend_pool_kernel), T_out))
        pool_s = max(1, int(self.trend_pool_stride))
        dec_pool = F.avg_pool1d(out.transpose(1, 2), kernel_size=pool_k, stride=pool_s).transpose(1, 2)
        steer_coarse_norm = self.steer_coarse_proj(dec_pool).squeeze(-1)
        if self.enable_manual_coarse_upsample:
            steer_coarse_up_norm = _manual_linear_upsample_1d_align_corners(
                steer_coarse_norm.unsqueeze(1),
                size=T_out,
            ).squeeze(1)
        else:
            steer_coarse_up_norm = F.interpolate(
                steer_coarse_norm.unsqueeze(1),
                size=T_out,
                mode="linear",
                align_corners=True,
            ).squeeze(1)
        steer_fine_out_norm = steer_fine_norm
        late_rev_gate = None
        late_rev_prob = None
        strong_pos_gate_logit = None
        strong_pos_gate_prob = None
        strong_pos_late_gate = None
        if self.enable_strong_pos_gate and self.strong_pos_gate_scale > 0.0:
            late_start_idx = _sec_to_future_idx(self.strong_pos_gate_start_sec, T_out)
            late_slice = out[:, late_start_idx:, :] if late_start_idx < T_out else out[:, -1:, :]
            late_feat = late_slice.mean(dim=1)
            gate_feat = torch.cat([ctx_enc, late_feat], dim=1)
            strong_pos_gate_logit = self.strong_pos_gate_head(gate_feat).squeeze(-1)
            strong_pos_gate_prob = torch.sigmoid(strong_pos_gate_logit).to(out.dtype).unsqueeze(1)
            centered_prob = (
                (strong_pos_gate_prob - self.strong_pos_gate_prob_center)
                / max(1e-6, 1.0 - self.strong_pos_gate_prob_center)
            ).clamp(0.0, 1.0)
            late_ramp = _build_late_ramp(
                T_out,
                self.strong_pos_gate_start_sec,
                device=out.device,
                dtype=out.dtype,
                power=self.strong_pos_gate_ramp_power,
            )
            strong_pos_late_gate = 1.0 + self.strong_pos_gate_scale * centered_prob * late_ramp
            steer_fine_out_norm = steer_fine_norm * strong_pos_late_gate
        elif self.enable_late_reversal_gate and self.late_rev_gate_scale > 0.0:
            late_ramp = _build_late_ramp(
                T_out,
                self.late_rev_gate_start_sec,
                device=out.device,
                dtype=out.dtype,
                power=self.late_rev_gate_ramp_power,
            )
            if torch.count_nonzero(late_ramp).item() > 0:
                late_rev_prob = torch.sigmoid(rev_logit).to(out.dtype).unsqueeze(1)
                late_rev_gate = 1.0 + self.late_rev_gate_scale * late_rev_prob * late_ramp
                # Keep coarse trend untouched and only amplify late fine residual on reversal-like samples.
                steer_fine_out_norm = steer_fine_norm * late_rev_gate
        steer_norm = steer_coarse_up_norm + steer_fine_out_norm
        other_norm = self.other_proj(out)
        y_hat_norm = torch.cat([steer_norm.unsqueeze(-1), other_norm], dim=-1)
        aux = {
            "steer_coarse_norm": steer_coarse_norm,
            "steer_coarse_up_norm": steer_coarse_up_norm,
            "steer_fine_raw_norm": steer_fine_norm,
            "steer_fine_norm": steer_fine_out_norm,
        }
        if strong_pos_gate_logit is not None:
            aux["strong_pos_gate_logit"] = strong_pos_gate_logit
            aux["strong_pos_gate_prob"] = strong_pos_gate_prob.squeeze(1)
            aux["strong_pos_late_gate"] = strong_pos_late_gate
        if late_rev_gate is not None:
            aux["late_rev_gate"] = late_rev_gate
            aux["late_rev_prob"] = late_rev_prob.squeeze(1)
        return y_hat_norm, z_veh, rev_logit, aux


def unpack_model_output(output):
    if not isinstance(output, tuple):
        raise TypeError(f"Unexpected model output type: {type(output)!r}")
    if len(output) == 3:
        y_hat_norm, z_veh, rev_logit = output
        return y_hat_norm, z_veh, rev_logit, {}
    if len(output) == 4:
        y_hat_norm, z_veh, rev_logit, aux = output
        return y_hat_norm, z_veh, rev_logit, (aux or {})
    raise ValueError(f"Unexpected model output length: {len(output)}")



