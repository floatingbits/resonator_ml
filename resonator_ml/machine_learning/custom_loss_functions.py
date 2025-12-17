import torch
import torch.nn.functional as F

eps = 1e-8

# ----------------------------
# 1) Relative MSE (per sample)
#    loss = mean( (pred - target)^2 / (target^2 + eps) )
# ----------------------------
def relative_mse(pred, target, eps=1e-8):
    denom = target.pow(2) + eps
    return ((pred - target).pow(2) / denom).mean()


# ----------------------------
# 2) Relative L1 (per sample) -- robuster gegen Ausreisser
#    loss = mean( |pred - target| / (|target| + eps) )
# ----------------------------
def relative_l1(pred, target, eps=1e-4):
    denom = target.abs() + eps
    return ( (pred - target).abs() / denom ).mean()


# ----------------------------
# 3) Log-amplitude MSE (dB-ähnlich)
#    loss = MSE( log(|pred|+eps), log(|target|+eps) )
#    gut für Pegelunterschiede in dB
# ----------------------------
def log_magnitude_mse(pred, target, eps=1e-6):
    return F.mse_loss(torch.log(pred.abs() + eps), torch.log(target.abs() + eps))


# ----------------------------
# 4) Per-window RMS normalised MSE
#    Für jedes Trainingsfenster normalisieren wir auf RMS=1, dann MSE.
# ----------------------------
def window_normalized_mse(pred, target, frame_size, eps=1e-8):
    # pred/target shapes: [B, T] or [T]
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    # compute RMS per window (over T)
    rms_t = torch.sqrt((target**2).mean(dim=1) + eps)  # [B]
    pred_n = pred / rms_t.unsqueeze(1)
    targ_n = target / rms_t.unsqueeze(1)
    return F.mse_loss(pred_n, targ_n)


# ----------------------------
# 5) Simple inverse-RMS weighting MSE
#    weight each frame by 1 / (rms_target + floor)
# ----------------------------
def inv_rms_weighted_mse(pred, target, frame_size, floor=1e-4, eps=1e-8):
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    rms_t = torch.sqrt((target**2).mean(dim=1) + eps)  # [B]
    weights = 1.0 / torch.clamp(rms_t, min=floor)
    mse_per_frame = ((pred - target)**2).mean(dim=1)  # [B]
    return (mse_per_frame * weights).mean()


# ----------------------------
# 6) SI-SNR (scale invariant) implementation for batch signals
#    Useful if pred/target are clips (waveform)
# ----------------------------
def si_snr_batch(est, ref, eps=1e-8):
    # est, ref: [B, T]
    if est.dim() == 1:
        est = est.unsqueeze(0)
        ref = ref.unsqueeze(0)
    # zero-mean
    est_zm = est - est.mean(dim=1, keepdim=True)
    ref_zm = ref - ref.mean(dim=1, keepdim=True)
    # projection
    proj = (torch.sum(est_zm * ref_zm, dim=1, keepdim=True) * ref_zm) / (torch.sum(ref_zm**2, dim=1, keepdim=True) + eps)
    e_noise = est_zm - proj
    si_snr_val = 10 * torch.log10((torch.sum(proj**2, dim=1) + eps) / (torch.sum(e_noise**2, dim=1) + eps))
    # return negative because we minimize loss
    return -si_snr_val.mean()
