"""
工具函数：Social LSTM 核心组件
  - 社交网格掩码 (getGridMask)
  - 社交张量聚合 (getSocialTensor)
  - 双变量高斯损失 (gaussian_likelihood_loss)
  - 评估指标 (compute_ade, compute_fde)
"""

import numpy as np
import torch
import math


# ─────────────────────────────────────────────
#  Social Pooling 相关
# ─────────────────────────────────────────────

def _grid_mask_single(positions, half, cell_size, grid_size, num_cells):
    """
    核心计算：单序列网格掩码（供 getGridMask / getGridMaskSequence 内部调用）。
    positions : (..., N, 2)  —— 前面可有任意维度（无维度=1帧，有T维度=多帧）
    返回与 positions 形状对应的 (..., N, N, num_cells)。
    """
    N = positions.shape[-2]
    if N == 0:
        return np.zeros(positions.shape[:-2] + (0, 0, num_cells), dtype=np.float32)
    pos_i = np.expand_dims(positions, -2)   # (..., N, 1, 2)
    pos_j = np.expand_dims(positions, -3)   # (..., 1, N, 2)
    diff  = pos_j - pos_i                   # (..., N, N, 2)
    in_hood = (
        (diff[..., 0] >= -half) & (diff[..., 0] < half) &
        (diff[..., 1] >= -half) & (diff[..., 1] < half)
    )
    # 排除自身（对角线）
    eye = np.eye(N, dtype=bool)
    in_hood[..., eye] = False
    out = np.zeros(diff.shape[:-1] + (num_cells,), dtype=np.float32)
    idx = np.where(in_hood)
    if len(idx[0]) > 0:
        cx = np.clip(((diff[idx + (0,)] + half) / cell_size).astype(np.int32), 0, grid_size - 1)
        cy = np.clip(((diff[idx + (1,)] + half) / cell_size).astype(np.int32), 0, grid_size - 1)
        out[idx + (cy * grid_size + cx,)] = 1.0
    return out


def getGridMask(frame_positions, neighborhood_size, grid_size, seq_start_end=None):
    """
    计算一帧内所有行人对之间的社交网格掩码。

    Args:
        frame_positions   : np.ndarray, shape (N, 2)
        neighborhood_size : float
        grid_size         : int
        seq_start_end     : list of (start, end)；提供时按序列分块计算，避免 O(total_N²)

    Returns:
        grid_mask : np.ndarray, shape (N, N, grid_size²)
    """
    N = frame_positions.shape[0]
    num_cells = grid_size * grid_size
    half      = neighborhood_size / 2.0
    cell_size = neighborhood_size / grid_size

    if N == 0:
        return np.zeros((0, 0, num_cells), dtype=np.float32)

    if seq_start_end is not None:
        # 分块计算：仅对每个序列内部做 O(N_i²) 运算，总量 O(Σ N_i²) ≪ O(total_N²)
        grid_mask = np.zeros((N, N, num_cells), dtype=np.float32)
        for s, e in seq_start_end:
            if e - s < 1:
                continue
            grid_mask[s:e, s:e, :] = _grid_mask_single(
                frame_positions[s:e], half, cell_size, grid_size, num_cells)
        return grid_mask
    else:
        return _grid_mask_single(frame_positions, half, cell_size, grid_size, num_cells)


def getGridMaskSequence(traj_positions, neighborhood_size, grid_size, seq_start_end=None):
    """
    批量计算 T 个时间步的社交网格掩码，用于整段序列的一次性预计算。

    Args:
        traj_positions    : np.ndarray, shape (T, N, 2)
        neighborhood_size : float
        grid_size         : int

    Returns:
        grid_masks : np.ndarray, shape (T, N, N, grid_size²)
    """
    T, N, _ = traj_positions.shape
    num_cells = grid_size * grid_size
    half      = neighborhood_size / 2.0
    cell_size = neighborhood_size / grid_size

    if N == 0 or T == 0:
        return np.zeros((T, N, N, num_cells), dtype=np.float32)

    if seq_start_end is not None:
        # 分块计算：O(Σ N_i² × T) 而非 O(total_N² × T)，内存节省可达 batch_size 倍
        grid_masks = np.zeros((T, N, N, num_cells), dtype=np.float32)
        for s, e in seq_start_end:
            if e - s < 1:
                continue
            grid_masks[:, s:e, s:e, :] = _grid_mask_single(
                traj_positions[:, s:e, :], half, cell_size, grid_size, num_cells)
        return grid_masks
    else:
        return _grid_mask_single(traj_positions, half, cell_size, grid_size, num_cells)


def getSocialTensor(grid_mask, hidden_states):
    """
    根据网格掩码聚合邻居的隐状态，生成社交张量。

    Args:
        grid_mask     : torch.Tensor, shape (N, N, num_cells)
        hidden_states : torch.Tensor, shape (N, rnn_size)

    Returns:
        social_tensor : torch.Tensor, shape (N, num_cells * rnn_size)
    """
    # (N, num_cells, N) @ (N, rnn_size) → (N, num_cells, rnn_size)
    # 等价于 grid_mask[i].T @ hidden_states  for each i
    # grid_mask: (N, N, C),  transpose(1,2): (N, C, N)
    # hidden_states.unsqueeze(0): (1, N, rnn_size) → broadcast
    N, _, num_cells = grid_mask.shape
    rnn_size = hidden_states.shape[1]

    # social_tensor[i, c] = sum_j grid_mask[i,j,c] * hidden_states[j]
    # (N, N, C).permute(0,2,1): (N, C, N)
    grid_t = grid_mask.permute(0, 2, 1)   # (N, C, N)
    # (N, C, N) x (N, rnn_size) → need (N, C, N) @ (N, N, rnn_size)? No.
    # Use einsum: 'icj, jd -> icd'  then reshape
    social = torch.einsum('icj,jd->icd', grid_t, hidden_states)  # (N, C, rnn_size)
    social_tensor = social.reshape(N, num_cells * rnn_size)
    return social_tensor


# ─────────────────────────────────────────────
#  GPU 版网格掩码（消除所有 CPU-GPU 同步）
# ─────────────────────────────────────────────

def getGridMask_torch(positions, neighborhood_size, grid_size, seq_start_end=None):
    """
    GPU 版单帧网格掩码，全程在 GPU Tensor 上运算，无 CPU-GPU 同步。

    Args:
        positions         : torch.Tensor (N, 2)，坐标已在 GPU 上
        neighborhood_size : float
        grid_size         : int
        seq_start_end     : list of (start, end)

    Returns:
        mask : torch.Tensor (N, N, grid_size²)，在同一 device 上
    """
    device = positions.device
    N = positions.shape[0]
    num_cells = grid_size * grid_size
    half = neighborhood_size / 2.0
    cell_size = neighborhood_size / grid_size

    if N == 0:
        return torch.zeros(0, 0, num_cells, device=device)

    with torch.no_grad():
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 2)
        in_hood = (diff[..., 0].abs() < half) & (diff[..., 1].abs() < half)
        in_hood.fill_diagonal_(False)

        if seq_start_end is not None:
            within = torch.zeros(N, N, dtype=torch.bool, device=device)
            for s, e in seq_start_end:
                within[s:e, s:e] = True
            in_hood = in_hood & within

        cx = ((diff[..., 0] + half) / cell_size).long().clamp(0, grid_size - 1)
        cy = ((diff[..., 1] + half) / cell_size).long().clamp(0, grid_size - 1)
        cell_idx = cy * grid_size + cx  # (N, N)

        mask = torch.zeros(N, N, num_cells, device=device)
        i_idx, j_idx = torch.where(in_hood)
        if len(i_idx) > 0:
            mask[i_idx, j_idx, cell_idx[i_idx, j_idx]] = 1.0

    return mask


def getGridMaskSequence_torch(traj_positions, neighborhood_size, grid_size, seq_start_end=None):
    """
    GPU 版序列网格掩码，全程在 GPU Tensor 上运算，无 CPU-GPU 同步。

    Args:
        traj_positions    : torch.Tensor (T, N, 2)，已在 GPU 上
        neighborhood_size : float
        grid_size         : int
        seq_start_end     : list of (start, end)

    Returns:
        masks : torch.Tensor (T, N, N, grid_size²)，在同一 device 上
    """
    device = traj_positions.device
    T, N, _ = traj_positions.shape
    num_cells = grid_size * grid_size
    half = neighborhood_size / 2.0
    cell_size = neighborhood_size / grid_size

    if N == 0 or T == 0:
        return torch.zeros(T, N, N, num_cells, device=device)

    with torch.no_grad():
        # (T, N, 1, 2) - (T, 1, N, 2) → (T, N, N, 2)
        diff = traj_positions.unsqueeze(2) - traj_positions.unsqueeze(1)
        in_hood = (diff[..., 0].abs() < half) & (diff[..., 1].abs() < half)

        eye = torch.eye(N, dtype=torch.bool, device=device)
        in_hood[:, eye] = False

        if seq_start_end is not None:
            within = torch.zeros(N, N, dtype=torch.bool, device=device)
            for s, e in seq_start_end:
                within[s:e, s:e] = True
            in_hood = in_hood & within.unsqueeze(0)

        cx = ((diff[..., 0] + half) / cell_size).long().clamp(0, grid_size - 1)
        cy = ((diff[..., 1] + half) / cell_size).long().clamp(0, grid_size - 1)
        cell_idx = cy * grid_size + cx  # (T, N, N)

        masks = torch.zeros(T, N, N, num_cells, device=device)
        t_idx, i_idx, j_idx = torch.where(in_hood)
        if len(t_idx) > 0:
            masks[t_idx, i_idx, j_idx, cell_idx[t_idx, i_idx, j_idx]] = 1.0

    return masks

def getCoef(outputs):
    """
    从模型输出提取双变量高斯的 5 个参数。

    Args:
        outputs : torch.Tensor, shape (..., 5)

    Returns:
        mux, muy : 均值
        sx, sy   : 标准差（经 exp 保证正值）
        corr     : 相关系数（经 tanh 限制在 (-1, 1)）
    """
    mux  = outputs[..., 0]
    muy  = outputs[..., 1]
    # 截断原始输出再取 exp，防止 σ → 0（loss 变负）或 σ → ∞（梯度爆炸）
    # exp(-4) ≈ 0.018m，exp(3) ≈ 20m，覆盖行人运动的合理位移范围
    sx   = torch.exp(torch.clamp(outputs[..., 2], min=-4.0, max=3.0))
    sy   = torch.exp(torch.clamp(outputs[..., 3], min=-4.0, max=3.0))
    corr = torch.tanh(outputs[..., 4])
    return mux, muy, sx, sy, corr


def gaussian_likelihood_loss(outputs, targets, eps=1e-20):
    """
    双变量高斯负对数似然损失。

    Args:
        outputs : torch.Tensor, shape (pred_len, N, 5)
        targets : torch.Tensor, shape (pred_len, N, 2)

    Returns:
        loss : scalar tensor
    """
    mux, muy, sx, sy, corr = getCoef(outputs)

    normx = targets[..., 0] - mux
    normy = targets[..., 1] - muy

    sxsy = sx * sy
    neg_rho = 1.0 - corr ** 2
    neg_rho = torch.clamp(neg_rho, min=1e-6)   # 防止数值不稳定

    z = (normx / sx) ** 2 + (normy / sy) ** 2 \
        - 2.0 * corr * normx * normy / sxsy

    exponent = torch.exp(-z / (2.0 * neg_rho))
    coeff    = 2.0 * math.pi * sxsy * torch.sqrt(neg_rho)
    prob     = exponent / coeff

    loss = -torch.log(torch.clamp(prob, min=eps))
    return loss.mean()


# ─────────────────────────────────────────────
#  从高斯分布采样预测坐标
# ─────────────────────────────────────────────

def sample_gaussian_2d(mux, muy, sx, sy, corr):
    """
    从双变量高斯分布采样（全 GPU 向量化，Cholesky 分解法）。
    """
    device = mux.device
    N = mux.shape[0]

    # 在 GPU 上一次性采样标准正态，无任何 CPU-GPU 同步
    z = torch.randn(N, 2, device=device, dtype=mux.dtype)   # (N, 2)

    # Cholesky 变换（数学等价于 multivariate_normal，无 Python for 循环）
    sy_sqrt_term = sy * torch.sqrt(torch.clamp(1.0 - corr ** 2, min=1e-6))
    dx = mux + sx * z[:, 0]
    dy = muy + corr * sy * z[:, 0] + sy_sqrt_term * z[:, 1]

    return dx, dy


# ─────────────────────────────────────────────
#  评估指标
# ─────────────────────────────────────────────

def compute_ade(pred_traj, gt_traj):
    """
    平均位移误差 (Average Displacement Error)。

    Args:
        pred_traj : torch.Tensor, shape (pred_len, N, 2)
        gt_traj   : torch.Tensor, shape (pred_len, N, 2)

    Returns:
        ade : scalar（米）
    """
    error = gt_traj - pred_traj                  # (pred_len, N, 2)
    error = torch.sqrt((error ** 2).sum(dim=2))  # (pred_len, N)
    return error.mean().item()


def compute_fde(pred_traj, gt_traj):
    """
    终点位移误差 (Final Displacement Error)。

    Args:
        pred_traj : torch.Tensor, shape (pred_len, N, 2)
        gt_traj   : torch.Tensor, shape (pred_len, N, 2)

    Returns:
        fde : scalar（米）
    """
    error = gt_traj[-1] - pred_traj[-1]          # (N, 2)
    error = torch.sqrt((error ** 2).sum(dim=1))  # (N,)
    return error.mean().item()
