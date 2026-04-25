"""
Social LSTM 模型
  - 每个行人一个 LSTMCell
  - Social Pooling Layer：4×4 网格聚合邻居隐状态
  - 输出双变量高斯分布的 5 个参数
  - 训练时使用 teacher forcing
  - 测试时自回归采样（双变量高斯采样）
"""

import torch
import torch.nn as nn
import numpy as np
import time

from utils import (getGridMask, getGridMaskSequence, getSocialTensor, getCoef, sample_gaussian_2d,
                   getGridMask_torch, getGridMaskSequence_torch)


class SocialLSTM(nn.Module):
    """
    Social LSTM for pedestrian trajectory prediction.

    超参数（默认值与原论文一致）：
        obs_len           : 8
        pred_len          : 12
        embedding_size    : 64
        rnn_size          : 128
        grid_size         : 4  (4×4=16 格)
        neighborhood_size : 32 (邻域范围，世界坐标单位)
        dropout           : 0.5
    """

    def __init__(self,
                 obs_len=8,
                 pred_len=12,
                 embedding_size=64,
                 rnn_size=128,
                 grid_size=8,
                 neighborhood_size=2.0,
                 dropout=0.5):
        super(SocialLSTM, self).__init__()

        self.obs_len           = obs_len
        self.pred_len          = pred_len
        self.embedding_size    = embedding_size
        self.rnn_size          = rnn_size
        self.grid_size         = grid_size
        self.neighborhood_size = neighborhood_size
        self.num_cells         = grid_size * grid_size   # 64

        # 位置嵌入：(x, y) → embedding_size
        self.input_embedding = nn.Sequential(
            nn.Linear(2, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 社交张量嵌入：num_cells * rnn_size → embedding_size
        self.social_embedding = nn.Sequential(
            nn.Linear(self.num_cells * rnn_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM 单元：输入 = 位置嵌入 + 社交嵌入 = 2 * embedding_size
        self.lstm_cell = nn.LSTMCell(
            input_size=2 * embedding_size,
            hidden_size=rnn_size
        )

        # 输出层：rnn_size → 5 个高斯参数
        self.output_layer = nn.Linear(rnn_size, 5)

        # ── 性能分析计数器（每 epoch 由 train.py 调用 reset_timing() 清零）──
        self.t_mask   = 0.0   # 掩码计算（CPU numpy）累积时间/秒
        self.t_xfer   = 0.0   # CPU→GPU 数据传输累积时间/秒
        self.t_lstm   = 0.0   # LSTM 计算（GPU）累积时间/秒
        self._n_calls = 0     # forward 调用次数

    def reset_timing(self):
        """每个 epoch 开始前调用，清零计时器。"""
        self.t_mask = self.t_xfer = self.t_lstm = 0.0
        self._n_calls = 0

    # ─────────────────────────────────────────────
    #  内部辅助：初始化隐状态
    # ─────────────────────────────────────────────

    def _init_hidden(self, N, device):
        h = torch.zeros(N, self.rnn_size, device=device)
        c = torch.zeros(N, self.rnn_size, device=device)
        return h, c

    # ─────────────────────────────────────────────
    #  内部辅助：计算社交张量（含 grid mask）
    # ─────────────────────────────────────────────

    def _social_tensor(self, positions_np, hidden_states):
        """
        Args:
            positions_np  : np.ndarray, shape (N, 2)，当前帧的绝对坐标
            hidden_states : torch.Tensor, shape (N, rnn_size)
        Returns:
            social_tensor : torch.Tensor, shape (N, num_cells * rnn_size)
        """
        device = hidden_states.device
        grid_mask = getGridMask(positions_np,
                                self.neighborhood_size,
                                self.grid_size)
        grid_mask_t = torch.tensor(grid_mask, device=device)  # (N, N, C)
        return getSocialTensor(grid_mask_t, hidden_states)     # (N, C * rnn_size)

    # ─────────────────────────────────────────────
    #  前向传播（训练，teacher forcing）
    # ─────────────────────────────────────────────

    def forward(self, obs_traj, obs_traj_rel, pred_traj_rel, seq_start_end=None):
        """
        训练模式：teacher forcing。
        优化：预计算整段序列的全部网格掩码，一次性传输至 GPU。

        Args:
            obs_traj      : (obs_len, N, 2)   观测段绝对坐标
            obs_traj_rel  : (obs_len, N, 2)   观测段相对位移
            pred_traj_rel : (pred_len, N, 2)  预测段相对位移（teacher forcing）
            seq_start_end : list of (start, end)，批处理时用于屏蔽跨序列社交交互

        Returns:
            outputs : (pred_len, N, 5)
        """
        device = obs_traj.device
        N = obs_traj.shape[1]
        h, c = self._init_hidden(N, device)

        # ── 预计算 decoder 的绝对坐标（teacher forcing 已知所有位移）──
        pred_rel_cumsum = torch.cat([
            torch.zeros(1, N, 2, device=device),
            torch.cumsum(pred_traj_rel[:-1], dim=0)
        ], dim=0)
        pred_abs_pool = obs_traj[-1].unsqueeze(0) + pred_rel_cumsum

        # ── [计时] 掩码计算（全 GPU，无 CPU-GPU 同步）──
        if device.type == 'cuda':
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        all_abs = torch.cat(
            [obs_traj.detach().float(), pred_abs_pool.detach().float()], dim=0)  # (seq_len, N, 2)
        all_masks_t = getGridMaskSequence_torch(
            all_abs, self.neighborhood_size, self.grid_size, seq_start_end)    # 全程 GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        self.t_mask += time.perf_counter() - _t0

        # ── [计时] LSTM Encoder + Decoder（GPU 计算）──
        _t0 = time.perf_counter()
        for t in range(self.obs_len):
            soc     = getSocialTensor(all_masks_t[t], h)
            inp_emb = self.input_embedding(obs_traj_rel[t])
            soc_emb = self.social_embedding(soc)
            h, c    = self.lstm_cell(torch.cat([inp_emb, soc_emb], dim=1), (h, c))

        outputs = []
        for t in range(self.pred_len):
            soc     = getSocialTensor(all_masks_t[self.obs_len + t], h)
            inp_emb = self.input_embedding(pred_traj_rel[t])
            soc_emb = self.social_embedding(soc)
            h, c    = self.lstm_cell(torch.cat([inp_emb, soc_emb], dim=1), (h, c))
            outputs.append(self.output_layer(h))
        if device.type == 'cuda':
            torch.cuda.synchronize()
        self.t_lstm += time.perf_counter() - _t0

        self._n_calls += 1
        return torch.stack(outputs, dim=0)  # (pred_len, N, 5)

    # ─────────────────────────────────────────────
    #  推理预测（自回归采样）
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, obs_traj, obs_traj_rel, num_samples=20, seq_start_end=None):
        """
        测试模式：自回归采样，重复 num_samples 次取均值。

        Args:
            obs_traj      : (obs_len, N, 2)  观测段绝对坐标
            obs_traj_rel  : (obs_len, N, 2)  观测段相对位移
            num_samples   : 采样次数（取均值以减少方差）
            seq_start_end : list of (start, end)，批处理时用于屏蔽跨序列社交交互

        Returns:
            pred_traj : (pred_len, N, 2)  预测的绝对坐标
        """
        device = obs_traj.device
        N = obs_traj.shape[1]

        # 预计算观测段所有时间步的网格掩码（全 GPU，num_samples 轮均可复用）
        obs_masks_t = getGridMaskSequence_torch(
            obs_traj.float(), self.neighborhood_size, self.grid_size, seq_start_end)

        all_preds = []

        for _ in range(num_samples):
            h, c = self._init_hidden(N, device)

            # Encoder（直接索引预计算掩码，无 CPU 开销）
            for t in range(self.obs_len):
                soc     = getSocialTensor(obs_masks_t[t], h)
                inp_emb = self.input_embedding(obs_traj_rel[t])
                soc_emb = self.social_embedding(soc)
                h, c    = self.lstm_cell(
                    torch.cat([inp_emb, soc_emb], dim=1), (h, c))

            # Decoder（自回归，每步一次向量化掩码计算）
            pred_positions = []
            curr_abs = obs_traj[-1].clone()
            curr_rel = obs_traj_rel[-1].clone()

            for t in range(self.pred_len):
                # GPU 版掩码：无 .cpu().numpy() 转换，无同步开销
                mask_t = getGridMask_torch(
                    curr_abs.float(), self.neighborhood_size, self.grid_size, seq_start_end)

                soc     = getSocialTensor(mask_t, h)
                inp_emb = self.input_embedding(curr_rel)
                soc_emb = self.social_embedding(soc)
                h, c    = self.lstm_cell(
                    torch.cat([inp_emb, soc_emb], dim=1), (h, c))

                out = self.output_layer(h)
                mux, muy, sx, sy, corr = getCoef(out)
                next_dx, next_dy = sample_gaussian_2d(mux, muy, sx, sy, corr)
                curr_rel = torch.stack([next_dx, next_dy], dim=1)
                curr_abs = curr_abs + curr_rel
                pred_positions.append(curr_abs.clone())

            all_preds.append(torch.stack(pred_positions, dim=0))

        return torch.stack(all_preds, dim=0).mean(dim=0)  # (pred_len, N, 2)
