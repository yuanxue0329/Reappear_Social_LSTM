"""
ETH/UCY 数据集加载器
  - 支持 train / val / test 三种模式
  - 滑动窗口采样：obs_len=8 + pred_len=12 = 20 帧
  - 只保留在完整 20 帧内均出现的行人
  - 输出绝对坐标和相对位移两种表示
"""

import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset


def read_trajectory_file(path, delim='\t'):
    """读取单个 txt 轨迹文件，返回 list of [frame, ped_id, x, y]。"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 兼容空格和 tab 混合
            parts = line.split()
            if len(parts) < 4:
                continue
            frame  = float(parts[0])
            ped_id = float(parts[1])
            x      = float(parts[2])
            y      = float(parts[3])
            data.append([frame, ped_id, x, y])
    return np.array(data, dtype=np.float32)


def _extract_sequences_from_array(raw, obs_len, pred_len, skip, min_ped):
    """从单个场景文件的原始数据中提取滑动窗口序列（逐文件调用）。"""
    seq_len = obs_len + pred_len
    frames = np.unique(raw[:, 0]).tolist()
    frame_data = [raw[raw[:, 0] == f] for f in frames]
    num_sequences = max(0, int(math.ceil((len(frames) - seq_len + 1) / skip)))

    obs_list, pred_list, obs_rel_list, pred_rel_list = [], [], [], []
    for idx in range(0, num_sequences * skip + 1, skip):
        curr_seq_data = np.concatenate(frame_data[idx: idx + seq_len], axis=0)
        peds_in_seq = np.unique(curr_seq_data[:, 1])
        valid_peds = []
        for ped in peds_in_seq:
            ped_data = curr_seq_data[curr_seq_data[:, 1] == ped]
            if len(ped_data) == seq_len:
                valid_peds.append(ped_data[ped_data[:, 0].argsort()])
        if len(valid_peds) < min_ped:
            continue
        N = len(valid_peds)
        curr_seq = np.zeros((seq_len, N, 2), dtype=np.float32)
        for i, pd_ in enumerate(valid_peds):
            curr_seq[:, i, :] = pd_[:, 2:4]
        curr_seq_rel = np.zeros_like(curr_seq)
        curr_seq_rel[1:] = curr_seq[1:] - curr_seq[:-1]
        obs_list.append(curr_seq[:obs_len])
        pred_list.append(curr_seq[obs_len:])
        obs_rel_list.append(curr_seq_rel[:obs_len])
        pred_rel_list.append(curr_seq_rel[obs_len:])
    return obs_list, pred_list, obs_rel_list, pred_rel_list


class TrajectoryDataset(Dataset):
    """
    滑动窗口轨迹数据集。

    每个样本包含：
      obs_traj      : (obs_len, N, 2)  观测段绝对坐标
      pred_traj     : (pred_len, N, 2) 预测段绝对坐标
      obs_traj_rel  : (obs_len, N, 2)  观测段相对位移
      pred_traj_rel : (pred_len, N, 2) 预测段相对位移
      seq_start_end : (batch, 2)       每个序列在 N 维上的起止索引
                                        （此 Dataset 每个 __getitem__ 只含一个序列）
    """

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
                 min_ped=1, delim='\t'):
        """
        Args:
            data_dir  : 包含 .txt 文件的目录
            obs_len   : 观测帧数（默认 8）
            pred_len  : 预测帧数（默认 12）
            skip      : 滑动窗口步长（默认 1）
            min_ped   : 每个序列最少行人数（默认 1）
            delim     : 文件分隔符
        """
        super().__init__()
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.seq_len  = obs_len + pred_len
        self.skip     = skip
        self.min_ped  = min_ped

        obs_traj_list, pred_traj_list = [], []
        obs_traj_rel_list, pred_traj_rel_list = [], []

        # ── 核心修复：逐文件独立处理，避免跨场景帧号冲突 ──
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith('.txt'):
                continue
            raw = read_trajectory_file(os.path.join(data_dir, fname), delim)
            if len(raw) == 0:
                continue
            raw = raw[raw[:, 0].argsort()]
            o, p, or_, pr = _extract_sequences_from_array(
                raw, obs_len, pred_len, skip, min_ped)
            obs_traj_list.extend(o)
            pred_traj_list.extend(p)
            obs_traj_rel_list.extend(or_)
            pred_traj_rel_list.extend(pr)
        
        if len(obs_traj_list) == 0:
            self._obs_traj = self._pred_traj = []
            self._obs_traj_rel = self._pred_traj_rel = []
            return

        self._obs_traj      = [torch.tensor(x) for x in obs_traj_list]
        self._pred_traj     = [torch.tensor(x) for x in pred_traj_list]
        self._obs_traj_rel  = [torch.tensor(x) for x in obs_traj_rel_list]
        self._pred_traj_rel = [torch.tensor(x) for x in pred_traj_rel_list]

    def __len__(self):
        return len(self._obs_traj)

    def __getitem__(self, idx):
        return (self._obs_traj[idx],       # (obs_len, N, 2)
                self._pred_traj[idx],      # (pred_len, N, 2)
                self._obs_traj_rel[idx],   # (obs_len, N, 2)
                self._pred_traj_rel[idx])  # (pred_len, N, 2)


def seq_collate(batch):
    """
    批处理 collate_fn：将 B 个序列的行人沿 N 维拼接，一次送入模型。

    每个序列的行人数 N_i 不同，拼接后 total_N = sum(N_i)。
    seq_start_end 记录各序列在 total_N 中的起止索引，
    用于确保 Social Pooling 不跨序列计算。

    返回：
        obs_traj      : (obs_len, total_N, 2)
        pred_traj     : (pred_len, total_N, 2)
        obs_traj_rel  : (obs_len, total_N, 2)
        pred_traj_rel : (pred_len, total_N, 2)
        seq_start_end : list of (start, end) tuples，长度 = B
    """
    obs_list, pred_list, obs_rel_list, pred_rel_list = [], [], [], []
    seq_start_end = []
    ped_count = 0

    for obs, pred, obs_rel, pred_rel in batch:
        N = obs.shape[1]
        obs_list.append(obs)
        pred_list.append(pred)
        obs_rel_list.append(obs_rel)
        pred_rel_list.append(pred_rel)
        seq_start_end.append((ped_count, ped_count + N))
        ped_count += N

    return (torch.cat(obs_list,     dim=1),   # (obs_len, total_N, 2)
            torch.cat(pred_list,    dim=1),   # (pred_len, total_N, 2)
            torch.cat(obs_rel_list, dim=1),   # (obs_len, total_N, 2)
            torch.cat(pred_rel_list,dim=1),   # (pred_len, total_N, 2)
            seq_start_end)                    # list of (start, end)
