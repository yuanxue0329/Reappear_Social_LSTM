"""
可视化脚本：3 种图表
  1. 轨迹对比图：历史(蓝) + 真实未来(绿) + 预测(红虚线)
  2. ADE/FDE 条形图：5 个场景分组展示
  3. Social Pooling 热力图：某帧的 4×4 网格邻居分布

用法：
    python visualize.py                     # 所有图表
    python visualize.py --scene eth         # 只画 eth 场景轨迹
    python visualize.py --mode bar          # 只画条形图（需先运行 evaluate.py）
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # Windows 非交互模式，保存图片不弹窗
import matplotlib.pyplot as plt
# 设置中文字体（优先使用 Windows 自带字体）
import matplotlib.font_manager as fm
_cjk_fonts = ['Microsoft YaHei', 'SimHei', 'STSong', 'WenQuanYi Micro Hei']
_available = {f.name for f in fm.fontManager.ttflist}
for _fn in _cjk_fonts:
    if _fn in _available:
        plt.rcParams['font.family'] = _fn
        break
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from dataset import TrajectoryDataset
from model import SocialLSTM
from utils import getGridMask


DATA_ROOT  = os.path.join(os.path.dirname(__file__),
                          '..', 'ETH_UCY dataset', 'datasets')
CKPT_DIR   = os.path.join(os.path.dirname(__file__), 'checkpoints')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
SCENES     = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  1. 轨迹对比图
# ─────────────────────────────────────────────

def plot_trajectories(scene, args, device, num_show=8):
    """绘制一个场景中若干行人的轨迹对比图。"""
    ckpt_path = os.path.join(CKPT_DIR, f'{scene}_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"  [{scene}] 无 checkpoint，跳过轨迹图")
        return

    ckpt      = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get('args', {})

    model = SocialLSTM(
        obs_len=saved_args.get('obs_len', args.obs_len),
        pred_len=saved_args.get('pred_len', args.pred_len),
        embedding_size=saved_args.get('embedding_size', args.embedding_size),
        rnn_size=saved_args.get('rnn_size', args.rnn_size),
        grid_size=saved_args.get('grid_size', args.grid_size),
        neighborhood_size=saved_args.get('neighborhood_size', args.neighborhood_size),
        dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_dir = os.path.join(DATA_ROOT, scene, 'test')
    test_ds  = TrajectoryDataset(
        test_dir,
        obs_len=saved_args.get('obs_len', args.obs_len),
        pred_len=saved_args.get('pred_len', args.pred_len),
        min_ped=1
    )
    if len(test_ds) == 0:
        print(f"  [{scene}] 测试集为空，跳过")
        return

    # 随机选几个 sequence index
    np.random.seed(42)
    indices = np.random.choice(len(test_ds), size=min(num_show, len(test_ds)),
                               replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Social LSTM — 轨迹预测对比  [{scene}]', fontsize=14)
    axes = axes.flatten()

    for ax_idx, seq_idx in enumerate(indices):
        obs_traj, pred_traj, obs_traj_rel, _ = test_ds[seq_idx]
        obs_traj     = obs_traj.to(device)
        pred_traj    = pred_traj.to(device)
        obs_traj_rel = obs_traj_rel.to(device)

        with torch.no_grad():
            pred = model.predict(obs_traj, obs_traj_rel,
                                 num_samples=args.num_samples)

        obs_np  = obs_traj.cpu().numpy()   # (obs_len, N, 2)
        gt_np   = pred_traj.cpu().numpy()  # (pred_len, N, 2)
        pred_np = pred.cpu().numpy()       # (pred_len, N, 2)

        ax = axes[ax_idx]
        N  = obs_np.shape[1]
        colors = plt.cm.Set1(np.linspace(0, 1, max(N, 1)))

        for ped_i in range(N):
            c = colors[ped_i % len(colors)]
            # 历史轨迹（实线，蓝色系）
            ax.plot(obs_np[:, ped_i, 0], obs_np[:, ped_i, 1],
                    '-o', color=c, alpha=0.9, linewidth=1.5,
                    markersize=3, label='_')
            # 真实未来（虚线，绿色）
            # 将观测末尾连接
            ax.plot([obs_np[-1, ped_i, 0], gt_np[0, ped_i, 0]],
                    [obs_np[-1, ped_i, 1], gt_np[0, ped_i, 1]],
                    '--', color='green', alpha=0.6, linewidth=1)
            ax.plot(gt_np[:, ped_i, 0], gt_np[:, ped_i, 1],
                    '--s', color='green', alpha=0.7, linewidth=1.5,
                    markersize=3)
            # 预测轨迹（虚线，红色）
            ax.plot([obs_np[-1, ped_i, 0], pred_np[0, ped_i, 0]],
                    [obs_np[-1, ped_i, 1], pred_np[0, ped_i, 1]],
                    ':', color='red', alpha=0.6, linewidth=1)
            ax.plot(pred_np[:, ped_i, 0], pred_np[:, ped_i, 1],
                    ':^', color='red', alpha=0.8, linewidth=1.5,
                    markersize=3)

        ax.set_title(f'序列 {seq_idx}  (N={N}行人)', fontsize=9)
        ax.set_xlabel('x (m)', fontsize=8)
        ax.set_ylabel('y (m)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal', adjustable='box')

    # 统一图例
    legend_elements = [
        Line2D([0], [0], color='blue',  lw=1.5, linestyle='-',  label='历史轨迹'),
        Line2D([0], [0], color='green', lw=1.5, linestyle='--', label='真实未来'),
        Line2D([0], [0], color='red',   lw=1.5, linestyle=':',  label='预测轨迹'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.01))

    # 隐藏多余子图
    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = os.path.join(OUTPUT_DIR, f'{scene}_trajectory_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] 轨迹对比图 → {save_path}")


# ─────────────────────────────────────────────
#  2. ADE/FDE 条形图
# ─────────────────────────────────────────────

def plot_ade_fde_bar(args, device):
    """加载各场景 checkpoint 并计算 ADE/FDE，绘制分组条形图。"""
    from utils import compute_ade, compute_fde

    ades, fdes, valid_scenes = [], [], []

    for scene in SCENES:
        ckpt_path = os.path.join(CKPT_DIR, f'{scene}_best.pth')
        if not os.path.exists(ckpt_path):
            print(f"  [{scene}] 无 checkpoint，跳过")
            continue

        ckpt       = torch.load(ckpt_path, map_location=device)
        saved_args = ckpt.get('args', {})
        obs_len    = saved_args.get('obs_len', args.obs_len)
        pred_len   = saved_args.get('pred_len', args.pred_len)

        model = SocialLSTM(
            obs_len=obs_len,
            pred_len=pred_len,
            embedding_size=saved_args.get('embedding_size', args.embedding_size),
            rnn_size=saved_args.get('rnn_size', args.rnn_size),
            grid_size=saved_args.get('grid_size', args.grid_size),
            neighborhood_size=saved_args.get('neighborhood_size', args.neighborhood_size),
            dropout=0.0
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        test_dir = os.path.join(DATA_ROOT, scene, 'test')
        test_ds  = TrajectoryDataset(test_dir, obs_len=obs_len,
                                     pred_len=pred_len, min_ped=1)
        if len(test_ds) == 0:
            continue

        total_ade, total_fde, n = 0.0, 0.0, 0
        with torch.no_grad():
            for i in range(len(test_ds)):
                obs_traj, pred_traj, obs_traj_rel, _ = test_ds[i]
                obs_traj     = obs_traj.to(device)
                pred_traj    = pred_traj.to(device)
                obs_traj_rel = obs_traj_rel.to(device)
                pred = model.predict(obs_traj, obs_traj_rel,
                                     num_samples=args.num_samples)
                total_ade += compute_ade(pred, pred_traj)
                total_fde += compute_fde(pred, pred_traj)
                n += 1

        if n > 0:
            ades.append(total_ade / n)
            fdes.append(total_fde / n)
            valid_scenes.append(scene)

    if len(valid_scenes) == 0:
        print("  [警告] 无有效场景数据，跳过条形图")
        return

    # 论文参考值
    paper_ade = {'eth': 1.09, 'hotel': 0.79, 'univ': 0.67,
                 'zara1': 0.47, 'zara2': 0.56}
    paper_fde = {'eth': 2.35, 'hotel': 1.76, 'univ': 1.40,
                 'zara1': 1.00, 'zara2': 1.17}

    x       = np.arange(len(valid_scenes))
    width   = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Social LSTM — ADE / FDE 评估结果', fontsize=13)

    for ax, metric, vals, paper_dict, ylabel in zip(
            axes,
            ['ADE', 'FDE'],
            [ades, fdes],
            [paper_ade, paper_fde],
            ['ADE (m)', 'FDE (m)']):

        paper_vals = [paper_dict.get(s, 0.0) for s in valid_scenes]
        bars1 = ax.bar(x - width/2, vals,        width, label='本实验', color='steelblue')
        bars2 = ax.bar(x + width/2, paper_vals,  width, label='论文参考', color='coral', alpha=0.7)

        # 数值标注
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}', ha='center', va='bottom',
                    fontsize=8, color='dimgray')

        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in valid_scenes], fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{metric} 对比')
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(max(vals + paper_vals), 0.1) * 1.25)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'ade_fde_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] ADE/FDE 条形图 → {save_path}")


# ─────────────────────────────────────────────
#  3. Social Pooling 热力图
# ─────────────────────────────────────────────

def plot_social_heatmap(scene, args):
    """取测试集第一个序列的第 8 帧（观测末尾），可视化 Social Pooling 网格。"""
    test_dir = os.path.join(DATA_ROOT, scene, 'test')
    test_ds  = TrajectoryDataset(test_dir, obs_len=args.obs_len,
                                 pred_len=args.pred_len, min_ped=2)
    if len(test_ds) == 0:
        print(f"  [{scene}] 行人数不足，跳过热力图")
        return

    # 找第一个 N >= 2 的序列
    seq = None
    for i in range(len(test_ds)):
        obs_traj, *_ = test_ds[i]
        if obs_traj.shape[1] >= 2:
            seq = obs_traj
            break
    if seq is None:
        print(f"  [{scene}] 未找到合适序列，跳过热力图")
        return

    # 取观测最后一帧的位置
    last_frame = seq[-1].numpy()  # (N, 2)
    N = last_frame.shape[0]

    grid_size = args.grid_size
    neighborhood_size = args.neighborhood_size
    num_cells = grid_size * grid_size

    # 绘制每个行人视角的 Social Pooling 网格（最多显示 4 个行人）
    n_show = min(N, 4)
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]
    fig.suptitle(f'Social Pooling 网格 热力图 [{scene}]  '
                 f'(grid={grid_size}×{grid_size}, '
                 f'neighborhood={neighborhood_size}m)', fontsize=12)

    grid_mask = getGridMask(last_frame, neighborhood_size, grid_size)
    # grid_mask: (N, N, num_cells)  → grid_mask[i, j, k] = 行人j落在行人i的第k格

    for i in range(n_show):
        ax = axes[i]
        # 聚合：对所有邻居求和 → 每格的邻居数
        occupancy = grid_mask[i].sum(axis=0).reshape(grid_size, grid_size)  # (4, 4)

        im = ax.imshow(occupancy, cmap='hot', vmin=0,
                       vmax=max(occupancy.max(), 1), origin='lower')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f'行人 {i}  @ ({last_frame[i,0]:.1f}, {last_frame[i,1]:.1f})',
                     fontsize=9)
        ax.set_xlabel('网格列 →')
        ax.set_ylabel('网格行 ↑')
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))

        # 邻居点标注
        cx, cy = last_frame[i]
        half = neighborhood_size / 2.0
        cell_size = neighborhood_size / grid_size
        for j in range(N):
            if j == i:
                continue
            ox, oy = last_frame[j]
            if (cx - half) <= ox < (cx + half) and (cy - half) <= oy < (cy + half):
                gx = (ox - (cx - half)) / cell_size
                gy = (oy - (cy - half)) / cell_size
                ax.plot(gx - 0.5, gy - 0.5, 'bx', markersize=10, markeredgewidth=2)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{scene}_social_pooling_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] Social Pooling 热力图 → {save_path}")


# ─────────────────────────────────────────────
#  命令行入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Social LSTM Visualization')
    # ETH/UCY 模式
    parser.add_argument('--scene', type=str, default='all',
                        choices=['all'] + SCENES)
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'trajectory', 'bar', 'heatmap'],
                        help='all=全部, trajectory=轨迹图, bar=条形图, heatmap=热力图')
    parser.add_argument('--num_show',          type=int,   default=8)
    parser.add_argument('--num_samples',       type=int,   default=20)
    parser.add_argument('--obs_len',           type=int,   default=8)
    parser.add_argument('--pred_len',          type=int,   default=12)
    parser.add_argument('--embedding_size',    type=int,   default=64)
    parser.add_argument('--rnn_size',          type=int,   default=128)
    parser.add_argument('--grid_size',         type=int,   default=4)
    parser.add_argument('--neighborhood_size', type=float, default=4.0)
    # nuScenes 模式
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='nuScenes 数据集根目录（含 train/ val/ test/）。'
                             '设置后切换为 nuScenes 可视化模式')
    parser.add_argument('--ckpt_name', type=str, default='nuscenes',
                        help='nuScenes checkpoint 前缀（默认 nuscenes）')
    return parser.parse_args()


# ─────────────────────────────────────────────
#  nuScenes 轨迹对比图
# ─────────────────────────────────────────────

def plot_trajectories_nuscenes(args, device, num_show=8):
    """nuScenes 模式：从 dataset_path/test/ 加载并绘制轨迹对比图。"""
    ckpt_name = args.ckpt_name or 'nuscenes'
    ckpt_path = os.path.join(CKPT_DIR, f'{ckpt_name}_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"  无 checkpoint {ckpt_path}，跳过轨迹图")
        return

    ckpt       = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get('args', {})
    obs_len           = saved_args.get('obs_len',           args.obs_len)
    pred_len          = saved_args.get('pred_len',          args.pred_len)
    embedding_size    = saved_args.get('embedding_size',    args.embedding_size)
    rnn_size          = saved_args.get('rnn_size',          args.rnn_size)
    grid_size         = saved_args.get('grid_size',         args.grid_size)
    neighborhood_size = saved_args.get('neighborhood_size', args.neighborhood_size)

    model = SocialLSTM(
        obs_len=obs_len, pred_len=pred_len,
        embedding_size=embedding_size, rnn_size=rnn_size,
        grid_size=grid_size, neighborhood_size=neighborhood_size,
        dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_dir = os.path.join(os.path.abspath(args.dataset_path), 'test')
    test_ds  = TrajectoryDataset(test_dir, obs_len=obs_len,
                                  pred_len=pred_len, min_ped=1)
    if len(test_ds) == 0:
        print("  测试集为空，跳过 nuScenes 轨迹图")
        return

    np.random.seed(42)
    indices = np.random.choice(len(test_ds), size=min(num_show, len(test_ds)),
                               replace=False)
    n_cols = min(4, len(indices))
    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle(f'Social LSTM — nuScenes 轨迹预测对比（{ckpt_name}）', fontsize=13)
    axes = np.array(axes).flatten()

    for ax_idx, seq_idx in enumerate(indices):
        obs_traj, pred_traj, obs_traj_rel, _ = test_ds[seq_idx]
        obs_traj     = obs_traj.to(device)
        pred_traj    = pred_traj.to(device)
        obs_traj_rel = obs_traj_rel.to(device)

        with torch.no_grad():
            pred = model.predict(obs_traj, obs_traj_rel,
                                 num_samples=args.num_samples)

        obs_np  = obs_traj.cpu().numpy()
        gt_np   = pred_traj.cpu().numpy()
        pred_np = pred.cpu().numpy()

        ax = axes[ax_idx]
        N  = obs_np.shape[1]
        colors = plt.cm.Set1(np.linspace(0, 1, max(N, 1)))

        for ped_i in range(N):
            c = colors[ped_i % len(colors)]
            ax.plot(obs_np[:, ped_i, 0], obs_np[:, ped_i, 1],
                    '-o', color=c, alpha=0.9, linewidth=1.5, markersize=3)
            ax.plot([obs_np[-1, ped_i, 0], gt_np[0, ped_i, 0]],
                    [obs_np[-1, ped_i, 1], gt_np[0, ped_i, 1]],
                    '--', color='green', alpha=0.6, linewidth=1)
            ax.plot(gt_np[:, ped_i, 0], gt_np[:, ped_i, 1],
                    '--s', color='green', alpha=0.7, linewidth=1.5, markersize=3)
            ax.plot([obs_np[-1, ped_i, 0], pred_np[0, ped_i, 0]],
                    [obs_np[-1, ped_i, 1], pred_np[0, ped_i, 1]],
                    ':', color='red', alpha=0.6, linewidth=1)
            ax.plot(pred_np[:, ped_i, 0], pred_np[:, ped_i, 1],
                    ':^', color='red', alpha=0.8, linewidth=1.5, markersize=3)

        ax.set_title(f'序列 {seq_idx}  (N={N}行人)', fontsize=9)
        ax.set_xlabel('x (m)', fontsize=8)
        ax.set_ylabel('y (m)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect('equal', adjustable='box')

    legend_elements = [
        Line2D([0], [0], color='blue',  lw=1.5, linestyle='-',  label='历史轨迹'),
        Line2D([0], [0], color='green', lw=1.5, linestyle='--', label='真实未来'),
        Line2D([0], [0], color='red',   lw=1.5, linestyle=':',  label='预测轨迹'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save_path = os.path.join(OUTPUT_DIR, f'{ckpt_name}_trajectory_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] nuScenes 轨迹对比图 → {save_path}")


# ─────────────────────────────────────────────
#  nuScenes ADE/FDE 条形图
# ─────────────────────────────────────────────

def plot_ade_fde_bar_nuscenes(args, device):
    """nuScenes 评估结果 vs ETH/UCY 论文均值，绘制分组条形图。"""
    from utils import compute_ade, compute_fde
    import matplotlib.patches as mpatches

    ckpt_name = args.ckpt_name or 'nuscenes'
    ckpt_path = os.path.join(CKPT_DIR, f'{ckpt_name}_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"  无 checkpoint {ckpt_path}，跳过条形图")
        return

    ckpt       = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get('args', {})
    obs_len    = saved_args.get('obs_len',  args.obs_len)
    pred_len   = saved_args.get('pred_len', args.pred_len)

    model = SocialLSTM(
        obs_len=obs_len, pred_len=pred_len,
        embedding_size=saved_args.get('embedding_size', args.embedding_size),
        rnn_size=saved_args.get('rnn_size', args.rnn_size),
        grid_size=saved_args.get('grid_size', args.grid_size),
        neighborhood_size=saved_args.get('neighborhood_size', args.neighborhood_size),
        dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_dir = os.path.join(os.path.abspath(args.dataset_path), 'test')
    test_ds  = TrajectoryDataset(test_dir, obs_len=obs_len,
                                  pred_len=pred_len, min_ped=1)
    if len(test_ds) == 0:
        print("  测试集为空，跳过条形图")
        return

    total_ade, total_fde, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(len(test_ds)):
            obs_traj, pred_traj, obs_traj_rel, _ = test_ds[i]
            obs_traj     = obs_traj.to(device)
            pred_traj    = pred_traj.to(device)
            obs_traj_rel = obs_traj_rel.to(device)
            pred = model.predict(obs_traj, obs_traj_rel,
                                 num_samples=args.num_samples)
            total_ade += compute_ade(pred, pred_traj)
            total_fde += compute_fde(pred, pred_traj)
            n += 1

    nuscenes_ade = total_ade / n
    nuscenes_fde = total_fde / n

    # ETH/UCY 论文各场景参考值
    eth_ucy_labels = ['ETH', 'Hotel', 'Univ', 'Zara1', 'Zara2']
    eth_ucy_ade    = [1.09,   0.79,   0.67,   0.47,    0.56]
    eth_ucy_fde    = [2.35,   1.76,   1.40,   1.00,    1.17]

    labels   = eth_ucy_labels + ['nuScenes']
    ade_vals = eth_ucy_ade    + [nuscenes_ade]
    fde_vals = eth_ucy_fde    + [nuscenes_fde]
    colors   = ['#4878CF'] * 5 + ['#E84646']

    x   = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Social LSTM — ADE / FDE 对比（nuScenes 本实验 vs ETH/UCY 论文）',
                 fontsize=13)

    for ax, metric, vals, ylabel in zip(
            axes, ['ADE', 'FDE'], [ade_vals, fde_vals], ['ADE (m)', 'FDE (m)']):
        bars = ax.bar(x, vals, 0.6, color=colors, alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{metric} 对比')
        ax.set_ylim(0, max(vals) * 1.3)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(handles=[
            mpatches.Patch(facecolor='#4878CF', label='ETH/UCY 论文实验值'),
            mpatches.Patch(facecolor='#E84646', label='nuScenes 本实验'),
        ], fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f'{ckpt_name}_ade_fde_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] nuScenes ADE/FDE 对比图 → {save_path}")


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"图片保存目录: {OUTPUT_DIR}\n")

    # ── nuScenes 可视化模式 ──
    if args.dataset_path is not None:
        if args.mode in ('all', 'trajectory'):
            print("── 1. nuScenes 轨迹对比图 ──────────────────────────")
            plot_trajectories_nuscenes(args, device, num_show=args.num_show)
        if args.mode in ('all', 'bar'):
            print("\n── 2. nuScenes ADE/FDE 对比条形图 ──────────────────")
            plot_ade_fde_bar_nuscenes(args, device)
        if args.mode in ('all', 'heatmap'):
            print("\n── 3. Social Pooling 热力图 ─────────────────────────")
            test_dir = os.path.join(os.path.abspath(args.dataset_path), 'test')
            test_ds  = TrajectoryDataset(test_dir, obs_len=args.obs_len,
                                         pred_len=args.pred_len, min_ped=2)
            if len(test_ds) > 0:
                # 复用原有热力图逻辑，直接传数据
                seq = None
                for i in range(len(test_ds)):
                    obs_traj, *_ = test_ds[i]
                    if obs_traj.shape[1] >= 2:
                        seq = obs_traj
                        break
                if seq is not None:
                    last_frame = seq[-1].numpy()
                    N = last_frame.shape[0]
                    grid_size = args.grid_size
                    neighborhood_size = args.neighborhood_size
                    n_show = min(N, 4)
                    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
                    if n_show == 1:
                        axes = [axes]
                    fig.suptitle(f'Social Pooling 网格 热力图 [nuScenes]  '
                                 f'(grid={grid_size}×{grid_size}, '
                                 f'neighborhood={neighborhood_size}m)', fontsize=12)
                    grid_mask = getGridMask(last_frame, neighborhood_size, grid_size)
                    for i in range(n_show):
                        ax = axes[i]
                        occupancy = grid_mask[i].sum(axis=0).reshape(grid_size, grid_size)
                        im = ax.imshow(occupancy, cmap='hot', vmin=0,
                                       vmax=max(occupancy.max(), 1), origin='lower')
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        ax.set_title(f'行人 {i}  @ ({last_frame[i,0]:.1f}, {last_frame[i,1]:.1f})',
                                     fontsize=9)
                    plt.tight_layout()
                    save_path = os.path.join(OUTPUT_DIR,
                                             f'{args.ckpt_name}_social_pooling_heatmap.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  [保存] Social Pooling 热力图 → {save_path}")
        print("\n全部完成！")
        return

    # ── ETH/UCY 可视化模式 ──
    scenes = SCENES if args.scene == 'all' else [args.scene]

    if args.mode in ('all', 'trajectory'):
        print("── 1. 轨迹对比图 ──────────────────────────")
        for scene in scenes:
            plot_trajectories(scene, args, device, num_show=args.num_show)

    if args.mode in ('all', 'bar'):
        print("\n── 2. ADE/FDE 条形图 ──────────────────────")
        plot_ade_fde_bar(args, device)

    if args.mode in ('all', 'heatmap'):
        print("\n── 3. Social Pooling 热力图 ────────────────")
        for scene in scenes:
            plot_social_heatmap(scene, args)

    print("\n全部完成！")


if __name__ == '__main__':
    main()
