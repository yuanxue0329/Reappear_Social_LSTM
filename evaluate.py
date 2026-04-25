"""
评估脚本：加载各场景最佳 checkpoint，输出 ADE/FDE 汇总表格
用法：
    python evaluate.py                     # 评估所有 5 个场景
    python evaluate.py --scene eth         # 只评估 eth 场景
"""

import os
import argparse
import torch

from dataset import TrajectoryDataset
from model import SocialLSTM
from utils import compute_ade, compute_fde


DATA_ROOT = os.path.join(os.path.dirname(__file__),
                         '..', 'ETH_UCY dataset', 'datasets')
CKPT_DIR  = os.path.join(os.path.dirname(__file__), 'checkpoints')
SCENES    = ['eth', 'hotel', 'univ', 'zara1', 'zara2']


def evaluate_scene(scene, args, device):
    ckpt_path = os.path.join(CKPT_DIR, f'{scene}_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"  [{scene}] 未找到 checkpoint：{ckpt_path}，请先运行 train.py")
        return None, None

    # 加载 checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get('args', {})

    # 优先使用保存时的模型超参
    obs_len           = saved_args.get('obs_len',           args.obs_len)
    pred_len          = saved_args.get('pred_len',          args.pred_len)
    embedding_size    = saved_args.get('embedding_size',    args.embedding_size)
    rnn_size          = saved_args.get('rnn_size',          args.rnn_size)
    grid_size         = saved_args.get('grid_size',         args.grid_size)
    neighborhood_size = saved_args.get('neighborhood_size', args.neighborhood_size)

    model = SocialLSTM(
        obs_len=obs_len,
        pred_len=pred_len,
        embedding_size=embedding_size,
        rnn_size=rnn_size,
        grid_size=grid_size,
        neighborhood_size=neighborhood_size,
        dropout=0.0   # 评估时关闭 dropout
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 测试集
    test_dir = os.path.join(DATA_ROOT, scene, 'test')
    test_ds  = TrajectoryDataset(
        test_dir,
        obs_len=obs_len,
        pred_len=pred_len,
        min_ped=1
    )
    if len(test_ds) == 0:
        print(f"  [{scene}] 测试集为空，跳过")
        return None, None

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

    ade = total_ade / n
    fde = total_fde / n
    return ade, fde


# ─────────────────────────────────────────────
#  nuScenes 评估
# ─────────────────────────────────────────────

def evaluate_nuscenes(args, device):
    """在 nuScenes test/ 上评估，使用 {ckpt_name}_best.pth。"""
    ckpt_name = args.ckpt_name or 'nuscenes'
    ckpt_path = os.path.join(CKPT_DIR, f'{ckpt_name}_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"  未找到 checkpoint：{ckpt_path}，请先运行 train.py --dataset_path ...")
        return None, None

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

    print(f"  checkpoint : {ckpt_path}")
    print(f"  obs_len={obs_len}  pred_len={pred_len}  "
          f"neighborhood_size={neighborhood_size}m  grid_size={grid_size}")
    print(f"  训练时 ADE={ckpt.get('ade', 'N/A')}  FDE={ckpt.get('fde', 'N/A')}")

    test_dir = os.path.join(os.path.abspath(args.dataset_path), 'test')
    test_ds  = TrajectoryDataset(test_dir, obs_len=obs_len,
                                  pred_len=pred_len, min_ped=1)
    print(f"  测试集 [test/] —— {len(test_ds)} 个序列\n")
    if len(test_ds) == 0:
        print("  [警告] 测试集为空，请检查 dataset_path 是否正确")
        return None, None

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

    return total_ade / n, total_fde / n


def parse_args():
    parser = argparse.ArgumentParser(description='Social LSTM Evaluation')
    # ETH/UCY 模式
    parser.add_argument('--scene', type=str, default='all',
                        choices=['all'] + SCENES)
    # 模型超参（checkpoint 中已保存，仅作兜底默认值）
    parser.add_argument('--obs_len',           type=int,   default=8)
    parser.add_argument('--pred_len',          type=int,   default=12)
    parser.add_argument('--embedding_size',    type=int,   default=64)
    parser.add_argument('--rnn_size',          type=int,   default=128)
    parser.add_argument('--grid_size',         type=int,   default=4)
    parser.add_argument('--neighborhood_size', type=float, default=4.0)
    parser.add_argument('--num_samples',       type=int,   default=20)
    # nuScenes 模式
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='nuScenes 数据集根目录（含 train/ val/ test/）。'
                             '设置后切换为 nuScenes 评估模式')
    parser.add_argument('--ckpt_name', type=str, default='nuscenes',
                        help='nuScenes checkpoint 前缀（默认 nuscenes）')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # ── nuScenes 评估模式 ──
    if args.dataset_path is not None:
        print("── nuScenes 评估 ─────────────────────────────────────────")
        ade, fde = evaluate_nuscenes(args, device)

        # ETH/UCY 原论文均值（作参考）
        ethucy_avg_ade = (1.09 + 0.79 + 0.67 + 0.47 + 0.56) / 5   # 0.716
        ethucy_avg_fde = (2.35 + 1.76 + 1.40 + 1.00 + 1.17) / 5   # 1.536

        print(f"{'数据集':<14} {'ADE (m)':>10} {'FDE (m)':>10}  "
              f"{'ETH/UCY论文均值':>16}")
        print('-' * 58)
        if ade is not None:
            print(f"{'nuScenes':<14} {ade:>10.4f} {fde:>10.4f}  "
                  f"ADE:{ethucy_avg_ade:.3f} / FDE:{ethucy_avg_fde:.3f}")
        else:
            print(f"{'nuScenes':<14} {'N/A':>10} {'N/A':>10}")
        return

    # ── ETH/UCY LOO 评估模式 ──
    scenes = SCENES if args.scene == 'all' else [args.scene]

    print(f"{'场景':<10} {'ADE (m)':>10} {'FDE (m)':>10}  {'来自论文ADE':>12}")
    print('-' * 50)
    # 原论文 Social LSTM 参考值（CVPR 2016）
    paper_ade = {'eth': 1.09, 'hotel': 0.79, 'univ': 0.67,
                 'zara1': 0.47, 'zara2': 0.56}

    valid_ades, valid_fdes = [], []
    for scene in scenes:
        ade, fde = evaluate_scene(scene, args, device)
        if ade is not None:
            ref = paper_ade.get(scene, '-')
            ref_str = f"{ref:.2f}" if isinstance(ref, float) else ref
            print(f"{scene:<10} {ade:>10.4f} {fde:>10.4f}  {ref_str:>12}")
            valid_ades.append(ade)
            valid_fdes.append(fde)
        else:
            print(f"{scene:<10} {'N/A':>10} {'N/A':>10}")

    if len(valid_ades) > 1:
        print('-' * 50)
        print(f"{'平均':<10} {sum(valid_ades)/len(valid_ades):>10.4f} "
              f"{sum(valid_fdes)/len(valid_fdes):>10.4f}")


if __name__ == '__main__':
    main()
