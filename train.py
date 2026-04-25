"""
训练脚本：Leave-One-Out 5 轮交叉验证
  - 每轮以一个场景为测试集，其他 4 个场景的 train/ 数据合并训练
  - 训练完成后在对应 test/ 上评估 ADE/FDE
  - 最佳 checkpoint 保存至 checkpoints/{scene}_best.pth
"""

import os
import argparse
import time
import torch
import torch.optim as optim
import torch.amp
from torch.utils.data import DataLoader, ConcatDataset

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[提示] 未安装 tqdm，使用简单进度输出。可运行: pip install tqdm")

from dataset import TrajectoryDataset, seq_collate
from model import SocialLSTM
from utils import gaussian_likelihood_loss, compute_ade, compute_fde


# ─────────────────────────────────────────────
#  数据集路径配置
# ─────────────────────────────────────────────

DATA_ROOT = os.path.join(os.path.dirname(__file__),
                         '..', 'ETH_UCY dataset', 'datasets')
SCENES = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  训练单个场景（LOO 其中一轮）
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  nuScenes 训练（train/val/test 直接划分模式）
# ─────────────────────────────────────────────

def train_nuscenes(args, device):
    """nuScenes 模式：直接使用 train/ val/ test/ 三目录，无需 LOO。"""
    dataset_path = os.path.abspath(args.dataset_path)
    ckpt_name    = args.ckpt_name or 'nuscenes'
    ckpt_path    = os.path.join(CKPT_DIR, f'{ckpt_name}_best.pth')

    print(f"\n{'='*60}")
    print(f"  nuScenes 训练模式")
    print(f"  数据目录: {dataset_path}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    # ── 训练集：train/ 下所有 txt ──
    train_dir = os.path.join(dataset_path, 'train')
    train_ds  = TrajectoryDataset(
        train_dir,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        min_ped=1
    )
    print(f"  加载训练集 [train/] —— {len(train_ds)} 个序列")
    if len(train_ds) == 0:
        print("  [错误] 训练集为空，请检查数据目录")
        return None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=seq_collate,
        num_workers=0
    )

    # ── 验证集：val/ 下所有 txt ──
    val_dir = os.path.join(dataset_path, 'val')
    val_ds  = TrajectoryDataset(
        val_dir,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        min_ped=1
    )
    print(f"  加载验证集 [val/]   —— {len(val_ds)} 个序列")

    # ── 模型 ──
    model = SocialLSTM(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_size=args.embedding_size,
        rnn_size=args.rnn_size,
        grid_size=args.grid_size,
        neighborhood_size=args.neighborhood_size,
        dropout=args.dropout
    ).to(device)

    optimizer  = optim.RMSprop(model.parameters(), lr=args.lr)
    amp_enabled = args.amp and (device.type == 'cuda')
    scaler      = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    if amp_enabled:
        print(f"  [AMP] 混合精度训练已启用")

    best_ade = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.reset_timing()
        epoch_loss = 0.0
        n_batches  = 0
        t_epoch    = time.perf_counter()
        t_backward_total = 0.0

        if HAS_TQDM:
            batch_iter = tqdm(
                train_loader,
                desc=f"  Ep {epoch:4d}/{args.epochs}",
                ncols=95, leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]'
            )
        else:
            batch_iter = train_loader

        for batch in batch_iter:
            (obs_traj, pred_traj,
             obs_traj_rel, pred_traj_rel,
             seq_start_end) = batch

            obs_traj      = obs_traj.to(device)
            pred_traj     = pred_traj.to(device)
            obs_traj_rel  = obs_traj_rel.to(device)
            pred_traj_rel = pred_traj_rel.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                outputs = model(obs_traj, obs_traj_rel, pred_traj_rel, seq_start_end)
                loss    = gaussian_likelihood_loss(outputs, pred_traj_rel)

            _t_bwd = time.perf_counter()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_backward_total += time.perf_counter() - _t_bwd

            epoch_loss += loss.item()
            n_batches  += 1

            if HAS_TQDM:
                mem_mb = torch.cuda.memory_allocated() / 1e6 if device.type == 'cuda' else 0
                n = model._n_calls
                if n > 0:
                    mask_pct = 100 * model.t_mask / (model.t_mask + model.t_lstm + 1e-9)
                    lstm_pct = 100 * model.t_lstm / (model.t_mask + model.t_lstm + 1e-9)
                else:
                    mask_pct = lstm_pct = 0
                batch_iter.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'mask%': f'{mask_pct:.0f}',
                    'lstm%': f'{lstm_pct:.0f}',
                    'GPU内存': f'{mem_mb:.0f}MB'
                }, refresh=False)

        avg_loss  = epoch_loss / max(n_batches, 1)
        t_epoch_s = time.perf_counter() - t_epoch
        n = model._n_calls
        timing_str = ''
        if n > 0:
            mask_ms  = model.t_mask     / n * 1000
            lstm_ms  = model.t_lstm     / n * 1000
            bwd_ms   = t_backward_total / n * 1000
            total_ms = t_epoch_s        / n * 1000
            timing_str = (
                f"  [时间分解/batch] "
                f"掩码(GPU):{mask_ms:.1f}ms "
                f"神经网:{lstm_ms:.1f}ms "
                f"反向:{bwd_ms:.1f}ms "
                f"共:{total_ms:.1f}ms"
            )

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            t_eval = time.perf_counter()
            ade, fde = evaluate(model, val_ds, device, args)
            t_eval_s = time.perf_counter() - t_eval
            print(f"  Epoch [{epoch:4d}/{args.epochs}] "
                  f"Loss={avg_loss:.4f}  "
                  f"ADE={ade:.4f}m  FDE={fde:.4f}m  "
                  f"训练:{t_epoch_s:.1f}s 评估:{t_eval_s:.1f}s")
            if timing_str:
                print(timing_str)
            if ade < best_ade:
                best_ade = ade
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ade': ade,
                    'fde': fde,
                    'args': vars(args)
                }, ckpt_path)
                print(f"    ✓ 保存最佳模型 → {ckpt_path}  (ADE={ade:.4f}m)")
        else:
            if epoch % 10 == 0:
                print(f"  Epoch [{epoch:4d}/{args.epochs}] Loss={avg_loss:.4f}  "
                      f"耗时:{t_epoch_s:.1f}s")
                if timing_str:
                    print(timing_str)

    print(f"\n  [完成] nuScenes 最佳 ADE = {best_ade:.4f}m")
    return best_ade


# ─────────────────────────────────────────────
#  ETH/UCY Leave-One-Out 训练（原有逻辑）
# ─────────────────────────────────────────────

def train_one_scene(test_scene, args, device):
    print(f"\n{'='*60}")
    print(f"  LOO: 测试场景 = {test_scene}")
    print(f"{'='*60}")

    # 训练集：其他 4 个场景的 train 目录
    train_datasets = []
    for scene in SCENES:
        if scene == test_scene:
            continue
        train_dir = os.path.join(DATA_ROOT, scene, 'train')
        if not os.path.isdir(train_dir):
            print(f"  [警告] 找不到目录: {train_dir}，跳过")
            continue
        ds = TrajectoryDataset(
            train_dir,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=args.skip,
            min_ped=1
        )
        if len(ds) > 0:
            train_datasets.append(ds)
            print(f"  加载训练集 [{scene}/train] —— {len(ds)} 个序列")

    if len(train_datasets) == 0:
        print(f"  [错误] 无可用训练数据，跳过 {test_scene}")
        return

    # 合并 4 个场景的训练集
    # ConcatDataset 要求 __len__ 和 __getitem__，TrajectoryDataset 已实现
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=seq_collate,
        num_workers=0
    )

    # 验证集（测试场景的 val/ 目录，若存在）
    val_dir = os.path.join(DATA_ROOT, test_scene, 'test')
    val_ds  = TrajectoryDataset(val_dir,
                                obs_len=args.obs_len,
                                pred_len=args.pred_len,
                                min_ped=1)
    print(f"  加载测试集 [{test_scene}/test]  —— {len(val_ds)} 个序列")

    # 模型
    model = SocialLSTM(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_size=args.embedding_size,
        rnn_size=args.rnn_size,
        grid_size=args.grid_size,
        neighborhood_size=args.neighborhood_size,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=args.lr
    )

    amp_enabled = args.amp and (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    if amp_enabled:
        print(f"  [AMP] 混合精度训练已启用")

    best_ade = float('inf')
    ckpt_path = os.path.join(CKPT_DIR, f'{test_scene}_best.pth')

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.reset_timing()   # 清零每 epoch 的计时器
        epoch_loss = 0.0
        n_batches  = 0
        t_epoch    = time.perf_counter()
        t_backward_total = 0.0

        # ── tqdm 进度条（如果安装了 tqdm）──
        if HAS_TQDM:
            batch_iter = tqdm(
                train_loader,
                desc=f"  Ep {epoch:4d}/{args.epochs}",
                ncols=95, leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]'
            )
        else:
            batch_iter = train_loader

        for batch in batch_iter:
            (obs_traj, pred_traj,
             obs_traj_rel, pred_traj_rel,
             seq_start_end) = batch

            obs_traj      = obs_traj.to(device)
            pred_traj     = pred_traj.to(device)
            obs_traj_rel  = obs_traj_rel.to(device)
            pred_traj_rel = pred_traj_rel.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                outputs = model(obs_traj, obs_traj_rel, pred_traj_rel, seq_start_end)
                # 目标为相对位移（与 decoder teacher-forcing 输入语义一致）
                # 推理时 predict() 将 (mux,muy) 作为位移累加，故此处必须对齐到 pred_traj_rel
                loss = gaussian_likelihood_loss(outputs, pred_traj_rel)

            _t_bwd = time.perf_counter()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_backward_total += time.perf_counter() - _t_bwd

            epoch_loss += loss.item()
            n_batches  += 1

            # ── 实时更新 tqdm 后缀信息 ──
            if HAS_TQDM:
                mem_mb = torch.cuda.memory_allocated() / 1e6 if device.type == 'cuda' else 0
                n = model._n_calls
                if n > 0:
                    mask_pct = 100 * model.t_mask / (model.t_mask + model.t_lstm + 1e-9)
                    lstm_pct = 100 * model.t_lstm / (model.t_mask + model.t_lstm + 1e-9)
                else:
                    mask_pct = lstm_pct = 0
                batch_iter.set_postfix({
                    'loss': f'{loss.item():.3f}',
                    'mask%': f'{mask_pct:.0f}',
                    'lstm%': f'{lstm_pct:.0f}',
                    'GPU内存': f'{mem_mb:.0f}MB'
                }, refresh=False)

        # ── Epoch 结束：打印详细源耗分析 ──
        avg_loss   = epoch_loss / max(n_batches, 1)
        t_epoch_s  = time.perf_counter() - t_epoch
        n = model._n_calls
        if n > 0:
            mask_ms  = model.t_mask         / n * 1000
            lstm_ms  = model.t_lstm         / n * 1000
            bwd_ms   = t_backward_total     / n * 1000
            total_ms = t_epoch_s            / n * 1000
            timing_str = (
                f"  [时间分解/batch] "
                f"掩码(GPU):{mask_ms:.1f}ms "
                f"神经网:{lstm_ms:.1f}ms "
                f"反向:{bwd_ms:.1f}ms "
                f"共:{total_ms:.1f}ms"
            )
        else:
            timing_str = ""

        # ── 每 eval_every epoch 评估一次 ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            t_eval = time.perf_counter()
            ade, fde = evaluate(model, val_ds, device, args)
            t_eval_s = time.perf_counter() - t_eval
            print(f"  Epoch [{epoch:4d}/{args.epochs}] "
                  f"Loss={avg_loss:.4f}  "
                  f"ADE={ade:.4f}m  FDE={fde:.4f}m  "
                  f"训练:{t_epoch_s:.1f}s 评估:{t_eval_s:.1f}s")
            if timing_str:
                print(timing_str)
            if ade < best_ade:
                best_ade = ade
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ade': ade,
                    'fde': fde,
                    'args': vars(args)
                }, ckpt_path)
                print(f"    ✓ 保存最佳模型 → {ckpt_path}  (ADE={ade:.4f}m)")
        else:
            if epoch % 10 == 0:
                print(f"  Epoch [{epoch:4d}/{args.epochs}] Loss={avg_loss:.4f}  "
                      f"耗时:{t_epoch_s:.1f}s")
                if timing_str:
                    print(timing_str)

    print(f"\n  [完成] {test_scene} 最佳 ADE = {best_ade:.4f}m")
    return best_ade


# ─────────────────────────────────────────────
#  评估函数
# ─────────────────────────────────────────────

def evaluate(model, dataset, device, args):
    """在数据集上计算平均 ADE 和 FDE。"""
    model.eval()
    total_ade, total_fde, n = 0.0, 0.0, 0

    for i in range(len(dataset)):
        (obs_traj, pred_traj, obs_traj_rel, _) = dataset[i]
        obs_traj     = obs_traj.to(device)
        pred_traj    = pred_traj.to(device)
        obs_traj_rel = obs_traj_rel.to(device)

        pred = model.predict(obs_traj, obs_traj_rel, num_samples=args.num_samples)

        total_ade += compute_ade(pred, pred_traj)
        total_fde += compute_fde(pred, pred_traj)
        n += 1

    if n == 0:
        return float('nan'), float('nan')
    return total_ade / n, total_fde / n


# ─────────────────────────────────────────────
#  命令行入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Social LSTM Training')
    # 数据
    parser.add_argument('--obs_len',   type=int, default=6)
    parser.add_argument('--pred_len',  type=int, default=8)
    parser.add_argument('--skip',      type=int, default=1)
    # 模型
    parser.add_argument('--embedding_size',    type=int,   default=64)
    parser.add_argument('--rnn_size',          type=int,   default=128)
    parser.add_argument('--grid_size',         type=int,   default=4)
    parser.add_argument('--neighborhood_size', type=float, default=4.0)
    parser.add_argument('--dropout',           type=float, default=0.5)
    # 训练
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=0.003)
    parser.add_argument('--weight_decay',type=float, default=5e-4)
    parser.add_argument('--clip',        type=float, default=10.0)
    parser.add_argument('--eval_every',  type=int,   default=20)
    parser.add_argument('--batch_size',  type=int,   default=64,
                        help='每次前向传播处理的序列数（沿N维拼接）')
    parser.add_argument('--num_samples', type=int,   default=10,
                        help='测试采样次数（取均值）')
    # 场景选择（默认训练所有场景，也可指定一个快速调试）
    parser.add_argument('--scene', type=str, default='all',
                        choices=['all'] + SCENES,
                        help='指定测试场景，"all" 表示全部 5 轮 LOO（仅 ETH/UCY 模式）')
    parser.add_argument('--amp', action='store_true',
                        help='启用混合精度训练 (AMP)，需 CUDA GPU')
    # nuScenes 模式
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='nuScenes 数据集根目录（含 train/ val/ test/ 子目录）。'
                             '设置后自动切换为 nuScenes 直接划分模式，跳过 LOO')
    parser.add_argument('--ckpt_name', type=str, default='nuscenes',
                        help='nuScenes 模式下 checkpoint 文件名前缀（默认 nuscenes）')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    t0 = time.time()

    # ── nuScenes 模式 ──
    if args.dataset_path is not None:
        best_ade = train_nuscenes(args, device)
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  nuScenes 训练完成，总耗时 {elapsed/60:.1f} 分钟")
        print(f"  最佳 ADE = {best_ade:.4f} m" if best_ade is not None else "  训练失败")
        print(f"{'='*60}")
        return

    # ── ETH/UCY LOO 模式 ──
    scenes_to_run = SCENES if args.scene == 'all' else [args.scene]

    results = {}
    for scene in scenes_to_run:
        best_ade = train_one_scene(scene, args, device)
        results[scene] = best_ade

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  所有场景训练完成，总耗时 {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")
    for s, a in results.items():
        if a is not None:
            print(f"  {s:<8}  最佳 ADE = {a:.4f} m")


if __name__ == '__main__':
    main()
