"""
nuScenes GIF 动态可视化脚本（Phase 5 — Section 9.4）

将 Social LSTM 预测轨迹叠加在 nuScenes v1.0-mini 真实相机图像上，
生成逐帧 GIF 动画：
  ━ 矢车菊蓝实线：历史轨迹（逐帧累积显示）
  ━ 绿色虚线    ：真实未来轨迹（进入预测段后全程显示）
  ━ 红色虚线    ：模型预测路径（进入预测段后全程显示）

用法：
    conda activate social_lstm
    cd "d:\\Python code\\car_control"

    # 自动遍历所有 test 场景，每场景最多 2 个 GIF
    python social_lstm/visualize_gif.py

    # 指定场景
    python social_lstm/visualize_gif.py --scene scene-0103 --max_gifs 3

    # 指定相机视角
    python social_lstm/visualize_gif.py --scene scene-1094 --camera CAM_FRONT_RIGHT

    # 使用 trainval checkpoint（脚本会自动检测 trainval 元数据路径）
    python social_lstm/visualize_gif.py --ckpt_name nuscenes_trainval \\
        --dataset_path "../nuScenes_trainval_dataset"

    # 若自动检测失败，可手动指定：
    python social_lstm/visualize_gif.py --ckpt_name nuscenes_trainval \\
        --dataset_path "../nuScenes_trainval_dataset" \\
        --dataroot "v1.0-trainval01_blobs/v1.0-trainval_meta" \\
        --version v1.0-trainval \\
        --image_root "v1.0-trainval01_blobs"
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw
from pyquaternion import Quaternion

# ── 确保能导入 Social LSTM 模块 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SocialLSTM

# ─────────────────────────────────────────────
#  路径常量
# ─────────────────────────────────────────────
_SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR         = os.path.join(_SCRIPT_DIR, 'checkpoints')
OUTPUT_DIR       = os.path.join(_SCRIPT_DIR, 'outputs', 'GIF')
DEFAULT_DATAROOT = os.path.join(_SCRIPT_DIR, '..', 'v1.0-mini')
DEFAULT_DATASET  = os.path.join(_SCRIPT_DIR, '..', 'nuScenes_dataset')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 自动检测 trainval 元数据候选路径（按优先级排序）
# 格式: (meta_root, version, image_root)
# meta_root/version/scene.json 必须存在
_WORK_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
TRAINVAL_CANDIDATES = [
    (
        os.path.join(_WORK_DIR, 'v1.0-trainval01_blobs', 'v1.0-trainval_meta'),
        'v1.0-trainval',
        os.path.join(_WORK_DIR, 'v1.0-trainval01_blobs'),
    ),
    (
        os.path.join(_WORK_DIR, 'v1.0-trainval_meta'),
        'v1.0-trainval',
        _WORK_DIR,
    ),
]

# ─────────────────────────────────────────────
#  颜色常量
# ─────────────────────────────────────────────
HIST_COLOR = (100, 149, 237)   # 矢车菊蓝（历史轨迹）
GT_COLOR   = ( 50, 200,  50)   # 绿（真实未来）
PRED_COLOR = (220,  50,  50)   # 红（预测）
TEXT_BG    = ( 20,  20,  20)   # 文字背景


# ─────────────────────────────────────────────
#  参数解析
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='nuScenes GIF 动态轨迹可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--ckpt_name',    type=str, default='nuscenes',
                        help='checkpoint 前缀（默认 nuscenes）')
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_DATASET,
                        help='转换后的 nuScenes txt 数据集目录')
    parser.add_argument('--dataroot',     type=str, default=DEFAULT_DATAROOT,
                        help='nuScenes 原始数据根目录（含 samples/ 和版本子目录）')
    parser.add_argument('--version',      type=str, default='v1.0-mini',
                        help='nuScenes 版本（v1.0-mini 或 v1.0-trainval）')
    parser.add_argument('--split',        type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='数据集分割（train/val/test，默认 test）')
    parser.add_argument('--image_root',   type=str, default=None,
                        help='相机图像根目录（默认同 --dataroot）；'
                             'trainval 中元数据与图像分开存放时需单独指定，'
                             '如 v1.0-trainval01_blobs')
    parser.add_argument('--scene',        type=str, default=None,
                        help='指定场景名（如 scene-0103）；默认遍历所有 test 场景')
    parser.add_argument('--camera',       type=str, default='CAM_FRONT',
                        choices=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                                 'CAM_BACK',  'CAM_BACK_LEFT',  'CAM_BACK_RIGHT'],
                        help='使用哪个相机视角（默认 CAM_FRONT）')
    parser.add_argument('--max_gifs',     type=int, default=2,
                        help='每个场景最多生成几个 GIF')
    parser.add_argument('--num_samples',  type=int, default=20,
                        help='模型采样次数（取均值）')
    parser.add_argument('--fps',          type=int, default=2,
                        help='GIF 帧率（Hz，建议与采样率一致）')
    parser.add_argument('--scale',        type=float, default=0.5,
                        help='图像缩放比例（0.5 = 半分辨率，减小文件体积）')
    parser.add_argument('--stride',       type=int, default=None,
                        help='窗口滑动步长（默认 obs_len，不重叠）')
    return parser.parse_args()


# ─────────────────────────────────────────────
#  nuScenes JSON 加载
# ─────────────────────────────────────────────

def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_nusc_luts(dataroot, version):
    """
    加载 nuScenes JSON，返回五张查找表：
      scene_lut      : scene_name  → scene_record
      sample_lut     : token       → sample_record
      sd_lut         : sample_token → {channel: sample_data_record}（仅关键帧）
      ego_pose_lut   : token       → ego_pose_record
      cs_lut         : token       → calibrated_sensor_record（含 channel 字段）

    注：sample_data.json 本身无 channel 字段，需通过：
        sample_data.calibrated_sensor_token
            → calibrated_sensor.sensor_token
            → sensor.channel
    """
    vdir = os.path.join(dataroot, version)
    print(f"  读取 JSON 目录: {vdir}")

    scenes          = _load_json(os.path.join(vdir, 'scene.json'))
    samples         = _load_json(os.path.join(vdir, 'sample.json'))
    sample_data_lst = _load_json(os.path.join(vdir, 'sample_data.json'))
    ego_poses       = _load_json(os.path.join(vdir, 'ego_pose.json'))
    cal_sensors     = _load_json(os.path.join(vdir, 'calibrated_sensor.json'))
    sensors         = _load_json(os.path.join(vdir, 'sensor.json'))

    scene_lut  = {s['name']: s for s in scenes}
    sample_lut = {s['token']: s for s in samples}

    # sensor_token → channel 名称（如 'CAM_FRONT'）
    sensor_channel = {s['token']: s['channel'] for s in sensors}

    # calibrated_sensor_token → record（附加 channel 字段）
    cs_lut = {}
    for cs in cal_sensors:
        cs = dict(cs)  # 避免修改原始对象
        cs['channel'] = sensor_channel.get(cs['sensor_token'], 'UNKNOWN')
        cs_lut[cs['token']] = cs

    ego_pose_lut = {ep['token']: ep for ep in ego_poses}

    # sample_token → {channel: sd_record}（仅 is_key_frame）
    sd_lut = {}
    for sd in sample_data_lst:
        if not sd['is_key_frame']:
            continue
        cs_token = sd['calibrated_sensor_token']
        channel  = cs_lut.get(cs_token, {}).get('channel', 'UNKNOWN')
        if channel == 'UNKNOWN':
            continue
        st = sd['sample_token']
        if st not in sd_lut:
            sd_lut[st] = {}
        sd_lut[st][channel] = sd

    print(f"  场景数: {len(scene_lut)}  样本数: {len(sample_lut)}"
          f"  关键帧数: {len(sd_lut)}")
    return scene_lut, sample_lut, sd_lut, ego_pose_lut, cs_lut


def scene_to_sample_tokens(scene_record, sample_lut):
    """沿链表展开 scene → 有序 sample_token 列表（索引即 frame_id）。"""
    tokens = []
    token  = scene_record['first_sample_token']
    while token:
        tokens.append(token)
        token = sample_lut[token]['next']
    return tokens


# ─────────────────────────────────────────────
#  坐标变换
# ─────────────────────────────────────────────

def world_to_pixel(point_3d, ego_pose, calibrated_sensor):
    """
    世界坐标 (x, y, z) → 像素坐标 (u, v)。
    若点位于相机后方则返回 None。

    三步变换：世界 → 自车 → 相机 → 像素
    """
    # Step 1: 世界 → 自车坐标系
    ego_rot   = Quaternion(ego_pose['rotation']).rotation_matrix
    ego_trans = np.array(ego_pose['translation'])
    p_ego     = ego_rot.T @ (point_3d - ego_trans)

    # Step 2: 自车 → 相机坐标系
    cam_rot   = Quaternion(calibrated_sensor['rotation']).rotation_matrix
    cam_trans = np.array(calibrated_sensor['translation'])
    p_cam     = cam_rot.T @ (p_ego - cam_trans)

    if p_cam[2] < 0.1:   # 点在相机后方
        return None

    # Step 3: 透视投影 → 像素
    K     = np.array(calibrated_sensor['camera_intrinsic'])
    pixel = K @ p_cam
    return int(pixel[0] / pixel[2]), int(pixel[1] / pixel[2])


# ─────────────────────────────────────────────
#  txt 数据读取与窗口提取
# ─────────────────────────────────────────────

def read_scene_txt(txt_path):
    """返回 {frame_id: [(ped_id, x, y), ...]}"""
    data = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            fid = int(float(parts[0]))
            pid = int(float(parts[1]))
            x, y = float(parts[2]), float(parts[3])
            data.setdefault(fid, []).append((pid, x, y))
    return data


def extract_windows(frame_data, obs_len, pred_len, stride=None):
    """
    提取所有满足条件的滑动窗口：
      - 窗口内帧连续
      - 至少有一个行人贯穿整个窗口（obs + pred）

    Returns:
        list of dict: {start_frame, ped_ids, obs_xy (obs_len,N,2), gt_xy (pred_len,N,2)}
    """
    total   = obs_len + pred_len
    frames  = sorted(frame_data.keys())
    if stride is None:
        stride = obs_len  # 不重叠窗口，避免生成相似 GIF

    windows = []
    for i in range(0, len(frames) - total + 1, stride):
        win = frames[i: i + total]
        # 检查帧连续性
        if win[-1] - win[0] != total - 1:
            continue
        # 所有窗口帧都出现的行人
        ped_sets = [set(p for p, _, _ in frame_data[f]) for f in win]
        common   = sorted(ped_sets[0].intersection(*ped_sets[1:]))
        if not common:
            continue

        N   = len(common)
        p2i = {pid: k for k, pid in enumerate(common)}
        obs_xy = np.zeros((obs_len,  N, 2), dtype=np.float32)
        gt_xy  = np.zeros((pred_len, N, 2), dtype=np.float32)

        for t, fid in enumerate(win[:obs_len]):
            for pid, x, y in frame_data[fid]:
                if pid in p2i:
                    obs_xy[t, p2i[pid]] = [x, y]

        for t, fid in enumerate(win[obs_len:]):
            for pid, x, y in frame_data[fid]:
                if pid in p2i:
                    gt_xy[t, p2i[pid]] = [x, y]

        windows.append(dict(start_frame=win[0], ped_ids=common,
                            obs_xy=obs_xy, gt_xy=gt_xy))
    return windows


# ─────────────────────────────────────────────
#  模型预测
# ─────────────────────────────────────────────

def run_predict(model, obs_xy, device, num_samples):
    """
    obs_xy: (obs_len, N, 2) numpy，绝对坐标
    返回:   (pred_len, N, 2) numpy，绝对坐标
    """
    obs_t   = torch.tensor(obs_xy, dtype=torch.float32, device=device)
    obs_rel = torch.zeros_like(obs_t)
    obs_rel[1:] = obs_t[1:] - obs_t[:-1]

    with torch.no_grad():
        pred = model.predict(obs_t, obs_rel, num_samples=num_samples)

    return pred.cpu().numpy()


# ─────────────────────────────────────────────
#  绘图辅助
# ─────────────────────────────────────────────

def _dot(draw, pt, color, r=5):
    if pt:
        u, v = pt
        draw.ellipse([u - r, v - r, u + r, v + r], fill=color)


def _line(draw, p1, p2, color, width=3):
    if p1 and p2:
        draw.line([p1, p2], fill=color, width=width)


def _project(xy_world, ego_pose, cs, scale, img_size, z=0.9):
    """世界坐标 → 缩放后像素坐标。超出图像范围返回 None。"""
    raw = world_to_pixel(np.array([xy_world[0], xy_world[1], z]),
                         ego_pose, cs)
    if raw is None:
        return None
    u = int(raw[0] * scale)
    v = int(raw[1] * scale)
    W, H = img_size
    return (u, v) if (0 <= u < W and 0 <= v < H) else None


# ─────────────────────────────────────────────
#  单窗口 GIF 生成
# ─────────────────────────────────────────────

def make_gif_for_window(scene_name, win_idx, window, pred_xy,
                        sample_tokens, sd_lut, ego_pose_lut, cs_lut,
                        camera, image_root, fps, scale, output_dir):
    """
    为一个滑动窗口生成 GIF，逐帧动态显示历史轨迹，
    进入预测段后一次性绘制完整 GT 和预测轨迹。
    image_root: 相机图像的根目录（samples/ 所在的父目录）。
    """
    obs_xy   = window['obs_xy']    # (obs_len,  N, 2)
    gt_xy    = window['gt_xy']     # (pred_len, N, 2)
    obs_len  = obs_xy.shape[0]
    pred_len = gt_xy.shape[0]
    total    = obs_len + pred_len
    N        = obs_xy.shape[1]
    start_f  = window['start_frame']

    # 对应帧的 sample_token 序列
    win_tokens = sample_tokens[start_f: start_f + total]
    if len(win_tokens) < total:
        print(f"    [跳过] sample_token 数量不足 ({len(win_tokens)} < {total})")
        return None

    gif_frames = []
    skipped    = 0

    for frame_idx in range(total):
        token  = win_tokens[frame_idx]
        cam_sd = sd_lut.get(token, {}).get(camera)

        if cam_sd is None:
            skipped += 1
            continue

        img_path = os.path.join(image_root, cam_sd['filename'])
        if not os.path.exists(img_path):
            skipped += 1
            continue

        img  = Image.open(img_path).convert('RGB')
        W0, H0 = img.size
        img  = img.resize((int(W0 * scale), int(H0 * scale)), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        ego_pose = ego_pose_lut[cam_sd['ego_pose_token']]
        cs       = cs_lut[cam_sd['calibrated_sensor_token']]
        isize    = img.size

        def P(xy, z=0.9):
            return _project(xy, ego_pose, cs, scale, isize, z)

        for ped_i in range(N):
            # ── 历史轨迹（逐帧累积） ──
            hist_end  = min(frame_idx + 1, obs_len)
            hist_pts  = [P(obs_xy[t, ped_i]) for t in range(hist_end)]
            for t in range(len(hist_pts) - 1):
                _line(draw, hist_pts[t], hist_pts[t + 1], HIST_COLOR, width=3)
            if hist_pts:
                _dot(draw, hist_pts[-1], HIST_COLOR, r=6)

            # ── 预测段：显示完整 GT（绿）和预测（红） ──
            if frame_idx >= obs_len - 1:
                last_obs = obs_xy[-1, ped_i]

                # 真实未来轨迹（从观测末尾连接）
                gt_chain = [last_obs] + [gt_xy[t, ped_i] for t in range(pred_len)]
                gt_pts   = [P(pt) for pt in gt_chain]
                for t in range(len(gt_pts) - 1):
                    _line(draw, gt_pts[t], gt_pts[t + 1], GT_COLOR, width=2)
                _dot(draw, gt_pts[-1], GT_COLOR, r=5)

                # 模型预测轨迹
                pr_chain = [last_obs] + [pred_xy[t, ped_i] for t in range(pred_len)]
                pr_pts   = [P(pt) for pt in pr_chain]
                for t in range(len(pr_pts) - 1):
                    _line(draw, pr_pts[t], pr_pts[t + 1], PRED_COLOR, width=2)
                _dot(draw, pr_pts[-1], PRED_COLOR, r=5)

        # ── 帧信息文字 ──
        t_sec  = frame_idx * 0.5   # 2Hz → 0.5s/帧
        phase  = f"观测 {frame_idx+1}/{obs_len}" \
                 if frame_idx < obs_len \
                 else f"预测 +{(frame_idx - obs_len + 1) * 0.5:.1f}s"
        header = (f"{scene_name} | {phase} | {camera} | "
                  f"N={N}行人 | t={t_sec:.1f}s")
        legend = "━ 历史(蓝)   ━ GT(绿)   ━ 预测(红)"

        W, H = img.size
        draw.rectangle([0, 0, W, 24], fill=TEXT_BG)
        draw.text((6, 5), header, fill=(255, 255, 255))
        draw.rectangle([0, H - 22, W, H], fill=TEXT_BG)
        draw.text((6, H - 18), legend, fill=(200, 200, 200))

        gif_frames.append(np.array(img))

    if not gif_frames:
        print(f"    [失败] 所有帧图像均缺失（{skipped}/{total} 帧跳过）")
        return None

    if skipped > 0:
        print(f"    [提示] {skipped}/{total} 帧图像不存在（可能不在 mini 中），已跳过")

    out_name = f"{scene_name}_w{win_idx:02d}_{camera}.gif"
    out_path = os.path.join(output_dir, out_name)
    imageio.mimsave(out_path, gif_frames, fps=fps, loop=0)
    return out_path


# ─────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # ── 加载 checkpoint ──
    ckpt_path = os.path.join(CKPT_DIR, f'{args.ckpt_name}_best.pth')
    if not os.path.exists(ckpt_path):
        print(f"[错误] 未找到 checkpoint: {ckpt_path}")
        print("  请先运行 train.py 训练模型，或通过 --ckpt_name 指定正确的前缀")
        return

    ckpt       = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get('args', {})
    obs_len    = saved_args.get('obs_len',           6)
    pred_len   = saved_args.get('pred_len',          8)
    emb_size   = saved_args.get('embedding_size',   64)
    rnn_size   = saved_args.get('rnn_size',         128)
    grid_size  = saved_args.get('grid_size',          4)
    n_size     = saved_args.get('neighborhood_size', 4.0)

    model = SocialLSTM(
        obs_len=obs_len, pred_len=pred_len,
        embedding_size=emb_size, rnn_size=rnn_size,
        grid_size=grid_size, neighborhood_size=n_size,
        dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"模型加载完成")
    print(f"  obs_len={obs_len}  pred_len={pred_len}  "
          f"neighborhood_size={n_size}m  grid_size={grid_size}")
    print(f"  checkpoint ADE={ckpt.get('ade','N/A')}  FDE={ckpt.get('fde','N/A')}\n")

    # ── 加载 nuScenes JSON（主数据源） ──
    primary_dataroot  = os.path.abspath(args.dataroot)
    primary_imgroot   = os.path.abspath(args.image_root) if args.image_root \
                        else primary_dataroot
    print(f"加载 nuScenes 元数据...")
    scene_lut, sample_lut, sd_lut, ego_pose_lut, cs_lut = \
        build_nusc_luts(primary_dataroot, args.version)
    print()

    # ── 确定目标场景列表 ──
    test_dir = os.path.join(os.path.abspath(args.dataset_path), args.split)
    if args.scene:
        scene_names = [args.scene]
    else:
        scene_names = [f.replace('.txt', '')
                       for f in sorted(os.listdir(test_dir))
                       if f.endswith('.txt')]

    # ── 自动检测并补充缺失场景的元数据（trainval 场景不在 mini JSON 中） ──
    missing = [s for s in scene_names if s not in scene_lut]
    # scene_name → 该场景图像所在的根目录
    scene_image_roots = {s: primary_imgroot for s in scene_lut}

    if missing:
        print(f"以下场景不在当前 JSON 中，尝试自动检测 trainval 元数据: {missing}")
        for meta_root, version, img_root in TRAINVAL_CANDIDATES:
            scene_json = os.path.join(meta_root, version, 'scene.json')
            if not os.path.exists(scene_json):
                continue
            print(f"  找到候选元数据: {meta_root} ({version})  图像根: {img_root}")
            extra_scene, extra_sample, extra_sd, extra_ep, extra_cs = \
                build_nusc_luts(meta_root, version)
            # 只合并仍然缺失的场景
            still_missing = []
            for s in missing:
                if s in extra_scene:
                    scene_lut[s]   = extra_scene[s]
                    sample_lut.update(extra_sample)
                    sd_lut.update(extra_sd)
                    ego_pose_lut.update(extra_ep)
                    cs_lut.update(extra_cs)
                    scene_image_roots[s] = img_root
                    print(f"    ✓ {s} 已从候选元数据加载")
                else:
                    still_missing.append(s)
            missing = still_missing
            if not missing:
                break
        if missing:
            print(f"  [警告] 以下场景在所有已知 JSON 中均未找到: {missing}")
            print(f"  可通过 --dataroot / --version / --image_root 手动指定路径")
        print()

    print(f"目标场景: {scene_names}")
    print(f"相机视角: {args.camera}  图像缩放: {args.scale}x  FPS: {args.fps}\n")

    total_gifs = 0
    stride     = args.stride  # None → extract_windows 自动设为 obs_len

    for scene_name in scene_names:
        txt_path = os.path.join(test_dir, f'{scene_name}.txt')
        if not os.path.exists(txt_path):
            print(f"[跳过] txt 文件不存在: {txt_path}")
            continue

        if scene_name not in scene_lut:
            print(f"[跳过] {scene_name}: 元数据未找到，无法生成 GIF")
            print(f"  请通过 --dataroot / --version / --image_root 指定对应的 nuScenes 元数据路径")
            continue

        image_root = scene_image_roots.get(scene_name, primary_imgroot)
        print(f"── {scene_name} {'─'*40}")

        # frame_id → sample_token
        sample_tokens = scene_to_sample_tokens(scene_lut[scene_name], sample_lut)
        print(f"  帧数: {len(sample_tokens)}  图像目录: {image_root}")

        # 读取 txt
        frame_data = read_scene_txt(txt_path)

        # 提取滑动窗口
        windows = extract_windows(frame_data, obs_len, pred_len, stride=stride)
        print(f"  有效窗口数: {len(windows)}  (obs={obs_len}+pred={pred_len}帧)")

        if not windows:
            print("  [跳过] 无满足条件的行人轨迹窗口\n")
            continue

        gif_count = 0
        for win_idx, window in enumerate(windows):
            if gif_count >= args.max_gifs:
                break

            N = window['obs_xy'].shape[1]
            print(f"  窗口 {win_idx:02d}: start_frame={window['start_frame']}  "
                  f"N={N}行人", end='  ')

            # 运行模型预测
            pred_xy = run_predict(model, window['obs_xy'], device, args.num_samples)

            # 生成 GIF
            out = make_gif_for_window(
                scene_name, win_idx, window, pred_xy,
                sample_tokens, sd_lut, ego_pose_lut, cs_lut,
                args.camera, image_root, args.fps, args.scale, OUTPUT_DIR
            )
            if out:
                size_kb = os.path.getsize(out) / 1024
                print(f"→ {os.path.basename(out)}  ({size_kb:.0f} KB)")
                gif_count  += 1
                total_gifs += 1
            else:
                print()

        print()

    print("=" * 60)
    print(f"全部完成！共生成 {total_gifs} 个 GIF")
    print(f"保存目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
