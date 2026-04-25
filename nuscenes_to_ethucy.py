"""
nuScenes → ETH/UCY 格式转换脚本（Phase 1）

将 nuScenes JSON 数据转换为 Social LSTM 兼容的 txt 格式：
  frame_id\tped_id\tx\ty  （制表符分隔，与 ETH/UCY 完全一致）

输出目录结构：
  nuScenes_dataset/
      train/   ← 7 个场景
      val/     ← 1 个场景
      test/    ← 2 个场景（包含行人丰富的场景）

用法：
  conda activate social_lstm
  python social_lstm/nuscenes_to_ethucy.py
  python social_lstm/nuscenes_to_ethucy.py --dataroot d:/path/to/nuscenes --version v1.0-mini
"""

import os
import json
import argparse


# ─────────────────────────────────────────────
#  纳入训练的行人子类别
# ─────────────────────────────────────────────
TARGET_PED_NAMES = {
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.police_officer',
}

# ─────────────────────────────────────────────
#  v1.0-mini 10 个场景的 train/val/test 划分
#  原则：将行人丰富的场景（含 peds 描述）放入 test，其余按顺序分配
# ─────────────────────────────────────────────
MINI_SPLIT = {
    'train': [
        'scene-0061',   # Parked truck, construction, intersection
        'scene-0553',   # Wait at intersection, peds crossing crosswalk
        'scene-0655',   # Parking lot, jaywalker
        'scene-0757',   # Busy intersection, bus
        'scene-0796',   # Scooter, peds on sidewalk
        'scene-0916',   # Parking lot, parked bicycles
        'scene-1077',   # Night, big street, bus stop
    ],
    'val': [
        'scene-1100',   # Night, peds in sidewalk, crosswalk
    ],
    'test': [
        'scene-0103',   # Many peds right（行人最丰富）
        'scene-1094',   # Night, many peds, PMD
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Convert nuScenes to ETH/UCY txt format')
    parser.add_argument('--dataroot', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'v1.0-mini'),
                        help='nuScenes 数据集根目录')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        help='nuScenes 版本（v1.0-mini / v1.0-trainval）')
    parser.add_argument('--outdir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'nuScenes_dataset'),
                        help='输出目录')
    parser.add_argument('--min_visibility', type=int, default=2,
                        help='最低可见性等级（1~4），低于此值的标注将被过滤')
    return parser.parse_args()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_lookup(records, key='token'):
    """将列表转为 {token: record} 字典，便于 O(1) 查询。"""
    return {r[key]: r for r in records}


def get_ped_category_tokens(categories):
    """返回目标行人类别的 token 集合。"""
    return {c['token'] for c in categories if c['name'] in TARGET_PED_NAMES}


def build_sample_to_anns(annotations):
    """从 sample_annotation 列表构建 {sample_token: [ann_token, ...]} 反向索引。
    
    raw sample.json 中没有 anns 字段，需要从 annotation 侧反向构建。
    """
    lut = {}
    for ann in annotations:
        st = ann['sample_token']
        if st not in lut:
            lut[st] = []
        lut[st].append(ann['token'])
    return lut


def convert_scene(scene, sample_lut, annotation_lut, instance_lut,
                  sample_to_anns, ped_cat_tokens, min_visibility):
    """
    将单个 scene 转换为 ETH/UCY 格式的行列表。

    Returns:
        rows: list of str，每行格式为 "frame_id\tped_id\tx\ty"
        stats: dict，统计信息
    """
    rows = []
    ped_id_map = {}     # instance_token → 整数 ped_id（从1开始）
    next_id = [1]       # 用列表以便在内部函数中修改

    def get_ped_id(instance_token):
        if instance_token not in ped_id_map:
            ped_id_map[instance_token] = next_id[0]
            next_id[0] += 1
        return ped_id_map[instance_token]

    # 沿链表还原时序
    token = scene['first_sample_token']
    frame_id = 0
    total_anns = 0
    kept_anns = 0

    while token:
        sample = sample_lut[token]
        ann_tokens = sample_to_anns.get(token, [])

        # 遍历该帧内所有标注
        for ann_token in ann_tokens:
            if ann_token not in annotation_lut:
                continue
            ann = annotation_lut[ann_token]
            total_anns += 1

            # 可见性过滤：visibility_token 是 "1"~"4" 的字符串
            try:
                vis = int(ann['visibility_token'])
            except (ValueError, KeyError):
                vis = 0
            if vis < min_visibility:
                continue

            # 类别过滤
            instance_token = ann['instance_token']
            if instance_token not in instance_lut:
                continue
            instance = instance_lut[instance_token]
            if instance['category_token'] not in ped_cat_tokens:
                continue

            # 提取全局坐标 (x, y)
            x, y = ann['translation'][0], ann['translation'][1]
            ped_id = get_ped_id(instance_token)

            rows.append(f"{float(frame_id)}\t{float(ped_id)}\t{x:.4f}\t{y:.4f}")
            kept_anns += 1

        token = sample['next']
        frame_id += 1

    stats = {
        'frames': frame_id,
        'total_anns': total_anns,
        'kept_anns': kept_anns,
        'unique_peds': len(ped_id_map),
    }
    return rows, stats


def main():
    args = parse_args()
    dataroot = os.path.abspath(args.dataroot)
    version_dir = os.path.join(dataroot, args.version)
    outdir = os.path.abspath(args.outdir)

    print(f"数据根目录: {dataroot}")
    print(f"版本目录  : {version_dir}")
    print(f"输出目录  : {outdir}")
    print(f"可见性阈值: {args.min_visibility}（过滤 < {args.min_visibility} 的标注）")
    print()

    # ── 加载所有 JSON ──
    print("加载 JSON 文件...")
    scenes       = load_json(os.path.join(version_dir, 'scene.json'))
    samples      = load_json(os.path.join(version_dir, 'sample.json'))
    annotations  = load_json(os.path.join(version_dir, 'sample_annotation.json'))
    instances    = load_json(os.path.join(version_dir, 'instance.json'))
    categories   = load_json(os.path.join(version_dir, 'category.json'))

    # ── 构建查找表 ──
    sample_lut     = build_lookup(samples)
    annotation_lut = build_lookup(annotations)
    instance_lut   = build_lookup(instances)
    scene_lut      = {s['name']: s for s in scenes}
    sample_to_anns = build_sample_to_anns(annotations)   # 反向索引

    ped_cat_tokens = get_ped_category_tokens(categories)
    print(f"目标行人类别 token 数量: {len(ped_cat_tokens)}")

    # ── 确定 train/val/test 划分 ──
    if args.version == 'v1.0-mini':
        split = MINI_SPLIT
    else:
        # trainval 版本：按场景名简单排序后 70/15/15 划分
        all_scene_names = sorted([s['name'] for s in scenes])
        n = len(all_scene_names)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        split = {
            'train': all_scene_names[:n_train],
            'val':   all_scene_names[n_train:n_train + n_val],
            'test':  all_scene_names[n_train + n_val:],
        }
        print(f"trainval 场景总数: {n}  train:{n_train}  val:{n_val}  test:{n - n_train - n_val}")

    # ── 创建输出目录 ──
    for split_name in ('train', 'val', 'test'):
        os.makedirs(os.path.join(outdir, split_name), exist_ok=True)

    # ── 逐场景转换 ──
    total_stats = {'frames': 0, 'total_anns': 0, 'kept_anns': 0, 'unique_peds': 0}
    all_scene_names_in_split = []
    for split_name, scene_names in split.items():
        all_scene_names_in_split.extend(scene_names)

    print(f"\n开始转换 {len(all_scene_names_in_split)} 个场景...\n")

    for split_name, scene_names in split.items():
        print(f"─── {split_name.upper()} ───")
        for scene_name in scene_names:
            if scene_name not in scene_lut:
                print(f"  [警告] 找不到场景: {scene_name}，跳过")
                continue

            scene = scene_lut[scene_name]
            rows, stats = convert_scene(
                scene, sample_lut, annotation_lut, instance_lut,
                sample_to_anns, ped_cat_tokens, args.min_visibility
            )

            out_path = os.path.join(outdir, split_name, f"{scene_name}.txt")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(rows))
                if rows:
                    f.write('\n')

            for k in total_stats:
                total_stats[k] += stats[k]

            status = "✓" if stats['kept_anns'] > 0 else "⚠ (无有效行人)"
            print(f"  {status} {scene_name:15s}  "
                  f"帧数:{stats['frames']:3d}  "
                  f"总标注:{stats['total_anns']:4d}  "
                  f"保留行人标注:{stats['kept_anns']:4d}  "
                  f"行人数:{stats['unique_peds']:3d}  "
                  f"→ {os.path.relpath(out_path, outdir)}")
        print()

    # ── 汇总统计 ──
    print("=" * 65)
    print(f"转换完成！")
    print(f"  场景总数    : {len(all_scene_names_in_split)}")
    print(f"  总帧数      : {total_stats['frames']}")
    print(f"  原始标注数  : {total_stats['total_anns']}")
    print(f"  保留标注数  : {total_stats['kept_anns']}  "
          f"（过滤率 {100*(1 - total_stats['kept_anns']/max(total_stats['total_anns'],1)):.1f}%）")
    print(f"  唯一行人数  : {total_stats['unique_peds']}")
    print(f"  输出目录    : {outdir}")

    # ── 格式验证 ──
    print("\n─── 格式验证（抽样前5行）───")
    sample_file = os.path.join(outdir, 'test',
                               f"{split.get('test', [''])[0]}.txt")
    if os.path.exists(sample_file):
        with open(sample_file, 'r') as f:
            lines = [f.readline().strip() for _ in range(5)]
        for line in lines:
            if line:
                parts = line.split('\t')
                assert len(parts) == 4, f"列数错误: {line}"
                float(parts[0]); float(parts[1]); float(parts[2]); float(parts[3])
                print(f"  OK: {line}")
        print("格式验证通过 ✓")
    else:
        print("  [提示] 测试集文件不存在，跳过格式验证")


if __name__ == '__main__':
    main()
