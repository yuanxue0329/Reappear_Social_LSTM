import os

test_dir = 'nuScenes_trainval_dataset/test'
obs_len, pred_len = 6, 8
total = obs_len + pred_len

target_seqs = {49, 188, 482, 613, 743, 892, 1393, 1605}

seq_idx = 0
results = {}

for fname in sorted(os.listdir(test_dir)):
    if not fname.endswith('.txt'):
        continue
    scene = fname.replace('.txt', '')

    frame_data = {}
    with open(os.path.join(test_dir, fname), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            fid = int(float(parts[0]))
            pid = int(float(parts[1]))
            frame_data.setdefault(fid, set()).add(pid)

    frames = sorted(frame_data.keys())
    stride = obs_len

    for i in range(0, len(frames) - total + 1, stride):
        win = frames[i: i + total]
        if win[-1] - win[0] != total - 1:
            continue
        ped_sets = [frame_data[f] for f in win]
        common = ped_sets[0].intersection(*ped_sets[1:])
        if not common:
            continue

        if seq_idx in target_seqs:
            results[seq_idx] = (scene, win[0], win[-1], sorted(common))
        seq_idx += 1

    if len(results) == len(target_seqs):
        break

print("扫描总序列数:", seq_idx)
print()
header = "%-6s  %-18s  %-12s  %s" % ("序列", "场景", "帧范围", "行人ID")
print(header)
print('-' * 65)
for sid in sorted(results):
    scene, f0, f1, peds = results[sid]
    print("%-6d  %-18s  %3d ~ %-3d     %s" % (sid, scene, f0, f1, peds))
