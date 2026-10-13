from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

CANONICAL_SCENE_TO_RAW = {
    'eth': ['eth/univ/true_pos_.csv'],
    'hotel': ['eth/hotel/true_pos_.csv'],
    'univ': ['ucy/univ/students001/true_pos_.csv', 'ucy/univ/students003/true_pos_.csv', 'ucy/univ/uni_examples/true_pos_.csv'],
    'zara1': ['ucy/zara/zara01/true_pos_.csv'],
    'zara2': ['ucy/zara/zara02/true_pos_.csv'],
}

def load_raw_table(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=',' if path.suffix.lower() == '.csv' else None)
    if arr.ndim != 2: raise ValueError(f'Expected a 2D table, got shape={arr.shape}')
    if arr.shape[0] == 4: arr = arr.T
    if arr.shape[1] < 4: raise ValueError(f'Expected at least 4 columns, got shape={arr.shape}')
    return arr[:, :4]

def build_scene_structures(path: Path):
    table = load_raw_table(path); tracks = defaultdict(list); frame_map = defaultdict(dict)
    for row in table:
        frame_id = int(round(float(row[0]))); ped_id = int(round(float(row[1]))); x = float(row[2]); y = float(row[3])
        tracks[ped_id].append((frame_id, x, y)); frame_map[frame_id][ped_id] = np.asarray([x, y], dtype=np.float32)
    for ped_id in tracks: tracks[ped_id].sort(key=lambda v: v[0])
    return tracks, frame_map

def infer_track_step(frames: list[int]) -> int:
    diffs = np.diff(frames)
    return 1 if len(diffs) == 0 else int(Counter(diffs.tolist()).most_common(1)[0][0])

def build_windows(tracks, frame_map, obs_len, pred_len, frame_step, sample_stride, seq_prefix, neighbor_topk):
    total_len = obs_len + pred_len; obs_list=[]; pred_list=[]; seq_list=[]; neigh_list=[]; neigh_mask_list=[]; sid=0
    for ped_id, track in tracks.items():
        if len(track) < total_len: continue
        coords = np.asarray([[x, y] for _, x, y in track], dtype=np.float32); frames = [f for f, _, _ in track]
        step = infer_track_step(frames) if frame_step is None else int(frame_step)
        if step <= 0: continue
        for start in range(0, len(track) - total_len + 1, sample_stride):
            wf = frames[start:start + total_len]
            if not np.all(np.diff(wf) == step): continue
            wc = coords[start:start + total_len]
            neigh = np.zeros((obs_len, neighbor_topk, 2), dtype=np.float32); neigh_mask = np.zeros((obs_len, neighbor_topk), dtype=np.float32)
            for t in range(obs_len):
                frame_id = wf[t]; target_xy = wc[t]; others = []
                for other_id, other_xy in frame_map[frame_id].items():
                    if other_id == ped_id: continue
                    rel = other_xy - target_xy; dist = float(np.linalg.norm(rel)); others.append((dist, rel))
                others.sort(key=lambda item: item[0])
                for j, (_, rel) in enumerate(others[:neighbor_topk]): neigh[t, j] = rel.astype(np.float32); neigh_mask[t, j] = 1.0
            obs_list.append(wc[:obs_len]); pred_list.append(wc[obs_len:]); seq_list.append(seq_prefix * 1_000_000 + sid); neigh_list.append(neigh); neigh_mask_list.append(neigh_mask); sid += 1
    if not obs_list: raise RuntimeError('No valid windows were extracted. Check data layout, obs/pred lengths, and frame_step.')
    return {'obs': np.stack(obs_list).astype(np.float32), 'pred': np.stack(pred_list).astype(np.float32), 'neighbor_rel': np.stack(neigh_list).astype(np.float32), 'neighbor_mask': np.stack(neigh_mask_list).astype(np.float32), 'seq_id': np.asarray(seq_list, dtype=np.int64)}

def save_npz(path: Path, arrs: dict):
    path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **arrs)

def combine_npz_records(npz_paths, out_path: Path):
    keys = ['obs', 'pred', 'neighbor_rel', 'neighbor_mask', 'seq_id']; combined = {k: [] for k in keys}
    for path in npz_paths:
        arr = np.load(path)
        for k in keys: combined[k].append(arr[k])
    np.savez_compressed(out_path, **{k: np.concatenate(v, axis=0) for k, v in combined.items()})

def main():
    parser = argparse.ArgumentParser(description='Build strict ETH-UCY five-fold leave-one-out .npz files with social neighbor context.')
    parser.add_argument('--raw_root', type=str, required=True); parser.add_argument('--out_dir', type=str, default='data/ethucy_5fold_social'); parser.add_argument('--obs_len', type=int, default=8); parser.add_argument('--pred_len', type=int, default=12); parser.add_argument('--frame_step', type=str, default='auto'); parser.add_argument('--sample_stride', type=int, default=1); parser.add_argument('--neighbor_topk', type=int, default=4)
    args = parser.parse_args(); raw_root = Path(args.raw_root); out_dir = Path(args.out_dir); per_scene_dir = out_dir / 'canonical_scenes'; frame_step = None if args.frame_step == 'auto' else int(args.frame_step); scene_npzs={}; manifest={'protocol':'strict_five_fold_leave_one_out','excluded_raw_sequences':['ucy/zara/zara03/true_pos_.csv'],'canonical_scenes':{},'obs_len':args.obs_len,'pred_len':args.pred_len,'neighbor_topk':args.neighbor_topk}
    for scene_idx, (scene_name, rel_paths) in enumerate(CANONICAL_SCENE_TO_RAW.items(), start=1):
        parts = {k: [] for k in ['obs','pred','neighbor_rel','neighbor_mask','seq_id']}; used_files=[]
        for part_idx, rel_path in enumerate(rel_paths, start=1):
            tracks, frame_map = build_scene_structures(raw_root / rel_path); arrs = build_windows(tracks, frame_map, args.obs_len, args.pred_len, frame_step, args.sample_stride, scene_idx * 100 + part_idx, args.neighbor_topk)
            for k in parts: parts[k].append(arrs[k])
            used_files.append(rel_path)
        scene_arrs = {k: np.concatenate(v, axis=0) for k,v in parts.items()}; scene_npz = per_scene_dir / f'{scene_name}.npz'; save_npz(scene_npz, scene_arrs); scene_npzs[scene_name]=scene_npz; manifest['canonical_scenes'][scene_name]={'raw_files':used_files,'num_windows':int(len(scene_arrs['obs']))}; print(f'Saved canonical scene {scene_name}: {len(scene_arrs["obs"])} windows -> {scene_npz}')
    folds={}; all_scenes=list(CANONICAL_SCENE_TO_RAW.keys())
    for test_scene in all_scenes:
        train_scenes=[s for s in all_scenes if s != test_scene]; fold_dir = out_dir / test_scene; fold_dir.mkdir(parents=True, exist_ok=True); combine_npz_records([scene_npzs[s] for s in train_scenes], fold_dir / 'train.npz'); combine_npz_records([scene_npzs[test_scene]], fold_dir / 'test.npz'); train_n=int(len(np.load(fold_dir / 'train.npz')['obs'])); test_n=int(len(np.load(fold_dir / 'test.npz')['obs'])); folds[test_scene]={'train_scenes':train_scenes,'test_scene':test_scene,'num_train_windows':train_n,'num_test_windows':test_n}; json.dump(folds[test_scene], open(fold_dir / 'manifest.json','w',encoding='utf-8'), indent=2); print(f'Built fold={test_scene}: train={train_n}, test={test_n}')
    manifest['folds']=folds; json.dump(manifest, open(out_dir / 'manifest.json','w',encoding='utf-8'), indent=2); print(f'Saved manifest to {out_dir / "manifest.json"}')
if __name__ == '__main__': main()
