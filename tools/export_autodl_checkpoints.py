#!/usr/bin/env python3
"""Inventory & explicitly copy *only* frozen paper checkpoints, never raw data.

Default is manifest-only. The --include-weights flag is an explicit opt-in
for a private staging destination. Do not commit such a destination to GitHub.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path

FOLDS=('eth','hotel','univ','zara1','zara2'); SEEDS=(42,43,44)

def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for buf in iter(lambda:f.read(2*1024*1024),b''):h.update(buf)
 return h.hexdigest()

def pairs():
 for variant in ('baseline','multihyp','multihyp_social'):
  for fold in FOLDS:
   for seed in SEEDS:
    rel=f'runs/tmm_stage3e_ethucy_final/ethucy_5fold_social/{variant}/{fold}/seed_{seed}/best_model.pt'
    yield f'ethucy/{variant}/{fold}/seed_{seed}.pt',rel
 for seed in SEEDS:
  yield f'sdd/generator/seed_{seed}.pt',f'runs/tmm_stage3b_raw_control/sdd_raw_visual/multihyp_social/sdd/seed_{seed}/best_model.pt'
  yield f'sdd/visual/seed_{seed}.pt',f'runs/tmm_stage3d_decoupled/decoupled_social_visual_adapter/m16_160px/mobilenet_v3_small/frozen/pretrained/cscale_0.2/soft_t0.05_d0.0001/seed_{seed}/geometry_best.pt'

def main():
 a=argparse.ArgumentParser()
 a.add_argument('--source',default='/root/autodl-tmp/TMM_Traj/source/tmm_v1')
 a.add_argument('--destination',default='/root/autodl-tmp/ContextGuided_export')
 a.add_argument('--include-weights',action='store_true',help='Copy model weights into a private staging folder. Never auto-publish.')
 args=a.parse_args();src=Path(args.source).resolve();out=Path(args.destination).resolve()
 if not src.exists():raise SystemExit(f'Source AutoDL project not found: {src}')
 if out==src or src in out.parents and 'runs' not in out.parts:
  print('Warning: destination resides within the original project; ensure it is not Git-tracked.')
 out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for dstrel,rel in pairs():
  p=src/rel; found=p.is_file()
  rec={'published_relative_path':dstrel,'source_relative_path':rel,'exists':found}
  if found:
   rec['size_bytes']=p.stat().st_size; rec['sha256']=sha256(p)
   if args.include_weights:
    dest=out/'checkpoints'/dstrel;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dest)
    if sha256(dest)!=rec['sha256']:raise RuntimeError('Copy hash mismatch '+str(dest))
  rows.append(rec)
 audit={'mode':'private_weight_staging' if args.include_weights else 'inventory_only','raw_datasets_exported':False,'weights_publicly_licensed':'NOT_VERIFIED','expected_count':len(rows),'found_count':sum(x['exists'] for x in rows),'files':rows}
 (out/'checkpoint_manifest.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'found':audit['found_count'],'expected':len(rows),'manifest':str(out/'checkpoint_manifest.json'),'copied_weights':bool(args.include_weights)},indent=2))
 if audit['found_count']<len(rows):print('Some paper checkpoint paths were not found. They may be under alternate AutoDL directories; do not fabricate placeholders.')
if __name__=='__main__':main()
