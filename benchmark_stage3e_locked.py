from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.dataset import build_dataloaders
from data.visual_dataset import build_visual_dataloaders
from models import build_model, build_visual_model
from train_visual_decoupled import load_generator_checkpoint
from utils.complexity import benchmark_callable, count_parameters, model_size_bytes


def load_standard(model, ckpt, device):
    d = torch.load(ckpt, map_location=device)
    model.load_state_dict(d.get("model_state_dict", d))
    model.eval()
    return d


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup",type=int,default=200)
    ap.add_argument("--repeats",type=int,default=2000)
    ap.add_argument("--out",default="runs/tmm_stage3e_latency.json")
    args=ap.parse_args()

    device=torch.device(args.device)
    seed=args.seed
    raw=Path("runs/tmm_stage3b_raw_control/sdd_raw_visual")
    social_ck=raw/"multihyp_social"/"sdd"/f"seed_{seed}"/"best_model.pt"
    geom=Path("runs/tmm_stage3d_decoupled/decoupled_social_visual_adapter/m16_160px/mobilenet_v3_small/frozen/pretrained/cscale_0.2/soft_t0.05_d0.0001")/f"seed_{seed}"/"geometry_best.pt"
    if not social_ck.exists(): raise FileNotFoundError(social_ck)
    if not geom.exists(): raise FileNotFoundError(geom)

    plain=build_dataloaders("data/sdd_raw_visual","sdd",batch_size=1,num_workers=0,val_source="train_group",split_seed=seed)
    b=next(iter(plain["val"]))
    obs=b["obs"].to(device)
    neigh=b["neighbor_rel"].to(device)
    mask=b["neighbor_mask"].to(device)

    social=build_model("multihyp_social",obs_len=8,pred_len=12,channels=2,individual=False,
                       num_modes=20,hyp_hidden_dim=64,social_hidden_dim=32,social_dim=32).to(device)
    load_standard(social,social_ck,device)

    visload=build_visual_dataloaders("data/sdd_raw_visual","sdd","m16_160px",batch_size=1,num_workers=0,
                                     val_source="train_group",split_seed=seed,pin_memory=False)
    vb=next(iter(visload["val"]))
    vobs=vb["obs"].to(device)
    vneigh=vb["neighbor_rel"].to(device)
    vmask=vb["neighbor_mask"].to(device)
    crop=vb["scene_crop"].to(device)

    visual=build_visual_model(
        "decoupled_social_visual_adapter",
        obs_len=8,pred_len=12,channels=2,individual=False,
        num_modes=20,hyp_hidden_dim=64,social_hidden_dim=32,social_dim=32,
        visual_dim=64,context_hidden_dim=128,reranker_hidden_dim=128,
        visual_backbone="mobilenet_v3_small",pretrained_backbone=True,
        backbone_trainable="frozen",correction_scale=0.20,
    ).to(device)
    load_generator_checkpoint(visual,social_ck)
    g=torch.load(geom,map_location=device)
    visual.load_state_dict(g["model_state_dict"])
    visual.eval()

    def f_social():
        return social(obs,neighbor_rel=neigh,neighbor_mask=mask)

    def f_visual():
        return visual(vobs,scene_crop=crop,neighbor_rel=vneigh,neighbor_mask=vmask,enable_reranker=False)

    out={
        "stage":"TMM-v1 Stage 3E",
        "device":str(device),
        "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,
        "warmup":args.warmup,
        "repeats":args.repeats,
        "social":{
            "parameters_total":sum(p.numel() for p in social.parameters()),
            "fp32_weight_bytes":model_size_bytes(social,4),
            "latency":benchmark_callable(f_social,device,args.warmup,args.repeats),
        },
        "locked_visual":{
            "parameters_total":sum(p.numel() for p in visual.parameters()),
            "fp32_weight_bytes":model_size_bytes(visual,4),
            "latency":benchmark_callable(f_visual,device,args.warmup,args.repeats),
        },
    }
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
