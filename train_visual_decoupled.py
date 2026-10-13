from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data.visual_dataset import build_visual_dataloaders
from models import build_visual_model
from models.components import augment_batch, augment_batch_social
from utils import compute_eval_metrics, compute_training_loss, quality_aware_targets, set_seed

VARIANTS = ("decoupled_visual_adapter", "decoupled_social_visual_adapter")


def _move(batch, device, key):
    v = batch.get(key)
    return v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v


def _forward(model, batch, device, enable_reranker=True):
    return model(
        _move(batch, device, "obs"),
        scene_crop=_move(batch, device, "scene_crop"),
        neighbor_rel=_move(batch, device, "neighbor_rel"),
        neighbor_mask=_move(batch, device, "neighbor_mask"),
        enable_reranker=enable_reranker,
    )


@torch.no_grad()
def evaluate(model, loader, device, best_k=20, miss_threshold=2.0, amp=True,
             enable_reranker=True):
    model.eval()
    sums = {}
    n = 0
    use_amp = bool(amp and device.type == "cuda")
    for batch in loader:
        target = _move(batch, device, "pred")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = _forward(model, batch, device, enable_reranker=enable_reranker)
            m = compute_eval_metrics(out, target, best_k=best_k, miss_threshold=miss_threshold)
        bs = int(target.size(0))
        mapping = {
            "ade": "top1_ade", "fde": "top1_fde",
            "best_k_ade": "minade_k", "best_k_fde": "minfde_k",
            "miss_rate": "top1_miss_rate", "best_k_miss_rate": "miss_rate_k",
            "nll_top1": "nll_top1", "nll_best": "nll_best",
        }
        for src, dst in mapping.items():
            if src in m:
                sums[dst] = sums.get(dst, 0.0) + float(m[src].detach()) * bs
        for key, outkey, mode in (
            ("context_correction", "context_correction_rms", "rms"),
            ("rank_delta", "rank_delta_rms", "rms"),
            ("context_gate", "context_gate_mean", "mean"),
        ):
            if key in out:
                val = out[key]
                v = float(val.square().mean().sqrt().detach()) if mode == "rms" else float(val.mean().detach())
                sums[outkey] = sums.get(outkey, 0.0) + v * bs
        n += bs
    if n == 0:
        raise RuntimeError("Empty validation loader")
    return {k: v / n for k, v in sums.items()}


def load_generator_checkpoint(model, path: Path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    current = model.state_dict()
    loadable = {}
    skipped = []
    for k, v in state.items():
        if k in current and tuple(current[k].shape) == tuple(v.shape):
            loadable[k] = v
        elif k in current:
            skipped.append((k, tuple(v.shape), tuple(current[k].shape)))
    prefixes = ["base.", "head."]
    if getattr(model, "use_social_base", False):
        prefixes.append("social_encoder.")
    expected = [k for k in current if any(k.startswith(p) for p in prefixes)]
    missing = [k for k in expected if k not in loadable]
    if missing or skipped:
        raise RuntimeError(f"Incompatible generator checkpoint: missing={missing[:8]} skipped={skipped[:4]}")
    result = model.load_state_dict(loadable, strict=False)
    return {
        "path": str(path), "loaded_keys": len(loadable),
        "missing_non_generator_keys": len(result.missing_keys),
        "checkpoint_args": ckpt.get("args", {}),
    }


def _optimizer_geometry(model, args):
    groups = []
    proj = [p for p in model.visual_encoder.proj.parameters() if p.requires_grad]
    feat = [p for p in model.visual_encoder.features.parameters() if p.requires_grad]
    adapter = [p for p in model.context_adapter.parameters() if p.requires_grad]
    if proj or adapter:
        groups.append({"params": proj + adapter, "lr": args.geometry_lr})
    if feat:
        groups.append({"params": feat, "lr": args.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def _optimizer_ranking(model, args):
    params = [p for p in model.visual_reranker.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=args.rank_lr, weight_decay=args.weight_decay)


def _augment(model, batch, device, args):
    obs = _move(batch, device, "obs")
    target = _move(batch, device, "pred")
    neigh = _move(batch, device, "neighbor_rel")
    mask = _move(batch, device, "neighbor_mask")
    if getattr(model, "uses_social_context", False):
        obs, target, neigh, mask, _ = augment_batch_social(
            obs, target, neigh, mask, ego_obs_rel=None,
            noise_std=args.noise_std, mask_prob=args.mask_prob,
            max_mask_steps=args.max_mask_steps, rotation_max_deg=0.0,
            use_inversion=False,
        )
    else:
        obs, target, _ = augment_batch(
            obs, target, ego_obs_rel=None,
            noise_std=args.noise_std, mask_prob=args.mask_prob,
            max_mask_steps=args.max_mask_steps, rotation_max_deg=0.0,
            use_inversion=False,
        )
    return obs, target, neigh, mask


def _geometry_train(model, loaders, device, args, outdir):
    model.set_phase("geometry")
    opt = _optimizer_geometry(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    use_amp = bool(args.amp and device.type == "cuda")
    best = float("inf")
    best_epoch = 0
    best_path = outdir / "geometry_best.pt"
    history = []
    stale = 0
    for epoch in range(1, args.geometry_epochs + 1):
        model.train()
        # Frozen generator always stays in eval mode.
        for mod in model.generator_modules():
            mod.eval()
        total = 0.0
        n = 0
        t0 = time.perf_counter()
        for batch in loaders["train"]:
            obs, target, neigh, mask = _augment(model, batch, device, args)
            crop = _move(batch, device, "scene_crop")
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(obs, scene_crop=crop, neighbor_rel=neigh, neighbor_mask=mask,
                            enable_reranker=False)
                # Geometry stage learns candidate corrections only. Base ranking
                # is retained and the classification/ranking term is disabled.
                loss, aux = compute_training_loss(
                    out, target, cls_lambda=0.0, nll_lambda=args.nll_lambda,
                    ranking_loss="hard_ce",
                )
                if args.correction_lambda > 0:
                    loss = loss + args.correction_lambda * out["context_correction"].square().mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            bs = int(obs.size(0)); total += float(loss.detach()) * bs; n += bs
        val = evaluate(model, loaders["val"], device, args.best_k, args.miss_threshold,
                       args.amp, enable_reranker=False)
        score = float(val["minade_k"] + args.coverage_fde_weight * val["minfde_k"])
        sec = time.perf_counter() - t0
        history.append({"epoch": epoch, "loss": total/max(1,n), "validation": val,
                        "coverage_score": score, "epoch_seconds": sec})
        print(f"GEOM {epoch:03d} loss={total/max(1,n):.5f} "
              f"Top1={val['top1_ade']:.4f}/{val['top1_fde']:.4f} "
              f"min@{args.best_k}={val['minade_k']:.4f}/{val['minfde_k']:.4f} "
              f"corr={val.get('context_correction_rms',float('nan')):.4f} score={score:.5f} {sec:.1f}s",
              flush=True)
        if score < best - 1e-7:
            best = score; best_epoch = epoch; stale = 0
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args),
                        "best_epoch": epoch, "best_score": score}, best_path)
        else:
            stale += 1
        if args.geometry_patience > 0 and stale >= args.geometry_patience:
            print(f"GEOMETRY_EARLY_STOP epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    selected = evaluate(model, loaders["val"], device, args.best_k, args.miss_threshold,
                        args.amp, enable_reranker=False)
    (outdir / "geometry_history.json").write_text(json.dumps(history, indent=2))
    return best_path, best_epoch, selected


def _ranking_loss(logits, candidates, target, mode, temperature, fde_weight):
    if mode == "hard":
        q, _, _, cost = quality_aware_targets(candidates, target, temperature=max(temperature,1e-4), fde_weight=fde_weight)
        idx = cost.argmin(dim=1)
        return F.cross_entropy(logits, idx)
    q, _, _, _ = quality_aware_targets(candidates, target, temperature=temperature, fde_weight=fde_weight)
    return -(q * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def _ranking_train(model, loaders, device, args, outdir, geometry_metrics):
    if args.ranking_epochs <= 0:
        return None, 0, geometry_metrics
    model.set_phase("ranking")
    opt = _optimizer_ranking(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    use_amp = bool(args.amp and device.type == "cuda")
    best = float("inf")
    best_epoch = 0
    best_path = outdir / "ranking_best.pt"
    history = []
    stale = 0
    for epoch in range(1, args.ranking_epochs + 1):
        model.train()
        # Every module other than the reranker remains frozen/eval.
        model.base.eval(); model.head.eval(); model.visual_encoder.eval(); model.context_adapter.eval()
        if getattr(model, "use_social_base", False):
            model.social_encoder.eval()
        total=0.0; n=0; t0=time.perf_counter()
        for batch in loaders["train"]:
            obs = _move(batch, device, "obs")
            target = _move(batch, device, "pred")
            neigh = _move(batch, device, "neighbor_rel")
            mask = _move(batch, device, "neighbor_mask")
            crop = _move(batch, device, "scene_crop")
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(obs, scene_crop=crop, neighbor_rel=neigh, neighbor_mask=mask,
                            enable_reranker=True)
                loss = _ranking_loss(out["mode_logits"], out["candidates"].detach(), target,
                                     args.ranking_loss, args.rank_temperature,
                                     args.rank_fde_weight)
                if args.rank_delta_lambda > 0:
                    loss = loss + args.rank_delta_lambda * out["rank_delta"].square().mean()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            bs=int(obs.size(0)); total += float(loss.detach())*bs; n += bs
        val=evaluate(model,loaders["val"],device,args.best_k,args.miss_threshold,args.amp,True)
        # Candidate geometry is frozen, so coverage must be unchanged apart from
        # floating-point reduction noise.
        if abs(val["minade_k"]-geometry_metrics["minade_k"]) > 2e-6 or abs(val["minfde_k"]-geometry_metrics["minfde_k"]) > 2e-6:
            raise RuntimeError("Ranking phase changed fixed candidate coverage; decoupling invariant violated")
        score=float(val["top1_ade"] + args.rank_select_fde_weight * val["top1_fde"])
        sec=time.perf_counter()-t0
        history.append({"epoch":epoch,"loss":total/max(1,n),"validation":val,
                        "ranking_score":score,"epoch_seconds":sec})
        print(f"RANK {epoch:03d} loss={total/max(1,n):.5f} "
              f"Top1={val['top1_ade']:.4f}/{val['top1_fde']:.4f} "
              f"min@{args.best_k}={val['minade_k']:.4f}/{val['minfde_k']:.4f} "
              f"delta={val.get('rank_delta_rms',float('nan')):.4f} score={score:.5f} {sec:.1f}s",
              flush=True)
        if score < best - 1e-7:
            best=score; best_epoch=epoch; stale=0
            torch.save({"model_state_dict":model.state_dict(),"args":vars(args),
                        "best_epoch":epoch,"best_score":score},best_path)
        else:
            stale += 1
        if args.ranking_patience > 0 and stale >= args.ranking_patience:
            print(f"RANKING_EARLY_STOP epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    ckpt=torch.load(best_path,map_location=device); model.load_state_dict(ckpt["model_state_dict"])
    selected=evaluate(model,loaders["val"],device,args.best_k,args.miss_threshold,args.amp,True)
    (outdir / "ranking_history.json").write_text(json.dumps(history, indent=2))
    return best_path,best_epoch,selected


def main():
    ap=argparse.ArgumentParser(description="Stage 3D decoupled visual geometry and candidate ranking")
    ap.add_argument("--variant",choices=VARIANTS,required=True)
    ap.add_argument("--generator_checkpoint",required=True)
    ap.add_argument("--geometry_checkpoint",default=None,
                    help="Optional Stage3D geometry checkpoint to reuse; skips geometry training when --skip_geometry is set.")
    ap.add_argument("--skip_geometry",action="store_true")
    ap.add_argument("--data_root",default="data/sdd_raw_visual")
    ap.add_argument("--test_set",default="sdd")
    ap.add_argument("--crop_tag",default="m16_160px")
    ap.add_argument("--val_source",choices=("train_group","train"),default="train_group")
    ap.add_argument("--val_ratio",type=float,default=0.1)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--geometry_epochs",type=int,default=30)
    ap.add_argument("--ranking_epochs",type=int,default=20)
    ap.add_argument("--geometry_patience",type=int,default=10)
    ap.add_argument("--ranking_patience",type=int,default=8)
    ap.add_argument("--batch_size",type=int,default=128)
    ap.add_argument("--num_workers",type=int,default=8)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir",default="runs/tmm_stage3d_decoupled")
    ap.add_argument("--best_k",type=int,default=20)
    ap.add_argument("--miss_threshold",type=float,default=2.0)
    ap.add_argument("--num_modes",type=int,default=20)
    ap.add_argument("--hyp_hidden_dim",type=int,default=64)
    ap.add_argument("--social_hidden_dim",type=int,default=32)
    ap.add_argument("--social_dim",type=int,default=32)
    ap.add_argument("--visual_dim",type=int,default=64)
    ap.add_argument("--context_hidden_dim",type=int,default=128)
    ap.add_argument("--reranker_hidden_dim",type=int,default=128)
    ap.add_argument("--visual_backbone",choices=("mobilenet_v3_small","resnet18","convnext_tiny"),default="mobilenet_v3_small")
    ap.add_argument("--backbone_trainable",choices=("full","last","frozen"),default="full")
    ap.add_argument("--pretrained_backbone",dest="pretrained_backbone",action="store_true")
    ap.add_argument("--no_pretrained_backbone",dest="pretrained_backbone",action="store_false")
    ap.set_defaults(pretrained_backbone=False)
    ap.add_argument("--correction_scale",type=float,default=0.20)
    ap.add_argument("--geometry_lr",type=float,default=1e-3)
    ap.add_argument("--backbone_lr",type=float,default=1e-4)
    ap.add_argument("--rank_lr",type=float,default=1e-3)
    ap.add_argument("--weight_decay",type=float,default=1e-4)
    ap.add_argument("--nll_lambda",type=float,default=0.1)
    ap.add_argument("--correction_lambda",type=float,default=0.0)
    ap.add_argument("--coverage_fde_weight",type=float,default=0.5)
    ap.add_argument("--ranking_loss",choices=("hard","soft"),default="soft")
    ap.add_argument("--rank_temperature",type=float,default=0.05)
    ap.add_argument("--rank_fde_weight",type=float,default=0.5)
    ap.add_argument("--rank_select_fde_weight",type=float,default=0.5)
    ap.add_argument("--rank_delta_lambda",type=float,default=1e-4)
    ap.add_argument("--noise_std",type=float,default=0.02)
    ap.add_argument("--mask_prob",type=float,default=0.3)
    ap.add_argument("--max_mask_steps",type=int,default=4)
    ap.add_argument("--amp",dest="amp",action="store_true")
    ap.add_argument("--no_amp",dest="amp",action="store_false")
    ap.set_defaults(amp=True)
    ap.add_argument("--max_train_samples",type=int,default=None)
    ap.add_argument("--max_val_samples",type=int,default=None)
    args=ap.parse_args()

    set_seed(args.seed); device=torch.device(args.device)
    loaders=build_visual_dataloaders(args.data_root,args.test_set,args.crop_tag,
        batch_size=args.batch_size,num_workers=args.num_workers,val_ratio=args.val_ratio,
        split_seed=args.seed,val_source=args.val_source,max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,pin_memory=device.type=="cuda")
    model=build_visual_model(args.variant,obs_len=8,pred_len=12,channels=2,individual=False,
        num_modes=args.num_modes,hyp_hidden_dim=args.hyp_hidden_dim,
        social_hidden_dim=args.social_hidden_dim,social_dim=args.social_dim,
        visual_dim=args.visual_dim,context_hidden_dim=args.context_hidden_dim,
        reranker_hidden_dim=args.reranker_hidden_dim,visual_backbone=args.visual_backbone,
        pretrained_backbone=args.pretrained_backbone,backbone_trainable=args.backbone_trainable,
        correction_scale=args.correction_scale).to(device)
    report=load_generator_checkpoint(model,Path(args.generator_checkpoint))

    first=next(iter(loaders["val"])); model.eval()
    with torch.no_grad():
        out=_forward(model,first,device,True)
        cand=float((out["candidates"]-out["base_candidates"]).abs().max().cpu())
        logit=float((out["mode_logits"]-out["base_mode_logits"]).abs().max().cpu())
    if cand>1e-7 or logit>1e-7:
        raise RuntimeError(f"Stage3D warm-start invariance failed candidate={cand} logits={logit}")
    initial=evaluate(model,loaders["val"],device,args.best_k,args.miss_threshold,args.amp,True)

    pretag="pretrained" if args.pretrained_backbone else "scratch"
    ctag=f"cscale_{args.correction_scale:g}"
    ranktag=f"{args.ranking_loss}_t{args.rank_temperature:g}_d{args.rank_delta_lambda:g}"
    outdir=Path(args.out_dir)/args.variant/args.crop_tag/args.visual_backbone/args.backbone_trainable/pretag/ctag/ranktag/f"seed_{args.seed}"
    outdir.mkdir(parents=True,exist_ok=True)
    print(f"Stage3D variant={args.variant} seed={args.seed} correction_scale={args.correction_scale} ranking={ranktag}",flush=True)
    print("Initial validation | "+json.dumps(initial,sort_keys=True),flush=True)

    if args.skip_geometry:
        if not args.geometry_checkpoint:
            raise ValueError("--skip_geometry requires --geometry_checkpoint")
        gck=torch.load(args.geometry_checkpoint,map_location=device)
        model.load_state_dict(gck["model_state_dict"])
        geometry_epoch=int(gck.get("best_epoch",0))
        geometry_metrics=evaluate(model,loaders["val"],device,args.best_k,args.miss_threshold,args.amp,False)
        geometry_path=Path(args.geometry_checkpoint)
    else:
        geometry_path,geometry_epoch,geometry_metrics=_geometry_train(model,loaders,device,args,outdir)

    ranking_path,ranking_epoch,ranked_metrics=_ranking_train(model,loaders,device,args,outdir,geometry_metrics)
    summary={
        "stage":"TMM-v1 Stage 3D","held_out_test_used":False,"variant":args.variant,
        "seed":args.seed,"generator_checkpoint":str(Path(args.generator_checkpoint).resolve()),
        "generator_load_report":report,"crop_tag":args.crop_tag,"visual_backbone":args.visual_backbone,
        "backbone_trainable":args.backbone_trainable,"pretrained_backbone":bool(args.pretrained_backbone),
        "correction_scale":args.correction_scale,"ranking_loss":args.ranking_loss,
        "rank_temperature":args.rank_temperature,"rank_delta_lambda":args.rank_delta_lambda,
        "initial_validation":initial,"geometry_best_epoch":geometry_epoch,"geometry_validation":geometry_metrics,
        "ranking_best_epoch":ranking_epoch,"validation":ranked_metrics,
        "geometry_checkpoint":str(geometry_path),"ranking_checkpoint":str(ranking_path) if ranking_path else None,
        "total_parameters":sum(p.numel() for p in model.parameters()),
        "reranker_parameters":sum(p.numel() for p in model.visual_reranker.parameters()),
    }
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2))
    print("SUMMARY_JSON="+str((outdir/"summary.json").resolve()))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
