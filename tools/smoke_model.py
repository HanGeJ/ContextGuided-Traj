"""Pure synthetic CPU smoke; not an accuracy or SDD model reproduction."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models import build_model

def main():
    torch.manual_seed(123)
    m=build_model('multihyp_social',obs_len=8,pred_len=12,channels=2,individual=False,num_modes=20,hyp_hidden_dim=64,social_hidden_dim=32,social_dim=32).eval()
    obs=torch.randn(2,8,2);rel=torch.randn(2,8,4,2);mask=torch.ones(2,8,4)
    with torch.no_grad(): out=m(obs,neighbor_rel=rel,neighbor_mask=mask)
    assert out['candidates'].shape==(2,20,12,2)
    assert out['mode_logits'].shape==(2,20)
    assert out['pred'].shape==(2,12,2)
    assert torch.isfinite(out['candidates']).all()
    params=sum(x.numel() for x in m.parameters())
    assert params==82529,(params,'paper parameters changed')
    print(json.dumps({'status':'CPU_SYNTHETIC_PASS','parameters':params,'candidates_shape':list(out['candidates'].shape),'actual_accuracy_reproduced':False}))
if __name__=='__main__':main()
