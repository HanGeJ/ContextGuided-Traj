#!/usr/bin/env python3
"""Rebuild paper Figures 6 and 7 from frozen numeric arrays and native scene pixels.
No vector/raster traces are extracted from old PDF/EPS figures.
Fig 6 uses frozen cached best-of-20 predictions; temporal/frame-anchor alignment remains unverified.
Fig 7 uses three outcome-selected train-group validation cases; see data/sdd/selection.json.
"""
from pathlib import Path
import json
import hashlib
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parent
plt.rcParams.update({
    'font.family':'DejaVu Sans',
    'pdf.fonttype':42,'ps.fonttype':42,
    'font.size':8.4,
    'text.usetex':False,
    'axes.linewidth':0.55,
    'savefig.facecolor':'white',
    'figure.facecolor':'white',
})
COL = {'history':'#06964e','truth':'#d83938','prediction':'#1879be',
       'after':'#d98a26','text':'#202a36','border':'#939aa0','gray':'#6b7279'}

def track(ax, pts, color, lw=.83, dashed=False, z=5, marker=False):
    pts = np.asarray(pts,dtype=float)
    assert pts.ndim==2 and pts.shape[1]==2 and np.isfinite(pts).all()
    ln,=ax.plot(pts[:,0],pts[:,1],color=color,lw=lw,
               linestyle=(0,(3.0,1.65)) if dashed else '-',
               solid_capstyle='round',dash_capstyle='round',zorder=z,
               antialiased=True)
    ln.set_path_effects([pe.Stroke(linewidth=lw+.55,foreground='white'),pe.Normal()])
    if marker:
        ax.plot(pts[-1,0],pts[-1,1],marker='o',markersize=2.6,
                markeredgecolor='white',markeredgewidth=.52,
                markerfacecolor=color,zorder=z+3)
    return ln

def native_ax(fig, loc, image):
    ax=fig.add_axes(loc)
    h,w=image.shape[:2]
    assert image.shape[2]==3
    ax.imshow(image,extent=(-.5,w-.5,h-.5,-.5),origin='upper',
              aspect='equal',interpolation='nearest',zorder=0)
    ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5)
    ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#9aa3a8');sp.set_linewidth(.47)
    return ax

def fig6(output_dir):
    folds=('ETH','HOTEL','UNIV','ZARA1','ZARA2')
    # Individual panel widths follow the actual native image aspect ratios.
    fh=4.06; row_height=1.00; xstart=.32; col_gap=.036; top=.37; row_gap=.07
    cols=[]; native_dims={}
    for f in folds:
        p=BASE/'data'/'ethucy'/f'{f.lower()}_example1.npz'
        with np.load(p,allow_pickle=False) as d:
            h,w=d['background_native'].shape[:2]
        native_dims[f]=(w,h)
        cols.append(row_height*w/h)
    fw=xstart+sum(cols)+col_gap*4+.075
    fig=plt.figure(figsize=(fw,fh))
    xx=[xstart]
    for cw in cols[:-1]:xx.append(xx[-1]+cw+col_gap)
    provenance={}
    for idx,fold in enumerate(folds):
        x=xx[idx];w=cols[idx]
        fig.text((x+w/2)/fw,(fh-.18)/fh,fold,ha='center',va='center',
                 color=COL['text'],fontsize=9.4,fontweight='bold')
    for row in range(3):
        bottom=fh-top-(row+1)*row_height-row*row_gap
        fig.text(.105/fw,(bottom+row_height/2)/fh,f'Example {row+1}',
                 ha='center',va='center',rotation=90,fontsize=8.6,fontweight='bold',color=COL['text'])
        for ci,fold in enumerate(folds):
            key=f'{fold.lower()}_example{row+1}'
            with np.load(BASE/'data'/'ethucy'/f'{key}.npz',allow_pickle=False) as d:
                bg=d['background_native'].copy();obs=d['obs_native'].copy()
                gt=d['gt_native'].copy();pred=d['pred_native'].copy()
                pid=d['ped_id'].copy()
                flag={k:bool(d[k].item()) for k in ('time_mapping_verified','anchor_semantics_verified','final_alignment_verified')}
                provenance[key]={"N":int(len(pid)),"background_WH":list(bg.shape[1::-1]),
                                "prediction_definition":str(d['prediction_definition'].item()),
                                **flag}
            assert obs.shape ==(len(pid),8,2) and gt.shape==(len(pid),12,2) and pred.shape==gt.shape
            ax=native_ax(fig,[xx[ci]/fw,bottom/fh,cols[ci]/fw,row_height/fh],bg)
            for i in range(len(pid)):
                track(ax,obs[i],COL['history'],lw=.83,z=4)
                track(ax,np.vstack([obs[i,-1],gt[i]]),COL['truth'],lw=.86,z=5)
                track(ax,np.vstack([obs[i,-1],pred[i]]),COL['prediction'],lw=.80,dashed=True,z=6)
                ax.plot(obs[i,-1,0],obs[i,-1,1],marker='o',markersize=1.7,
                        markerfacecolor=COL['history'],markeredgecolor='white',markeredgewidth=.3,zorder=7)
    legend=[Line2D([0],[0],color=COL['history'],lw=1.8,label='Observed history'),
            Line2D([0],[0],color=COL['truth'],lw=1.8,label='Ground truth'),
            Line2D([0],[0],color=COL['prediction'],lw=1.8,ls=(0,(3,1.65)),label='Oracle minADE@20')]
    fig.legend(handles=legend,ncol=3,frameon=False,fontsize=8.6,loc='lower center',
               bbox_to_anchor=(.53,.043),columnspacing=2.0,handlelength=2.6)
    # No temporary internal preview watermark; alignment qualification is in provenance.
    for typ in ('eps','pdf','png'):
        fig.savefig(output_dir/f'Fig6_ETHUCY_RawData_Redrawn.{typ}',
                    dpi=300 if typ=='png' else 220)
    plt.close(fig)
    return provenance

def load_sdd(n):
    d=np.load(BASE/'data'/'sdd'/f'case{n}.npz',allow_pickle=False)
    keys=('background_rgb','obs_crop_px','gt_crop_px',
          'candidates_before_crop_px','candidates_after_crop_px','selected_candidate_id')
    data={k:d[k].copy() for k in keys}
    data['selected_candidate_id']=int(data['selected_candidate_id'].item())
    return data

def sdd_paths(d):
    idx=d['selected_candidate_id']
    obs=d['obs_crop_px'];gt=d['gt_crop_px'];before=d['candidates_before_crop_px'][idx];after=d['candidates_after_crop_px'][idx]
    assert obs.shape==(8,2) and gt.shape==(12,2) and before.shape==(12,2) and after.shape==(12,2)
    return obs,gt,before,after

def sdd_panel(fig,position,img,obs,gt,before,after,mode):
    ax=native_ax(fig,position,img)
    if mode=='before':
        track(ax,obs,COL['history'],1.1,z=4,marker=True)
        track(ax,np.vstack([obs[-1],gt]),COL['truth'],lw=1.02,dashed=True,z=5)
        track(ax,np.vstack([obs[-1],before]),COL['prediction'],lw=1.0,z=6)
        ax.scatter(before[-1,0],before[-1,1],marker='o',s=8,color=COL['prediction'],edgecolors='white',linewidths=.36,zorder=9)
    elif mode=='after':
        track(ax,obs,COL['history'],1.1,z=4,marker=True)
        track(ax,np.vstack([obs[-1],gt]),COL['truth'],lw=1.02,dashed=True,z=5)
        track(ax,np.vstack([obs[-1],after]),COL['after'],lw=1.08,z=6)
        ax.scatter(after[-1,0],after[-1,1],marker='o',s=8,color=COL['after'],edgecolors='white',linewidths=.36,zorder=9)
    else:
        endpoints=np.stack([before[-1],after[-1],gt[-1]])
        xmin=min(endpoints[:,0])-12; xmax=max(endpoints[:,0])+12
        ymin=min(endpoints[:,1])-10; ymax=max(endpoints[:,1])+10
        centerx=(xmin+xmax)/2;centery=(ymin+ymax)/2
        # Preserve geographic image geometry while zooming into endpoints.
        width=max(27,xmax-xmin);height=width
        xmin=centerx-width/2;xmax=centerx+width/2
        ymin=centery-height/2;ymax=centery+height/2
        ax.set_xlim(xmin,xmax);ax.set_ylim(ymax,ymin)
        for pts,color,is_dashed in [(gt,COL['truth'],True),(before,COL['prediction'],False),(after,COL['after'],False)]:
            track(ax,pts,color,lw=1.18,dashed=is_dashed,z=5)
        for point,color,shape in [(gt[-1],COL['truth'],'D'),(before[-1],COL['prediction'],'o'),(after[-1],COL['after'],'^')]:
            ax.scatter(point[0],point[1],marker=shape,s=20,zorder=9,
                       facecolors=color,edgecolors='white',linewidths=.75)
    return ax

def fig7(output_dir):
    manifest=json.loads((BASE/'data'/'sdd'/'manifest.json').read_text(encoding='utf-8'))
    casedata=sorted(manifest['cases'], key=lambda c:int(c['case_id'][-1]))
    cw=1.26;ch=1.26; gap=.07; marginleft=.43; metricw=2.28; marginright=.055; head=.36; rowgap=.09;foot=.34
    fw=marginleft+3*cw+3*gap+metricw+marginright
    fh=head+3*ch+2*rowgap+foot
    fig=plt.figure(figsize=(fw,fh))
    xx=[marginleft+i*(cw+gap) for i in range(3)]
    metricx=marginleft+3*(cw+gap)
    heads=('Before correction','After correction','Endpoint detail','Paired metrics')
    for col,x in enumerate(xx+[metricx]):
        ww=cw if col<3 else metricw
        fig.text((x+ww/2)/fw,(fh-.13)/fh,heads[col],fontsize=8.35,
                 fontweight='bold',ha='center',va='center',color=COL['text'])
    used=[]
    for i,metadata in enumerate(casedata):
        d=load_sdd(i+1)
        assert int(metadata['selected_candidate_id']) == d['selected_candidate_id']
        obs,gt,before,after=sdd_paths(d)
        y=fh-head-(i+1)*ch-i*rowgap
        fig.text(.137/fw,(y+ch/2)/fh,f'Case {i+1}',va='center',ha='center',rotation=90,
                 color=COL['text'],fontsize=8.9,fontweight='bold')
        img=d['background_rgb']
        for j,mode in enumerate(('before','after','detail')):
            sdd_panel(fig,[xx[j]/fw,y/fh,cw/fw,ch/fh],img,obs,gt,before,after,mode)
        mm=metadata
        metrics=[('Top-1 ADE (m)',f"{mm['top1_ade_before_m']:.3f}  →  {mm['top1_ade_after_m']:.3f}"),
                 ('Top-1 FDE (m)',f"{mm['top1_fde_before_m']:.3f}  →  {mm['top1_fde_after_m']:.3f}"),
                 ('minADE (m)',f"{mm['minade_before_m']:.3f}  →  {mm['minade_after_m']:.3f}"),
                 ('ADE reduction',f"{mm['top1_gain_m']:.3f} m ({100*mm['top1_gain_m']/mm['top1_ade_before_m']:.1f}%)"),
                 ('Endpoint shift',f"{mm['end_shift_native_px']:.2f} px")]
        # Plain neutral metric panel; all text remains vector text.
        panel=fig.add_axes([metricx/fw,y/fh,metricw/fw,ch/fh])
        panel.set_xlim(0,1);panel.set_ylim(0,1)
        panel.axis('off')
        panel.add_patch(Rectangle((0,0),1,1,facecolor='#fafafa',edgecolor='#aab1b7',linewidth=.55))
        panel.text(.042,.865,f"{mm['scene']} / video{mm['video']}",fontsize=8.25,fontweight='bold',color=COL['text'])
        for k,(label,value) in enumerate(metrics):
            yy=.685-.136*k
            panel.text(.042,yy,label,fontsize=7.05,color=COL['gray'],ha='left')
            panel.text(.50,yy,value,fontsize=7.22,fontweight='bold' if k==3 else 'normal',
                       color='#188047' if k==3 else COL['text'],ha='left')
        used.append({'case':i+1,'source_video':f"{mm['scene']}/video{mm['video']}",
                     'raw_index':mm['raw_index'],
                     'selected_candidate_id':mm['selected_candidate_id'],
                     'top1_gain_raw_m':mm['top1_gain_m'],
                     'gain_percent_unrounded':mm['top1_gain_m']/mm['top1_ade_before_m']*100})
    legend=[Line2D([0],[0],color=COL['history'],lw=1.7,label='History'),
            Line2D([0],[0],color=COL['truth'],lw=1.7,ls='--',label='Ground truth'),
            Line2D([0],[0],color=COL['prediction'],lw=1.7,label='Before'),
            Line2D([0],[0],color=COL['after'],lw=1.7,label='After')]
    fig.legend(handles=legend,ncol=4,loc='lower center',bbox_to_anchor=(.475,.016),
               frameon=False,fontsize=8.15,handlelength=2.5,columnspacing=1.7)
    for ext in ('eps','pdf','png'):
        fig.savefig(output_dir/f'Fig7_SDD_RawData_Redrawn.{ext}',dpi=300 if ext=='png' else 220)
    plt.close(fig)
    return used

def main():
    output_dir=BASE/'output';output_dir.mkdir(exist_ok=True)
    p6=fig6(output_dir)
    p7=fig7(output_dir)
    sourcehash={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((BASE/'data').rglob('*')) if p.is_file()}
    report={'creation':'Native Matplotlib figure rebuilding from original case arrays; no PDF/EPS tracing',
            'backgrounds':'Original source scene rasters embedded into EPS without JPEG recompression; tracks, markers, labels and rules are native vector primitives',
            'fig6_source':'TMM ETHUCY V6 reprojection review bundle, 15 case npz objects; frozen V3 cached per-agent oracle forecasts',
            'fig6_caveat':'V6 audit: time_mapping_verified=false; anchor_semantics_verified=false; final_alignment_verified=false; requires visual/temporal source verification prior to scientific final acceptance',
            'fig6_cases':p6,
            'fig7_source':'TMM SDD Visual Analysis V3.1 review bundle, 3 cached validation cases',
            'fig7_caveat':'Positive-gain case selection and single recording (bookstore/video0); static cropped images are scene context, not synchronized video frames',
            'fig7_case_info':p7,
            'fig7_case3_rounding':'Raw ADE reduction 0.057611343679773566 m -> 0.058 m (36.0%). Previous draft showed 0.057 m (35.6%) by subtracting already-rounded ADE values; update paper sentence if using raw-exact redraw.',
            'source_sha256':sourcehash}
    (output_dir/'PROVENANCE_AND_LIMITATIONS.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('FIG6: 15 cases, raw source frames + native coordinate arrays')
    print('FIG7: 3 cases, raw 160x160 crops + native source trajectories')
    for p in output_dir.glob('*'):print(p.name,p.stat().st_size)

if __name__=='__main__':main()
