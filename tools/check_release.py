"""Offline safety & documentation check for a source-only GitHub release."""
from pathlib import Path
import json,re,sys
ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','__pycache__','.venv'}
BAD_EXT={'.pt','.pth','.ckpt','.npz','.npy','.pkl','.pickle','.jpg','.jpeg','.png','.pdf','.eps','.mp4','.avi','.mov','.tar','.tif','.tiff'}
TEXT_EXT={'.py','.sh','.md','.txt','.json','.yaml','.yml','.toml','.cfg','.cff','.gitignore','.bib','.csv'}
BANNED=['-'*5+'BEGIN PRIVATE KEY'+'-'*5,'gh'+'p_','github'+'_pat_','AK'+'IA']
files=[];fail=[]
for p in ROOT.rglob('*'):
 if not p.is_file() or any(x in SKIP for x in p.parts):continue
 rel=p.relative_to(ROOT).as_posix();files.append(rel)
 if p.suffix.lower() in BAD_EXT:fail.append('disallowed binary/media '+rel)
 if p.stat().st_size>3_000_000:fail.append('oversized source file '+rel)
 if p.suffix.lower() in TEXT_EXT or p.name in {'.gitignore','LICENSE','CITATION.cff'}:
  s=p.read_text(encoding='utf8',errors='ignore')
  for token in BANNED:
   if token in s: fail.append(f'potential credential marker {token} in {rel}')
for path in ('train.py','evaluate.py','prepare_ethucy_5fold_social.py','models/factory.py','models/components.py','data/dataset.py','utils/metrics.py','README.md'):
 if not (ROOT/path).exists():fail.append('missing required '+path)
report={'status':'PASS' if not fail else 'FAIL','scanned_files':len(files),'violations':fail,'not_yet_verified':['clean GPU end-to-end paper reproduction','public code license','checkpoint redistribution rights','Fig6 frame alignment']}
print(json.dumps(report,indent=2))
sys.exit(0 if not fail else 1)
