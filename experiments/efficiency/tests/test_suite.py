"""Offline-only tests; no asserted GPU timing or trained checkpoint is manufactured."""
import os,sys,tempfile,subprocess,json,zipfile,hashlib
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from core import q95,sha256_file,write_json,read_json

class Tests(unittest.TestCase):
 def test_quantile(self):
  self.assertAlmostEqual(q95([1,2,3,4,5]),4.8)
 def test_atomic_write(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x.json'
   write_json(p,{'n':20});self.assertEqual(read_json(p)['n'],20)
   self.assertFalse(Path(str(p)+'.tmp').exists())
 def test_flow_fails_closed(self):
  import matched_adapters
  with self.assertRaisesRegex(RuntimeError,'FLOWCHAIN_K20_ADAPTER_PENDING'):
   matched_adapters.flow('/does/not/exist')
 def test_source_generator_rejects_missing_repo(self):
  with tempfile.TemporaryDirectory() as td:
   p=subprocess.run([sys.executable,str(ROOT/'tools/prepare_univ.py'),'--root',td,
       '--out',str(Path(td)/'scene.npz')],capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0)
   self.assertNotIn('SUCCESS',p.stdout)
 def test_missing_scene_still_produces_review_zip(self):
  with tempfile.TemporaryDirectory() as td:
   env={**os.environ,'TMM_ROOT':td}
   p=subprocess.run([sys.executable,str(ROOT/'tools/suite.py'),'run','--root',td,'--targets','4',
       '--warmup','5','--repeats','30','--wall-repeats','10'],env=env,capture_output=True,text=True,timeout=30)
   self.assertEqual(p.returncode,0,(p.stdout+p.stderr)[-2000:])
   package=Path(td)/'runs/TMM_Efficiency_AutoBench_v1_3_1_review_bundle.zip'
   self.assertTrue(package.is_file())
   with zipfile.ZipFile(package) as z:
    names=z.namelist();self.assertIn('qualification.json',names)
    qual=json.loads(z.read('qualification.json'))
    self.assertFalse(qual['strict_all_five'])
    self.assertNotIn('canonical_univ_observations.npz',names)
    self.assertIn('FLOWCHAIN_SOURCE_AUDIT.json',names)
 def test_no_model_is_certified_from_fake_success_without_ckpt(self):
  import suite
  with tempfile.TemporaryDirectory() as td:
   ns=type('Args',(),{'root':td,'scene_index':0,'min_agents':1,'seed':42,
       'targets':'4','gpu':'0','warmup':50,'repeats':300,'wall_repeats':60,
       'force_scene':False,'force':False,'timeout_sec':200})()
   r=suite.TaskRunner(ns)
   # Syntactically success-shaped outputs with NO checkpoints must not qualify.
   r.scene.write_bytes(b'fake canonical file')
   write_json(r.scene_meta,{'agents_total':8,'observation_sha256':'hash'})
   for model in ('proposed','tutr','eigen','vite'):
    write_json(r.out/f'{model}_univ_N4.json',{'status':'MEASURED','method':model,'fold':'univ',
     'n_targets':4,'candidate_count_per_target':20,'gpu':'NVIDIA GeForce RTX 4090',
     'global_coordinate_predictions_verified':True,'trained_checkpoint_loaded':True,
     'precision':'float32','tf32':False,'gpu_repeats':300,'wall_repeats':60,
     'observed_steps':8,'future_steps':12,'physical_observation_sha256':'hash',
     'source_sha256':sha256_file(r.scene),'checkpoint_sha256':'fake',
     'gpu_forward_median_ms':1.0,'prepared_pipeline_wall_median_ms':2.0})
   r.summarize()
   d=read_json(r.out/'qualification.json')
   self.assertEqual(d['same_input_k20_results'],0)
   self.assertFalse(d['strict_all_five'])
 def test_no_data_or_checkpoint_in_zip_source(self):
  forbidden={'.pt','.pth','.ckpt','.npz','.pkl','.safetensors'}
  for p in ROOT.rglob('*'):
   if p.is_file():self.assertNotIn(p.suffix,forbidden)

if __name__=='__main__':unittest.main(verbosity=2)
