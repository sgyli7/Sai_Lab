import json,tempfile,unittest
from pathlib import Path
from experiments.pi05_g1_round02.video_index import make_index,digest,KEYS
class VideoIndexTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.summary=self.root/'summary.json';self.output=self.root/'all.csv';self.data={'schema':'pi05_g1_round02_multi_policy_benchmark_v2','complete':True,'evidence_qualified':True,'errors':[],'infrastructure_incomplete':[],'policies':{p:{'completed':20} for p in ['groot','pi1','pi2']},'video_evidence':[]}
  for policy in self.data['policies']:
   for i in range(20):
    mp=self.root/f'{policy}_{i}.json';mp.write_text(json.dumps({'schema':'pi05_g1_round02_video_manifest_v1','policy':policy,'case_id':str(i),'raw_result_sha256':'a'*64,'source_case_dir':'/not_read_raw_case'}));videos={k:{'path':'/not_read_video/'+k+'.mp4','sha256':'b'*64,'fps':50,'frames':401} for k in KEYS};self.data['video_evidence'].append({'policy':policy,'case_id':str(i),'complete':True,'errors':[],'manifest_path':str(mp),'manifest_sha256':digest(mp),'videos':videos})
  self.save()
 def save(self):self.summary.write_text(json.dumps(self.data))
 def tearDown(self):self.tmp.cleanup()
 def test_indexes_all_without_reading_raw(self):
  r=make_index(self.summary,self.output);self.assertEqual(r['indexed_cases'],60);self.assertEqual(r['indexed_video_rows'],240);self.assertIn('raw_result_sha256',self.output.read_text().splitlines()[0])
 def test_available_wall_views_also_indexed_without_selection(self):
  self.data['video_evidence'][0]['actual_wall_clock_videos']={k:{'path':'/never_open/'+k+'.mp4','sha256':'c'*64,'fps':25,'frames':1000} for k in ['head_wall_clock','observer_wall_clock']};self.save()
  r=make_index(self.summary,self.output);self.assertEqual(r['indexed_cases'],60);self.assertEqual(r['indexed_video_rows'],242)
 def test_incomplete_rejected(self):
  self.data['complete']=False;self.save()
  with self.assertRaises(ValueError):make_index(self.summary,self.output)
  self.assertFalse(self.output.exists())
 def test_manifest_tamper_rejected(self):
  Path(self.data['video_evidence'][0]['manifest_path']).write_text('{}')
  with self.assertRaises(ValueError):make_index(self.summary,self.output)
  self.assertFalse(self.output.exists())
if __name__=='__main__':unittest.main()
