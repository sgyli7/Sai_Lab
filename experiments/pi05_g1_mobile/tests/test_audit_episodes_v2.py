"""Real CPU H264 normal/fragmented container timing and unchanged episode checks."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import av
from experiments.pi05_g1_mobile import audit_episodes_v2 as a,acceptance
from experiments.pi05_g1_mobile.tests import test_audit_episodes as legacy_fixture
write=legacy_fixture.write


def encode(path,*,fragmented=False,fps=50,ticks=None,n=6):
    options={'movflags':'frag_keyframe+empty_moov+default_base_moof'} if fragmented else {}
    with av.open(str(path),'w',options=options) as container:
        stream=container.add_stream('libx264',rate=fps);stream.width=64;stream.height=48;stream.pix_fmt='yuv420p';stream.codec_context.thread_count=2;stream.codec_context.gop_size=50
        for i in (range(n) if ticks is None else ticks):
            frame=av.VideoFrame.from_ndarray(np.full((48,64,3),int(i)%255,np.uint8),format='rgb24');frame.pts=i;frame.time_base=Fraction(1,fps)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)


class VideoOriginTests(unittest.TestCase):
    def test_real_normal_and_fragmented_h264_both_relative50hz(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);streams={}
            for fragment in [False,True]:
                path=p/('fragment.mp4' if fragment else 'normal.mp4');encode(path,fragmented=fragment)
                streams[str(fragment)]=a.video_probe(path,6)
            self.assertAlmostEqual(streams['False']['first_pts_seconds'],0.)
            self.assertGreater(streams['True']['first_pts_seconds'],0.)
            self.assertTrue(a.check_video_pair(streams)['passed'])
            self.assertTrue(all(v['frames']==6 and v['fps']==50 for v in streams.values()))

    def test_missing_frames_wrongfps_discontinuity_excess_origin_rejected(self):
        for kind in ['missing','fps','gap']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'bad.mp4';kwargs={'fragmented':True}
                if kind=='missing':kwargs['n']=5
                elif kind=='fps':kwargs['fps']=25
                elif kind=='gap':kwargs['ticks']=[0,1,2,4,5,6]
                encode(path,**kwargs)
                with self.assertRaises(ValueError):a.video_probe(path,6)

    def test_nonfinite_negative_and_reordered_pts_rejected_even_with_nominal_rate(self):
        from types import SimpleNamespace
        class Frame:
            def __init__(self,pts):self.pts=pts
            def to_ndarray(self,format):return np.zeros((2,2,3),np.uint8)
        for times in [[float('nan'),1,2],[-1,0,1],[0,1,1],[0,2,1],[0,1,3],[6,7,8]]:
            stream=SimpleNamespace(average_rate=50,time_base=Fraction(1,50),codec_context=SimpleNamespace(thread_count=0))
            container=SimpleNamespace(streams=SimpleNamespace(video=[stream]),decode=lambda _:iter([Frame(t) for t in times]))
            with patch.object(av,'open') as mocked:
                mocked.return_value.__enter__.return_value=container
                with self.assertRaises(ValueError):a.video_probe('CPU_MOCK_NO_FILE',3)

    def test_pair_different_relative_timeline_rejected(self):
        head={'frames':3,'fps':50,'relative_pts_seconds':[0,.02,.04]};observer={**head,'relative_pts_seconds':[0,.02,.041]}
        with self.assertRaises(ValueError):a.check_video_pair({'head':head,'observer':observer})


class V2EpisodeTests(unittest.TestCase):
    def fixture(self,p):
        old=legacy_fixture.MobileEpisodeAuditTests();fixture,rows,result=old.fixture(p)
        config=a.read(p/'config.json');marker=p/'frozen_source.py';marker.write_text('# synthetic frozen source\n');config['frozen_files']={str(marker):a.sha(marker)};write(p/'config.json',config)
        interface={'synthetic_cpu_only':True};write(p/'interface.json',interface)
        digest=hashlib.sha256(json.dumps(interface,sort_keys=True).encode()).hexdigest()
        result.update(wbc_config={'enable_waist':False},offline_unused_assets={'synthetic':True},video_encoding={'synthetic':True},frozen_interface_sha256=digest)
        from experiments.pi05_g1_mobile.simulator import SOURCE_BUILD
        runtime={'schema':'pi05_g1_mobile_runtime_identity_v1','sdk_build':SOURCE_BUILD,'frozen_files':config['frozen_files'],'interface':interface,**{k:result[k] for k in ['source_joint_names','body_names','contact_view','wbc_config','offline_unused_assets','video_encoding','frozen_interface_sha256']}}
        write(p/'runtime_identity.json',runtime);result['runtime_identity_path']=str(p/'runtime_identity.json');result['runtime_identity_sha256']=a.sha(p/'runtime_identity.json');write(p/'result.json',result)
        return fixture,rows,result

    def test_all_original_checks_failed_task_and_runtime_roster_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixture,rows,result=self.fixture(p);rawsha=a.sha(p/'result.json')
            with patch.object(acceptance,'geometry',return_value=fixture.geometry):report=a.audit_case(p)
            self.assertTrue(report['passed'],report['anomalies']);self.assertEqual(report['schema'],'pi05_g1_mobile_episode_cpu_audit_v2');self.assertFalse(report['checks']['acceptance']['success']);self.assertTrue(report['checks']['actual_runtime_identity']['full_roster_verified']);self.assertEqual(a.sha(p/'result.json'),rawsha)
            (p/'frozen_source.py').write_text('# changed')
            with patch.object(acceptance,'geometry',return_value=fixture.geometry):report=a.audit_case(p)
            self.assertFalse(report['passed']);self.assertIn('Frozen execution input changed',str(report['anomalies']))

if __name__=='__main__':unittest.main()
