"""CPU logical fixtures only, not evidence of any real physical episode."""
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from experiments.pi05_g1_mobile import acceptance_pose_speed as subject


def deep_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=True).encode()).hexdigest()


class PoseSpeedTests(unittest.TestCase):
    def fixture(self):
        actual=['left_hand_'+str(i) for i in range(8)]+['right_hand_'+str(i) for i in range(8)]+['body'+str(i) for i in range(16,53)]
        names=list(reversed(actual));robot={'bodies':[{'name':n} for n in names],
            'collisions':[{'body':i,'shape':'toy','points':[[0,0,0]]} for i in range(52)]}
        box=np.array([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (-.02,.02)])
        floor=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])
        polygon=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
        inward=np.array([[0,1],[-1,0],[0,-1],[1,0]])
        def bounds(collider,pose):return (np.asarray(pose[:3])-.01,np.asarray(pose[:3])+.01)
        def gap(a,b):return float(np.linalg.norm(np.maximum(0.,np.maximum(a[0]-b[1],b[0]-a[1]))))
        bundle=(box,floor,polygon,inward,robot,bounds,gap)
        context={'geometry_definition':'toy','geometry_definition_sha256':'toy','robot_geometry':'toy','robot_geometry_sha256':'toy','source_harness_dir':'toy',
            'audit_body_names':actual,'audit_contact_metadata':{'schema_writes':0,'robot_filter_indices':list(range(1,54)),'hand_filter_indices':list(range(1,17)),
                'bin_filter_indices':[0],'filter_paths':['bin']+actual,'robot_body_names':actual},
            'source_ground_z':0.,'acceptance':{'task_family':'toy','minimum_source_target_distance_m':5.,'minimum_carried_displacement_m':4.,'minimum_carry_turn_degrees':60.,
                'stable_seconds':2.,'stable_position_m':.005,'stable_angle_degrees':5.}}
        rows=[]
        for tick in range(151):
            carried=1<=tick<=10;x=.5*tick if carried else (0. if tick==0 else 5.);z=.1 if carried else .02
            angle=math.radians(min(tick,10)*7);matrix=np.zeros((54,3));points=[]
            indices=[1,9] if carried else [0]
            for idx in indices:
                matrix[idx,2]=1.;points.append({'filter_index':idx,'normal_world':[0,0,1],'normal_force_n':1.,'separation_m':0.})
            rows.append({'control_tick':tick,'physics_steps_since_reset':4*tick,'sim_time_seconds':tick/50,'monotonic_ns':1000000000+tick*20000000,'wall_time_ns':2000000000+tick*20000000,
                'root_link_pose_w':[0,0,.8,0,0,math.sin(angle/2),math.cos(angle/2)],'root_link_velocity_w':[0]*6,'root_com_velocity_w':[0]*6,'upright':1.,
                'acceptance_robot_body_link_poses_w':[[100,0,0,0,0,0,1] for _ in range(53)],
                'acceptance_truth_only':{'brown_box':{'pose':[x,0,z,0,0,0,1],'velocity':[.03,0,0,0,0,0]},'blue_sorting_bin':{'pose':[5,0,0,0,0,0,1],'velocity':[0]*6}},
                'acceptance_contacts_only':{'box_filter_force_matrix_world_n':[matrix.tolist()],'box_robot_force_matrix_world_n':matrix[1:].tolist(),
                    'box_hand_force_matrix_world_n':matrix[1:17].tolist(),'box_bin_force_world_n':matrix[0].tolist(),'points':points},'original_terms':{'success':{'ever':True}}})
        return rows,context,bundle

    def evaluate(self,rows,config,bundle):
        with patch.object(subject.body_ids,'_ORIGINAL_GEOMETRY',return_value=bundle) as fake,patch.object(subject.original,'geometry',fake):
            return subject.evaluate_episode(rows,config)

    def test_success_requires_transport_contact_and_full_stability_input_unchanged(self):
        rows,config,bundle=self.fixture();before=deep_sha(rows);result=self.evaluate(rows,config,bundle)
        self.assertTrue(result['success']);self.assertEqual(result['qualified_stability_window'],{'start_tick':12,'end_tick':112,'seconds':2.})
        self.assertFalse(result['samples'][0]['placement_ready']);self.assertEqual(result['measurement_source']['schema'],subject.SCHEMA)
        self.assertEqual(deep_sha(rows),before);self.assertEqual(rows[100]['acceptance_truth_only']['brown_box']['velocity'][0],.03)
        self.assertEqual(result['samples'][100]['box_speed_m_s'],0.)

    def test_individual_negative_predicates_and_pose_drift_block_success(self):
        for mode in ('moving','drift','contact','support','separation','angular','fall'):
            with self.subTest(mode=mode):
                rows,config,bundle=self.fixture()
                for row in rows[11:]:
                    tick=row['control_tick'];truth=row['acceptance_truth_only']['brown_box'];c=row['acceptance_contacts_only']
                    if mode=='moving':truth['pose'][0]+=max(0,tick-11)*.04
                    if mode=='drift':truth['pose'][0]+=max(0,tick-11)*.0001
                    if mode=='support':
                        c['box_bin_force_world_n']=[0,0,0];c['box_filter_force_matrix_world_n'][0][0]=[0,0,0];c['points']=[]
                    if mode=='contact':
                        c['box_filter_force_matrix_world_n'][0][1]=[1,0,0];c['box_robot_force_matrix_world_n'][0]=[1,0,0];c['box_hand_force_matrix_world_n'][0]=[1,0,0]
                        c['points'].append({'filter_index':1,'normal_world':[1,0,0],'normal_force_n':1.,'separation_m':0.})
                    if mode=='separation':row['acceptance_robot_body_link_poses_w'][52]=[5,0,.02,0,0,0,1]
                    if mode=='angular':truth['velocity'][3]=.2
                    if mode=='fall':row['upright']=.4
                before=deep_sha(rows);result=self.evaluate(rows,config,bundle)
                self.assertFalse(result['success']);self.assertEqual(deep_sha(rows),before)
                if mode=='drift':self.assertFalse(result['stable_placement'])

    def test_raw_invalid_values_and_clock_gaps_rejected_before_masking(self):
        for mode in ('nan_velocity','nan_pose','body_pose_shape','root_pose_nan','tick','physics','dt','mono','wall'):
            rows,config,bundle=self.fixture()
            if mode=='nan_velocity':rows[1]['acceptance_truth_only']['brown_box']['velocity'][0]=float('nan')
            if mode=='nan_pose':rows[1]['acceptance_truth_only']['blue_sorting_bin']['pose'][0]=float('nan')
            if mode=='body_pose_shape':rows[1]['acceptance_robot_body_link_poses_w'].pop()
            if mode=='root_pose_nan':rows[1]['root_link_pose_w'][0]=float('nan')
            if mode=='tick':rows[1]['control_tick']=2
            if mode=='physics':rows[1]['physics_steps_since_reset']=8
            if mode=='dt':rows[1]['sim_time_seconds']=.021
            if mode=='mono':rows[1]['monotonic_ns']=rows[0]['monotonic_ns']
            if mode=='wall':rows[1]['wall_time_ns']=rows[0]['wall_time_ns']
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.evaluate(rows,config,bundle)

    def test_pose_only_windows_do_not_make_preplaced_oracle_fullsuccess(self):
        rows,config,bundle=self.fixture()
        for row in rows:
            row['acceptance_truth_only']['brown_box']['pose']=[5,0,.02,0,0,0,1]
            c=row['acceptance_contacts_only'];m=np.zeros((54,3));m[0,2]=1
            c.update(box_filter_force_matrix_world_n=[m.tolist()],box_robot_force_matrix_world_n=m[1:].tolist(),box_hand_force_matrix_world_n=m[1:17].tolist(),box_bin_force_world_n=m[0].tolist(),
                points=[{'filter_index':0,'normal_world':[0,0,1],'normal_force_n':1.,'separation_m':0.}])
        result=self.evaluate(rows,config,bundle)
        self.assertGreaterEqual(result['max_ready_span_seconds'],2.);self.assertFalse(result['success']);self.assertFalse(result['stable_placement'])

    def test_all52_collider_identities_remain_correct_with_reordered_sdk(self):
        rows,config,bundle=self.fixture();robot=bundle[4];mapped=subject.body_ids.adapt_robot_geometry(robot)
        for before,after in zip(robot['collisions'],mapped['collisions']):
            self.assertEqual(after['body'],robot['bodies'][before['body']]['name'])
            self.assertEqual(config['audit_body_names'].index(after['body']),52-before['body'])
            self.assertEqual({k:v for k,v in before.items() if k!='body'},{k:v for k,v in after.items() if k!='body'})

    def test_prefilter_uses_derived_last_but_final_receives_original_raw_rows(self):
        rows,config,bundle=self.fixture();rows=rows[:151];config.update(mode='policy',stop_on_verified_success=True,evaluate_acceptance=True)
        before=deep_sha(rows);seen=[]
        def final(raw,ctx):self.assertIs(raw,rows);self.assertEqual(raw[-1]['acceptance_truth_only']['brown_box']['velocity'][0],.03);seen.append(True);return {'success':True}
        oldeval=subject.original.evaluate_episode;oldpref=subject.simulator.verified_success_check;oldgeom=subject.original.geometry
        with subject.pose_speed_measurement():
            result=subject.simulator.verified_success_check(rows,config,final)
            self.assertTrue(result['success']);self.assertEqual(seen,[True])
            self.assertIsNot(subject.original.evaluate_episode,oldeval)
            # Prebound aliases stay old, per documented foreground binding contract.
            self.assertIs(oldpref,subject._RAW_PREFILTER)
        self.assertIs(subject.original.evaluate_episode,oldeval);self.assertIs(subject.simulator.verified_success_check,oldpref);self.assertIs(subject.original.geometry,oldgeom)
        self.assertEqual(before,deep_sha(rows))
        with self.assertRaises(RuntimeError):
            with subject.pose_speed_measurement():raise RuntimeError('test')
        self.assertIs(subject.original.evaluate_episode,oldeval);self.assertIs(subject.simulator.verified_success_check,oldpref)

    def test_actual_frozen_simulator_uses_runtime_import_and_module_global_prefilter(self):
        tree=ast.parse(Path(subject.simulator.__file__).read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        evaluator_imports=[n for n in ast.walk(run) if isinstance(n,ast.ImportFrom) and n.module=='experiments.pi05_g1_mobile.acceptance' and any(a.name=='evaluate_episode' for a in n.names)]
        checks=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='verified_success_check']
        self.assertEqual(len(evaluator_imports),2);self.assertEqual(len(checks),1)

    def test_cpu_cli_enters_context_before_dispatch_and_restores(self):
        previous=sys.argv
        def fake(module,run_name):
            self.assertIs(subject.original.evaluate_episode,subject._evaluate_with_pose_speed)
            self.assertIs(subject.simulator.verified_success_check,subject._prefilter_with_pose_speed)
            return {'cpu':'only'}
        with patch.object(subject.runpy,'run_module',side_effect=fake),contextlib.redirect_stderr(io.StringIO()):
            result=subject.main(['--module','experiments.pi05_g1_mobile.transport_analysis','--','--help'])
        self.assertEqual(result,{'cpu':'only'});self.assertIs(sys.argv,previous)
        self.assertIs(subject.original.evaluate_episode,subject._RAW_EVALUATE)

if __name__=='__main__':unittest.main()
