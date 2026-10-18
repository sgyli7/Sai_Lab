"""Adversarial synthetic traces only: this is NOT physical validation.

Geometry/contact samples here are deliberate test fixtures, never simulator
results or evidence of a grasp. Two tests expose required ordering/hand rules.
"""
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from experiments.pi05_g1_mobile import acceptance


def quat(degrees):
    angle=math.radians(degrees)/2;return [0.,0.,math.sin(angle),math.cos(angle)]


class AcceptanceAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.box=np.array(list(itertools.product([-.05,.05],repeat=3)),float)
        self.floor=np.array([[-.4,-.4,0],[.4,-.4,0],[.4,.4,0],[-.4,.4,0]],float)
        polygon=acceptance._hull(self.floor);edge=np.roll(polygon,-1,axis=0)-polygon;inward=np.column_stack((-edge[:,1],edge[:,0]));inward/=np.linalg.norm(inward,axis=1)[:,None]
        self.robot={'bodies':[{'name':f'body{i}'} for i in range(53)],'collisions':[{'body':f'body{i}','index':i} for i in range(53)]};self.collision_calls=[]
        def bounds(collider,pose):
            self.collision_calls.append(collider['index']);center=np.asarray(pose[:3]);return center-.1,center+.1
        def gap(first,second):return float(np.max(np.maximum(np.asarray(first[0])-second[1],np.asarray(second[0])-first[1])))
        self.geometry=(self.box,self.floor,polygon,inward,self.robot,bounds,gap)
        self.config={'geometry_definition':'synthetic-unused','geometry_definition_sha256':'synthetic','robot_geometry':'synthetic-unused','robot_geometry_sha256':'synthetic','source_harness_dir':'synthetic-unused','source_ground_z':-.795,'acceptance':{'task_family':'synthetic_mobile_transport','stable_seconds':2.,'stable_position_m':.005,'stable_angle_degrees':5.,'minimum_source_target_distance_m':4.,'minimum_carried_displacement_m':4.,'minimum_carry_turn_degrees':60.},'audit_body_names':[f'body{i}' for i in range(53)],'audit_contact_metadata':{'schema_writes':0,'robot_filter_indices':list(range(53)),'hand_filter_indices':list(range(16)),'left_hand_filter_indices':list(range(8)),'right_hand_filter_indices':list(range(8,16)),'bin_filter_indices':[53],'filter_paths':[('/Robot/left_hand/'+str(i) if i<8 else '/Robot/right_hand/'+str(i) if i<16 else '/Robot/body'+str(i)) for i in range(53)]+['/blue_sorting_bin']}}
        names=[f'left_hand_part{i}' if i<8 else f'right_hand_part{i}' if i<16 else f'body{i}' for i in range(53)]
        self.robot['bodies']=[{'name':name} for name in names]
        for collider,name in zip(self.robot['collisions'],names):collider['body']=name
        self.config['audit_body_names']=names;self.config['audit_contact_metadata']['robot_body_names']=names

    def row(self,tick,position,*,held=False,supported=False,yaw=0.,hand_indices=(0,8)):
        matrix=np.zeros((1,54,3));points=[]
        if held:
            for index in hand_indices:
                matrix[0,index]=[1,0,0];points.append({'filter_index':index,'normal_world':[1,0,0],'normal_force_n':1.,'separation_m':-.001})
        if supported:
            matrix[0,53]=[0,0,10];points.append({'filter_index':53,'normal_world':[0,0,1],'normal_force_n':10.,'separation_m':0.})
        return {'control_tick':tick,'physics_steps_since_reset':tick*4,'monotonic_ns':tick*20_000_000,'root_link_pose_w':[0.,0.,0.,*quat(yaw)],'upright':1.,'acceptance_truth_only':{'brown_box':{'pose':[*position,0.,0.,0.,1.],'velocity':[0.]*6},'blue_sorting_bin':{'pose':[4.2,0.,0.,0.,0.,0.,1.]}},'acceptance_contacts_only':{'box_hand_force_matrix_world_n':matrix[0,:16].tolist(),'box_robot_force_matrix_world_n':matrix[0,:53].tolist(),'box_bin_force_world_n':matrix[0,53].tolist(),'box_filter_force_matrix_world_n':matrix.tolist(),'points':points},'acceptance_robot_body_link_poses_w':[[10.,10.,10.,0.,0.,0.,1.] for _ in range(53)]}

    def valid(self):
        return [self.row(0,[0,0,.1]),self.row(1,[0,0,.2],held=True),self.row(2,[4.2,0,.2],held=True),self.row(3,[4.2,0,.2],held=True,yaw=70)]+[self.row(tick,[4.2,0,.05],supported=True,yaw=70) for tick in range(4,105)]

    def evaluate(self,rows):
        with patch.object(acceptance,'geometry',return_value=self.geometry):return acceptance.evaluate_episode(rows,self.config)

    def test_complete_synthetic_proxy_requires101_samples_and_all53_colliders(self):
        result=self.evaluate(self.valid());self.assertTrue(result['success']);self.assertEqual(result['first_grasp_control_tick'],1)
        self.assertEqual(result['qualified_stability_window'],{'start_tick':4,'end_tick':104,'seconds':2.})
        self.assertEqual(result['physics_integrations'],104*4);self.assertEqual(result['actual_control_ticks'],104)
        self.assertEqual(set(self.collision_calls),set(range(53)));self.assertGreaterEqual(result['max_carried_box_displacement_m'],4)
        self.assertFalse(self.evaluate(self.valid()[:-1])['success'])

    def test_signed_contact_matrix_reconstruction_and_group_order(self):
        for field in ('raw_sign','derived_order','point_sum'):
            rows=self.valid();contact=rows[-1]['acceptance_contacts_only']
            if field=='raw_sign':contact['box_filter_force_matrix_world_n'][0][53][2]=-10
            elif field=='derived_order':contact['box_robot_force_matrix_world_n'][52][0]=1
            else:contact['points'][0]['normal_force_n']=9
            with self.subTest(field=field),self.assertRaises(ValueError):self.evaluate(rows)

    def test_all_robot_aabb_overlap_even_nonhand_last_collider_rejects_release(self):
        rows=self.valid()
        for row in rows[4:]:row['acceptance_robot_body_link_poses_w'][52][:3]=[4.2,0,.05]
        result=self.evaluate(rows);self.assertFalse(result['success']);self.assertFalse(result['stable_placement'])
        self.assertIn(52,self.collision_calls);self.assertFalse(result['samples'][-1]['released_separation_proved'])

    def test_proximity_and_side_force_are_not_upward_bin_support(self):
        for kind in ('none','side'):
            rows=self.valid()
            for row in rows[4:]:
                c=row['acceptance_contacts_only'];vector=[0,0,0] if kind=='none' else [10,0,0];c['box_filter_force_matrix_world_n'][0][53]=vector;c['box_bin_force_world_n']=vector;c['points']=[] if kind=='none' else [{'filter_index':53,'normal_world':[1,0,0],'normal_force_n':10.,'separation_m':0.}]
            with self.subTest(kind=kind):self.assertFalse(self.evaluate(rows)['success'])

    def test_position_and_rotation_pairwise_out_and_back_drift_rejected(self):
        for actor,kind in itertools.product(('brown_box','blue_sorting_bin'),('position','angle')):
            rows=self.valid()
            for j,row in enumerate(rows[4:]):
                pose=row['acceptance_truth_only'][actor]['pose'];phase=2*math.pi*j/100
                if kind=='position':pose[0]+=.0031*math.sin(phase) #6.2mm pairwise; equal endpoints.
                else:pose[3:]=quat(3.1*math.sin(phase)) #6.2deg pairwise.
            with self.subTest(actor=actor,kind=kind):self.assertFalse(self.evaluate(rows)['success'])

    def test_quaternion_sign_alias_is_same_pose_not_rotation_drift(self):
        rows=self.valid()
        for row in rows[4:]:
            if row['control_tick']%2:
                for actor in ('brown_box','blue_sorting_bin'):
                    pose=row['acceptance_truth_only'][actor]['pose'];pose[3:]=[-value for value in pose[3:]]
        self.assertTrue(self.evaluate(rows)['success'])

    def test_repeated_small_left_right_sways_do_not_accumulate_required_turn(self):
        rows=[self.row(0,[0,0,.1]),self.row(1,[0,0,.2],held=True)]
        rows.extend(self.row(tick,[4.2,0,.2],held=True,yaw=10 if tick%2 else -10) for tick in range(2,22))
        rows.extend(self.row(tick,[4.2,0,.05],supported=True) for tick in range(22,123))
        result=self.evaluate(rows);self.assertFalse(result['success']);self.assertAlmostEqual(result['carry_turn_degrees'],20.)
        self.assertIn('carry_turn_short',result['termination_reasons'])

    def test_long_oscillating_path_is_not_four_metre_carried_displacement(self):
        rows=[self.row(0,[0,0,.1]),self.row(1,[0,0,.2],held=True)]
        rows.extend(self.row(tick,[.5 if tick%2 else -.5,0,.2],held=True,yaw=70) for tick in range(2,22))
        rows.extend(self.row(tick,[4.2,0,.05],supported=True,yaw=70) for tick in range(22,123))
        result=self.evaluate(rows);self.assertFalse(result['success']);self.assertGreater(result['contact_qualified_carried_path_m'],4.)
        self.assertLess(result['max_carried_box_displacement_m'],4.);self.assertIn('carry_distance_short',result['termination_reasons'])

    def test_supported_but_moving_box_does_not_pass_stable_release(self):
        for angular in (False,True):
            rows=self.valid()
            for row in rows[4:]:row['acceptance_truth_only']['brown_box']['velocity'][3 if angular else 0]=.11 if angular else .021
            with self.subTest(angular=angular):self.assertFalse(self.evaluate(rows)['success'])

    def test_drop_and_fall_memory_cannot_be_cleared_by_later_recovery(self):
        for kind in ('drop','fall'):
            rows=self.valid()
            if kind=='drop':rows[2]=self.row(2,[0,0,-.80])
            else:rows[0]['root_link_pose_w'][2]=-.60
            result=self.evaluate(rows)
            with self.subTest(kind=kind):
                self.assertFalse(result['success']);self.assertTrue(result['dropped' if kind=='drop' else 'fell'])

    def test_trace_counter_and_nonfinite_contact_guards(self):
        for kind in ('extra_physics','reset','nan_contact','bad_quaternion'):
            rows=self.valid()
            if kind=='extra_physics':rows[2]['physics_steps_since_reset']+=1
            elif kind=='reset':rows[2]['control_tick']=0
            elif kind=='nan_contact':rows[2]['acceptance_contacts_only']['box_filter_force_matrix_world_n'][0][0][0]=float('nan')
            else:rows[2]['acceptance_truth_only']['brown_box']['pose'][3:]=[0,0,0,2]
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.evaluate(rows)

    def test_original_task_success_flags_do_not_substitute_for_independent_proof(self):
        rows=[self.row(tick,[0,0,.1]) for tick in range(105)]
        for row in rows:row.update(original_success=True,success=True,source_task_success=True)
        result=self.evaluate(rows);self.assertFalse(result['success']);self.assertIn('no_verified_grasp',result['termination_reasons'])

    def test_same_hand_two_fingers_must_not_count_as_two_hand_grasp(self):
        rows=self.valid()
        for tick in (1,2,3):
            old=rows[tick];position=old['acceptance_truth_only']['brown_box']['pose'][:3]
            rows[tick]=self.row(tick,position,held=True,yaw=70 if tick==3 else 0,hand_indices=(0,1))
        self.assertFalse(self.evaluate(rows)['success'],'Both force channels belong to the left hand, not a two-hand contact proxy')

    def test_placement_before_grasp_and_carry_must_not_qualify_later_success(self):
        rows=[self.row(0,[0,0,.1])]+[self.row(tick,[4.2,0,.05],supported=True) for tick in range(1,102)]
        rows.extend([self.row(102,[4.2,0,.2],held=True),self.row(103,[8.4,0,.2],held=True),self.row(104,[8.4,0,.2],held=True,yaw=70)])
        self.assertFalse(self.evaluate(rows)['success'],'Stable bin window precedes first grasp and later carry; final box is not placed')

    def test_placement_before_carry_and_turn_milestones_must_not_count(self):
        rows=[self.row(0,[0,0,.1]),self.row(1,[0,0,.2],held=True)]+[self.row(tick,[4.2,0,.05],supported=True) for tick in range(2,103)]
        rows.extend([self.row(103,[8.4,0,.2],held=True),self.row(104,[8.4,0,.2],held=True,yaw=70)])
        self.assertFalse(self.evaluate(rows)['success'],'Even after a grasp, an early placement window cannot precede qualified carry/turn')

    def test_geometry_uses_convex_part2_floor_and_checks_hash(self):
        acceptance.geometry.cache_clear()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);definition={'objects':[{'kind':'t2_box','convex_parts':[{'points':self.box.tolist()}]},{'kind':'t2_bin','convex_parts':[{'points':(self.floor*10).tolist()},{'points':(self.floor+3).tolist()},{'points':self.floor.tolist()}]}]}
            source=root/'geometry.json';source.write_text(json.dumps(definition));robot=root/'robot.json';robot.write_text(json.dumps(self.robot))
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            module=SimpleNamespace(collision_bounds=self.geometry[-2],aabb_gap=self.geometry[-1])
            try:
                with patch.dict(sys.modules,{'unitree_g1_t1_source_placement':module}):
                    data=acceptance.geometry(str(source),sha(source),str(robot),sha(robot),str(root));np.testing.assert_array_equal(data[1],self.floor)
                    with self.assertRaisesRegex(ValueError,'geometry changed'):acceptance.geometry(str(source),'0'*64,str(robot),sha(robot),str(root))
            finally:acceptance.geometry.cache_clear()

    def test_single_sensor_nested_physx_filter_metadata_matches_flat_fixture(self):
        rows=self.valid();flat=self.evaluate(rows)
        self.config['audit_contact_metadata']['filter_paths']=[self.config['audit_contact_metadata']['filter_paths']]
        nested=self.evaluate(rows)
        self.assertEqual(nested['success'],flat['success']);self.assertEqual(nested['qualified_stability_window'],flat['qualified_stability_window'])
        self.config['audit_contact_metadata']['filter_paths']*=2
        with self.assertRaisesRegex(ValueError,'coverage mismatch'):self.evaluate(rows)

    def test_real_development_producer1501_rows_nested_metadata_cpu_reaudit(self):
        # Authorized development evidence only; never formal/heldout observations.
        root=Path(__file__).resolve().parents[3]
        folder=root/'.scratch/pi05_g1/round03_mobile/development/source_native_seed200_attempt02'
        if not (folder/'result.json').exists():self.skipTest('Local development producer fixture unavailable')
        raw=json.loads((folder/'result.json').read_text());config=json.loads((folder/'config.json').read_text())
        rows=[json.loads(line) for line in (folder/'trajectory.jsonl').read_text().splitlines()]
        config['audit_contact_metadata']=raw['contact_view'];config['audit_body_names']=raw['body_names']
        self.assertEqual(raw['status'],'failed');self.assertIn('Raw contact coverage mismatch',raw['error'])
        self.assertEqual(len(rows),1501);self.assertEqual(rows[-1]['physics_steps_since_reset'],6000)
        self.assertEqual(len(raw['contact_view']['filter_paths']),1);self.assertEqual(len(raw['contact_view']['filter_paths'][0]),54)
        result=acceptance.evaluate_episode(rows,config)
        self.assertTrue(result['grasped']);self.assertFalse(result['stable_placement']);self.assertFalse(result['success'])
        self.assertAlmostEqual(result['max_carried_box_displacement_m'],1.7747,places=3)
        self.assertAlmostEqual(result['carry_turn_degrees'],90.63,places=1)


if __name__=='__main__':unittest.main()
