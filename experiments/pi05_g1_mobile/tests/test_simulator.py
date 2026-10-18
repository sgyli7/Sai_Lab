import copy
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

from experiments.pi05_g1_mobile.simulator import validate_config,apply_initial_scene,suppress_term,sample_contacts,verified_success_check,PolicyProcess


class Value:
    def __init__(self, active):self.active=active
    def any(self):return self.active
    def __and__(self, other):return Value(self.active and other)


class SimulatorTests(unittest.TestCase):
    def config(self):
        return {'mode':'diagnostic','ticks':10,'case':{'case_id':'reset','seed':42},'scene':{}}

    def test_original_timing_and_bounded_diagnostic(self):
        x=self.config();original=copy.deepcopy(x);out=validate_config(x)
        self.assertEqual(out['episode_seconds'],30.)
        self.assertEqual(x,original)
        x['ticks']=11
        with self.assertRaises(ValueError):validate_config(x)
        x['ticks']=0
        self.assertEqual(validate_config(x)['ticks'],0)
        x['mode']='diagnostic_hold'
        self.assertEqual(validate_config(x)['mode'],'diagnostic')

    def test_contact_groups_are_measured_matrix_slices(self):
        import numpy as np
        class Array:
            def __init__(self,v):self.v=np.asarray(v)
            def detach(self):return self
            def cpu(self):return self
            def numpy(self):return self.v
        matrix=np.arange(54*3,dtype=float).reshape(1,54,3)
        view=SimpleNamespace(filter_count=54,max_contact_data_count=4096,
            get_net_contact_forces=lambda dt:Array(matrix.sum(axis=1)),
            get_contact_force_matrix=lambda dt:Array(matrix),
            get_contact_data=lambda dt:(Array(np.zeros((0,1))),Array(np.zeros((0,3))),
                Array(np.zeros((0,3))),Array(np.zeros((0,1))),Array(np.zeros((1,54))),Array(np.zeros((1,54)))))
        metadata={'hand_filter_indices':list(range(1,17)),
                  'robot_filter_indices':list(range(1,54)),'bin_filter_indices':[0]}
        result=sample_contacts(view,lambda x:x,metadata)
        self.assertEqual(np.asarray(result['box_hand_force_matrix_world_n']).shape,(16,3))
        self.assertEqual(np.asarray(result['box_robot_force_matrix_world_n']).shape,(53,3))
        self.assertEqual(result['box_bin_force_world_n'],matrix[0,0].tolist())

    def test_extension_explicit_and_policy_limits_cadence_required(self):
        x=self.config();x.update(mode='policy',ticks=3000,episode_seconds=60,
            native_action_horizon=50,replan_interval_ticks=10,action_limits_file='frozen.json')
        self.assertEqual(validate_config(x)['episode_seconds'],60.)
        for key,value in [('native_action_horizon',40),('replan_interval_ticks',51),
                          ('action_limits_file',None),('episode_seconds',float('nan'))]:
            bad={**x,key:value}
            with self.assertRaises(ValueError):validate_config(bad)

    def test_scene_config_applies_only_prebuild_initial_poses(self):
        calls=[]
        def asset(name):return SimpleNamespace(set_initial_pose=lambda pose:calls.append((name,pose)))
        description=SimpleNamespace(scene=SimpleNamespace(assets={
            'brown_box':asset('box'),'blue_sorting_bin':asset('bin')}),embodiment=asset('robot'))
        pose={'position_xyz':[0,5,0],'rotation_xyzw':[0,0,0,1]}
        config=self.config();config['scene']={'bin_pose':pose}
        validate_config(config)
        apply_initial_scene(description,config['scene'],SimpleNamespace)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],'bin')
        self.assertEqual(calls[0][1].position_xyz,(0,5,0))
        apply_initial_scene(description,{},SimpleNamespace)
        self.assertEqual(len(calls),1)
        config['scene']['bin_pose']['rotation_xyzw']=[0,0,0,2]
        with self.assertRaises(ValueError):validate_config(config)

    def test_original_predicate_preserved_but_autoreset_suppressed(self):
        cfg=SimpleNamespace(func=lambda:Value(True))
        saved={}
        manager=SimpleNamespace(get_term_cfg=lambda name:cfg,
            set_term_cfg=lambda name,value:saved.update({name:value}))
        raw=SimpleNamespace(termination_manager=manager,_sim_step_counter=20)
        events={}
        suppress_term(raw,'success',events)
        self.assertFalse(saved['success'].func().any())
        self.assertTrue(cfg.func().any())
        self.assertEqual(events['success']['first_control_tick'],5)
        raw._sim_step_counter=24;saved['success'].func()
        self.assertEqual(events['success']['calls'],2)
        self.assertEqual(events['success']['first_control_tick'],5)

    def success_row(self,tick=100):
        return {'control_tick':tick,'original_terms':{'success':{'ever':True}},
            'acceptance_contacts_only':{'box_bin_force_world_n':[0,0,1],
                'box_robot_force_matrix_world_n':[[0,0,0] for _ in range(53)]},
            'acceptance_truth_only':{'brown_box':{'velocity':[0]*6}}}

    def test_success_prefilter_only_invokes_independent_judge(self):
        config={'mode':'policy','stop_on_verified_success':True,'evaluate_acceptance':True}
        row=self.success_row();rows=[row];before=copy.deepcopy(rows)
        judge=MagicMock(return_value={'success':True,'actions':'must never consume actions'})
        result=verified_success_check(rows,config,judge)
        self.assertTrue(result['success']);judge.assert_called_once_with(rows,config)
        self.assertNotIn('actions',result);self.assertEqual(rows,before)
        judge.reset_mock();judge.return_value={'success':False}
        self.assertFalse(verified_success_check(rows,config,judge)['success'])
        self.assertEqual(rows,before)

    def test_success_checks_have_no_diagnostic_replay_or_unqualified_trigger(self):
        config={'mode':'policy','stop_on_verified_success':True,'evaluate_acceptance':True}
        judge=MagicMock(return_value={'success':True})
        for change in ({'mode':'diagnostic'},{'mode':'replay'},{'evaluate_acceptance':False},{'stop_on_verified_success':False}):
            self.assertIsNone(verified_success_check([self.success_row()],{**config,**change},judge))
        for tick in (0,99,101,149):
            self.assertIsNone(verified_success_check([self.success_row(tick)],config,judge))
        rows=[]
        for key in ('source','support','contact','speed','angular'):
            row=self.success_row()
            if key=='source':row['original_terms']['success']['ever']=False
            if key=='support':row['acceptance_contacts_only']['box_bin_force_world_n'][2]=0
            if key=='contact':row['acceptance_contacts_only']['box_robot_force_matrix_world_n'][1][0]=1e-9
            if key=='speed':row['acceptance_truth_only']['brown_box']['velocity'][0]=.02
            if key=='angular':row['acceptance_truth_only']['brown_box']['velocity'][3]=.1
            rows.append(row)
        for row in rows:self.assertIsNone(verified_success_check([row],config,judge))
        judge.assert_not_called()

    def test_policy_wire_has_no_acceptance_or_object_truth(self):
        import numpy as np
        policy=object.__new__(PolicyProcess);policy.timeout=1
        incoming={'episode_id':'case','control_tick':100,'actions':np.zeros((50,32)).tolist(),'inference_seconds':.1}
        policy.process=SimpleNamespace(stdin=io.StringIO(),stdout=io.StringIO(json.dumps(incoming)+'\n'))
        selector=MagicMock();selector.__enter__.return_value=selector;selector.select.return_value=[('ready',None)]
        with patch('experiments.pi05_g1_round02.simulator.selectors.DefaultSelector',return_value=selector):
            policy.infer(np.zeros((480,640,3),dtype=np.uint8),np.zeros(32),{'case_id':'case','seed':42},100,'task',np.zeros(43),['joint']*43)
        wire=json.loads(policy.process.stdin.getvalue())
        self.assertEqual(set(wire),{'schema','episode_id','control_tick','rgb_shape','rgb_uint8_b64','state','prompt','seed','measured_joint_positions43','measured_joint_names43'})
        self.assertNotIn('acceptance_truth_only',wire)
        self.assertNotIn('audit_contact_metadata',wire)


if __name__=='__main__':unittest.main()
