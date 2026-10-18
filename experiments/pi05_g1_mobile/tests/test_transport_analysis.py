import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.pi05_g1_mobile import transport_analysis as subject


class TransportTests(unittest.TestCase):
    def fixture(self):
        context={'acceptance':{'minimum_source_target_distance_m':5.,'minimum_carried_displacement_m':4.,'minimum_carry_turn_degrees':60.},
                 'audit_contact_metadata':{'robot_filter_indices':[1,2],'bin_filter_indices':[0]}}
        rows=[];samples=[]
        for tick,(x,angle) in enumerate([(0,0),(2,30),(4.2,70)]):
            rows.append({'control_tick':tick,'physics_steps_since_reset':4*tick,'monotonic_ns':10**9+tick*10**8,
                'root_link_pose_w':[0,0,0,0,0,math.sin(math.radians(angle)/2),math.cos(math.radians(angle)/2)],
                'acceptance_truth_only':{'brown_box':{'pose':[x,0,1,0,0,0,1]},'blue_sorting_bin':{'pose':[5,0,0,0,0,0,1]}},
                'acceptance_contacts_only':{'points':[{'filter_index':1,'separation_m':-.001,'normal_force_n':1}],
                    'box_robot_force_matrix_world_n':[[1,0,0],[0,1,0]],'box_hand_force_matrix_world_n':[[1,0,0],[0,1,0]]}})
            samples.append({'tick':tick,'grasp_contact_lift_proxy':True,'left_hand_contact':True,'right_hand_contact':True,
                'box_inside_bin_prism':False,'bin_upward_force_n':0.,'released_separation_proved':False,'placement_ready':False,
                'standing':True,'fell':False,'ground_drop':False})
        judgment={'samples':samples,'first_grasp_control_tick':0,'source_target_distance_m':5.,'carry_qualified_control_tick':2,
            'max_carried_box_displacement_m':4.2,'contact_qualified_carried_path_m':4.2,'carry_turn_degrees':70.,
            'hand_contact_gap_events_before_arrival':0,'largest_hand_contact_gap_seconds':0.,'contact_gap_note':'sampled only',
            'grasped':True,'stable_placement':False,'fell':False,'dropped':False,'success':False,'timeout':True}
        return rows,judgment,context

    def test_existing_transport_condition_is_not_arrival(self):
        rows,judgment,context=self.fixture();out=subject.summarize_rows(rows,judgment,context)
        self.assertEqual(out['first_4m_60deg_qualified_tick'],2)
        self.assertTrue(out['transport_condition_is_not_arrival_or_task_success'])
        self.assertAlmostEqual(out['box_target_actor_origin_xy_error_m']['final'],.8)
        self.assertEqual(out['last10seconds']['predicates']['bin_support_contact']['true_samples'],0)
        self.assertEqual(out['last10seconds']['predicates']['released_no_sampled_robot_contact']['true_samples'],0)
        self.assertNotIn('arrived',out)

    def test_failed_infrastructure_retains_unknown_full_task(self):
        rows,judgment,context=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);geometry=folder/'geometry.json';geometry.write_text('{}')
            config={'mode':'policy','case':{'case_id':'synthetic'},'policy_id':'p','acceptance':context['acceptance'],
                'geometry_definition':str(geometry),'robot_geometry':str(geometry)}
            (folder/'config.json').write_text(json.dumps(config))
            (folder/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result={'status':'failed','success':None,'termination_reason':'infra_wall_deadline','source_interface_passed':True,
                'body_names':['hand'],'contact_view':context['audit_contact_metadata'],
                'trace_sha256':subject.sha(folder/'trajectory.jsonl')}
            (folder/'result.json').write_text(json.dumps(result))
            with patch.object(subject.acceptance,'evaluate_episode',return_value=judgment):out=subject.analyze_case(folder)
            self.assertIsNone(out['full_task_verdict']);self.assertIsNone(out['full_task_timeout'])
            self.assertTrue(out['diagnostic_prefix_only']);self.assertEqual(out['result_status'],'failed')
            self.assertTrue(out['observed_physical_flags']['grasped'])

    def test_gap_count_and_sample_span_are_distinct(self):
        out=subject.intervals([4,8,9])
        self.assertEqual(out[0]['sample_span_seconds'],0.)
        self.assertEqual(out[0]['missing_sample_count_over_50Hz_seconds'],.02)
        self.assertEqual(out[1]['sample_span_seconds'],.02)
        self.assertEqual(out[1]['missing_sample_count_over_50Hz_seconds'],.04)


if __name__=='__main__':unittest.main()
