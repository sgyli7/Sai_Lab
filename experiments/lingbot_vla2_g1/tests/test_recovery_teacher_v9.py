"""Keep the successful v8 controller, with the actual matched R6 prefix."""
import copy
import json
from pathlib import Path

import numpy as np

from experiments.lingbot_vla2_g1.data import OWN, sha
from experiments.lingbot_vla2_g1.tests import test_recovery_teacher_v8 as legacy
from experiments.lingbot_vla2_g1.diagnostics.privileged_teacher_v9 import PlacementTeacher, validate_source
from experiments.lingbot_vla2_g1.diagnostics.privileged_teacher_v8 import PlacementTeacher as PreviousTeacher
from experiments.lingbot_vla2_g1.diagnostics.teacher_measurement_v9 import attribute_teacher


class MatchedRecoveryTests(legacy.R6RecoveryTests):
    def teacher(self):
        return PlacementTeacher(self.rows, self.objects, self.robot, self.result['body_names'],
                                action_bounds=self.result['actual_action_bounds'])

    def test_attribution_uses_actual_declared_prefix_and_query_boundary(self):
        config = {'privileged_TRAIN_teacher': {'TRAIN_only': True, 'collector_version': 9,
                                               'initial_replay_controls': 1800}}
        for ticks in [0, 1799, 1800, 1801, 2750, 3000]:
            actual = attribute_teacher({'success': False}, [None]*(ticks+1), config)
            self.assertEqual(actual['recorded_TRAIN_action_prefix_controls'], min(1800, ticks))
            self.assertEqual(actual['privileged_truth_feedback_queries'], len(range(1800, ticks, 50)))
            self.assertFalse(actual['counts_as_LingBot_success'])
        config['privileged_TRAIN_teacher']['initial_replay_controls'] = 1700
        with self.assertRaises(ValueError):
            attribute_teacher({}, [None]*2751, config)

    def matched_source(self):
        folder = OWN / 'matched_parent_R6_against_R7_plan/episodes/round07_fresh_full60'
        rows = [json.loads(s) for s in (folder/'trajectory.jsonl').read_text().splitlines()]
        result = json.loads((folder/'result.json').read_text())
        config = json.loads((folder/'config.json').read_text())
        objects = json.loads(Path(config['geometry_definition']).read_text())
        robot = json.loads(Path(config['robot_geometry']).read_text())
        return folder, rows, result, objects, robot

    def test_new_matched_source_prefix_exactly_replays_all1800_applied_controls(self):
        _, rows, result, objects, robot = self.matched_source()
        teacher = PlacementTeacher(rows, objects, robot, result['body_names'], action_bounds=result['actual_action_bounds'])
        for tick in range(0, 1800, 50):
            actual, info = teacher.actions(rows[tick])
            np.testing.assert_array_equal(actual, np.asarray([r['applied_action32'] for r in rows[tick+1:tick+51]], np.float32))
            self.assertFalse(info['privileged_truth_used'])
        np.testing.assert_array_equal(teacher.last, teacher.replay[-1])

    def test_v9_commands_and_diagnostics_equal_successful_v8_algorithm_on_new_rows(self):
        _, rows, result, objects, robot = self.matched_source()
        args = (rows, objects, robot, result['body_names'])
        old = PreviousTeacher(*args, action_bounds=result['actual_action_bounds'])
        new = PlacementTeacher(*args, action_bounds=result['actual_action_bounds'])
        for tick in range(0, 3000, 50):
            previous, previous_info = old.actions(rows[tick])
            actual, info = new.actions(rows[tick])
            np.testing.assert_array_equal(actual, previous)
            self.assertEqual(info, previous_info)
            self.assertTrue(np.all(actual >= np.asarray(result['actual_action_bounds']['lower'])-1e-6))
            self.assertTrue(np.all(actual <= np.asarray(result['actual_action_bounds']['upper'])+1e-6))

    def test_only_declared_new_longcarry_source_admits_not_R7_shortcarry_or_old_seed(self):
        folder, _, _, _, _ = self.matched_source()
        declaration = copy.deepcopy(json.loads((OWN/'placement_recovery_teacher08/config.json').read_text())['privileged_TRAIN_teacher'])
        declaration.update(collector_version=9, source_trace=str(folder/'trajectory.jsonl'), source_result=str(folder/'result.json'),
                           reclassification=str(OWN/'diagnostics/development_to_TRAIN_recovery_v9.json'))
        for key in ('source_trace', 'source_result', 'reclassification'):
            declaration[key+'_sha256'] = sha(declaration[key])
        validate_source(declaration)
        for wrong in (OWN/'round07_development_plan/episodes/round07_fresh_full60',
                      OWN/'round06_development_plan/episodes/round06_fresh_full60'):
            changed = copy.deepcopy(declaration)
            for key, file in [('source_trace','trajectory.jsonl'), ('source_result','result.json')]:
                changed[key] = str(wrong/file)
                changed[key+'_sha256'] = sha(changed[key])
            with self.assertRaisesRegex(ValueError, 'reviewed nonformal'):
                validate_source(changed)
