import copy
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from sim2sim.standalone.sprint import metrics,templates,write_cases
from sim2sim.standalone.sprint_accept import paired_acceptance


def straight_trace():
    case=templates()['shift_first'];case['sprint_intervals']=[[1,9]]
    rows=[];actions=[];x=0.
    for step in range(600):
        t=step*.02;moving=1<=t<9;speed=.24 if moving else 0.;x+=speed*.02
        rows.append(dict(time=t+.02,vel=np.array([speed,0,0]),xy=np.array([x,0]),
                         yaw=0.,tilt=3.,z=.115))
        actions.append(dict(t=t,skill='sprint' if moving else 'walking',requested_command=[.4 if moving else 0.,0,0]))
    return rows,actions,case


class SprintScore(unittest.TestCase):
    def test_paired_acceptance_rejects_missing_falling_or_mismatched_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=write_cases(tmp,[919000],paired=True,selected=['shift_first'])
            episodes=[]
            for path in paths:
                case=json.loads(path.read_text());ordinary=case['ordinary_control']
                episodes.append(dict(case_path=str(path),completed=True,models={'walking':'a','sprint':'b'},
                    runtime_id='frozen',task_metrics=dict(success=True,fell=False,
                    actor_selection_passed=True,sustained_mean_vx=.2 if ordinary else .24)))
            self.assertTrue(paired_acceptance({'episodes':episodes},paths)['accepted'])
            for failure in ['missing','duplicate','fall','slow','runtime']:
                changed=copy.deepcopy(episodes)
                if failure=='missing':changed.pop()
                elif failure=='duplicate':changed.append(changed[0])
                elif failure=='fall':changed[0]['task_metrics'].update(success=False,fell=True)
                elif failure=='slow':changed[0]['task_metrics']['sustained_mean_vx']=.21
                else:changed[0]['runtime_id']='different'
                self.assertFalse(paired_acceptance({'episodes':changed},paths)['accepted'],failure)

    def test_stable_straight_and_settled_release(self):
        rows,actions,case=straight_trace();result=metrics(rows,actions,case)
        self.assertTrue(result['success']);self.assertAlmostEqual(result['sustained_mean_vx'],.24)
        self.assertTrue(result['stops'][0]['passed'])

    def test_speed_cannot_hide_fall_direction_or_wrong_actor(self):
        for error in ['fall','direction','actor']:
            rows,actions,case=straight_trace()
            if error=='fall':rows[200]['tilt']=61.
            elif error=='direction':
                for row in rows[100:450]:row['yaw']=.4
            else:actions[200]['skill']='walking'
            self.assertFalse(metrics(rows,actions,case)['success'],error)

    def test_incomplete_replay_is_error(self):
        rows,actions,case=straight_trace()
        with self.assertRaises(ValueError):metrics(rows[:-1],actions[:-1],case)
