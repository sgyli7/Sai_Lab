"""Create an isolated, explicit neck-target intervention for native replay.

Research only: never changes the source runtime, model, or physical parameters.
Negative neck_pitch moves the head/neck forward for this robot's joint axes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from sim2sim.research.budget import require_supervision
from sim2sim.standalone.suite import runtime_inputs


def patch_driver(source, degrees):
    if degrees not in (0, -5, -10):
        raise ValueError('Frozen experiment only supports 0, -5, -10 degrees')
    replacements = [
        ('const CONTROL_DT := 0.02\n',
         f'const CONTROL_DT := 0.02\nconst EXPERIMENT_NECK_DEG: float = {float(degrees)}\nvar experiment_neck_bias_rad: float = 0.0\n'),
        ('func _reset_controller() -> void:\n',
         'func _reset_controller() -> void:\n\texperiment_neck_bias_rad = 0.0\n'),
        ('\tif skill == "roulade": return _bodies.keys()\n',
         '\tif session.mode == "walk" and session.trace_path != "": return _bodies.keys()\n'
         '\tif skill == "roulade": return _bodies.keys()\n'),
        ('\tvar ctrl := Contract.control(action,home,float(robot_config.action_scale))\n',
         '\t# Research intervention: actual final action remains the recurrent feedback.\n'
         '\tvar experiment_policy_action := action.duplicate()\n'
         '\tvar experiment_target: float = deg_to_rad(EXPERIMENT_NECK_DEG) if skill == "sprint" and requested_command[0] > 0.0 else 0.0\n'
         '\texperiment_neck_bias_rad = move_toward(experiment_neck_bias_rad,experiment_target,abs(deg_to_rad(EXPERIMENT_NECK_DEG))*CONTROL_DT/0.5)\n'
         '\tif experiment_neck_bias_rad != 0.0:\n'
         '\t\taction[5] += experiment_neck_bias_rad/float(robot_config.action_scale)\n'
         '\tvar ctrl := Contract.control(action,home,float(robot_config.action_scale))\n'),
        ('\t\t\t"obs":Array(obs),"action":Array(action),"last_action":Array(last_action),\n',
         '\t\t\t"experiment_policy_action":Array(experiment_policy_action),"experiment_neck_bias_rad":experiment_neck_bias_rad,\n'
         '\t\t\t"obs":Array(obs),"action":Array(action),"last_action":Array(last_action),\n'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Driver seam changed; preserve the source and review the patch')
        source = source.replace(old, new)
    return source


def prepare(project, output, degrees):
    before = runtime_inputs(project)
    if output.exists():
        raise FileExistsError(output)
    subprocess.run(['cp', '--reflink=auto', '-a', str(project), str(output)], check=True, timeout=60)
    driver = output / 'standalone/driver.gd'
    if driver.is_symlink():
        raise ValueError('Experiment driver must be an owned regular file')
    driver.write_text(patch_driver(driver.read_text(), degrees))
    after = runtime_inputs(output)
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    if changed != {'standalone/driver.gd'} or runtime_inputs(project) != before:
        raise ValueError('Intervention changed more than the isolated driver')
    record = dict(degrees=degrees, ramp_seconds=.5, changed_inputs=sorted(changed),
                  source_inputs=before, candidate_inputs=after,
                  driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
                  physical_parameters_changed=False, models_changed=False)
    output.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--degrees', type=int, choices=[0, -5, -10], required=True)
    args = parser.parse_args()
    require_supervision(args.session)
    if not args.output.resolve().is_relative_to(args.session.resolve()):
        raise ValueError('Output must belong to the supervised session')
    print(prepare(args.project.resolve(), args.output.resolve(), args.degrees)['driver_sha256'])
