"""CPU gate and owned Docker launcher for TRAIN-only scripted demonstration."""
import argparse
import json
from pathlib import Path

from experiments.lingbot_vla2_g1.data import OWN, sha
from experiments.lingbot_vla2_g1 import sim_launcher_pose_speed as launcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (config.get('policy_id') != 'privileged_scripted_TRAIN_teacher'
            or config.get('privileged_TRAIN_teacher', {}).get('counts_as_LingBot_success') is not False
            or not Path(config['output_dir']).is_relative_to(OWN)
            or config['policy_command'][2] != 'experiments.lingbot_vla2_g1.diagnostics.privileged_teacher'):
        raise ValueError('Only explicit owned TRAIN collector admitted')
    if Path(config['output_dir']).exists():
        raise FileExistsError('Preserve every prior raw collection')
    for path, expected in config['frozen_files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen TRAIN collector input changed: ' + path)
    original = launcher.docker_command
    def command(*args, **kwargs):
        result = original(*args, **kwargs)
        index = result.index('experiments.pi05_g1_mobile.sim_launcher_pose_speed')
        result[index] = 'experiments.lingbot_vla2_g1.diagnostics.teacher_sim_entry'
        return result
    launcher.docker_command = command
    try:
        launcher.run_owned_docker(args.config)
    finally:
        launcher.docker_command = original


if __name__ == '__main__':
    main()
