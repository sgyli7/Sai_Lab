"""Owned launcher for the explicitly privileged versioned TRAIN collector."""
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
    declaration = config.get('privileged_TRAIN_teacher', {})
    if (config.get('policy_id') != 'privileged_scripted_TRAIN_teacher'
            or declaration.get('counts_as_LingBot_success') is not False
            or declaration.get('collector_version') != 2
            or not Path(config['output_dir']).is_relative_to(OWN)
            or config['policy_command'][2] != 'experiments.lingbot_vla2_g1.diagnostics.privileged_teacher_v2'):
        raise ValueError('Only explicit owned v2 TRAIN collector admitted')
    if Path(config['output_dir']).exists():
        raise FileExistsError('Preserve every prior collection')
    for path, expected in config['frozen_files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen collector input changed: ' + path)
    original = launcher.docker_command
    def command(*args, **kwargs):
        argv = original(*args, **kwargs)
        index = argv.index('experiments.pi05_g1_mobile.sim_launcher_pose_speed')
        argv[index] = 'experiments.lingbot_vla2_g1.diagnostics.teacher_sim_entry'
        return argv
    launcher.docker_command = command
    try:
        launcher.run_owned_docker(args.config)
    finally:
        launcher.docker_command = original


if __name__ == '__main__':
    main()
