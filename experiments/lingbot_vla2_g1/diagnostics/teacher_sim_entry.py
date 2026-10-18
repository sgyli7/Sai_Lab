"""Owned source-image entry for explicitly privileged TRAIN data collection."""
import argparse
from pathlib import Path
import subprocess

from experiments.pi05_g1_mobile.sim_launcher_pose_speed import PYTHON, ROOT, probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', type=Path, default=PYTHON)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    receipt = probe(args.python)
    if not receipt['ready']:
        raise RuntimeError('Frozen source runtime unavailable: ' + str(receipt['blockers']))
    subprocess.run([str(args.python), '-m', 'experiments.lingbot_vla2_g1.diagnostics.teacher_measurement',
                    '--config', str(args.config.absolute())], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
