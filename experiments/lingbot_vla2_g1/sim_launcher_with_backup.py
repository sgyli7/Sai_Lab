"""The same owned source simulator, with this task's backups mounted read-only."""
from pathlib import Path

from . import sim_launcher_pose_speed as base

BACKUP = Path('/home/ethan/ProjectBackups/LingBotVLA2_G1_checkpoints_20261017T045028Z')
_source_docker_command = base.docker_command


def docker_command(*args, **kwargs):
    command = _source_docker_command(*args, **kwargs)
    index = command.index('--workdir')
    return [*command[:index], '--volume', f'{BACKUP}:{BACKUP}:ro', *command[index:]]


def run_owned_docker(config):
    previous = base.docker_command
    base.docker_command = docker_command
    try:
        return base.run_owned_docker(config)
    finally:
        base.docker_command = previous
