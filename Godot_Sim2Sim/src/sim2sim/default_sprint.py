"""Install the accepted walking/sprint pair without replacing other robot skills."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from sim2sim.research.queue import atomic_json


PROFILE = Path(__file__).parent / 'assets' / 'microduck_sprint_v1'


def apply_default_sprint(project, *, profile_directory=PROFILE):
    project, profile_directory = Path(project), Path(profile_directory)
    assets = project / 'runtime_assets'
    deployment_path = assets / 'deployment.json'
    deployment = json.loads(deployment_path.read_text())
    profile = json.loads((profile_directory / 'profile.json').read_text())
    if (deployment['physics_hz'], deployment['decimation']) != (200, 4):
        raise ValueError('Default sprint requires 200 Hz physics and 50 Hz decisions')
    robot = deployment['robots']['walk']
    spec = project / robot['spec'].removeprefix('res://')
    if hashlib.sha256(spec.read_bytes()).hexdigest() != profile['robot_spec_sha256']:
        raise ValueError('Default sprint robot does not match its accepted physical asset')
    fixtures = json.loads((assets / 'self_test.json').read_text())
    additions = json.loads((profile_directory / 'self_test.json').read_text())['cases']
    policies = profile['policies']
    if set(policies) != {'walking', 'sprint'}:
        raise ValueError('Default sprint profile must contain exactly walking and sprint')
    if {v['skill']: v['sha256'] for v in additions} != {k: v['sha256'] for k,v in policies.items()}:
        raise ValueError('Default sprint numerical fixtures do not match its models')
    transfers = []
    for skill, item in policies.items():
        name = Path(item['path']).name
        if item['path'] != 'res://runtime_assets/policies/' + name:
            raise ValueError('Invalid default sprint model resource path')
        source = profile_directory / name
        manifest = json.loads(source.with_suffix('.manifest.json').read_text())
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256'] or manifest != item['manifest']:
            raise ValueError('Default sprint model or manifest checksum mismatch: ' + skill)
        target = assets / 'policies' / name
        if target.is_symlink() or target.parent.is_symlink():
            raise ValueError('Refusing to modify shared model symlinks')
        transfers.append((source, target))
    before = {k:v['sha256'] for k,v in deployment['policies'].items()}
    updated = copy.deepcopy(deployment)
    updated['policies'].update(policies)
    updated.setdefault('control_config', {})['walk'] = profile['walk_control']
    updated['default_sprint_profile'] = profile['id']
    fixtures['cases'] = [v for v in fixtures['cases'] if v['skill'] not in policies] + additions
    for source, target in transfers:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        shutil.copyfile(source.with_suffix('.manifest.json'), target.with_suffix('.manifest.json'))
    atomic_json(assets / 'self_test.json', fixtures)
    atomic_json(deployment_path, updated)
    return dict(profile=profile['id'], before=before,
                after={k:v['sha256'] for k,v in updated['policies'].items()},
                unchanged_skills=[k for k in before if k not in policies])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    print(json.dumps(apply_default_sprint(parser.parse_args().project)))
