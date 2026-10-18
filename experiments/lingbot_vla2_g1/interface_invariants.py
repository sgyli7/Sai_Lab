"""Keep full per-case identities while allowing only frozen box-X/Y variants."""
import copy
import hashlib
import json
from pathlib import Path


def digest_identity(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def invariant_identity(identity, config, claimed_sha):
    if digest_identity(identity) != claimed_sha:
        raise ValueError('Measured interface content does not reproduce its full SHA')
    if identity.get('scene_config') != config['scene']:
        raise ValueError('Measured scene_config differs from frozen case')
    actual = identity['scene']['brown_box']['init_state']['pos']
    planned = config['scene']['box_pose']['position_xyz']
    if actual != planned:
        raise ValueError('Measured SDK box initialization differs from frozen position')
    # The frozen source harness explicitly enforces raw.cfg.sim.dt == .005;
    # physics_hz is a result/contract field, not a case-config field.
    if identity['physics']['dt'] != .005:
        raise ValueError('Measured physics timestep differs from frozen cadence')
    if identity['replan_interval_ticks'] != config['replan_interval_ticks'] or identity['episode_seconds'] != config['episode_seconds']:
        raise ValueError('Measured control/task cadence differs from frozen case')
    # Exactly four leaves are normalized, after checking their actual numeric
    # values against this case. Every other leaf remains in the comparison.
    value = copy.deepcopy(identity)
    for axis, label in [(0, '<FROZEN_CASE_BOX_X>'), (1, '<FROZEN_CASE_BOX_Y>')]:
        value['scene']['brown_box']['init_state']['pos'][axis] = label
        value['scene_config']['box_pose']['position_xyz'][axis] = label
    return value


def validate_position_interfaces(rows, plan):
    expected = {job['case_id']: job for job in plan['jobs']}
    by_cluster = {}
    common = None
    common_sha = None
    inputs = {}
    for row in rows:
        job = expected[row['case_id']]
        config = json.loads(Path(job['config_path']).read_text())
        path = Path(row['result_path']).with_name('interface.json')
        identity = json.loads(path.read_text())
        invariant = invariant_identity(identity, config, row['frozen_interface_sha256'])
        if common is None:
            common, common_sha = invariant, digest_identity(invariant)
        elif invariant != common:
            raise ValueError('Measured non-position interface changed across cases')
        cluster = row['cluster_id']
        by_cluster.setdefault(cluster, set()).add(row['frozen_interface_sha256'])
        if len(by_cluster[cluster]) != 1:
            raise ValueError('Measured full interface changed within a fixed position cluster')
        with path.open('rb') as stream:
            inputs[str(path.absolute())] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'schema': 'lingbot_frozen_position_interface_verification_v1', 'passed': True,
            'common_nonposition_interface_sha256': common_sha,
            'full_interfaces_by_position_cluster': {key: next(iter(value)) for key, value in sorted(by_cluster.items())},
            'allowed_variation': ['scene.brown_box.init_state.pos[0]', 'scene.brown_box.init_state.pos[1]',
                                  'scene_config.box_pose.position_xyz[0]', 'scene_config.box_pose.position_xyz[1]'],
            'variation_values_checked_against_each_frozen_case': True,
            'all_other_measured_interface_leaves_exactly_equal': True,
            'full_original_per_case_hashes_retained': True,
            'physics_actions_robot_assets_cameras_and_scoring_not_modified': True,
            'inputs_sha256': inputs}
