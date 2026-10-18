"""Original physical acceptance with declaration-derived collector attribution."""
from contextlib import contextmanager

from experiments.pi05_g1_mobile import acceptance
from experiments.pi05_g1_mobile.acceptance_pose_speed import pose_speed_measurement


def attribute_teacher(physical, rows, config):
    declaration = config.get('privileged_TRAIN_teacher', {})
    prefix = declaration.get('initial_replay_controls')
    if (declaration.get('TRAIN_only') is not True
            or declaration.get('collector_version') != 6
            or type(prefix) is not int or prefix != 1800):
        raise ValueError('Explicit v6 TRAIN collector and exact declared prefix required')
    ticks = len(rows) - 1
    first_truth_query = ((prefix + 49) // 50) * 50
    return {**physical, 'policy_external_interventions': ticks,
            'intervention_rule': f'Every action is explicitly scripted TRAIN collection; initial{prefix} applied controls replay the declared nonformal student prefix. Subsequent queries use actor/link truth. No autonomous model ability is claimed.',
            'autonomous_model_policy': False, 'counts_as_LingBot_success': False,
            'TRAIN_only': True,
            'privileged_truth_feedback_queries': len(range(first_truth_query, ticks, 50)),
            'recorded_TRAIN_action_prefix_controls': min(prefix, ticks)}


@contextmanager
def teacher_measurement():
    with pose_speed_measurement():
        original = acceptance.evaluate_episode
        def evaluate(rows, config):
            return attribute_teacher(original(rows, config), rows, config)
        acceptance.evaluate_episode = evaluate
        try:
            yield
        finally:
            acceptance.evaluate_episode = original


def main():
    from experiments.pi05_g1_mobile import simulator
    with teacher_measurement():
        simulator.main()


if __name__ == '__main__':
    main()
