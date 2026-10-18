"""Unchanged physical acceptance, with truthful privileged-teacher attribution."""
from contextlib import contextmanager

from experiments.pi05_g1_mobile import acceptance
from experiments.pi05_g1_mobile.acceptance_pose_speed import pose_speed_measurement


def attribute_teacher(physical, rows, config):
    if config.get('privileged_TRAIN_teacher', {}).get('TRAIN_only') is not True:
        raise ValueError('Never apply collector attribution to a student evaluation')
    ticks = len(rows) - 1
    return {**physical, 'policy_external_interventions': ticks,
            'intervention_rule': 'Every action is an explicitly scripted TRAIN teacher action; initial1462 are replayed TRAIN commands, subsequent placement uses actor/link truth. No autonomous model ability is claimed.',
            'autonomous_model_policy': False, 'counts_as_LingBot_success': False,
            'TRAIN_only': True,
            'privileged_truth_feedback_queries': len(range(1500, ticks, 50)),
            'recorded_TRAIN_action_prefix_controls': min(1462, ticks)}


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
