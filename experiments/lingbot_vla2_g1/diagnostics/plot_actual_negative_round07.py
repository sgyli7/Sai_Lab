"""Present the audited R7 failure without changing a score or running a model."""
from pathlib import Path
import csv
import hashlib
import json
import os

import numpy as np

OWN = Path(__file__).resolve().parents[3] / '.scratch/lingbot_vla2_g1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    episode = OWN / 'round07_development_plan/episodes/round07_fresh_full60'
    audit_path = OWN / 'round07_development_preparation/execution/case_01_independent_audit.json'
    result_path, trace = episode / 'result.json', episode / 'trajectory.jsonl'
    result, audit = [json.loads(p.read_text()) for p in (result_path, audit_path)]
    if (result.get('status') != 'completed' or result.get('synthetic') is True
            or audit.get('passed') is not True or audit.get('autonomous_policy_trial') is not True
            or len(audit['checks']) != 7 or not all(c['passed'] for c in audit['checks'].values())
            or audit['recomputed_acceptance']['success'] is not False
            or result['trace_sha256'] != sha(trace)):
        raise ValueError('An actual complete independently audited negative is required')
    outcome = audit['recomputed_acceptance']
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    if len(rows) != 3001 or [r['control_tick'] for r in rows] != list(range(3001)):
        raise ValueError('Complete original60s/50Hz trace required')
    output = OWN / 'diagnostics/R7_actual_failure_paths'
    if output.exists():
        raise FileExistsError('Preserve existing failure presentation')
    os.environ.setdefault('MPLCONFIGDIR', str(OWN / 'presentation_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    box = np.asarray([r['acceptance_truth_only']['brown_box']['pose'][:3] for r in rows])
    root = np.asarray([r['root_link_pose_w'][:3] for r in rows])
    target = np.asarray(rows[0]['acceptance_truth_only']['blue_sorting_bin']['pose'][:3])
    actions = np.asarray([r['applied_action32'] for r in rows[1:]])
    output.mkdir()
    csv_path = output / 'actual_path_and_commands.csv'
    with csv_path.open('x') as stream:
        writer = csv.writer(stream)
        writer.writerow(['control_tick', 'sim_seconds', 'box_x_m', 'box_y_m', 'robot_x_m', 'robot_y_m',
                         'forward_command_m_s', 'lateral_command_m_s', 'yaw_command_rad_s'])
        for i, row in enumerate(rows):
            writer.writerow([i, i/50, *box[i, :2], *root[i, :2], *([] if i == 0 else actions[i-1, 29:32])])
    fig = plt.figure(figsize=(12, 7), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.2])
    path_ax = fig.add_subplot(grid[:, 0])
    path_ax.plot(root[:, 0], root[:, 1], color='#9ca3af', lw=1, label='机器人位置')
    path_ax.plot(box[:, 0], box[:, 1], color='#d97706', lw=2, label='箱子位置')
    path_ax.scatter(*box[0, :2], color='#16a34a', s=55, label='箱子起点', zorder=5)
    path_ax.scatter(*target[:2], color='#2563eb', marker='*', s=170, label='目标筐位置', zorder=5)
    path_ax.set(xlabel='世界 X（米）', ylabel='世界 Y（米）', title='60 秒实际路径（物体记录原点）')
    path_ax.set_aspect('equal', adjustable='datalim')
    path_ax.grid(alpha=.2)
    path_ax.legend(fontsize=9)
    times = np.arange(50, 3001)/50
    smooth = lambda values: np.convolve(values, np.ones(50)/50, mode='valid')
    move_ax = fig.add_subplot(grid[0, 1])
    move_ax.plot(times, smooth(actions[:, 29]), label='向前指令', color='#2563eb')
    move_ax.plot(times, smooth(actions[:, 30]), label='侧向指令', color='#16a34a', alpha=.8)
    move_ax.set(ylabel='米／秒', title='实际行走指令（1 秒滑动平均）')
    move_ax.grid(alpha=.2)
    move_ax.legend(fontsize=9)
    yaw_ax = fig.add_subplot(grid[1, 1], sharex=move_ax)
    yaw_ax.plot(times, smooth(actions[:, 31]), color='#dc2626')
    yaw_ax.set(xlabel='仿真时间（秒）', ylabel='弧度／秒', title='实际转向指令（1 秒滑动平均）')
    yaw_ax.grid(alpha=.2)
    fig.suptitle(f'R7 自主测试：有效携物 {outcome["max_carried_box_displacement_m"]:.2f} 米，未完成任务', fontsize=17)
    fig.supxlabel('实际仿真失败记录｜抓取成功，无跌倒／掉箱｜正式 20 场尚未开始｜不作唯一根因判断', fontsize=10)
    for suffix in ('png', 'pdf'):
        fig.savefig(output / ('actual_negative.' + suffix), dpi=160)
    plt.close(fig)
    manifest = {'schema': 'lingbot_R7_actual_negative_path_presentation_v1', 'actual_negative_only': True,
                'formal20_results': False, 'model_or_simulator_run': False, 'raw_sources_modified': False,
                'success_claimed': False, 'goal_completed': False,
                'source_sha256': sha(__file__), 'raw_inputs_sha256': {str(p): sha(p) for p in (result_path, trace, audit_path)},
                'generated_files_sha256': {str(p): sha(p) for p in output.iterdir() if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'actual_negative_only': True}))


if __name__ == '__main__':
    main()
