"""Seal existing ground_drag_load_002 evidence. No simulation or controller execution."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import resource
import shutil
import subprocess
import time
import zipfile

START = time.time()
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
import numpy as np
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
def read(path):
    return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
def write(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
def now():
    return datetime.now(timezone.utc)

A = read(ROOT / 'acceptance.json')
I = read(ROOT / 'inputs/inputs.json')
ORDER = read(ROOT / 'inputs/order.json')
PKG = ROOT / 'sealed04/physics_v0_2_20261018_04'
assert sha(I['package']['path']) == I['package']['sha256']
core = {}
for name, identity in I['files'].items():
    source, isolated = sha(identity['path']), sha(PKG / name)
    assert source == isolated == identity['sha256'], name
    core[name] = {'expected_SHA256': identity['sha256'], 'source_SHA256': source,
                  'isolated_SHA256': isolated, 'unchanged': True}
archive = []
with zipfile.ZipFile(I['package']['path']) as z:
    for item in z.infolist():
        if item.is_dir():
            continue
        path = ROOT / 'sealed04' / item.filename
        digest = hashlib.sha256(z.read(item)).hexdigest()
        assert sha(path) == digest, item.filename
        archive.append({'member': item.filename, 'SHA256': digest, 'identical': True})
expected = {x['member'] for x in archive}
extras = [str(p.relative_to(ROOT / 'sealed04')) for p in (ROOT / 'sealed04').rglob('*')
          if p.is_file() and str(p.relative_to(ROOT / 'sealed04')) not in expected]
assert all('__pycache__' in p or p.endswith('.pyc') for p in extras), extras
write('final_input_immutability.json', {
    'status': 'PASS', 'checked_UTC': now().isoformat(), 'package': I['package'],
    'seven_core_files': core, 'archive_files_checked': len(archive),
    'archive_members': archive, 'isolated_extra_generated_cache_files': extras,
    'note': 'All archived members are byte-identical. Generated Python caches are not package members.'})

cases = []
for n in range(1, 6):
    p = ROOT / f'attempt_{n:03d}'
    s, a, declaration = (read(p / x) for x in ['summary.json', 'analysis.json', 'declaration.json'])
    with np.load(p / 'derived_trace.npz') as f:
        z = f['data']
    assert s['actual_steps'] == a['rows_checked'] == len(z)
    assert declaration['harness_SHA256'] == sha(ROOT / 'drag_pilot.py')
    assert declaration['frozen_scene_readback']['all_named_robot_fields_and_original_internal_equalities_exact']
    pull = a['phases']['pull_1_to3s']
    loaded = a['phases']['all_loaded']
    free = s['mode'] == 'FREE_BASE_BACKDRAG'
    label = ('FREE_BASE_FALL_AIRBORNE_DISTANCE_EXCLUDED' if free else
             'LOCAL_SHORT_NECK_PULL_WITH_5MS_CONTACT_INTERRUPTS' if s['mass_kg'] <= 1 else
             'LOCAL_LOADED_MOTION_UNDER50MM_WITH_5MS_CONTACT_INTERRUPTS')
    c = {
        'attempt': n, 'mode': s['mode'], 'mass_kg': s['mass_kg'],
        'classification': label, 'sim_duration_s': s['sim_duration_s'],
        'physics_steps': s['actual_steps'], 'command_updates': s['actual_command_updates'],
        'bilateral_loaded_contact_duty': s['bilateral_loaded_duty'],
        'case_center_backward_from_cold_m': float(z[-1, 1]),
        'case_center_backward_pull_entry_to_end_m': a['case_center_net_backward_from_pull_entry_m'],
        'case_center_backward_1_to3s_m': pull['case_center_backward_net_m'],
        'case_return_forward_after3s_m': a['case_net_return_forward_after3s_m'],
        'handle_origin_backward_from_cold_m': s['final_object_backwards_m'],
        'torso_max_backward_m': a['max_torso_backward_m'],
        'torso_final_backward_m': float(z[-1, 4]),
        'case_startup20mm_time_from_cold_s': next((float(row[0]) for row in z if row[1] >= .020), None),
        'ground_force_support_duty_pull_1_to3s': pull['ground_force_above1pct_mg_duty'],
        'ground_force_support_duty_all_loaded': loaded['ground_force_above1pct_mg_duty'],
        'longest_ground_support_loss_s': a['ground_contact_gate']['max_force_support_loss_s'],
        'ground_support_loss_intervals': len(a['ground_contact_gate']['force_loss_intervals']),
        'mean_ground_N_pull': pull['mean_ground_normal_N'],
        'mean_ground_N_all_loaded': loaded['mean_ground_normal_N'],
        'mean_mouth_backward_N_pull': pull['mean_mouth_backward_force_N'],
        'mean_mouth_downward_N_pull': pull['mean_downward_mouth_force_N'],
        'mean_mouth_downward_N_capture': a['phases']['capture']['mean_downward_mouth_force_N'],
        'torque_curve_clipping_duty18': a['curve_clipping_duty18'],
        'max_stance_contact_slip_m': a['max_per_stance_force_weighted_contact_slip_integral_m'],
        'max_beak_pin_coordinate_residual_m': a['max_internal_and_fixture_eq_coordinate_residuals']['beak_distal_pin'],
        'max_jaw_penetration_m': s['max_jaw_payload_penetration_m'],
        'max_object_floor_penetration_m': s['max_object_floor_penetration_m'],
        'min_projected_grasp_X_margin_m': s['min_grasp_margin_m'],
        'first_failure': s['first_failure'], 'runtime_error': s['runtime_error'],
        'whole_robot_0p5m_qualification': 'NOT_PASSED', 'continuous60s': 'NOT_EXECUTED',
        'free_fall_displacement_excluded': free,
        'data_directory': str(p), 'video': str(p / 'actual_continuous_1x_50fps.mp4'),
    }
    cases.append(c)
assert sum(x['physics_steps'] for x in cases) == 5548
assert sum(x['command_updates'] for x in cases) == 1387
write('final_classification.json', {
    'status': 'EXPLORATORY_EVIDENCE_ONLY_NO_WHOLE_ROBOT_QUALIFICATION',
    'authoritative_closeout_interpretation': True,
    'raw_summary_and_analysis_preserved': True,
    'window_definition': 'Completed states follow each5ms integration. Forces/contacts were solved from incoming states. Capture incoming t=[0,1), pull=[1,3), stop=[3,end). All loaded=[1,end). Force support threshold is1%mg; geometric and positive-contact gaps are also in each analysis.json.',
    'mean_only_FLAT_NOMINAL_note': 'Original meanN>=90%mg label is not continuous-contact qualification. No new universal hardware gate is inferred from5ms interruptions.',
    'case_vs_handle_note': 'Case is the box geometric center. Original summary.object_backwards is the body origin at its handle. Rotations make these distances different.',
    'case_reference_note': 'Cold includes capture; pull_entry_to_end includes the stop phase.1_to3s is active pulling only. Free-case distances during the later fall are excluded from achievements.',
    'cases': cases})

video_results = []
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
for c in cases:
    receipt = read(ROOT / f"attempt_{c['attempt']:03d}/video_receipt.json")
    video = Path(c['video'])
    assert sha(video) == receipt['video_SHA256']
    result = subprocess.run([ffmpeg, '-v', 'error', '-threads', '1', '-filter_threads', '1',
        '-i', str(video), '-map', '0:v:0', '-an', '-threads', '1', '-progress', 'pipe:1',
        '-nostats', '-f', 'null', '-'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and not result.stderr.strip(), (video, result.stderr)
    progress = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    frames = int(progress['frame']); duration = int(progress['out_time_us']) / 1e6
    assert frames == receipt['frames'] and abs(duration - receipt['duration_s']) < .021
    video_results.append({'attempt': c['attempt'], 'file': str(video), 'SHA256': sha(video),
        'decoded_frames': frames, 'decoded_duration_s': duration, 'expected_fps': 50,
        'decode_exit_code': result.returncode, 'error_output': result.stderr,
        'trajectory_SHA256': receipt['trajectory_SHA256'], 'physics_integrations': 0})
write('video_validation.json', {'status': 'PASS_ALL5_FULL_DECODE', 'threads': 1,
    'GPU': 0, 'videos': video_results, 'total_frames': sum(x['decoded_frames'] for x in video_results)})

snapshot = ROOT / 'source_snapshot'
snapshot.mkdir(exist_ok=True)
sources = {}
for name in ['drag_pilot.py', 'frozen_helpers.py', 'prepare_kinematics.py',
             'prepare_kinematics_flat_rejected.py', 'analyze_and_render.py', 'read_only_verify.py', 'closeout.py']:
    shutil.copy2(ROOT / name, snapshot / name)
    sources[name] = {'source_path': str(ROOT / name), 'SHA256': sha(ROOT / name),
                     'snapshot_path': str(snapshot / name)}
write('source_identity.json', {'source_files': sources, 'dynamics_harness_all5_declarations_match': True,
    'engine': 'native MuJoCo3.10.0', 'no_policy_inference_or_training': True,
    'executed_requests': str(ROOT / 'executed_requests.jsonl'),
    'original_adapter': I['files']['runtime_adapter.py'],
    'earlier_analysis_failure_preserved': str(ROOT / 'postprocess_first_failure'),
    'earlier_analysis_failure_note': 'TypeError abs(list), corrected to numpy array before successful analysis. Both calls contained zero physics integrations.'})

new_target = {'human_message': '你的拖拽重物太轻了 至少比自己体重要重吧，比如超市拖车啥的',
    'source': 'direct human message in Lab chat', 'robot_mass_kg': I['nominal_robot_mass_kg'],
    'next_benchmark_minimum_gross_mass_kg': 20,
    'separate_conditions': ['wheeled_cart', 'unwheeled_sliding_load'],
    'required_metrics': ['actual startup pull force', 'actual sustained pull force', 'real displacement and speed',
                         'wheel rolling/caster behavior', 'ground support', 'grasp retention', 'body/foot stability'],
    'current_order_not_extended_or_reset': True,
    'cart_CAD_request_in_Lab': 'Human immediately withdrew the misdirected request; design subagent interrupted. No cart design is part of002.'}
write('human_target_update.json', new_target)

rows = []
for c in cases:
    mode = '自由整机' if c['mode'].startswith('FREE') else '躯干固定、真实颈驱动'
    rows.append(f"| {c['attempt']} | {mode} | {c['mass_kg']:g} | {1000*c['case_center_backward_from_cold_m']:.2f} | {1000*c['case_center_backward_pull_entry_to_end_m']:.2f} | {1000*c['handle_origin_backward_from_cold_m']:.2f} | {100*c['ground_force_support_duty_pull_1_to3s']:.2f}% / {100*c['ground_force_support_duty_all_loaded']:.2f}% | {1000*c['longest_ground_support_loss_s']:.0f} |")
report = f'''# ground_drag_load_002 — 已结束，探索证据回传

04机体质量 **{I['nominal_robot_mass_kg']:.5f}kg**。本轮0.5–4kg只完成短行程摸底，**未通过自由整机拖拽资格，也未测得硬件最大负荷**。局部各质量均保住双侧嘴部接触；自由0.5kg在停止阶段后仰失稳，不能把跌倒/物体离地造成的位移计入拖拽成绩。

人类已把正式重载目标提高到超过机体自重，下一阶段从20kg级起，带轮车与无轮滑块分别验。Rotbots已接收并写入唯一主计划。002五次额度已用完；下一单由PM审核后定义，不延长或重置002。误发到Lab的低矮购物车设计请求已被人类撤回，子任务已叫停。

## 本轮实际结果

位移单位mm，正值向后。箱体冷态位移含0–1s夹持/沉降；“1s到末态”包含1–3s主动牵拉和其后的停止保持。把手位移是原始summary的object位移，不能当箱体位移。力/接触按incoming5ms状态记录，位姿按积分后状态记录。

| 次数 | 模式 | 物体kg | 箱体：冷态到末态 | 箱体：1s到末态 | 把手：冷态到末态 | 有效地面支撑：主动拉 / 全带载 | 最长失支撑ms |
|---|---|---:|---:|---:|---:|---|---:|
{chr(10).join(rows)}

第2次表中箱体417mm属于含跌倒的原始轨迹，**排除成绩**。躯干最远仅后移60.60mm，末态44.18mm；3.195–3.740s物体持续离地545ms，3.535s首次触发后仰超过50°，在3.74s结束。双侧嘴部仍保持接触；支撑脚每支撑段实际接触滑移最大8.77/8.43mm。短双支撑IK路径没有完成受载后退/停步控制验收，因此此失败不能单独归因为嘴部或驱动硬件不足。

局部0.5/1kg箱体净位移约54/53mm；2/4kg约47/39mm。局部失地面支撑均是反复出现的单个5ms步，积分后记录的最高离地量约24–35微米；与自由整机连续545ms离地不同。不能用包含停留段的平均支撑率掩盖主动拉阶段的间歇，也不能把平均法向力满足90%mg的原始`FLAT_NOMINAL`标签解释为连续平拖合格。全部5次的正式0.5m整机拖拽、60s耐久和重复可靠性均未通过或未执行。

## 硬件判断的边界

4kg主动拉平均嘴部向后力9.26N、向下力2.44N，平均地面法向36.36N；停止段向下力升至6.05N。闭合/沉降阶段约11N的下压力会改变实际摩擦负载，故不能只用物重乘摩擦系数预测此次受力。4kg停止后箱体前回1.075mm；把手/头部也在移动，二者不能混同。

全部局部0.5–4kg过程均没有触发实际扭矩曲线限幅。4kg末态嘴部实际TCP比名义终点向前约16.2mm，显示受载跟踪/姿态与接触运动需要拆开验证；现有数据不证明结构柔性、选型不足或全局载重上限。自由整机后期出现限幅也不能取代对失稳前状态的判断。优先由PM闭合自由基座受载姿态/停步前提，再用可解释的启动和持续牵引力决定是否修改头颈传力链、咬杆接口或轮车几何。保留鹅的仿生外观要求。

最大嘴部销约束单坐标残差：局部0.11599–0.17238mm，自由0.43280mm；最大嘴部穿透：局部0.437–0.650mm，自由1.131mm；最大物体地面穿透：局部0.210–0.240mm，自由0.501mm。它们是本轮实际求解量，不能当作实体材料变形或已标定精度。保守嘴部X投影余量始终为正且双侧带载接触100%，但该投影不等于三维力闭合认证。

## 物理、输入与控制身份

冻结04机器人、14个凸碰撞叶、质量惯量、轴、传动、执行器、材质与求解器未改。每场实际读回796项具名机器人/驱动字段与原3个内部约束；局部仅额外加入静止躯干软约束，头/颈用真实关节驱动。自由场没有根约束、根位姿写入、外力辅助或非嘴推物。50Hz请求、200Hz实际驱动和积分，连续限值，禁用峰值模式，22.2V/350W原封套。

原生MuJoCo3.10.0，5ms、implicitfast、elliptic、impratio1、Newton100、无NoSlip。实际接触读回嘴/物体μ0.65，脚/地μ0.65，物/地显式材质对μ0.3，condim3。摩擦和软接触参数是未实测假设。完整物体由把手、两臂、箱体和箱内配重组成；各档独立重算完整质量、重心与fullI。4kg配重等效密度6589kg/m³；20kg需要另建合理体积/轮组/重心场景，不能塞入旧70×130×70mm小箱。

控制是本地离线FK/IK生成的短请求路径，不是已验收的整机倒走控制。局部1–3s名义60mm颈部回拉，3–6s目标保持；自由仅短双支撑腿部请求并补偿颈高，无步态转移。冷态已在嘴边，仅检查接触承载，不代表自动搜寻/抓取全过程通过。固定躯干结果不能代表自由机体牵引能力。

## 证据与复现

- [最终分类]({ROOT}/final_classification.json)：定义、全部5次位移/受力/脚滑/约束/失败和资格边界。
- [原始逐步数据]({ROOT}/attempt_005/actual_substeps.jsonl.gz)：每次目录均保留全部5ms状态、实际接触6D力、驱动和异常；NPZ含真实t0和全部积分状态。
- [4kg局部原速视频]({ROOT}/attempt_005/actual_continuous_1x_50fps.mp4)；[自由整机失败原速视频]({ROOT}/attempt_002/actual_continuous_1x_50fps.mp4)。全部5条连续1x/50fps视频1387帧已完整解码，绑定轨迹与模型SHA。
- [只读复现说明]({ROOT}/REPRODUCE.md)；[只读复核收据]({ROOT}/verification_readonly.json)：5548步、10个JSON输出和10个NPZ数组独立重算一致；不重新求解接触或证明硬件能力。
- [输入不变核验]({ROOT}/final_input_immutability.json)；[执行源码身份]({ROOT}/source_identity.json)；[文件哈希清单]({ROOT}/results_manifest.json)。源ZIP与7项核心文件、隔离解包的全部{len(archive)}个归档文件不变；额外Python缓存单列。
- [动力学退出]({ROOT}/resource_exit_dynamics.json)、[处理/渲染退出]({ROOT}/resource_exit_postprocess.json)、[最终总退出]({ROOT}/resource_exit_closeout.json)。失败的第一次只读分析调用保存在postprocess_first_failure，修正abs(list)后成功复核，无新增动力学。

总计5次、5548个积分/实际驱动更新、1387次命令、27.74s仿真；GPU0、神经网络推理0、训练0。动力学于11:46:00UTC退出，后续只读处理/渲染于11:57:06UTC退出。接单11:29:31UTC，硬截止12:29:31UTC；总墙钟和实际资源见最终退出收据。未启动第6次、额外诊断或自动续跑。
'''
(ROOT / 'SUMMARY.md').write_text(report)

python = '/home/ethan/ProjectBackups/2026-10-03/Sai_Lab/goose_warp_replay_admission_001/environments/mjlab_cuda_py312/bin/python'
(ROOT / 'REPRODUCE.md').write_text(f'''# 只读复核与历史入口

下列命令仅重算并对比现有5次结果，默认只向stdout打印收据，不改原始证据，不调用mj_step，不启动训练，可在已关闭订单截止后运行。

```bash
env PYTHONPATH='' CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  {python} \\
  {ROOT}/read_only_verify.py \\
  {ROOT}
```

运行依赖该环境中的MuJoCo3.10.0和numpy，根目录内sealed04、5次MJB/JSONL/NPZ、kinematic_plan.json和分析源码须完整。若复制到新位置，将最后参数改为复制后的绝对路径。脚本只编译分析源码内mat/gaps/analyze三函数，禁止执行其顶层渲染/截止逻辑；numpy输出代理只比对不写盘。可显式加`--receipt /新路径/receipt.json`写独立复核收据。现有verification_readonly.json已证明5548行、10个JSON和10个NPZ数组全部一致。

只读核验不重新求解接触力，不支持据此宣称重复可靠性。原始每步解算接触力在actual_substeps.jsonl.gz；incoming_time_s对应求解时刻，completed_time_s位姿已积分5ms。

drag_pilot.py、executed_requests.jsonl、场景XML/MJB、cold_state.npz、commands50Hz.json和source_snapshot保存实际执行入口及全部请求。该动力学入口已明确停止，原订单5次上限用完；不要直接重启worker或在旧目录覆盖数据。新的动力学需要新订单/新输出目录/新资源和身份声明。这里没有复活旧训练或后台自动继续的入口。

summary.json与final_classification.json是本次收口解释；各attempt原始summary.json和analysis.json保留。原始mean-only FLAT_NOMINAL不是连续平拖通过。原始object位移测的是把手，箱体位移看derived_trace.npz第一位移列及final_classification。
''')

wt = Path(A['worktree'])
git_status = subprocess.run(['git', 'status', '--porcelain=v1', '-uno'], cwd=wt,
                             capture_output=True, text=True, check=True).stdout
head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=wt,
                      capture_output=True, text=True, check=True).stdout.strip()
known = [read(ROOT / 'resource_exit_dynamics.json')['pid'], read(ROOT / 'resource_exit_postprocess.json')['pid']]
active = [pid for pid in known if Path(f'/proc/{pid}').exists()]
assert not active, active
finished = now()
elapsed = finished.timestamp() - datetime.fromisoformat(A['received_at_UTC']).timestamp()
assert elapsed < 3600
r = resource.getrusage(resource.RUSAGE_SELF)
write('resource_exit_closeout.json', {
    'status': 'CLOSED_NO_AUTOMATIC_CONTINUATION', 'finished_UTC': finished.isoformat(),
    'acceptance_UTC': A['received_at_UTC'], 'absolute_deadline_UTC': A['absolute_deadline_UTC'],
    'wall_s_from_acceptance': elapsed, 'within3600s': True,
    'closeout_process_wall_s': time.time() - START, 'closeout_CPU_user_s': r.ru_utime,
    'closeout_CPU_system_s': r.ru_stime, 'closeout_maxRSS_KiB': r.ru_maxrss,
    'closeout_CPU_affinity': sorted(os.sched_getaffinity(0)), 'address_space_cap_bytes': 8 * 1024**3,
    'total_counts': {'dynamic_attempts': 5, 'physics_integrations': 5548, 'drive_updates': 5548,
                     'command_updates': 1387, 'sim_seconds': 27.74, 'NN': 0, 'GPU': 0, 'training': 0},
    'closeout_physics_integrations': 0, 'known_worker_PIDs': known, 'known_workers_still_present': active,
    'dynamics_peakRSS_KiB': read(ROOT / 'resource_exit_dynamics.json')['maxRSS_KiB'],
    'dynamics_max_observed_threads': 1,
    'per_phase_receipts': ['resource_exit_prepare.json', 'resource_exit_dynamics.json',
       'postprocess_first_failure/resource_exit_postprocess.json', 'resource_exit_postprocess.json',
       'verification_readonly.json'],
    'accounting_boundary': 'Acceptance wall covers all preparation, waits, failures, dynamics, rendering and reporting. Preparation retry CPU was not exhaustively accumulated; per-process receipts are not claimed as an exact total CPU time.',
    'lab_worktree': str(wt), 'HEAD': head, 'git_status_porcelain': git_status,
    'no_further_dynamic_attempts': True})
write('summary.json', {'experiment_id': ORDER['experiment_id'], 'status': 'LAB_RETURNED_CLOSED_PENDING_PM_REVIEW',
    'finished_UTC': finished.isoformat(), 'source_release': I['release_id'], 'package_SHA256': I['package']['sha256'],
    'order_SHA256': sha(ROOT / 'inputs/order.json'), 'robot_mass_kg': I['nominal_robot_mass_kg'],
    'formal_whole_robot_drag': 'NOT_QUALIFIED', 'maximum_hardware_load': 'NOT_DETERMINED',
    'observed_range_kg': [0.5, 4], 'all_local_grips_retained': True, 'free_base_pilot': 'FAILED_BODY_TILT',
    'actual_dynamic_attempts': 5, 'physics_integrations': 5548, 'NN': 0, 'GPU': 0, 'training': 0,
    'cases': cases, 'new_human_target': new_target,
    'unexecuted': ['whole_robot0.5m drag with stop and3s retention', '60s endurance', 'repeat reliability',
                   '20kg+ wheeled cart', '20kg+ sliding load', 'full autonomous acquisition'],
    'remaining_control_prerequisite': 'Qualified loaded free-base posture/backstep/stop control before attributing motor or mouth capacity.',
    'final_resources': str(ROOT / 'resource_exit_closeout.json'), 'automatic_continuation': False})

files = []
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and p.name != 'results_manifest.json' and '__pycache__' not in p.parts and p.suffix != '.pyc':
        files.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)})
write('results_manifest.json', {'schema': 'goose_lab_results_manifest_v1', 'root': str(ROOT),
    'generated_UTC': now().isoformat(), 'file_count': len(files), 'files': files,
    'excluded': ['results_manifest.json (self)', '**/__pycache__/**', '**/*.pyc'],
    'physics_integrations_during_seal': 0})
print(json.dumps({'status': 'CLOSED', 'files': len(files), 'manifest_SHA256': sha(ROOT / 'results_manifest.json'),
                  'wall_s_from_acceptance': elapsed, 'summary': str(ROOT / 'SUMMARY.md')}, ensure_ascii=False))
