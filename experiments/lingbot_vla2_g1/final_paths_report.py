"""Export twenty actual accepted-case paths, figures and a human-readable PDF.

This is read-only evidence presentation. It never runs a model or simulator and
does not construct a performance verdict from demonstrations or partial cases.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.pi05_g1_mobile.acceptance import rotation_xyzw

OWN = Path(__file__).resolve().parents[2]/'.scratch/lingbot_vla2_g1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def export(plan_path, report_path, output):
    plan_path, report_path, output = [Path(p).absolute() for p in (plan_path, report_path, output)]
    if not output.is_relative_to(OWN) or output.exists():
        raise FileExistsError('Fresh owned final presentation directory required')
    plan, report = read(plan_path), read(report_path)
    if (report.get('synthetic') is True or report.get('status') != 'completed'
            or report.get('expected_cases') != 20 or report.get('completed_evidence_cases') != 20
            or report.get('goal_performance_passed') is not True
            or report.get('target_success_count') != 16 or report.get('success_count',0) < 16
            or report.get('errors') or report.get('actual_robot_success_claimed') is not False):
        raise ValueError('An actual accepted twenty-case report is required before final presentation')
    if (plan.get('schema') != 'lingbot_vla2_g1_pose_speed_benchmark_plan_v1'
            or plan.get('primary_policies') != ['lingbot_vla2_selected']
            or report.get('inputs_sha256', {}).get(str(plan_path)) != sha(plan_path)
            or len(plan.get('jobs', [])) != 20 or len(report.get('episodes', [])) != 20):
        raise ValueError('Actual accepted twenty-case plan/report binding required')
    episodes = {episode['case_id']: episode for episode in report['episodes']}
    if (len(episodes) != 20 or set(episodes) != {job['case_id'] for job in plan['jobs']}
            or sum(episode['success'] for episode in episodes.values()) != report['success_count']):
        raise ValueError('Actual case coverage or score differs')
    import matplotlib
    os.environ.setdefault('MPLCONFIGDIR',str(OWN/'presentation_cache'))
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Polygon
    from matplotlib import font_manager
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    chinese = font.exists()
    if chinese:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
        plt.rcParams['axes.unicode_minus'] = False
    output.mkdir(parents=True)
    saved, cases = {}, []
    report_pdf = output/'验收结果与20条路径.pdf'
    def retain(path):
        saved[str(path)] = sha(path)
    with PdfPages(report_pdf) as book:
        fig = plt.figure(figsize=(8.27, 11.69))
        title = 'LingBot-VLA 2.0 本地 G1 仿真验收' if chinese else 'LingBot-VLA 2.0 local G1 simulation acceptance'
        lines = [title, f"{report['success_count']} / 20", '门槛：20 场全部有效，至少 16 场完整成功。' if chinese else 'Target: all 20 valid, at least 16 full successes.',
                 '同一 NVIDIA GB10；Unitree G1 / Homie v2。' if chinese else 'Same NVIDIA GB10; Unitree G1 / Homie v2.',
                 '5 米任务；携物至少 4 米；转弯至少 60 度。' if chinese else '5 m task; >=4 m contact-qualified carry; >=60 degree turn.',
                 '箱体完整入槽、支撑、释放分离并稳定 2 秒；无跌倒或掉箱。' if chinese else 'Whole box in bin, supported, separated and stable 2 s; no fall/drop.',
                 '5 个物理位置，每个位置重复 4 个种子。' if chinese else '5 physical positions with 4 seeds each.',
                 '推理时仿真暂停；仿真时间与实际耗时分别记录。' if chinese else 'Simulation pauses for inference; simulation and wall time are distinct.',
                 '本报告是仿真验收，不代表实机或墙钟实时运行验证。' if chinese else 'Simulation acceptance; physical-robot deployment and real-time wall execution unverified.',
                 '训练包含额外的 TRAIN 恢复示范；正式轨迹没有用于训练。' if chinese else 'Training includes additional TRAIN recovery demos; formal trajectories were not training data.']
        for index, text in enumerate(lines):
            fig.text(.08, .94-index*.055, text, fontsize=18 if index in (0,1) else 11, va='top')
        fig.text(.08,.26,'完整原始证据与复现身份见 manifest.json。' if chinese else 'See manifest.json for raw evidence and reproduction identities.',fontsize=11)
        book.savefig(fig); plt.close(fig)
        for index, job in enumerate(plan['jobs']):
            result_path = Path(job['result_path']); folder = result_path.parent
            config_path, trace = Path(job['config_path']), folder/'trajectory.jsonl'
            config, result = read(config_path), read(result_path)
            if (sha(config_path) != job['config_sha256']
                    or report['inputs_sha256'].get(str(result_path)) != sha(result_path)
                    or sha(trace) != result['trace_sha256']
                    or report['inputs_sha256'].get(str(trace)) != result['trace_sha256']
                    or result['success'] is not episodes[job['case_id']]['success']):
                raise ValueError('Actual path evidence changed: '+job['case_id'])
            geometry = read(config['geometry_definition'])
            if config['frozen_files'].get(config['geometry_definition']) != sha(config['geometry_definition']):
                raise ValueError('Original physical object geometry changed')
            kinds = {item['kind']:item for item in geometry['objects']}
            box = np.concatenate([part['points'] for part in kinds['t2_box']['convex_parts']])
            center = (box.min(0)+box.max(0))/2
            floor = np.asarray(kinds['t2_bin']['convex_parts'][2]['points'])
            path = output/(job['case_id']+'.csv')
            xs, ys, rx, ry = [], [], [], []
            count = 0; first = final = None
            fields = ['tick','sim_seconds','box_actor_x_m','box_actor_y_m','box_actor_z_m',
                      'box_center_x_m','box_center_y_m','box_center_z_m',
                      'robot_root_x_m','robot_root_y_m','robot_root_z_m','upright']
            with trace.open() as source, path.open('x',newline='') as target:
                writer=csv.writer(target);writer.writerow(fields)
                for line in source:
                    row=json.loads(line);pose=np.asarray(row['acceptance_truth_only']['brown_box']['pose'])
                    geo=center@rotation_xyzw(pose).T+pose[:3]
                    root=np.asarray(row['root_link_pose_w'][:3]);tick=row['control_tick']
                    if tick != count or row['physics_steps_since_reset'] != 4*tick:
                        raise ValueError('Path counters are not the actual 50/200 Hz stream')
                    writer.writerow([tick,row['sim_time_seconds'],*pose[:3],*geo,*root,row['upright']])
                    xs.append(geo[0]);ys.append(geo[1]);rx.append(root[0]);ry.append(root[1])
                    if first is None:first=row
                    final=row;count+=1
            if count != result['completed_control_ticks']+1:
                raise ValueError('Path length differs from finalized actual result')
            retain(path)
            target_pose=np.asarray(first['acceptance_truth_only']['blue_sorting_bin']['pose'])
            lo,hi=floor.min(0),floor.max(0)
            rectangle=np.array([[lo[0],lo[1],hi[2]],[hi[0],lo[1],hi[2]],
                                [hi[0],hi[1],hi[2]],[lo[0],hi[1],hi[2]]])
            rectangle=rectangle@rotation_xyzw(target_pose).T+target_pose[:3]
            fig,ax=plt.subplots(figsize=(6,8))
            ax.add_patch(Polygon(rectangle[:,:2],facecolor='#d6e9ff',edgecolor='#2365a4',label='目标槽' if chinese else 'Target bin'))
            ax.plot(rx,ry,color='#777777',lw=1,label='机器人根部' if chinese else 'Robot root')
            ax.plot(xs,ys,color='#d77b1f',lw=1.8,label='箱子几何中心' if chinese else 'Box geometric center')
            ax.scatter([xs[0],xs[-1]],[ys[0],ys[-1]],c=['#396c3b','#ac3030'],s=30,zorder=5)
            verdict=('成功' if result['success'] else '失败') if chinese else ('PASS' if result['success'] else 'FAIL')
            ax.set_title(f"{index+1:02d}  {job['case_id']}\n{verdict} | {result['acceptance']['task_sim_seconds']:.2f} s")
            ax.set_xlabel('世界 X (m)' if chinese else 'World X (m)');ax.set_ylabel('世界 Y (m)' if chinese else 'World Y (m)')
            ax.axis('equal');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
            png=output/(job['case_id']+'.png');pdf=png.with_suffix('.pdf')
            fig.savefig(png,dpi=150);fig.savefig(pdf);book.savefig(fig);plt.close(fig)
            retain(png);retain(pdf)
            cases.append({'case_id':job['case_id'],'success':result['success'],'csv':str(path),
                          'png':str(png),'pdf':str(pdf),'actual_path_rows':count,
                          'head_video':result['head_video_path'],'observer_video':result['observer_video_path'],
                          'result':str(result_path),'result_sha256':sha(result_path),'trace_sha256':result['trace_sha256']})
    retain(report_pdf)
    manifest={'status':'completed','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'plan':str(plan_path),'plan_sha256':sha(plan_path),'report':str(report_path),'report_sha256':sha(report_path),
              'actual_full_successes':report['success_count'],'expected_cases':20,'cases':cases,
              'generated_files_sha256':saved,'raw_sources_modified':False,'model_or_simulator_run':False,
              'source_tool_sha256':sha(__file__),'renderer_matplotlib_version':matplotlib.__version__,
              'simulation_pauses_for_inference':True,'actual_robot_success_claimed':False,
              'overall_goal_completion_requires_separate_owned_cleanup_and_permanent_archive':True}
    with (output/'manifest.json').open('x') as stream:json.dump(manifest,stream,indent=2,allow_nan=False)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','report','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=export(args.plan,args.report,args.output)
    print(json.dumps({'status':result['status'],'actual_cases':len(result['cases']),'generated_files':len(result['generated_files_sha256'])}))


if __name__=='__main__':main()
