from pathlib import Path
import json,gzip,hashlib,datetime
import numpy as np
R=Path(__file__).parent;L=Path('/home/ethan/Projects/TempWorktree/Sai_Lab/goose_v02_hardware_validation/experiments/goose_v0_2/hardware_validation/analysis/cart_drag_probe_005');L.mkdir(exist_ok=True)
allrows=json.loads((R/'all_attempts_analysis.json').read_text());table=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for a in allrows:
 S=Path(a['directory']);rows=[json.loads(x) for x in gzip.open(S/'substeps.jsonl.gz','rt')];phaseforces={}
 for p in ['open','approach','close','pull','hold','release']:
  rr=[r for r in rows if r['phase']==p]
  if not rr:continue
  forces=[]
  for r in rr:
   F=np.zeros(3)
   for c in r['contacts']:
    if c['a']=='ground' and c['b'].startswith('cart_wheel_'):
     # Stored contact normal/tangent force; reconstruct frame from cart-wheel contact by identical incoming kinematics unavailable here.
     # Ground plane normal is world+Z, so retain contact-frame vector, do not invent world tangent signs.
     F+=np.array(c['force_contact'][:3])
   forces.append(F)
  phaseforces[p]={'ground_wheel_sum_contact_frame3_mean_N':np.mean(forces,axis=0).tolist(),'warning':'Contact frames were not logged; do not interpret tangential sum as world force or calibrated axle resistance.'}
 a['wheel_ground_contact_frame_force_diagnostic']=phaseforces
 # Exact pin residual separated from belt radians.
 a['closure_components']={rows[0]['equalities'][i]['name']:{'all_max_abs':max(abs(x) for r in rows for x in r['equalities'][i]['residual']),'hold_max_abs':max(abs(x) for r in rows if r['phase']=='hold' for x in r['equalities'][i]['residual'])} for i in range(3)}
 a['hold_conservative_x_margin_m']=min(.0392-.006-abs(r['rod_rel_head'][0]-.13750001972253326) for r in rows if r['phase']=='hold')
 (S/'analysis.json').write_text(json.dumps(a,indent=2));table.append({'attempt':a['attempt'],'steps':a['steps'],'variant':['3_pitch_fixed_orientation','5_axis_actual_pose_follow','5_axis_open_jaw_countercheck','4_axis_pitch_follow_roll_neutral','5_axis_plus_passive_pitch_spring'][a['attempt']-1],'pull_back_mm':a['phases']['pull']['cart_backward_delta_m']*1000,'hold_back_mm':a['phases']['hold']['cart_backward_delta_m']*1000,'hold_s':a['phases']['hold']['duration_s'],'hold_bilateral_duty':a['phases']['hold']['bilateral_duty'],'hold_slip_end_mm':None if a['phases']['hold']['slip_end_m'] is None else a['phases']['hold']['slip_end_m']*1000,'pull_mean_tow_x_N':a['phases']['pull']['tow_on_rod_mean_world_N'][0],'hold_margin_mm':a['hold_conservative_x_margin_m']*1000,'peak_power_W':a['max_positive_power_W'],'video':a['video']})
(R/'all_attempts_analysis.json').write_text(json.dumps(allrows,indent=2));summary={'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempts':5,'physics_steps':sum(a['steps'] for a in allrows),'scope':'local fixed torso only; no free root or leg retreat, no whole robot haul','table':table,'source_robot53_unchanged':True,'order7_inputs_unchanged':True,'training':0,'GPU_physics':0,'formal_hardware_acceptance':False,'primary_finding':'Real bilateral contact can retain rod during roughly37mm actual local pull and5.18s hold in declared passive-pitch-spring variant; residual11mm slip and dynamic contact/loop compliance preclude formal acceptance. Unbalanced draghead sag and Teacher orientation strongly confound mouth attribution.','remaining_dynamic_budget':0,'resource_evidence':{'dynamic_processes_concurrent':1,'cpu_affinity_cores':4,'BLAS_threads_env':1,'max_dynamic_RSS_KiB':max(a['summary']['maxRSS_KiB'] for a in allrows),'actual_dynamic_thread_count_sample':'NOT_SAMPLED; no strict observed-thread receipt claim','render':'OSMesa llvmpipe CPU; EGL setup failed before usable output','original_deadline_utc':'2026-10-18 13:51:04 UTC'}}
(L/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(L/'immutable_sources_verified.json').write_bytes((R/'immutable_sources_verified.json').read_bytes());(L/'all_attempts_analysis.json').write_bytes((R/'all_attempts_analysis.json').read_bytes())
lines=['# Goose嘴部Teacher/IK局部物理抓取005','', '## 结论','', '原嘴机构确实能通过纯接触夹住真实车杆。配平实验变体中，约37.3mm真实局部牵引后，双侧接触持续保持5.18s、未脱嘴；保持末相对捕获点仍滑11.2mm。不能据此宣称无滑、硬件验收通过、整机倒退、20kg运营能力或最大负载。','', '首先暴露的是自由拖头重力下落与Teacher姿态跟随耦合，不足以判定嘴摩擦材料或电机必须升级。原未配平车在张嘴时杆高由420mm下降到约314mm；单独弹簧配平减少初始车漂移并改善牵引，但接近后杆仍约316mm，不能宣称弹簧维持420mm。','', '## 真实尝试','', '|次|唯一主要变体|物理步|pull车后移mm|hold车后移mm|hold双侧占比|hold末滑移mm|pull均值杆X力N|峰正功率W|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for t in table:lines.append(f"|{t['attempt']}|{t['variant']}|{t['steps']}|{t['pull_back_mm']:.2f}|{t['hold_back_mm']:.2f}|{t['hold_bilateral_duty']:.0%}|{str(round(t['hold_slip_end_mm'],2)) if t['hold_slip_end_mm'] is not None else '未捕获'}|{t['pull_mean_tow_x_N']:.2f}|{t['peak_power_W']:.2f}|")
lines+=['','后移正号=世界-X。相位位移严格从相位进入前一物理步起算；不把张嘴/接近时车的漂移加成牵引。第3开嘴负对照pull仅0.60mm且无双侧捕获，证明总车位移不等于嘴拖车。第3只匹配第2未配平工况；配平负对照未做，5次额度已用完。第4为四轴pitch跟随、head_roll目标中立，无重力补偿、无增益修改。第5仅相对第2加入原pitch轴被动弹簧k=2Nm/rad、qref=.276930475rad；q0时弹簧+.55386095Nm、重力广义力-.55386095012Nm。未采购/未标定的机械配平假设，不是Goose嘴升级。','', '## 物理与数值边界','', '20kg自由实箱600×400×150mm，惯量(.3041667,.6375,.8666667)kg·m²，真台面181.5mm、初始中心cart+[.325,0,.2565]。车7.815342443kg，车货27.815342443kg；箱平均台面支撑196.20N，相对车最大位移不足0.4mm。无物体焊接、吸附、外力搬物或在线qpos改写。静止躯干fixture明确为局部台架，原root_z=.005000002806m对齐；头颈全部真驱动。','', '只在初始化写qpos。主动张嘴→IK跟踪实际杆→合嘴→请求约50mm缓慢颈短拉→保持5.18s→放开；第3负对照在无捕获判据5.54s停止。所有请求经原18轴runtime_adapter、200Hz物理/驱动、50Hz命令、slew、原扭矩-速度曲线、350W连续功率限幅，禁用峰值，beak_input_rotor真驱动和四杆约束原样。','', '机器人/地面/杆实际混合μ=.65，未降材料摩擦。四轮frictionloss=.125Nm对应名义8N等效阻力（R=.0625m），是显式集总轴阻力变体，不是已标定实车阻力；原始轴摩擦约束力没有单独存证，不能把8N说成逐步实测牵引。实际嘴→杆世界力逐步记录。','', '五场均831项原编译物理字段/原关节阻尼摩擦/网格/驱动/闭环等精确；53原文件及原订单7源哈希完全匹配。原XML未改，车弹簧仅位于raw第5场景。全场warning=0、驱动曲线和正功率上限复核通过。','', '数值限制必须保留：第5合嘴2.15s的beak_distal_pin最大单轴残差1.68mm，保持最大0.148mm；第5嘴杆动态最大穿透1.17mm、保持0.866mm。均使用原solver，不能把软接触仿真当无变形制造验证。其他场景逐项残差/穿透见JSON。第5pull/hold无嘴-轴/叉额外承载，保持双侧接触100%；第1/2保持末杆已接近/略越保守工作面安全边界，第5仍约26mm余量。','', '逐步日志中的qpos/相对姿态为积分后状态；接触力/接触距离/约束残差为该5ms步的求解值。记录真扭矩、实际输入转速、正功率、所有接触力/摩擦、真实箱/车/根姿态、闭环反力和警告。足滑/车箱支撑等离线由完整轨迹核验。单独50Hz原始请求未另存，逐200Hz实际slew目标已保存；可复现实验由每场fresh harness与场景、原适配器和cold_state确定。','', '## 下一步最小工作','', '暂不改嘴形或静默提高μ。先把被动pitch配平/限位与咬杆姿态做真实器件和静态力矩校准，再复做同Teacher抓取+配平开嘴负对照；以固定接触面余量、保持滑移和杆X力为嘴硬件改进判据。若校准后仍有明显滑移，再单因素评估工作面浅凹导向/可更换接触垫，保持轴线与原驱动不变。','', '用户游戏参考表达低头叼持、压低身体、用腿持续倒退。本轮只有局部嘴台架诊断，没有复现该整机动作。后续须具备受载平衡的全身Teacher/IK退步控制，原限制下自由根、无足/root绑定，先验证小幅退步和稳定传力；本轮不研发步态、不做PPO/网络训练。未执行R2、20N、free-root；局部稳定传力与数值质量尚不足以放行整机后退。','', '## 视频和原始证据','']
for t in table:lines.append(f"- 第{t['attempt']}原速连续视频：{t['video']}")
lines+=['',f'原始轨迹、每物理步日志、每场harness/cold_state/场景、CPU渲染代码：{R}','', '视频均640×480、20fps，13s试验包含端点为13.05s视频，第3为5.55s；全部逐帧解码通过，OSMesa llvmpipe纯CPU渲染。EGL初始化失败后转OSMesa，未产出可用EGL视频。','', '资源：单动态执行进程、4核亲和、BLAS线程环境=1、峰RSS约2.72GiB、0 GPU物理/0训练；未采样实际动态线程数，不能声称有完整≤4线程观测回执。准备与执行继承13:31:04起算/13:51:04硬截止，5次总11508步，已停止全部动态。','']
(L/'report.md').write_text('\n'.join(lines));(R/'final_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
manifest=[]
for S in [R,R/'attempt_002',R/'attempt_003',R/'attempt_004',R/'attempt_005']:
 for n in ['probe.py','render.py','cold_state.npz','actual_scene.mjb','trajectory.npz','substeps.jsonl.gz','summary.json','preflight.json','analysis.json','continuous_original_speed.mp4']:
  p=S/n;manifest.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
for p in sorted(L.iterdir()):
 if p.is_file():manifest.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
(L/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2));print('FINAL_EVIDENCE',L,summary['physics_steps'])
