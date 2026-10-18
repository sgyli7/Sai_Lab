"""Seal this bounded experiment's evidence. No dynamics or training."""
from pathlib import Path
import os, sys, json, time, datetime, hashlib, zipfile, subprocess, resource, math
ROOT=Path(__file__).resolve().parent
START=time.time()
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
import imageio_ffmpeg
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc)
def link(label,p):return f'[{label}]({p})'
A=read(ROOT/'acceptance.json');I=read(ROOT/'input_identity.json');PRE=read(ROOT/'preflight_audit.json');VERIFY=read(ROOT/'verification.json')
assert VERIFY['status']=='PASS_RECORDED_DATA_CONSISTENCY_ONLY'
PKG=Path((ROOT/'isolated_model_root.txt').read_text().strip())
zip_path=Path('/home/ethan/.codex/worktrees/goose-delivery-navigation/Sai_Rotbots/artifacts/goose_v0_2/physics_freeze_20261018_01/integration_release/packages/physics_v0_2_20261018_04_native_reference.zip')
assert sha(zip_path)==I['package_SHA256']
keycheck=[]
for row in I['key_files']:
    source=Path(row['path']);local=PKG/row['name'];ok=sha(source)==sha(local)==row['sha256'];assert ok
    keycheck.append({**row,'source_and_isolated_unchanged':ok})
entries=[]
with zipfile.ZipFile(zip_path) as z:
    for info in z.infolist():
        if info.is_dir():continue
        rel=Path(info.filename)
        if rel.parts[0]==PKG.name:rel=Path(*rel.parts[1:])
        p=PKG/rel;expected=hashlib.sha256(z.read(info)).hexdigest();assert p.is_file(),str(p);assert sha(p)==expected,str(p)
        entries.append({'archive_entry':info.filename,'isolated_relative_path':str(rel),'SHA256':expected})
sourcecheck={'ZIP_path':str(zip_path),'ZIP_SHA256':sha(zip_path),'ZIP_unchanged':True,'key_source_files':keycheck,'all_archive_members_exact_count':len(entries),'archive_members':entries,'generated_local_python_cache_files':len(list(PKG.rglob('*.pyc'))),'cache_note':'Python import caches in the isolated copy are extra generated files, not source edits; no archived file bytes changed.'}
write(ROOT/'final_input_immutability.json',sourcecheck)
wt=Path(A['worktree']);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=wt,text=True).strip();status=subprocess.check_output(['git','status','--porcelain'],cwd=wt,text=True)
assert head==A['actual_source_HEAD'] and status==''
serve=read(ROOT/'resource_exit_serve.json');post=read(ROOT/'resource_exit_postprocess.json')
assert not Path(f"/proc/{serve['pid']}").exists() and not Path(f"/proc/{post['pid']}").exists()
assert serve['counters']['physics_integrations']==1540 and serve['counters']['attempts_started']==4
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();video_checks=[];attempts=[]
classifications={1:'ACQUISITION_FAILED_AND_FELL',2:'BILATERAL_RETAINED_BUT_STATIC_HOLD_FAILED',3:'BILATERAL_HOLD_CRITERION_FAILED_OBSERVATION_CENSORED_NO_DROP',4:'EXPECTED_FREE_FALL_NEGATIVE_CONTROL'}
for n in range(1,5):
    p=ROOT/f'attempt_{n:03d}';s=read(p/'summary.json');e=read(p/'hardware_evidence.json');k=read(p/'kinematic_evidence.json');v=read(p/'video_receipt.json');corr=read(p/'true_cold_reference_correction.json')
    r=subprocess.run([ffmpeg,'-v','error','-threads','1','-filter_threads','1','-i',v['path'],'-map','0:v:0','-an','-threads','1','-progress','pipe:1','-nostats','-f','null','-'],capture_output=True,text=True,timeout=45,check=True)
    fields={}
    for line in r.stdout.splitlines():
        if '=' in line:a,b=line.split('=',1);fields[a]=b.strip()
    count=int(fields['frame']);secs=int(fields['out_time_us'])/1e6
    assert count==v['video_frames'] and abs(secs-v['video_duration_s'])<.0001
    video_checks.append({'attempt':n,'decoded_frames':count,'decoded_duration_s':secs,'expected_trajectory_duration_s':s['sim_duration_s'],'actual_video_SHA256':sha(Path(v['path'])),'decode_returncode':r.returncode,'CPU_threads_flag':1,'video_path':v['path']})
    if s['hold5s'] is not None:five='FAIL_FULL_5S_OBSERVED'
    elif s['hold_first_failure'] is not None:five='NOT_COMPLETED_CRITERION_ALREADY_FAILED'
    else:five='NOT_STARTED_NO_CAPTURE'
    attempts.append({'attempt':n,'classification':classifications[n],'is_negative_control':n==4,'admission':'PASS_COLD_CONTACT_ADMISSION','sim_duration_s':s['sim_duration_s'],'actual_steps':s['actual_steps'],'commands50Hz':s['actual_50Hz_commands'],'drive_updates200Hz':s['actual_200Hz_drive_updates'],'first_bilateral_completed_step_time_s':None if s['first_positive_bilateral_pose'] is None else s['first_positive_bilateral_pose']['time_s'],'hold_start_s':s['hold_start_s'],'hold5s_status':five,'hold5s':s['hold5s'],'hold_first_failure':s['hold_first_failure'],'physical_latch_original':s['first_physical_failure'],'hold30s':'NOT_EXECUTED','drop_observed':n in [1,4],'min_full_object_clearance_m':s['min_full_object_clearance_m'],'max_jaw_penetration_m':e['max_jaw_payload_penetration_m'],'max_ground_penetration_m':e['max_ground_payload_penetration_m'],'mechanism_max_coordinate_residual_m':s['mechanism_max_position_error_m_or_rad'],'selected_speed_excursion':k['largest_selected_speed_excursion'],'selected_speed_exceeded_substeps':s['selected_speed_exceeded_substeps'],'worst_soft_joint_limit_violation':k['worst_actual_joint_soft_limit_violation'],'positive_mechanical_power_max_w':s['max_total_positive_mechanical_power_after_w'],'true_cold_correction':corr,'summary_path':str(p/'summary.json'),'raw_path':str(p/'actual_substeps.jsonl.gz'),'trajectory_path':str(p/'actual_trajectory.npz'),'actual_scene_MJB':str(p/'actual_scene.mjb'),'video_path':v['path']})
write(ROOT/'video_validation.json',{'all_four_decoded_successfully':True,'all_duration_and_frame_counts_match_1x_receipts':True,'videos':video_checks})
metadata={'raw_originals_preserved':True,'issue_1':{'description':'Original relative_from_cold fields use first post-step t=5ms pose, not true t0 birth. Corrected values saved separately from actual t0 state.','affects_hold_reference_or_5s_failure':False,'attempt3_30mm_crossing_original_s':.490,'attempt3_true_t0_reference_crossing_s':.495},'issue_2':{'description':'scene_declaration.json has a nominal cold_jaw_angle_rad=.25 for every scene. Per-attempt declaration.json, summary.json and cold_state.npz contain the actual qstart.','actual_qstart_rad_by_attempt':[read(ROOT/f'attempt_{n:03d}/summary.json')['initial_qjaw_rad'] for n in range(1,5)],'physics_affected':False},'contact_time_alignment':'Recorded contact geometry and solved forces belong to incoming state t-5ms; stored completed qpos/relative pose belong to t. Both are retained and independently checked.'}
write(ROOT/'metadata_corrections.json',metadata)
old=read(ROOT/'historical_readback.json')
differences={'comparison_type':'HISTORICAL_ONLY_NOT_CAUSAL_AB','qualification_transfer':False,'old':{'robot_mass_kg':old['robot_body_mass_kg'],'collision_leaves':22,'jaw_geometry':'V3 paired crowned ribs,3 per side plus core/rear leaves','engine':'3.15.0','integrator_enum':4,'integrator_name':'mjINT_DISCRETE','impratio':1000.,'jaw_condim':6,'jaw_solimp':[.95,.99,.001,.5,2.],'jaw_friction':[.65,.01,.002],'jaw_solref':[.005,1.],'commands_Hz':50,'physics_Hz':200,'drive_timing':'digital prepare/held ctrl and bounds across4 substeps; actuator-specific interpretation remains in archived source','original_internal_equalities':old['equalities'],'NN':0},'sealed04_active':{'robot_mass_kg':PRE['base_robot_mass_kg'],'collision_leaves':14,'jaw_geometry':'one convex upper envelope plus one convex lower envelope, no oldribs imported','engine':'3.10.0','integrator_enum':3,'integrator_name':'mjINT_IMPLICITFAST','impratio':1.,'jaw_condim':3,'jaw_payload_effective_solimp':[.9,.95,.001,.5,2.],'jaw_friction':[.65,.01,.002],'configured_jaw_solref':[.005,1.],'refsafe_enabled':True,'refsafe_used_standard_timeconst_s':.01,'commands_Hz':50,'drive_Hz':200,'physics_Hz':200,'drive_timing':'target slew, logical PD, actual motor speed cap and350W power envelope recomputed every5ms','original_internal_equalities':['beak_distal_pin','left_yaw_input_rotor_belt_relation','right_yaw_input_rotor_belt_relation'],'autoreset_disabled':True,'NN':0},'multiccd_correction':'Both versions use disable-bit flags. NativeCCD/MULTICCD are enabled by default here; 04 disableflags65536 disables only AUTORESET. enableflags0 is NOT evidence of single-contact collision. Actual case2 has2 contact points per jaw.','enum_sources':{'3.10.0':'/home/ethan/ProjectBackups/2026-10-03/Sai_Lab/goose_warp_replay_admission_001/environments/mjlab_cuda_py312/lib/python3.12/site-packages/mujoco/include/mujoco/mjtype.h','3.15.0':'/home/ethan/ProjectBackups/2026-10-07/Sai_Lab/goose_native_discrete_admission_001/upstream315/mujoco/include/mujoco/mjtype.h'},'refsafe_primary_source':'https://raw.githubusercontent.com/google-deepmind/mujoco/3.10.0/src/engine/engine_core_constraint.c','unsupported_causal_claims':['ribs alone explain old success','larger motor required','mu=.65 is physically calibrated','selected speed is a measured absolute motor hard speed limit']}
write(ROOT/'old_new_audit.json',differences)
PY='/home/ethan/ProjectBackups/2026-10-03/Sai_Lab/goose_warp_replay_admission_001/environments/mjlab_cuda_py312/bin/python'
verify_command=f"env PYTHONPATH='' CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 {PY} {ROOT}/verify_results.py {ROOT}"
angle_deg=math.degrees(math.atan2(.07335,.1059))
now=utc();elapsed=now.timestamp()-datetime.datetime.fromisoformat(A['received_at_UTC']).timestamp()
assert elapsed<3600
summary={'schema':'goose_lab_validation_result_v1','experiment_id':'grip_100g_pilot_001','state':'RESULT_RETURNED_EXECUTION_STOPPED','main_objective':'通过物理证据优化硬件与夹持设计；本单不训练、不发布机体改型','result':'TESTED_LOCAL_STATIC_HOLD_NOT_PASSED_HARDWARE_ROOT_UNDETERMINED','accepted_UTC':A['received_at_UTC'],'finished_UTC':now.isoformat(),'finished_Asia_Shanghai':now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'absolute_deadline_UTC':A['absolute_deadline_UTC'],'actual_wall_from_acceptance_s':elapsed,'within3600s':True,'stop_reason':'Scoped frozen04 exploratory question answered;4 attempts sufficient; fifth slot intentionally unused. Return for Rotbots review.','input_release':'physics_v0_2_20261018_04','ZIP_SHA256':I['package_SHA256'],'source_HEAD':head,'Lab_worktree_clean':True,'result_package_path':str(ROOT),'report_path':str(ROOT/'SUMMARY.md'),'reproduction_document':str(ROOT/'REPRODUCE.md'),'read_only_verification_entry':verify_command,'original_dynamics_entry':A['reproducible_entry'],'original_request_stream':str(ROOT/'executed_requests.jsonl'),'original_harness_SHA256':A['harness_SHA256'],'controller_SHA256':I['key_files'][3]['sha256'],'engine':'MuJoCo3.10.0 native CPU','historical_engine_usage':'3.15.0 read-only object/oldmodel identity only, zero integrations; not active experiment','scope':{'local_fixture':'two declared soft welds: torso and head_roll to world','payload':'full100g four-component free body; no attachment, adhesion, online pose writes or applied forces','qualifies_free_robot':False,'fixture_reactions_recorded_every5ms':True},'counts':{'all_dynamic_attempts':4,'normal_contact_attempts':3,'negative_control_attempts':1,'max_authorized_attempts':5,'fifth_attempt':'NOT_EXECUTED','physics_integrations':1540,'command50Hz_updates':385,'drive200Hz_updates':1540,'NN_forwards':0,'training_updates':0,'GPU':0,'simulated_seconds_total':7.7},'resources':{'dynamic_executor_processes':1,'dynamic_executor_PID':serve['pid'],'dynamic_executor_exited':True,'dynamics_max_observed_threads':serve['counters']['max_observed_threads'],'dynamics_CPU_affinity':serve['CPU_affinity'],'address_space_cap_GiB':8,'dynamics_maxRSS_GiB':serve['self_maxRSS_KiB']/1024**2,'dynamics_CPU_user_s':serve['CPU_user_s'],'dynamics_CPU_system_s':serve['CPU_system_s'],'CPU_video_postprocessing_after_executor_exit':True,'postprocess_maxRSS_GiB':post['self_maxRSS_KiB']/1024**2,'video_encoder_child_maxRSS_GiB':post['child_maxRSS_KiB']/1024**2,'receipt_scope_note':'Per-process receipts are retained. MaxRSS values are not a sampled simultaneous total. Preparatory shell and initial historical-object read-only export CPU times are not exhaustively metered.','absolute_deadline_and_closeout_reserve_honored':True,'automatic_continuation':False},'pass':['sealed ZIP/key source and all archived isolated files unchanged','796 named robot/drive fields and original3 equalities exactly matched','full100g object COM/inertia reconstructed from4+3+3+90g primitives','actual200Hz drive and350W continuous envelope reconstructed exactly from1540 recorded incoming states','expected negative-control free fall with zero jaw-payload contacts','four videos decoded with exact frame counts and1x durations','recorded relative poses/wrenches and drive timing independent consistency check passed'],'failed':['attempt1 acquisition, then fall','attempt2 full5s static hold:11.6558mm and6.30281deg','attempt3 stability criterion exceeded before5s observation completion'],'unexecuted':['fifth attempt','all30s holds','formal repeat/statistical acceptance','free-standing robot grasp','100g full pickup/lift/place/release','backward locomotion or dragging','heavier than100g','original16s whole-body Teacher shaking replay or speed optimization','hardware modification/release','training'],'attempts':attempts,'metadata_corrections':str(ROOT/'metadata_corrections.json'),'independent_verification':str(ROOT/'verification.json'),'video_validation':str(ROOT/'video_validation.json'),'interpretation':{'direct_evidence':['bilateral contact does not meet stability thresholds','case2 slow slip occurs inside the sliding friction cone','case2 normal directions oppose at7..8deg and drift mainly along mouth X','complete object carries approximately0.07196Nm initial gravity moment about handle axis','old/new geometry, solver, integrator, body mass and drive differ together'],'not_excluded':['cold free-release acquisition transient and fixed-target closure choice','uncalibrated contact softness/material parameters and persistent tangential regularization','beak4bar constraint compliance and soft joint stops','real torque-speed and pad deformation calibration'],'not_supported':['hardware incapable of100g','motor saturated or too small','a particular rib/groove is proven necessary','original16s shake is fixed','whole-robot or hardware qualification']},'next_smallest_discriminating_test':{'status':'PROPOSED_ONLY_REQUIRES_NEW_PM_ORDER_NOT_EXECUTED','proposal':f'Keep04 geometry/material/solver/drive and the same full100g assembly; change only initial payload rotation about its12mm handle Y axis to approximately{angle_deg:.4f}deg so its COM starts below the handle. Repeat the near-contact birth with no support/attachment, then observe5s and log contact-relative velocity, force-cone utilization and constraint regularizer. This removes the initial gravity-moment/acquisition-swing factor; it does not itself calibrate contact softness or prove a hardware redesign.','hardware_decision_owner':'Sai_Rotbots PM; no geometry or torque change selected by this pilot'}}
write(ROOT/'summary.json',summary)
rows=[]
for a in attempts:
    h=a['hold5s'];htext='未开始' if a['hold_start_s'] is None else ('5秒失败：%.3f mm / %.3f°'%(h['max_relative_translation_m']*1000,h['max_relative_rotation_deg']) if h else '未完成5秒；已有2°越界')
    rows.append(f"| {a['attempt']} | {a['classification']} | {a['sim_duration_s']:.3f} | {a['actual_steps']} | {htext} | {'是' if a['drop_observed'] else '否'} |")
report=f'''# 04／100g 夹持小样本结果

目标是为硬件与夹持设计提供证据。本单**没有通过局部静态保持门槛**，没有证明硬件带不动100g，也没有选定卡扣、凹槽或更大电机。04原包保持不变；结果交回Sai_Rotbots PM审核。本单执行已停止。

接单：{A['received_at_Asia_Shanghai']}。收口：{summary['finished_Asia_Shanghai']}。含准备、分析、渲染与收口的墙钟约{elapsed/60:.2f}分钟，限额60分钟；未使用第五次尝试。动力学进程已于18:58:12退出，此后只有只读分析与轨迹渲染。GPU、NN推理、训练更新均为0。

## 通过、失败与未执行

| 次数 | 实际状态 | 仿真时长(s) | 物理步 | 5秒观察 | 掉落 |
|---|---|---:|---:|---|---|
{chr(10).join(rows)}

1. 第1次：嘴角度0.25rad、上侧初始间隙1mm、下侧约2.406mm。没有形成正力双侧夹持，完成步t=0.095s时偏离安全区，此后自由掉到地面。
2. 第2次：近相切出生，嘴角度0.1968252300rad，初始穿透约纳米量级。首个双侧正力记录步t=0.105s；目标达到0后的保持参考只设一次，t=0.215s。t=0.270s、保持仅0.055s时转角2.042°越界。完整观察5秒，到t=5.215s最大位移11.6558mm、转角6.30281°。始终未掉落、没有其他物体支撑，完整物体最低离地仍约396.6mm。因此是“咬住但不稳定”，不是稳定通过。
3. 第3次：夹点由x=204.950mm后移到190mm，其他冻结设置相同；相切出生所需嘴角度0.2565554738rad。t=0.435s、保持0.155s时转角2.057°越界；t=0.490s触发原安全偏离锁存，观察于1.100s停止，**没有完成5秒，也没有掉落**。原安全偏离参考有5ms标注问题，真t0参考首次30mm越界在0.495s，详见修正文件。
4. 第4次：排除嘴部与载荷接触的负对照，无双侧夹力，0.080s偏离安全区后自由落地，行为符合预期。它验证本场景载荷不靠嘴部隐式连接留在空中，不能替代完整驱动／硬件验证。

所有30秒保持、第五次尝试、正式重复统计、整机自由站立抓取、100g完整举起／放置／释放、倒退拖运、重于100g、旧16秒Teacher晃动重放及速度优化均**未执行**。不把不同初态的单次探索换算成成功率。

## 硬件判断所需证据

第2次的保持阶段，双侧法向力约16–22N；t=1s约19.850/20.847N，t=5.2s约15.904/16.891N。这不是实测夹爪额定能力，而是该冻结场景内点法向力的和。实际嘴部力矩约1.5–1.7Nm，未用满当时4.4Nm连续上限；“应换更大电机”没有得到本单支持。

完整100g对象由12mm直径把手、两根连接臂和盒体组成，质量分别4g、3g、3g、90g。盒体70×130×70mm；把手中段120mm，总长132mm。COM相对把手原点为(73.35,0,-105.9)mm，初态重力绕把手轴力矩约0.07196Nm。完整惯量由四个实体重新计算，与历史对象一致到浮点误差；没有只用一根100g杆替代原对象。其转至约{angle_deg:.2f}°时COM接近把手正下方，不能将这种摆正过程算成抗转保持。

第2次末端主要沿头部坐标X方向漂移约11.67mm，Y方向约0.002mm。上颚接触法向近竖直，下颚偏斜约7–8°。t=5.2s时四个点的滑动摩擦锥利用率约0.093–0.109，均远未饱和；相对切向速度约1.97–2.05mm/s。早期捕获阶段则出现摩擦锥饱和与大角度转动。抗转力矩、接触法向、捕获瞬态与后期慢滑应分别评估。

MuJoCo的软接触可在摩擦锥内部仍产生慢滑；几何的法向支撑、所需力矩和接触软化都需核查，不能仅凭力的大小判定材料或硬件根因。此处是与文档一致的诊断线索，**本单未做单因素求解器对照，尚无唯一因果结论**。[MuJoCo官方防滑说明](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip)

## 穿透、闭环残差与驱动

| 次数 | 嘴–对象最大穿透(mm) | 地面–对象最大穿透(mm) | 嘴部连接最大坐标残差(mm) | 实际驱动峰值速度 / 选定slew速度 | 超过选定速度步数 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(f"| {a['attempt']} | {a['max_jaw_penetration_m']*1000:.4f} | {a['max_ground_penetration_m']*1000:.4f} | {a['mechanism_max_coordinate_residual_m']*1000:.4f} | {a['selected_speed_excursion']['actual_drive_velocity_after_rad_s']:.6f} / 0.925 rad/s | {a['selected_speed_exceeded_substeps']} |" for a in attempts)}

第1次的13.489mm是掉落后的**地面撞击穿透**，不是嘴部穿透。内部残差的最坏分量均定位到`beak_distal_pin`；两侧髋yaw传动等式残差远小。连接残差与软接触穿透相对2mm门槛不可忽略，不等同于真实结构的制造间隙。

第1次嘴逻辑关节软限位最大越界1.802mrad；负对照输入转子最大越界5.678mrad，均发生在已记录的自由偏离之后。第2、3次未出现关节范围越界。四次MuJoCo警告均为0，但警告为0不代表刚性闭环误差为0。

驱动采用原始04适配器：50Hz动作、每个20ms内4次5ms目标限速／PD／电压降额转矩速度限制／350W正机械功率检查。逻辑嘴轴v11与真正施力的输入转子v12分开处理；髋yaw同样按名称映射实际输入转子。没有速度投影、在线物体写入或浮根补偿力。

实际输入转子的短时速度超过0.925rad/s选定目标slew值。该值在04合同中用于目标限速，不是物理速度投影；实际力矩按每步开始的真实电机速度与22.2V曲线重算。1540步重建误差为0，正机械功率最高{max(a['positive_mechanical_power_max_w'] for a in attempts):.6f}W，峰值模式关闭。不能把超出slew直接误报为合同被绕开，也不能据此认定真实硬件绝对转速边界已验证。

配置solref=(0.005,1)，但3.10的refsafe开启，正格式时间常数按max(0.005,2×0.005)=0.010s使用；存储的contact.solref仍是0.005。材料／压陷参数尚未实测标定。[MuJoCo3.10源码](https://raw.githubusercontent.com/google-deepmind/mujoco/3.10.0/src/engine/engine_core_constraint.c)

## 与旧Pickup的差异

| 项目 | 历史100g Teacher | 本单04 |
|---|---|---|
| 机器人质量 | 10.430691kg | 16.970204kg |
| 机器人凸碰撞叶 | 22；上下各3道冠状筋 | 14；嘴上下各一个凸包 |
| 实际动力学引擎 | 3.15.0 | 3.10.0 |
| 积分器 | mjINT_DISCRETE，枚举4 | mjINT_IMPLICITFAST，枚举3 |
| 嘴接触维数 | condim6，含额外点力矩 | condim3，点力矩为0 |
| impratio | 1000 | 1 |
| 接触solimp | .95/.99/... | 自动嘴–对象对为.9/.95/... |
| 摩擦系数 | .65/.01/.002 | 相同数值；condim3仅使用滑动维 |
| 命令／物理 | 50Hz／200Hz | 50Hz／200Hz |
| 驱动时序 | 历史prepare及四步held包 | 每5ms重算目标slew、PD与限制 |
| 内部等式 | 嘴部连接1个 | 嘴部及双侧髋yaw传动3个 |
| 本次资格 | 历史参考 | 仅局部台架 |

两版本的nativeCCD和MULTICCD在这些输入中均开启；04的disableflags=65536只关闭自动复位，enableflags=0不能解读为只找单点。旧引擎本次只用于只读MJB身份／对象导出，积分次数为0。旧成功不构成新04的通过证据，更不能把差异仅归因于筋条。详见{link('旧新审计',ROOT/'old_new_audit.json')}。

## 输入、台架与数据可信度

- ZIP SHA256：`{I['package_SHA256']}`。7个核心源文件及隔离副本仍逐字节相同；归档内全部{len(entries)}个文件亦与隔离副本一致。额外Python缓存是导入生成文件，未改归档内容。{link('输入终审',ROOT/'final_input_immutability.json')}。
- 04的23个非world刚体MCI逐体回读一致；14个凸体共547支持顶点、1038三角面。场景保留796项机器人／驱动字段及原3条等式完全一致。只增加自由对象及明示的躯干、head_roll到world软焊接台架；原robot root未被隐藏锁定。它不代表自由整机。
- 每5ms保存输入与完成状态、接触框架及6D力、穿透、真实驱动转矩／速度／限制／功率，以及台架约束行力和6D反力。台架反力重构误差约1e-13；其重量支撑必须计入诊断边界。
- 第2次近相切出生时初始穿透约1.4nm，未采用深重叠预紧。真实出生状态见各次`cold_state.npz`；没有给载荷焊接、粘附或额外支持。
- 原`relative_from_cold`字段误以5ms首个完成步为参考；逐次真t0修正另存。各`scene_declaration.json`还保留了名义0.25rad标注，实际出生角度由各次declaration、summary及cold_state核实。原始失败与原日志未改写。保持参考无重置，5秒失败结果不受两项标注错误影响。{link('标注修正',ROOT/'metadata_corrections.json')}。
- 独立只读复核覆盖1540步：状态连续性、所有18轴slew／PD／实际限制力矩、相对位姿和接触扳手重建误差均为0。它核验记录一致性，未重新生成动力学接触力。{link('复核记录',ROOT/'verification.json')}。

## 视频、复现与下一步

四次视频均为实际轨迹50fps抽样，原速连续、无插值／剪辑／时间拉伸，原5ms轨迹全部保留。时长分别0.700、5.220、1.100、0.680秒。画面持续标明局部固定头／躯干台架。50fps最后一帧覆盖最后20ms；原始最后完成步保存在NPZ中。CPU OSMesa软渲染在实验进程退出后串行执行，GPU为0。全部视频已实际解码核对帧数及持续时间。

{chr(10).join('- '+link('尝试'+str(a['attempt'])+'原速视频',Path(a['video_path'])) for a in attempts)}

{link('复现说明与只读入口',ROOT/'REPRODUCE.md')}；{link('结构化结果',ROOT/'summary.json')}；{link('全部视频校验',ROOT/'video_validation.json')}。

下一最小判别试验仅作为建议交给PM：保持04设置与完整对象，单独将初始对象绕把手轴旋转约{angle_deg:.2f}°，让COM开始就在把手正下方，对照近相切出生的捕获转动和后期慢滑，增加约束regularizer记录。它可去掉一个初态力矩因素，不能替代真实接触标定。**未执行，不延续本单。** 本单不预定凹槽、卡扣、材料或力矩调整。

![诊断曲线]({ROOT}/hardware_diagnostics.png)
'''
(ROOT/'SUMMARY.md').write_text(report)
repro=f'''# 证据复核与原始执行入口

结果目录：`{ROOT}`。活动引擎严格为MuJoCo3.10.0 CPU；旧3.15仅只读导出历史对象／身份，不是活动动力学环境。

## 可重复的只读复核

以下命令没有mj_step、没有NN或训练，重建每5ms的相对位姿、18轴驱动与限制、接触扳手并核对原始轨迹及哈希：

```bash
{verify_command}
```

输出{link('verification.json',ROOT/'verification.json')}。可在本机事后复核，过程会刷新该只读复核的结果／资源时间字段，因此如需保持当前结果目录封存，请先复制结果目录再运行，并使复制目录的isolated_model_root.txt指向所附sealed04/physics_v0_2_20261018_04目录。

## 原始动力学复现资料

当时实际入口：

```bash
{A['reproducible_entry']}
```

当时按{link('executed_requests.jsonl',ROOT/'executed_requests.jsonl')}顺序显式送入4次attempt及stop，不自动批跑。各次declaration保存配置、场景／控制器／执行脚本SHA；cold_state.npz是真实t0出生姿态及速度，actual_scene.mjb是真正编译模型，actual_substeps.jsonl.gz是每5ms实际记录，actual_trajectory.npz含真实t0及每个完成步。

原实验脚本包含这次订单的绝对截止时间、已有次数与只建新目录保护，**本单已经停止，不能把原入口作为追加实验授权**。重新积分应由PM另开有界实验单、在新输出目录提供新订单／时间边界，并保留本轮输入及记录。当前目录可用于只读核查，不启动旧Teacher／长期目标或自动训练。

postprocess.py是这次预算内运行的CPU渲染程序；视频已保存并通过实际解码。轨迹重放只调用mj_forward，不积分。脚本本身保留原订单截止时间，未设置事后自动续跑入口。

## 一致性与已知标注修正

{link('输入不变性终审',ROOT/'final_input_immutability.json')}核对原ZIP、核心源文件和全部隔离归档文件；{link('metadata_corrections.json',ROOT/'metadata_corrections.json')}说明5ms冷态参考标注及scene_declaration名义角度。后者均不改变真实初态与保持失败判定，原始证据保留。
'''
(ROOT/'REPRODUCE.md').write_text(repro)
A.update(status='completed_result_returned_execution_stopped',finished_UTC=now.isoformat(),actual_attempts=4,actual_integrations=1540,automatic_continuation=False,summary_path=str(ROOT/'summary.json'),stop_reason=summary['stop_reason'])
write(ROOT/'acceptance.json',A)
r=resource.getrusage(resource.RUSAGE_SELF);ch=resource.getrusage(resource.RUSAGE_CHILDREN)
write(ROOT/'resource_exit_closeout.json',{'pid':os.getpid(),'purpose':'input/hash audit, video decode, reports; no dynamics','finished_UTC':utc().isoformat(),'process_wall_s':time.time()-START,'CPU_affinity':sorted(os.sched_getaffinity(0)),'max_address_space_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],'self_maxRSS_KiB':r.ru_maxrss,'child_maxRSS_KiB':ch.ru_maxrss,'CPU_user_s':r.ru_utime,'CPU_system_s':r.ru_stime,'children_CPU_s':ch.ru_utime+ch.ru_stime,'physics_integrations':0,'GPU':0,'NN':0,'training':0,'ffmpeg_decode_threads':1})
manifest=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name=='results_manifest.json':continue
    manifest.append({'path':str(p),'relative_path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'SHA256':sha(p)})
write(ROOT/'results_manifest.json',{'purpose':'sealed result file inventory; self excluded, generated Python caches excluded','files':manifest,'count':len(manifest),'total_bytes':sum(x['bytes'] for x in manifest),'created_UTC':utc().isoformat()})
print(json.dumps({'state':summary['state'],'summary':str(ROOT/'summary.json'),'report':str(ROOT/'SUMMARY.md'),'result_package':str(ROOT),'files':len(manifest),'bytes':sum(x['bytes'] for x in manifest),'wall_s_from_acceptance':elapsed,'dynamics_exited':True,'attempts':4,'integrations':1540,'GPU':0,'training':0},ensure_ascii=False,indent=2))
