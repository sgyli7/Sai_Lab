from pathlib import Path
import json,hashlib,collections
from sim2sim.research.queue import atomic_json
r=Path('results/sprint_gpu_match_20260912');d=Path('docs/sprint_gpu_match_20260912');d.mkdir(exist_ok=True)
comparison=json.loads((r/'native_comparison.json').read_text())
labels={'native_s05':'S05 基准','native_source_graph':'原脚底 GPU 训练','native_jolt_graph':'游戏脚底 GPU 训练','native_positive':'同代码正值奖励对照','native_cat':'CaT 约束学习'}
training={name:json.loads((r/name/'completed.json').read_text()) for name in ['train_source_graph','train_jolt_graph','train_positive','train_cat']}
rows=[];table=[]
for name,label in labels.items():
 item=comparison[name];s=item['completed'];long=next(x for x in s['comparisons'] if x['case']=='sprint_long')
 record=dict(name=name,label=label,passes=s['candidate_pass'],count=s['candidate_count'],falls=s['candidate_falls'],minimal_success=s['mini']['success'],long_speed_mps=long['candidate_vx'],ordinary_long_mps=long['ordinary_vx'],paired_speed_ratio=long['ratio'],lost_old_successes=len(item['lost_successes']),gained_old_failures=len(item['gained_successes']),ordinary_identical=item['ordinary_identical'],eligible=item['eligible'])
 rows.append(record);table.append(f"| {label} | {record['passes']}/128 | {record['long_speed_mps']:.4f} | {record['falls']} | {'通过' if record['minimal_success'] else '失败'} | {record['lost_old_successes']} |")
matched={}
for name in ['s05_source','s05_jolt','source_candidate','jolt_candidate','cat_candidate']:
 s=json.loads((r/('matched_start_'+name)/'completed.json').read_text());counts=collections.Counter()
 for x in s['results']:
  if x['ordinary']:continue
  for key in ['turns','stops','straight']:counts[key]+=sum(not y['passed'] for y in x['metrics'].get(key,[]))
 matched[name]=dict(passes=s['passes'],count=s['candidate_count'],falls=s['falls'],failed_segments=dict(counts),parity=s['parity_max_abs'])
result=dict(objective_complete=False,status='frozen development results; no promotion',native=rows,training=training,matched_starts=matched,dev_seeds=[925000,925015],final_seeds=[926000,926049],final_seeds_used=False,old_unused_final_seeds=[924000,924049],game_physics_changed=False,default_models_replaced=False,os_keyboard='not reverified, previous lockscreen acceptance outstanding',formal_samples=sum(x['samples'] for x in training.values()),formal_training_seconds=sum(x['elapsed_s'] for x in training.values()))
for filename in ['candidate_numerical_verification.json','candidate_freeze.json','cat_pair_audit.json','environment.json','handoff_audit.json']:
 if (r/filename).exists():result[filename.removesuffix('.json')]=json.loads((r/filename).read_text())
for filename in ['tests.json','defaults_and_physics.json','resources.json','prepared_cleanup.json']:
 if (r/'closeout'/filename).exists():result[filename.removesuffix('.json')]=json.loads((r/'closeout'/filename).read_text())
from sim2sim.research.budget import ActiveBudget
result['budget']=ActiveBudget(r).status()
atomic_json(r/'RESULT.json',result);atomic_json(d/'RESULT.json',result)
text='''# GPU 采样、训练域对齐与约束学习

本轮打通了 **CUDA 物理采样 + CUDA 策略更新**，此前约99%时间消耗在CPU/Jolt采样的训练路径没有继续追加。游戏仍运行冻结的Godot/Jolt与原生ORT。CaT候选在游戏中122/128、零跌倒，长直行0.2585m/s，比S05基准快约14.8%；同预算正值奖励对照只有48/128。旧8个失败和最小失败均修复，但新增6个失败，仍未达到晋级条件；默认模型与前轮试验包保持原样。

## 游戏中的固定终点对照

开发种子925000–925015，八种按键课程各16次，另有128条配对普通行走和1条已知最小失败。每个模型共257条独立原生回放，普通actor均为S05。以下“通过”同时要求方向、转向、停止、技能选择及无跌倒；最终晋级另要求每组加速平均至少为普通的115%且不丢失旧成功。

| 候选 | 完整通过 | 长直行 m/s | 跌倒 | 已知最小回放 | 丢失旧成功 |
|---|---:|---:|---:|---|---:|
'''+ '\n'.join(table)+'''

“丢失旧成功”比较同case/seed，包含最小回放；基准自身最小回放失败。普通128条指标逐项对照见机器记录。这里的S05是前轮试验基准，不是工作区默认walking的同义词。最终926000–926049及前轮924000–924049均未使用，旧920/922保留集也没有用于本轮调参。

![开发集速度与任务成功对照](native_comparison.png)

## 已定位的瓶颈

1. **运行分工**：旧路径只把PPO更新放在GPU，物理采样仍是Jolt/CPU。新512世界全GPU链已跑通。直接Warp逐步调度的瓶颈经对照定位到调度/分配开销；CUDA Graph微基准8.405→0.408秒，约20.58倍，缩小约束缓冲区无帮助。接触轨迹并非逐位一致，微基准不等同于全流水线加速比例。
2. **训练域资产**：旧GPU任务缺六个源碰撞体、两个shin过滤设置不同，脚底也不是游戏实际32点凸棱柱。新代理匹配游戏源资产、执行器公式和转子惯量近似，导入真实脚底，顶点误差小于1e-7m。仅换训练脚底，GPU基准长直行约0.1933→0.2240m/s，接近历史Jolt0.2256m/s；这不是严格同种子改善证明。90个短响应状态中关节速度误差改善而COM速度略恶化，不能称为物理完全对齐；更硬接触反而更差，已拒绝。
3. **训练目标取舍**：使用原生初态与相同按键，S05在精确脚底GPU代理中129/129，速度训练后66/129；退步在GPU内已经出现。奖励为速度增加提供的收益可以抵消转弯损失，游戏中转弯常降至约0.39–0.42rad/s。问题不只是GPU到Jolt迁移，更不是多跑几个小时就能自动修复。

## 约束学习对照

参考[CaT原论文](https://arxiv.org/html/2403.18765v1)，把约束违规映射为学习收益的存活概率，折扣当前奖励和后续回报；它不触发物理复位。工程采用现有验收的转向比例、路径偏差、朝向与停止速度作局部训练约束，物理与原生评分器未改。两组使用同代码、相同非负奖励、精确脚底、S05和相同4194304次转移，仅比较概率为零的对照与CaT。归一化EMA=.95，概率按预注册128轮从.05升至.25。

CaT在相同原生初态/按键的GPU诊断中恢复到129/129，零跌倒。游戏中剩余6个失败全部位于退出sprint、重新运行冻结普通S05的阶段：3个侧移越界、2个转向不足、1个停止失败；6个原始S05配对回放均通过。影子对照验证控制/推理契约。它支持“新加速状态超出普通S05可接住的状态分布”的假设，尚未证明具体机械原因，也未验证联合策略能全部解决。

这些局部约束与完整轨迹评分仍有差别，CaT不构成绝对安全保证。单元回归发现浮点累计会错过第50个决策的1秒边界，已改成整数决策计数；收益折扣、真实终止不自举和复位窗口均有针对性检查。研究来源、近期人形/梯度优先级工作及适用边界见[补充研究](constraint_research.md)。

## 实验与数值证据

四组正式终点各4194304次转移，总计16777216次；只测试固定终点，不挑checkpoint。全部训练无跌倒，完成导出最大误差均1.192e-6。正式训练总时间、采样/学习拆分、SHA、原生/Python影子对照与checkpoint检查见RESULT.json。

冻结Torch ONNX执行器保留原精度和Cast，未知算子拒绝；S05在606个随机/真实输入上GPU最大误差2.384e-6，原生700行控制命令对照误差0。外部公开高速donor误差9.155e-5未过1e-5门槛，已排除，未偷偷降精度或换模型。公开高速策略的旧Jolt跌倒证据仍见前轮报告。

第一组raw-dispatch训练在第16轮完整checkpoint主动中断，不计为合格候选；Graph重启使用新目录、同预算和种子。冒烟、续训、失败尝试、原始轨迹、冻结runtime与代码哈希均保留。异机复现需要本地模型和microduck_rl依赖，不能只clone后忽略这些输入。

## 收尾与剩余工作

代码和研究记录进入本地Git；未推送，未覆盖默认九模型，也未更换Jolt。保留最终种子，未把开发结果包装成最终验收通过。下一轮优先比较普通/加速/停止共享策略的CaT训练，配合来自普通参考轨迹的教师保留及原有95%速度/无回退检查；避免把修复新状态的动作也强行锁死为旧教师。先验证剩余6个交接失败及同初态GPU/Jolt差距，再考虑扩大速度范围。此方案尚未实施，不宣称已经解决交接问题。

本轮耗时、测试和进程/容器清理以RESULT.json及账本为准。删除的只是与canonical runtime逐文件核验相同的staging副本，原始轨迹、模型、checkpoint和失败尝试保留。宿主真实键盘仍未重新核定，完整用户目标未完成。
'''
(d/'RESULT.md').write_text(text)
print('Summary assembled:',[(x['name'],x['passes'],x['eligible']) for x in rows],flush=True)
