# 证据复核与原始执行入口

结果目录：`/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/grip_100g_pilot_001`。活动引擎严格为MuJoCo3.10.0 CPU；旧3.15仅只读导出历史对象／身份，不是活动动力学环境。

## 可重复的只读复核

以下命令没有mj_step、没有NN或训练，重建每5ms的相对位姿、18轴驱动与限制、接触扳手并核对原始轨迹及哈希：

```bash
env PYTHONPATH='' CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/ethan/ProjectBackups/2026-10-03/Sai_Lab/goose_warp_replay_admission_001/environments/mjlab_cuda_py312/bin/python /home/ethan/ProjectBackups/2026-10-18/Sai_Lab/grip_100g_pilot_001/verify_results.py /home/ethan/ProjectBackups/2026-10-18/Sai_Lab/grip_100g_pilot_001
```

输出[verification.json](/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/grip_100g_pilot_001/verification.json)。可在本机事后复核，过程会刷新该只读复核的结果／资源时间字段，因此如需保持当前结果目录封存，请先复制结果目录再运行，并使复制目录的isolated_model_root.txt指向所附sealed04/physics_v0_2_20261018_04目录。

## 原始动力学复现资料

当时实际入口：

```bash
PYTHONPATH='' CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/ethan/ProjectBackups/2026-10-03/Sai_Lab/goose_warp_replay_admission_001/environments/mjlab_cuda_py312/bin/python /home/ethan/ProjectBackups/2026-10-18/Sai_Lab/grip_100g_pilot_001/pilot.py serve
```

当时按[executed_requests.jsonl](/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/grip_100g_pilot_001/executed_requests.jsonl)顺序显式送入4次attempt及stop，不自动批跑。各次declaration保存配置、场景／控制器／执行脚本SHA；cold_state.npz是真实t0出生姿态及速度，actual_scene.mjb是真正编译模型，actual_substeps.jsonl.gz是每5ms实际记录，actual_trajectory.npz含真实t0及每个完成步。

原实验脚本包含这次订单的绝对截止时间、已有次数与只建新目录保护，**本单已经停止，不能把原入口作为追加实验授权**。重新积分应由PM另开有界实验单、在新输出目录提供新订单／时间边界，并保留本轮输入及记录。当前目录可用于只读核查，不启动旧Teacher／长期目标或自动训练。

postprocess.py是这次预算内运行的CPU渲染程序；视频已保存并通过实际解码。轨迹重放只调用mj_forward，不积分。脚本本身保留原订单截止时间，未设置事后自动续跑入口。

## 一致性与已知标注修正

[输入不变性终审](/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/grip_100g_pilot_001/final_input_immutability.json)核对原ZIP、核心源文件和全部隔离归档文件；[metadata_corrections.json](/home/ethan/ProjectBackups/2026-10-18/Sai_Lab/grip_100g_pilot_001/metadata_corrections.json)说明5ms冷态参考标注及scene_declaration名义角度。后者均不改变真实初态与保持失败判定，原始证据保留。
