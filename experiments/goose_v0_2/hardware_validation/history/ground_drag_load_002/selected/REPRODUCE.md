# 只读复核与历史入口

下列命令仅重算并对比现有5次结果，默认只向stdout打印收据，不改原始证据，不调用mj_step，不启动训练，可在已关闭订单截止后运行。

```bash
env PYTHONPATH='' CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/ethan/ProjectBackups/2026-10-03/Sai_Lab/goose_warp_replay_admission_001/environments/mjlab_cuda_py312/bin/python \
  /home/ethan/ProjectBackups/2026-10-18/Sai_Lab/ground_drag_load_002/read_only_verify.py \
  /home/ethan/ProjectBackups/2026-10-18/Sai_Lab/ground_drag_load_002
```

运行依赖该环境中的MuJoCo3.10.0和numpy，根目录内sealed04、5次MJB/JSONL/NPZ、kinematic_plan.json和分析源码须完整。若复制到新位置，将最后参数改为复制后的绝对路径。脚本只编译分析源码内mat/gaps/analyze三函数，禁止执行其顶层渲染/截止逻辑；numpy输出代理只比对不写盘。可显式加`--receipt /新路径/receipt.json`写独立复核收据。现有verification_readonly.json已证明5548行、10个JSON和10个NPZ数组全部一致。

只读核验不重新求解接触力，不支持据此宣称重复可靠性。原始每步解算接触力在actual_substeps.jsonl.gz；incoming_time_s对应求解时刻，completed_time_s位姿已积分5ms。

drag_pilot.py、executed_requests.jsonl、场景XML/MJB、cold_state.npz、commands50Hz.json和source_snapshot保存实际执行入口及全部请求。该动力学入口已明确停止，原订单5次上限用完；不要直接重启worker或在旧目录覆盖数据。新的动力学需要新订单/新输出目录/新资源和身份声明。这里没有复活旧训练或后台自动继续的入口。

summary.json与final_classification.json是本次收口解释；各attempt原始summary.json和analysis.json保留。原始mean-only FLAT_NOMINAL不是连续平拖通过。原始object位移测的是把手，箱体位移看derived_trace.npz第一位移列及final_classification。
