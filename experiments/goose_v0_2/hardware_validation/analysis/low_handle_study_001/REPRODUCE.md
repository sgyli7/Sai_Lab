# 原执行源与路径映射

`static_height_study.py`和`verify_selected_clearance.py`是实际执行的固定路径程序。它们不是已验证的portable CLI，归档后**不要就地运行**：前者从文件位置推导Rotbots根和输出目录，后者同样使用文件旁的已选接口；二者还读取固定Lab002路径。输入路径和当前输出见input_path_map.json，原文件哈希见results/source记录。

如需要有界复算，先由Rotbots另发实验单和全新的不存在输出目录；用以下方式复制源码并显式重映射根位置，不在任何旧输出目录执行：

1. 给新目录复制实际源码和selected_420_interface.json；保持原归档源字节。
2. 仅在新副本把`W=Path(__file__).resolve().parents[3]`替换为input_path_map中实际Rotbots根的显式Path；`O`仍使用新副本所在目录。固定Lab002输入路径保持原值或按映射指向完整同哈希副本。若要改R或context入口，同样只改新副本并记录差异。
3. 使用记录的MuJoCo3.10 Python运行新副本。内存模型会更改新设计的cart pose/mount fixture，但不保存或修改输入模型文件。程序从不调用mj_step、训练或GPU。
4. 先核对输出路径映射和所有输入哈希，再复算。未经新实验单不要自行重算144候选矩阵。当前交付只完成已有有限筛查。

Python: `/home/ethan/ProjectBackups/2026-10-03/Sai_Lab/goose_warp_replay_admission_001/environments/mjlab_cuda_py312/bin/python`。MuJoCo3.10、numpy、scipy；MUJOCO_GL=disable，CPU亲和最多4核，数值库线程1、地址空间8GiB。资源收据记录了精确可重建的forward/LP次数；优化器内部有限差分kinematics次数未记录，不补造。
