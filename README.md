# Sai_Lab

Sai_Lab 汇集两个独立的 Sim2Sim 工程：

| 目录 | 工程 | 说明 |
| --- | --- | --- |
| [Godot_Sim2Sim](Godot_Sim2Sim/README.md) | MuJoCo ↔ Godot/Jolt | MicroDuck、轮滑版和 Sai Robot 的场景、控制与训练。 |
| [Unity_Sim2Sim](Unity_Sim2Sim/README.md) | MuJoCo ↔ Unity/团结引擎 | MicroDuck 的 Unity/团结引擎实现。 |

两个工程有各自的依赖、运行命令和测试。克隆后请先进入对应目录，再按其 README 操作。

```bash
git clone https://github.com/sgyli7/Sai_Lab.git
cd Sai_Lab/Godot_Sim2Sim  # 或 cd Sai_Lab/Unity_Sim2Sim
```

## 来源与许可

Godot 工程保留原 `Robot_Godot_Sim2Sim` 的提交历史、[LICENSE](Godot_Sim2Sim/LICENSE) 和 [NOTICE](Godot_Sim2Sim/NOTICE)。Unity 工程从 [MicroDuck-Unity-Sim2Sim](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim) 导入并保留其提交历史；其 [NOTICE](Unity_Sim2Sim/NOTICE) 说明 MicroDuck 3D 模型及衍生网格的非商业许可限制。使用或分发前请分别查看各目录的许可文件。

## 迁移期间的开发

已有任务如果还在旧的 `Robot_Godot_Sim2Sim` 本地检出目录工作，请先把自己的改动提交到独立分支，再同步新的 `main`。不要在有未提交改动时直接切换到迁移后的 `main`；文件路径已经整体移动，Git 可能需要人工处理冲突。新的 Godot 改动应放在 `Godot_Sim2Sim/`，Unity 改动应放在 `Unity_Sim2Sim/`。原 GitHub 仓库地址会重定向到 `Sai_Lab`，但建议将本地 `origin` 更新为新地址。
