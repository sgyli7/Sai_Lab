# Leviathan 003

模型与训练源位于 RobotDesign/Leviathan_001/vehicles/Leviathan_003。

从此游戏根目录运行 `./run-leviathan003.sh`。此目录是可导入资产和运行脚本，不是另一份游戏工程。
003 启动入口复用现有 03 极地地形与天空。原来的 001/F7 Sai 切换入口保留；本版 003 尚未接入 Sai 登车作业。

100 km/h 是平整硬地仿真速度。双刚体/32接触点运输近似；上装固定。
GLB 包含命名的 front/rear 和八个 bogie 组。源坐标 +X 前、+Y 左、+Z 上；runtime.gd 在 Godot 边界转换。
独立加载 assets/vehicle.xml 不会自动运行 Python 的接触控制；请使用设计包的 source/physics.py。

完整来源、简化边界和实测以设计包 README.md 和 reports/ 为准。
