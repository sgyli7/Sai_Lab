# cart_static_screen_001 · STATIC_ONLY

归档已经执行的 Goose04 理货车接口静力筛查。零物理积分；归档时没有导入MuJoCo、没有重跑FK/LP、没有新动力学订单、接触重放或训练。仅此目录由D写入，未改索引/输入/模型，未提交。

- `executed_stdin_static_screen.py`：先前实际工具调用的完整复算源码，保留原相对模型路径；只输出stdout，本次没有执行此文件。
- `input_and_method.json`：原cwd/Python、冻结输入SHA、完整29维qpos与全部命名角度、TCP、12支持点及力平衡定义。既有Lab04输入逐文件一致性也已记录。
- `recorded_static_results.json`：已完成调用的7个静力场景stdout数值转录，以及上颈链单分量边界；不是新计算。

复算需另获主线授权。原入口要求cwd为input_and_method记录的Rotbots工作树；该文件也记录完整命令。若使用本Lab已冻结输入，只需显式改脚本唯一p赋值指向`../../inputs/physics_v0_2_20261018_04`（相对本目录），不得改变物理或执行器合同。该路径调整尚未执行验证，不标为已重跑。

车重9.21kg是本次审查当时实际首CAD沟通值，后续新CAD需按需求公式更新；该静力模型没有装入购物车或货物刚体。侧力、竖向舌载与Mx仍需实测与自由整机动态验证，不因被动接口而假定为零。
