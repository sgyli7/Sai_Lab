# Cart collision 001 · R2 / no dynamics

用户询问购物车碰撞数量和外观。本次只交一份可实际加载、低复杂度的首版代理及实际代理图，不启动拖拽或训练。

实际结果见 `loaded_collision_stats.json`。购物车18叶（8 box、6 cylinder、4凸mesh），鹅14，合计32，地面另计。车上4凸mesh合计188支持点/360凸包面，单块上限64；14个解析体没有运行面数。`cart_collision.png`、`goose_cart_collision.png`来自实际 MuJoCo 加载代理的 CPU OSMesa 渲染，不是CAD图或AI图片。作者已亲自查看两图；白底，独立车与同景皆完整可见。

`source_input_immutability.json`核对R2来源和全部53个冻结04文件。`functional_checks.json`含27个装货内腔探点、18个嘴前接近探点、轮径/轮宽检查以及叉架遗漏包络。有限探点通过不是完整Hausdorff、连续扫掠或动力学资格。质量/完整惯量独立于碰撞体积，预留与运动件划分假设见资产包 `body_mass_map.json`。

## 复现（另行授权的静态构建）

在本 Sai_Lab 工作树根目录执行：

```sh
LD_LIBRARY_PATH=/home/ethan/ProjectBackups/2026-10-13/Sai_Lab/goose_picking_001/render_deps/osmesa/usr/lib/aarch64-linux-gnu \
/home/ethan/ProjectBackups/2026-10-03/Sai_Lab/goose_warp_replay_admission_001/environments/mjlab_cuda_py312/bin/python \
experiments/goose_v0_2/hardware_validation/analysis/cart_collision_001/build_collision.py
```

代码输入仍指向已锁Rotbots R2源；改动源后须另发候选，不覆盖本次身份。MuJoCo3.10、numpy/trimesh/scipy、Pillow；CPU软渲染，不用GPU。单进程最多4CPU亲和、数值库线程1、地址空间8GiB。主调用import OSMesa第一次因库路径缺失停止；加现有只读库路径后成功；随后仅修正显示背景/机位再成功。两个成功构建共4次forward，0次mj_step，无训练或策略推理。最终调用资源收据为 `resource_receipt.json`，失败/前图不作为新动力学运行。

待下一次有界实验确认：真实轮阻、转向/轴承/弹簧参数与自由整机夹持/支撑/倒退控制。此结果不能证明20kg、0.5m、60s、性能吞吐或跨引擎通过。

## 002近嘴修正

G审核发现001将整段钢轴用R7包络，缩小嘴上净空。原001全包存于资产history ZIP，002保留18角色、全部MCI/轴，局部shaft改真实R4端/局部R7 socket的低顶点凸壳。仅增加一次重新构建的2次forward与一次局部间距核查forward，共新增3次forward、仍0积分。新最小距3.770124547mm；旧1.073322869mm；G独立真实CAD钢轴为3.854477mm。凸锥接与真实台阶差异保留，未扩为扫掠或动力学结论。

省略接触清单原32条漏列4根axle，这次明确补全36条，总CAD质量1.722290933kg，所有质量一直在MCI中，没有删除。源R2质量元数据修正support rectangle x下界30→80，当前来源SHA be6945；源几何及质量分布逐字节不变的逆替证明由CAD专家提供，非碰撞几何改动。
