# Cart collision003 · low handle420mm · CPU static only

实际产物在`../../assets/low_profile_stock_cart_02`，独立于旧002。18车叶、14鹅叶；8box/6解析cylinder/4凸mesh，188支持点/360凸包面、单块64。当前关键帧`low_handle_420_pitch10`将真实杆置于world(300,0,420)mm，车端真实pitch+10°，取已选完整29维鹅姿态；需要显式resetKeyframe0。

- `loaded_collision_stats.json`：实际加载计数、源04编译MCI/轴/驱动/闭环不变。
- `initial_mci_reconstruction.json`：源02完整122件＋明确预留惯量模型，与实际加载总质量/COM/完整I复核；没有代理密度替代。
- `functional_checks.json`：空腔/轮径与近嘴功能有限检查，36项省略接触包络及限制。
- `source_input_immutability.json`：02几何源与全部53个04输入哈希。
- `previous_002_immutability.json`：旧002已登记22件逐项字节不变。
- `cart_collision.png`、`goose_cart_collision.png`：实际引擎CPU白底渲染，作者亲自查看同场景完整图。
- `resource_receipt.json`：单进程、4CPU亲和、数值线程1、8GiB地址空间，2次forward，0积分/训练/GPU。

`build_collision.py`是本次实际执行程序；路径显式读取当前独立Lab树的冻结输入，并从Rotbots读已锁02源。只写自己A02/D003。静态构建需另开新的输出或保留本候选；不要重写已发布002。环境MuJoCo3.10（记录的mjlab_cuda_py312）、numpy/scipy/trimesh/Pillow，CPU OSMesa softpipe。原命令：

```sh
LD_LIBRARY_PATH=/home/ethan/ProjectBackups/2026-10-13/Sai_Lab/goose_picking_001/render_deps/osmesa/usr/lib/aarch64-linux-gnu \
/home/ethan/ProjectBackups/2026-10-03/Sai_Lab/goose_warp_replay_admission_001/environments/mjlab_cuda_py312/bin/python \
experiments/goose_v0_2/hardware_validation/analysis/cart_collision_003/build_collision.py
```

近嘴实际读回在资产`initial_state.json`：杆对上/下嘴近切，上嘴对细轴3.77013mm、叉臂≥8.639mm，属于保存FK的代理检查，非保持力、摩擦、连续扫掠或动态稳定验收。新02真实CAD另由CAD专家与G独审。轮阻/弹簧0占位及36项接触省略限制原样保留。不能声称20kg、整机0.5m/60s或转弯通过。
