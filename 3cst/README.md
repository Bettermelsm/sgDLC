# 三箱社交测试（3CST）模块

三箱社交测试（Three-Chamber Social Test）是评估小鼠社交偏好的经典范式：
一只实验鼠在三连通箱内依次经历三个相位，最终以它在"陌生鼠"与"新颖物体"
两个刺激间的探索分配来量化社交偏好。

本模块包含：指标计算管线（输入为 DLC 推理产物）+ 一键合成 demo。

## 协议（3-phase S-NS）

| 相位 | 名称 | 两侧刺激 | 作用 |
|---|---|---|---|
| 1 | Habituation | 两个空杯 | 习惯化 + 活动量基线（阴性对照） |
| 2 | Pre-test | 两个相同物体 | 位置偏好对照（阴性对照） |
| 3 | Test | 一侧陌生鼠（S）、另一侧新颖物体（NS） | 正式测试，产出 SPI |

## 核心指标

| 指标 | 定义 | 说明 |
|---|---|---|
| 各区停留时间 | 身体中心落在左/中/右箱的累计秒数 | 分区探索基本面 |
| 嗅探时间 T_social / T_object | 鼻尖进入左右杯判定圈的累计秒数 | 嗅探是啮齿类社交探索的主要形式 |
| **SPI** | `(T_social − T_object) / (T_social + T_object)`，范围 −1..1 | **仅在 Phase 3 有意义**；Phase 1/2 同公式只输出几何量 `side_index`，SPI 字段留空并写明原因 |
| 穿梭次数 | 箱室间切换计数（带滞回） | 探索/运动对照 |
| 运动距离/速度 | 逐帧位移求和（剔除跳变） | 排除"运动能力受损"替代解释 |

## 方法学硬约束（实践踩坑固化）

1. **杯位必须独立于被测行为。** 从轨迹反推杯位（鼻尖密度峰）是循环论证：
   对回避社交的动物系统性失效，把结果推向"检出偏好"。杯位一律来自人工标定
   （`core/roi_calibration/pick_roi.py`）或图像检测。
2. **嗅探判定圈跨组一致。** 圈的大小直接决定嗅探时长，进而决定 SPI——
   实测同一条轨迹把半径放大一倍，SPI 可从 −0.54 变到 −0.30。标定工具自带
   跨组几何一致性核查。
3. **每个视频用它自己的 fps。** 同批视频 fps 常有多个取值，用常数换算会让
   所有时间类指标系统性偏掉。
4. **分析时间窗从数据里找。** 用 likelihood 滚动合格率检测入场时刻，避免把
   放鼠前的操作段算进指标。
5. **social_side 约定必须核实。** 搞反会让 SPI 整体翻符号且下游无感；
   脚本把所用约定原样写进输出。
6. **坐标空间先过闸门。** 参考帧被图像工具"自动旋转"会让 ROI 全盘错位且
   不报错，先跑 `core/orientation_check/rotcheck_demo.py` 的朝向核查。

## 统计管线（对标文献方法）

详见 [`docs/method_benchmark_PMC8103520.md`](../docs/method_benchmark_PMC8103520.md)：

```
Shapiro-Wilk 正态性前置
   ↓（决定参数/非参检验走向）
Two-way ANOVA（stimulus × group，看交互效应）
   ↓
Kruskal-Wallis（SPI 组间整体）→ Mann-Whitney U（各组和参照组两两比较）
   ↓
作图：SPI 组间箱线图（Fig.A）+ 标准 S/NS 柱状图（Fig.N，双重显著性
符号系统：*/**/*** 组内 S vs NS；组间比较只进表格不上图，防误读）
+ 三相位一致性核查（Fig.B）+ 运动量对照（Fig.D）
```

## 目录内容

```
3cst/
├── analysis_pipeline/
│   ├── compute_3cst_metrics.py   # 单视频指标计算（核心库，也可 CLI 单跑）
│   └── run_batch_metrics.py      # 批量：manifest + ROI 标定 + h5 → 指标长表 + QC
└── demo/
    ├── generate_demo_data.py     # 合成 180 行长表 + 全套统计表/图（开箱即跑）
    ├── generate_demo_heatmap.py  # 基于同一份长表生成组间占用热图（QC 闸门）
    └── sample_output/            # demo 产物示例
```

## demo 复现

```bash
pip install numpy pandas matplotlib scipy statsmodels

cd 3cst/demo
python generate_demo_data.py      # 输出 ./demo_output/（路径在脚本顶部一行改）
python generate_demo_heatmap.py   # 读取上一步长表，输出 ./demo_heatmap_output/
```

所有输入/输出路径都集中在脚本**顶部的变量块**，按自己的目录布局修改即可。

## 免责声明

`demo/` 与 `sample_output/` 中的全部数据与图均为**合成演示数据**，
文件名/图内标题/水印均显式标注 `DEMO`，不代表任何真实实验结果。
真实分析请使用你自己的 DLC 推理产物并遵守上述方法学约束。
