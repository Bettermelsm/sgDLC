# sgDLC · 深度学习动物行为学分析 ｜ 三君科技（Sengene）

基于 DeepLabCut（DLC）姿态估计的动物行为学定量分析能力展示与方法论仓库：

```
视频 → DLC 姿态估计 → 人工 ROI 标定 → 行为学指标计算 → 统计检验 → 可视化
```

## 业务能力

- **DLC 环境部署与训练链路把关**：版本选型、GPU 环境搭建、标注→训练→评估→推理
  全链路校验，见 [`core/env_setup/`](core/env_setup/README.md)。
- **人工 ROI 标定**：竞技场边界、隔板、刺激杯位置三合一人工标定工具，
  内置自洽性检查与跨组几何一致性核查，见 [`core/roi_calibration/`](core/roi_calibration/)。
- **朝向核查闸门**：参考帧坐标系与 DLC 推理坐标系一致性检查，
  拦截"自动旋转"类静默错误，见 [`core/orientation_check/`](core/orientation_check/)。
- **三箱社交测试（3CST）指标管线**：分区停留、嗅探时长、社交偏好指数（SPI）、
  穿梭次数、运动量，批量计算 + 质量红旗汇总，
  见 [`3cst/`](3cst/README.md)。
- **统计与图表（对标文献方法）**：Shapiro-Wilk 正态性前置 → Two-way ANOVA
  （stimulus × group 交互）→ Kruskal-Wallis / Mann-Whitney → 标准柱状图
  双重显著性符号系统，见 [`docs/method_benchmark_PMC8103520.md`](docs/method_benchmark_PMC8103520.md)。

## 仓库结构

```
sgDLC/
├── core/                        # 通用基础设施
│   ├── env_setup/               # DLC 环境部署要点（版本选择/GPU/网络受限环境）
│   ├── roi_calibration/         # 人工 ROI 标定工具（pick_roi.py）
│   └── orientation_check/       # 朝向核查闸门演示（rotcheck_demo.py）
├── 3cst/                        # 三箱社交实验模块
│   ├── README.md                # 方法说明：协议、SPI 定义、统计方法
│   ├── analysis_pipeline/       # 指标批量计算（输入为 DLC 推理产物）
│   └── demo/                    # 合成数据 demo：一键生成统计图表与占用热图
└── docs/                        # 对标文献方法说明
```

`oft/`、`grooming/` 等模块待相应业务产出后开放，本仓库不放空壳目录。

## 快速开始（demo，全合成数据）

```bash
pip install numpy pandas matplotlib scipy statsmodels

cd 3cst/demo
python generate_demo_data.py      # 生成 180 行合成数据长表 + 统计表 + 4 组图/天
python generate_demo_heatmap.py   # 基于同一份长表生成组间占用热图（jet 共享色标）
```

两个脚本的所有输入/输出路径都在文件**顶部单行变量**处，按需修改即可。
示例产出见 [`3cst/demo/sample_output/`](3cst/demo/sample_output/)。

## 数据声明

**本仓库所有数据与图均为合成演示数据（synthetic / demo）。**
每个输出文件名、图内标题与水印均显式标注 `DEMO`，不代表任何真实实验结果；
本仓库不含任何真实实验数据（轨迹文件、指标长表、ROI 坐标等一律不入库）。

## License

本仓库采用 [CC BY 4.0](LICENSE) 协议开源，转载/修改/商用请注明来源：三君科技 sgDLC。
