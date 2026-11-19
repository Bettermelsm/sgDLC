# sample_output — demo 产物示例

以下为 `generate_demo_data.py` / `generate_demo_heatmap.py` 的代表性输出示例，
**全部为合成数据**（图内水印已标注），完整产物请本地运行两个脚本重新生成。

| 文件 | 内容 | 生成命令 |
|---|---|---|
| `DEMO_FigA_SPI_groups_day1.png` | Phase 3 SPI 组间箱线图（+个体散点；标记 = vs BLANK，Mann-Whitney U） | `python generate_demo_data.py` |
| `DEMO_FigN_TS_TNS_bars_day1.png` | 标准 S/NS 柱状图（每组一格，bracket + 组内配对检验符号；对标 PMC8103520 图型） | 同上 |
| `DEMO_FigF_heatmap_day1.png` | Phase 3 组间占用热图（jet 共享色标、虚线隔板、杯位圈；含 QC 闸门） | `python generate_demo_heatmap.py` |
