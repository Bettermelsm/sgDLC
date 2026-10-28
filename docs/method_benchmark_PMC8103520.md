# 对标文献方法说明：三箱社交测试标准化协议

> 本文是本仓库 3CST 统计/图表实现的方法学依据说明，**不是原文献的翻译或替代**；
> 实验设计与统计口径请以原文为准。

## 文献

**Rein B, Yan Z.** *A Standardized Social Preference Protocol for Measuring Social
Deficits in Mouse Models of ASD.* Current Protocols in Neuroscience (2020).
[PMC8103520](https://pmc.ncbi.nlm.nih.gov/articles/PMC8103520/)

三阶段（habituation → social stimulus → non-social stimulus）三箱社交偏好
标准化协议。我们对标的是其**统计与作图规范**，核对方式是抓取原文
Methods/Statistics 段落逐条对照实现，不凭记忆。

## 四个对标要点 → 我们的实现

| # | 原文要求 | 我们的实现 | 代码位置 |
|---|---|---|---|
| 1 | **正态性前置检验**：先检验各变量分布，再决定走参数检验（t/ANOVA）还是非参检验 | 对每 组×天×变量（T_social / T_object / SPI）跑 Shapiro-Wilk，结果落表，并**自动驱动** Fig.N 组内检验的选择 | `3cst/demo/generate_demo_data.py` 第 1 步 → `DEMO_shapiro_normality_check.csv` |
| 2 | **Two-way ANOVA**：因子1 = 刺激类型（social/object，组内），因子2 = 组别（组间），重点看**交互效应**（不同组对 S vs NS 的响应差异） | statsmodels OLS（type II）实现；同时如实说明：TS/TNS 是同一只动物的两次测量（重复测量），严格应为 repeated-measures 设计，故另配组内配对检验（paired t / Wilcoxon）作为对照口径 | 同上第 2 步 → `DEMO_two_way_anova_results.csv` |
| 3 | **标准 S/NS 柱状图**（原文核心图型，不是箱线图）：每组一格，两根柱（mean±SEM）+ 个体散点 + 配对连线 | Fig.N 实现，5 组并排、共享 y 轴 | 同上第 5 步 |
| 4 | **双重显著性符号系统**：`*`/`**`/`***` = 组内 S vs NS 配对比较；`#`/`##` = 组间比较（原文为 WT vs mutant） | 组内符号以**连接两柱的 bracket** 画在图上；组间比较（vs 参照组，Mann-Whitney U）**不画进 Fig.N**、只落 `DEMO_mannwhitney_vs_blank.csv` 与 Fig.A 标记——两根柱不在同一面板时，浮动符号容易被误读成组内结果 | 同上第 3/4/5 步 |

## 作图工程备注（多子图防重叠）

Fig.N 每格柱高不同，若显著性符号用"数据坐标倍数"定位，符号会撞上子图标题。
实现采用三件套：

1. bracket/符号位置钉在 **axes fraction 坐标**（`ax.annotate(..., xycoords="axes fraction")`），
   与数据范围解耦，保证每格与标题的间距一致；
2. `fig.subplots_adjust(top=...)` 预留 suptitle 空间；
3. `savefig(..., bbox_inches="tight")` 收边。

## 原文要点之外的两条实现约定

- **Phase 1/2 的同形比值不叫 SPI。** 两杯为空/相同时，左右嗅探之比是随机侧偏，
  数值范围与真 SPI 相同但语义无效——输出为 `side_index` 并作 Fig.B 三相位
  一致性核查（Phase 1/2 应贴近 0，仅 Phase 3 呈组间分离）。
- **所有 demo 输出显式标注 DEMO**（文件名/图内标题/水印三处），
  与真实结果不可混淆。

## 在本仓库的位置

- 统计+作图完整可跑版本：`3cst/demo/generate_demo_data.py`（合成数据）
- 效果示例：`3cst/demo/sample_output/`
- 管线方法背景：`3cst/README.md`
