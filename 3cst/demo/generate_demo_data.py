# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-4.0
"""
[DEMO - SYNTHETIC DATA] 三箱社交（3CST）分析 demo 数据与统计图表生成。

生成一套"趋势合理"的合成数据（180 行长表 = 5 组 x 6 只 x 2 天 x 3 相位），
并复现完整统计/图表管线：
  Shapiro-Wilk 正态性 -> Two-way ANOVA(group x stimulus) -> Kruskal-Wallis
  -> Mann-Whitney vs 对照组 -> SPI 组间箱线图 -> 标准 S/NS 柱状图（对标
  PMC8103520，双重显著性符号系统）-> 三相位一致性核查 -> 运动量对照。

所有数值均为模拟，不代表任何真实实验结果。每个输出文件名、图内标题、
水印均显式标注 DEMO。数据结构（样本量/分组/相位/统计方法）与真实管线
一致，仅数值趋势按预期方向设定，用于演示最终交付物的形态。

合成趋势链（组间分离度演示）：
  WT     正常偏好基线            SPI ~ 0.55-0.65
  BLANK  无偏好（组间检验参照组） SPI ~ 0.00-0.10
  NC     部分恢复                SPI ~ 0.20-0.30
  OV     强恢复（接近 WT）       SPI ~ 0.50-0.60
  SI     恢复消失（接近 BLANK）  SPI ~ 0.05-0.15
Phase 1/2（阴性对照）：各组围绕 0 波动，无组间偏置。

图内文字仅用英文，标题/显著性符号用黑色；跨面板比较显式标注为
"vs BLANK"，不在 S/NS 两柱之间画连接线（那是组内比较的位置），
避免两类比较被误读。

用法：
    python generate_demo_data.py
依赖：numpy pandas matplotlib scipy statsmodels
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ====== 用户只需改这一行 ======
OUTPUT_DIR = r"./demo_output"   # demo 产物输出目录（自动创建）
# ==============================

DEMO_TAG = "DEMO - SYNTHETIC DATA, FOR PRESENTATION ONLY"
TITLE_COLOR = "black"
WATERMARK_COLOR = "#555555"

GROUPS = ["WT", "BLANK", "NC", "OV", "SI"]
GLABEL = {"WT": "WT", "BLANK": "BLANK", "NC": "NC-MSC", "OV": "OV-SOD2", "SI": "SI-SOD2"}
COLOR = {"WT": "#4C72B0", "BLANK": "#DD8452", "NC": "#55A868", "OV": "#C44E52", "SI": "#8172B2"}
DAYS = ["DAY 1", "DAY 2"]
DAY_TAG = {"DAY 1": "day1", "DAY 2": "day2"}
N = 6

BASE_SPI = {
    "DAY 1": {"WT": 0.60, "BLANK": 0.05, "NC": 0.25, "OV": 0.55, "SI": 0.10},
    "DAY 2": {"WT": 0.58, "BLANK": 0.08, "NC": 0.28, "OV": 0.52, "SI": 0.12},
}
BASE_ENTRY = {"WT": 15, "BLANK": 13, "NC": 14, "OV": 16, "SI": 13}
BASE_DIST1 = {"WT": 21000, "BLANK": 20000, "NC": 20500, "OV": 21500, "SI": 20200}

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(2026)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

rows = []
for day in DAYS:
    for g in GROUPS:
        for m in range(1, N + 1):
            subj = f"{g}-{DAY_TAG[day].upper()}-M{m:02d}"
            si1 = float(np.clip(np.random.normal(0.0, 0.08), -1, 1))
            si2 = float(np.clip(np.random.normal(0.0, 0.08), -1, 1))
            dist1 = max(5000, np.random.normal(BASE_DIST1[g], 4000))
            entries = max(5, int(np.random.normal(BASE_ENTRY[g], 2.5)))
            spi3 = float(np.clip(np.random.normal(BASE_SPI[day][g], 0.16), -0.95, 0.95))
            total_sniff = max(25, np.random.normal(70, 20))
            t_social = total_sniff * (1 + spi3) / 2
            t_object = total_sniff - t_social
            for phase, phase_name, si, extra in [
                (1, "Habituation", si1, {"distance_px": round(dist1, 1)}),
                (2, "Pre-test", si2, {}),
                (3, "Test", spi3, {"T_social_s": round(t_social, 2), "T_object_s": round(t_object, 2),
                                    "entries_total": entries, "SPI": round(spi3, 4)}),
            ]:
                rec = {"day": day, "group_code": g, "subject_id": subj, "phase": phase,
                       "phase_name": phase_name, "side_index": round(si, 4)}
                rec.update(extra)
                rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(f"{OUTPUT_DIR}/DEMO_all_metrics_long.csv", index=False, encoding="utf-8-sig")
print(f"[{DEMO_TAG}] generated {len(df)} synthetic rows "
      f"(5 groups x 6 animals x 2 days x 3 phases)")


def stars(p):
    if pd.isna(p):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def hashes(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "##"
    if p < 0.05:
        return "#"
    return ""


all_shapiro_rows = []
all_anova_rows = []
all_mw_rows = []

for day in DAYS:
    tag = DAY_TAG[day]
    p1 = df[(df.day == day) & (df.phase == 1)]
    p2 = df[(df.day == day) & (df.phase == 2)]
    p3 = df[(df.day == day) & (df.phase == 3)].copy()

    # ---- 1. Shapiro-Wilk normality ----
    for g in GROUPS:
        sub = p3[p3.group_code == g]
        for col, label in (("T_social_s", "T_social"), ("T_object_s", "T_object"), ("SPI", "SPI")):
            vals = sub[col].dropna().values
            w, p = stats.shapiro(vals) if len(vals) >= 3 else (np.nan, np.nan)
            all_shapiro_rows.append({"day": day, "group": g, "variable": label, "n": len(vals),
                                      "shapiro_W": round(w, 4) if not pd.isna(w) else None,
                                      "shapiro_p": round(p, 4) if not pd.isna(p) else None,
                                      "normal_at_0.05": (not pd.isna(p)) and p > 0.05})

    # ---- 2. Two-way ANOVA (group x stimulus) ----
    long_rows = []
    for _, row in p3.iterrows():
        long_rows.append({"group": row.group_code, "stimulus": "social", "time_s": row.T_social_s})
        long_rows.append({"group": row.group_code, "stimulus": "object", "time_s": row.T_object_s})
    long_df = pd.DataFrame(long_rows)
    model = ols("time_s ~ C(group) + C(stimulus) + C(group):C(stimulus)", data=long_df).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    for term in ["C(group)", "C(stimulus)", "C(group):C(stimulus)"]:
        all_anova_rows.append({"day": day, "term": term.replace("C(", "").replace(")", ""),
                                "F": round(aov.loc[term, "F"], 4), "p": round(aov.loc[term, "PR(>F)"], 6)})
    print(f"\n[{DEMO_TAG}] [{day}] Two-way ANOVA:")
    print(aov.round(4))

    # ---- 3. SPI group comparisons: Kruskal-Wallis + Mann-Whitney vs BLANK ----
    h, p_kw = stats.kruskal(*[p3[p3.group_code == g]["SPI"].values for g in GROUPS])
    blank_spi = p3[p3.group_code == "BLANK"]["SPI"].values
    mw_results = {}
    for g in GROUPS:
        if g == "BLANK":
            mw_results[g] = {"p": np.nan, "sig": "-"}
            continue
        u, p = stats.mannwhitneyu(p3[p3.group_code == g]["SPI"].values, blank_spi, alternative="two-sided")
        mw_results[g] = {"p": p, "sig": stars(p)}
        all_mw_rows.append({"day": day, "group": g, "vs_BLANK_p": round(p, 5), "vs_BLANK_sig": stars(p)})
    print(f"[{DEMO_TAG}] [{day}] Kruskal-Wallis H={h:.3f} p={p_kw:.5f}")
    for g, r in mw_results.items():
        print(f"  {g} vs BLANK: p={r['p']} {r['sig']}" if g != "BLANK" else f"  {g}: reference group")

    # ---- 4. Fig A: SPI group comparison (box + scatter) ----
    fig, ax = plt.subplots(figsize=(8, 6.5))
    data = [p3[p3.group_code == g]["SPI"].values for g in GROUPS]
    bp = ax.boxplot(data, tick_labels=[GLABEL[g] for g in GROUPS], patch_artist=True, showmeans=True, widths=0.55)
    for patch, g in zip(bp["boxes"], GROUPS):
        patch.set_facecolor(COLOR[g]); patch.set_alpha(0.55)
    for i, g in enumerate(GROUPS):
        y = p3[p3.group_code == g]["SPI"].values
        x = np.random.RandomState(0).normal(i + 1, 0.05, size=len(y))
        ax.scatter(x, y, s=26, color=COLOR[g], edgecolor="k", linewidth=0.4, zorder=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    data_max = p3["SPI"].max()
    data_min = p3["SPI"].min()
    span = data_max - data_min
    ax.set_ylim(data_min - 0.12 * span, data_max + 0.30 * span)
    sig_y = data_max + 0.14 * span
    for i, g in enumerate(GROUPS):
        if g == "BLANK":
            continue
        ax.text(i + 1, sig_y, mw_results[g]["sig"], ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color="black")
    ax.set_ylabel("SPI = (T_social - T_object) / (T_social + T_object)")
    ax.set_title(f"[DEMO] Fig.A  {day}  Phase 3 SPI by group\n"
                 "(significance mark: vs BLANK, Mann-Whitney U)",
                 pad=14, color=TITLE_COLOR, fontsize=12)
    fig.text(0.5, 0.005, DEMO_TAG, ha="center", fontsize=9, color=WATERMARK_COLOR, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(f"{OUTPUT_DIR}/DEMO_FigA_SPI_groups_{tag}.png", dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    # ---- 5. Fig N: standard S/NS bar chart (benchmarked against PMC8103520) ----
    paired_sig = {}
    for g in GROUPS:
        sub = p3[p3.group_code == g]
        g_normal = all(r["normal_at_0.05"] for r in all_shapiro_rows
                        if r["day"] == day and r["group"] == g and r["variable"] in ("T_social", "T_object"))
        ts, tn = sub["T_social_s"].values, sub["T_object_s"].values
        if g_normal:
            _, p = stats.ttest_rel(ts, tn); test_name = "paired t-test"
        else:
            _, p = stats.wilcoxon(ts, tn); test_name = "Wilcoxon"
        paired_sig[g] = {"p": p, "sig": stars(p), "test": test_name}

    fig, axes = plt.subplots(1, 5, figsize=(19, 5.3), sharey=True)
    for ax, g in zip(axes, GROUPS):
        sub = p3[p3.group_code == g]
        ts, tn = sub["T_social_s"].values, sub["T_object_s"].values
        means = [np.mean(tn), np.mean(ts)]
        sems = [np.std(tn, ddof=1) / np.sqrt(len(tn)), np.std(ts, ddof=1) / np.sqrt(len(ts))]
        ax.bar([0, 1], means, yerr=sems, capsize=5, width=0.55,
               color=["#999999", COLOR[g]], alpha=0.85, edgecolor="k", linewidth=0.8)
        rng = np.random.RandomState(0)
        for i in range(len(ts)):
            jx = rng.uniform(-0.05, 0.05, size=2)
            ax.plot([0 + jx[0], 1 + jx[1]], [tn[i], ts[i]], color="k", alpha=0.25, linewidth=0.8, zorder=2)
            ax.scatter([0 + jx[0], 1 + jx[1]], [tn[i], ts[i]], s=18, color="k", alpha=0.5, zorder=3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["NS", "S"])
        top = max(ts.max(), tn.max())
        # Headroom scales with the TALLEST bar in THIS panel only (data
        # coords), but the bracket/star position is pinned in AXES FRACTION
        # coords (0..1 of the panel's own box), which is independent of the
        # data range. That guarantees the same visual gap to the subplot
        # title in every panel, regardless of how tall its bars are.
        ax.set_ylim(0, top * 1.35)
        bracket_y = 0.86   # axes-fraction y for the bracket line
        tick_h = 0.04       # axes-fraction height of the bracket's end-ticks
        ax.plot([0, 0, 1, 1],
                 [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
                 color="black", linewidth=1.1, transform=ax.transAxes, clip_on=False)
        ax.annotate(paired_sig[g]["sig"], xy=(0.5, bracket_y), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=12, fontweight="bold",
                    color="black", annotation_clip=False)
        ax.set_title(GLABEL[g], fontsize=11, pad=16)
    axes[0].set_ylabel("Sniffing time T (s)")
    fig.suptitle(f"[DEMO] Fig.N  {day}  Standard S/NS bar chart (benchmarked against PMC8103520)\n"
                 "bracket + */**/*** = within-group S vs NS (paired test)",
                 fontsize=11.5, color=TITLE_COLOR, y=1.1)
    fig.text(0.5, 0.005, DEMO_TAG, ha="center", fontsize=9, color=WATERMARK_COLOR, style="italic")
    fig.subplots_adjust(top=0.72, bottom=0.14, wspace=0.3)
    fig.savefig(f"{OUTPUT_DIR}/DEMO_FigN_TS_TNS_bars_{tag}.png", dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    # ---- 6. Fig B: 3-phase position-bias consistency check ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), sharey=True)
    titles = ["Phase 1 Habituation", "Phase 2 Pre-test", "Phase 3 Test"]
    for ax, ph, col, title in zip(axes, [p1, p2, p3], ["side_index", "side_index", "SPI"], titles):
        data = [ph[ph.group_code == g][col].values for g in GROUPS]
        bp = ax.boxplot(data, tick_labels=[GLABEL[g] for g in GROUPS], patch_artist=True, widths=0.55)
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(COLOR[g]); patch.set_alpha(0.55)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.2, color="red", linestyle=":", linewidth=0.8)
        ax.axhline(-0.2, color="red", linestyle=":", linewidth=0.8)
        ax.set_title(title, fontsize=10, pad=10)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("side_index (Phase1/2) or SPI (Phase3)")
    fig.suptitle(f"[DEMO] Fig.B  {day}  3-phase position-bias consistency check",
                 fontsize=12.5, color=TITLE_COLOR)
    fig.text(0.5, 0.005, DEMO_TAG, ha="center", fontsize=9, color=WATERMARK_COLOR, style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(f"{OUTPUT_DIR}/DEMO_FigB_phase_consistency_{tag}.png", dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    # ---- 7. Fig D: locomotor / entries control ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
    means = [p3[p3.group_code == g]["entries_total"].mean() for g in GROUPS]
    sds = [p3[p3.group_code == g]["entries_total"].std() for g in GROUPS]
    axes[0].bar([GLABEL[g] for g in GROUPS], means, yerr=sds, capsize=5, color=[COLOR[g] for g in GROUPS], alpha=0.85)
    axes[0].set_ylabel("Entries (mean +/- SD)"); axes[0].set_title("Chamber entries (Phase 3)", pad=10)
    means2 = [p1[p1.group_code == g]["distance_px"].mean() for g in GROUPS]
    sds2 = [p1[p1.group_code == g]["distance_px"].std() for g in GROUPS]
    axes[1].bar([GLABEL[g] for g in GROUPS], means2, yerr=sds2, capsize=5, color=[COLOR[g] for g in GROUPS], alpha=0.85)
    axes[1].set_ylabel("Distance (px, mean +/- SD)"); axes[1].set_title("Phase 1 locomotor baseline", pad=10)
    fig.suptitle(f"[DEMO] Fig.D  {day}  Locomotor / exploration control", fontsize=13, color=TITLE_COLOR)
    fig.text(0.5, 0.005, DEMO_TAG, ha="center", fontsize=9, color=WATERMARK_COLOR, style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(f"{OUTPUT_DIR}/DEMO_FigD_entries_baseline_{tag}.png", dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    print(f"[{DEMO_TAG}] [{day}] all figures generated")

# ---- summary tables ----
pd.DataFrame(all_shapiro_rows).to_csv(f"{OUTPUT_DIR}/DEMO_shapiro_normality_check.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(all_anova_rows).to_csv(f"{OUTPUT_DIR}/DEMO_two_way_anova_results.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(all_mw_rows).to_csv(f"{OUTPUT_DIR}/DEMO_mannwhitney_vs_blank.csv", index=False, encoding="utf-8-sig")

summary_rows = []
for day in DAYS:
    p3 = df[(df.day == day) & (df.phase == 3)]
    summ = p3.groupby("group_code").agg(
        n=("subject_id", "count"), SPI_mean=("SPI", "mean"), SPI_sd=("SPI", "std"),
        T_social_mean=("T_social_s", "mean"), T_object_mean=("T_object_s", "mean"),
        Entries_mean=("entries_total", "mean")).reindex(GROUPS)
    summ["day"] = day
    summary_rows.append(summ.reset_index())
pd.concat(summary_rows, ignore_index=True).round(4).to_csv(
    f"{OUTPUT_DIR}/DEMO_summary_stats.csv", index=False, encoding="utf-8-sig")

print(f"\n[{DEMO_TAG}] all files written to {OUTPUT_DIR}")
