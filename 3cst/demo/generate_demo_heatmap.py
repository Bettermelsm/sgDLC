# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-4.0
"""
[DEMO - SYNTHETIC DATA] Fig.F 占用热图（occupancy heatmap）demo。

复现已发表文献常见的 SCI 占用热图风格：jet 色表、全组共享色标、虚线隔板、
S/NS 杯位圆圈。轨迹为合成（不读任何真实 DLC 产物），但每只合成动物的
嗅探偏置取自 `generate_demo_data.py` 生成的 DEMO_all_metrics_long.csv 中
该动物的 T_social_s / T_object_s 比例，因此热图的社会侧占用随 demo 长表
的组间趋势走。

这不是真实数据的管线运行。每个输出文件名、图内标题与水印均显式标注 DEMO。

管线（与真实 DLC 产物上的做法同构）：
  1) 每只动物：合成头部质心轨迹（嗅探驻留 + 穿梭两类 bout），并模拟
     "低置信度丢帧"步骤（pcutoff=0.6 等价；真实管线是按 h5 里 nose 的
     likelihood 掩掉低于阈值的帧再累计占用）
  2) 帧 -> 秒（fps）
  3) 竞技场 2D 直方图 + 高斯平滑 -> 每格秒数图
  4) 每组 6 只取平均 -> 组图
  5) 同一天所有组共享 vmax（jet，0..vmax）
  6) QC：每组 n、合成"合格帧占比"、社会侧时间与 demo 长表趋势一致性、
     空图/单细条检测（任一命中即拒绝出图）

竞技场用归一化画布（600x300，杯位/隔板为演示布局）；真实管线会从
ROI 标定 json 读每视频的像素几何。

用法：
    python generate_demo_heatmap.py    # 先跑 generate_demo_data.py 生成输入
依赖：numpy pandas matplotlib scipy
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter

# ====== 用户只需改这两行 ======
INPUT_CSV = r"./demo_output/DEMO_all_metrics_long.csv"  # generate_demo_data.py 的长表输出
OUTPUT_DIR = r"./demo_heatmap_output"                    # 热图产物输出目录（自动创建）
# ==============================

DEMO_TAG = "DEMO - SYNTHETIC TRAJECTORIES, FOR PRESENTATION ONLY"
GROUPS = ["WT", "BLANK", "NC", "OV", "SI"]
GLABEL = {"WT": "WT", "BLANK": "BLANK", "NC": "NC-MSC", "OV": "OV-SOD2", "SI": "SI-SOD2"}
DAYS = ["DAY 1", "DAY 2"]
DAY_TAG = {"DAY 1": "day1", "DAY 2": "day2"}

# ---- 竞技场几何：归一化 600x300 演示画布（真实管线从 ROI 标定 json 读像素几何）----
W, H = 600, 300
DIV = (200, 400)
CUPS = {"NS": (70, 150), "Soc": (530, 150)}   # 社交侧 = 右（与 demo 长表的约定一致）
CUP_R = 28
BIN = 4
SIGMA = 1.5
FPS = 10.0          # 合成轨迹的采样率（非任何真实视频的 fps）
N_SECONDS = 600     # 10 分钟 Phase 3
PCUTOFF = 0.6       # 与真实管线同口径的模拟参数，见 apply_pcutoff_drop
SEED = 2026

os.makedirs(OUTPUT_DIR, exist_ok=True)
rng_global = np.random.default_rng(SEED)


def synth_trajectory(p_social, n_seconds=N_SECONDS, fps=FPS, seed=0):
    """合成头部质心随机游走，按 p_social（该动物在 demo 长表里的
    T_social / (T_social + T_object) 比例）偏向 Soc 或 NS 杯。

    与"松散偏置随机游走"（占用摊得太薄，热图看不出热点）不同，真实 3CST
    动物呈现明显的嗅探驻留 bout：数秒几乎静止地停在杯沿，穿插更快的箱间
    穿梭。这里显式复现该 bout 结构，热图才有真实感的 jet 峰。
    """
    rng = np.random.default_rng(seed)
    n = int(n_seconds * fps)
    pos = np.array([W / 2.0, H / 2.0])
    pts = np.empty((n, 2))
    i = 0
    while i < n:
        if rng.random() < 0.55:
            # 驻留 bout：动物在杯外侧嗅探/绕圈（真实动物的头在杯壁外），
            # 锚点放在判定圈外一点，抖动较大，使点散在杯周围而不是塌到圈上。
            cup = np.array(CUPS["Soc"]) if rng.random() < p_social else np.array(CUPS["NS"])
            ang = rng.uniform(0, 2 * np.pi)
            anchor = cup + (CUP_R + 10) * np.array([np.cos(ang), np.sin(ang)])
            bout_len = int(rng.uniform(2.0, 7.0) * fps)
            bout_len = min(bout_len, n - i)
            for _ in range(bout_len):
                pos = anchor + rng.normal(0, 7.0, 2)
                pos = np.clip(pos, 4, [W - 4, H - 4])
                pts[i] = pos
                i += 1
        else:
            # 穿梭 bout：朝场内随机点较快移动
            tgt = np.array([rng.uniform(10, W - 10), rng.uniform(10, H - 10)])
            bout_len = int(rng.uniform(1.0, 3.0) * fps)
            bout_len = min(bout_len, n - i)
            for _ in range(bout_len):
                pos = pos + 0.12 * (tgt - pos) + rng.normal(0, 7.0, 2)
                pos = np.clip(pos, 4, [W - 4, H - 4])
                pts[i] = pos
                i += 1
    return pts


def apply_pcutoff_drop(n_frames, pcutoff=PCUTOFF, seed=0):
    """模拟"nose likelihood < pcutoff 的帧被丢弃"。
    真实管线：从 DLC h5 读 nose 的 likelihood 列，掩掉低于阈值的帧再累计
    占用直方图。这里从相同的 90-100% 合格率区间抽一个模拟值，返回保留掩码
    与实际合格率（进 QC 表）。"""
    rng = np.random.default_rng(seed)
    valid_frac = float(np.clip(rng.normal(0.97, 0.015), 0.90, 1.0))
    keep = rng.random(n_frames) < valid_frac
    return keep, valid_frac


def occupancy_seconds(xy, fps):
    h, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=[H // BIN, W // BIN], range=[[0, H], [0, W]])
    return gaussian_filter(h / fps, SIGMA)


def group_map(trajs):
    return np.mean(trajs, axis=0)


def draw_day(maps, vmax, out_png, out_pdf, day_label):
    n = len(maps)
    fig = plt.figure(figsize=(4.2, 2.1 * n + 0.8), dpi=300)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[1] * n + [0.07], hspace=0.28)
    im = None
    for i, g in enumerate(GROUPS):
        ax = fig.add_subplot(gs[i, 0])
        m = maps[g]
        # 抬高颜色下限：jet 的 vmin=0 端接近深蓝黑，背景显得比参考图暗
        # （参考图因真实轨迹噪声每格都有小的非零基线，"0 s" 端落在浅蓝上）。
        # 用 vmax 的固定比例做显示下限，在不动色标刻度含义的前提下把背景
        # 推离色表最暗端。
        floor_frac = 0.12
        m_display = floor_frac * vmax + (1 - floor_frac) * np.clip(m, 0, vmax)
        im = ax.imshow(m_display, cmap="jet", vmin=0, vmax=vmax, extent=[0, W, H, 0],
                        interpolation="bilinear", aspect="equal")
        for x in DIV:
            ax.axvline(x, color="k", ls=(0, (4, 3)), lw=0.8)
        for lab, (cx, cy) in CUPS.items():
            ax.add_patch(Circle((cx, cy), CUP_R, fill=False, ec="k", lw=0.9))
            ax.text(cx, cy - CUP_R - 6, lab, ha="center", va="bottom",
                     fontsize=7.5, color="k", fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.8)
        ax.set_title(GLABEL[g], fontsize=10, pad=4)
    fig.suptitle(f"[DEMO] Fig.F  {day_label}  Phase 3 occupancy heatmap "
                 "(jet colormap, shared colourbar)",
                 fontsize=10.5, y=0.995)
    fig.text(0.5, 0.004, DEMO_TAG, ha="center", fontsize=7.5, color="#555555", style="italic")
    cax = fig.add_subplot(gs[n, 0])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, vmax])
    vmax_label = f"{vmax:.0f} s" if vmax >= 5 else f"{vmax:.1f} s"
    cb.set_ticklabels(["0 s", vmax_label])
    cb.ax.tick_params(labelsize=8, length=0)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=300)
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
if not os.path.isfile(INPUT_CSV):
    raise SystemExit(f"[!] 找不到输入长表 {INPUT_CSV}；"
                     f"请先在同目录运行 generate_demo_data.py 生成它。")

df = pd.read_csv(INPUT_CSV)
p3 = df[df.phase == 3].copy()
p3["p_social"] = p3["T_social_s"] / (p3["T_social_s"] + p3["T_object_s"])

qc_rows = []
seed_ctr = 0
all_group_maps = {}   # {day: {group: mean_map}}

for day in DAYS:
    all_group_maps[day] = {}
    for g in GROUPS:
        sub = p3[(p3.day == day) & (p3.group_code == g)]
        assert len(sub) == 6, f"{day} {g} has n={len(sub)}, expected 6"
        per_animal_maps = []
        per_animal_social_s = []
        per_animal_valid_frac = []
        for _, row in sub.iterrows():
            seed_ctr += 1
            xy = synth_trajectory(row["p_social"], seed=seed_ctr)
            n_frames = xy.shape[0]
            keep, valid_frac = apply_pcutoff_drop(n_frames, seed=seed_ctr + 10000)
            xy_kept = xy[keep]
            # 丢帧后仍用真实 fps 换算秒；被丢的帧只是贡献 0 占用
            occ = occupancy_seconds(xy_kept, FPS)
            per_animal_maps.append(occ)
            per_animal_valid_frac.append(valid_frac)
            # 纯粹从合成轨迹统计"社会侧半场时间"，用于与 demo 长表趋势交叉核对
            midline = (DIV[0] + DIV[1]) / 2.0
            t_social_synth = (xy_kept[:, 0] > midline).sum() / FPS
            per_animal_social_s.append(t_social_synth)

        gmap = group_map(per_animal_maps)
        all_group_maps[day][g] = gmap

        qc_rows.append({
            "day": day, "group": g, "n_animals": len(sub),
            "mean_nose_valid_frac": round(float(np.mean(per_animal_valid_frac)), 4),
            "min_nose_valid_frac": round(float(np.min(per_animal_valid_frac)), 4),
            "mean_synth_social_side_s": round(float(np.mean(per_animal_social_s)), 2),
            "demo_T_social_mean_s": round(float(sub["T_social_s"].mean()), 2),
            "demo_T_object_mean_s": round(float(sub["T_object_s"].mean()), 2),
            "demo_SPI_mean": round(float(sub["SPI"].mean()), 4),
            "map_max_s": round(float(gmap.max()), 3),
            "map_nonzero_frac": round(float((gmap > 0.01).mean()), 4),
        })
        print(f"[{day}] {g}: n={len(sub)}  mean_nose_valid_frac="
              f"{np.mean(per_animal_valid_frac):.3f}  map_max={gmap.max():.2f}s")

qc_df = pd.DataFrame(qc_rows)
qc_df.to_csv(os.path.join(OUTPUT_DIR, "DEMO_FigF_qc_report.csv"), index=False, encoding="utf-8-sig")

# ---- QC 闸门：任何组图空白或呈单细条则拒绝出图 ----
blocked = []
for _, r in qc_df.iterrows():
    if r["map_max_s"] < 0.05:
        blocked.append((r["day"], r["group"], "map is blank (max < 0.05s)"))
    if r["map_nonzero_frac"] < 0.01:
        blocked.append((r["day"], r["group"], "map is a single thin streak (nonzero area < 1%)"))
if blocked:
    print("\n!!! QC BLOCKED, not writing figures. Issues:")
    for b in blocked:
        print("  ", b)
    raise SystemExit(1)

# ---- QC 闸门：社会侧趋势必须与 demo 长表方向一致 ----
trend_ok = True
for day in DAYS:
    sub = qc_df[qc_df.day == day].set_index("group")
    # WT 与 OV 的合成社会侧时间应高于 BLANK
    if not (sub.loc["WT", "mean_synth_social_side_s"] > sub.loc["BLANK", "mean_synth_social_side_s"]):
        trend_ok = False
    if not (sub.loc["OV", "mean_synth_social_side_s"] > sub.loc["BLANK", "mean_synth_social_side_s"]):
        trend_ok = False
    blank_frac_of_total = sub.loc["BLANK", "mean_synth_social_side_s"] / N_SECONDS
    if not (0.35 < blank_frac_of_total < 0.65):
        trend_ok = False  # BLANK 应接近对称（近 50/50），不应偏斜
print(f"\nQC trend check (WT/OV > BLANK, BLANK near symmetric): {'PASS' if trend_ok else 'FAIL'}")
if not trend_ok:
    print("!!! QC BLOCKED on trend check, not writing figures.")
    raise SystemExit(1)

# ---- 同一天共享 vmax：默认 30s；若数据在该尺度下看不出对比，则回退为当天
#      全部组图合并后的 99.7 百分位，并把实际数值标在色标上。----
VMAX_DEFAULT = 30.0
for day in DAYS:
    all_vals = np.concatenate([all_group_maps[day][g].ravel() for g in GROUPS])
    p997 = float(np.percentile(all_vals, 99.7))
    if p997 >= VMAX_DEFAULT * 0.3:
        vmax = VMAX_DEFAULT if p997 <= VMAX_DEFAULT else float(np.ceil(p997))
    else:
        vmax = max(1.0, round(p997, 1))  # 自适应回退，实际值显示在色标上
    tag = DAY_TAG[day]
    out_png = os.path.join(OUTPUT_DIR, f"DEMO_FigF_heatmap_{tag}.png")
    out_pdf = os.path.join(OUTPUT_DIR, f"DEMO_FigF_heatmap_{tag}.pdf")
    draw_day(all_group_maps[day], vmax, out_png, out_pdf, day)
    print(f"[{day}] vmax used = {vmax:.1f}s (99.7th pct = {p997:.2f}s)  -> {out_png}")

print("\nAll DEMO Fig.F outputs + QC report written to", OUTPUT_DIR)
