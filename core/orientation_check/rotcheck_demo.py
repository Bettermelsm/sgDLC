#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-4.0
"""朝向核对闸门（orientation check gate）方法演示。

## 要防的错误

参考帧若经图像工具"自动旋转"导出（或从不同坐标系来源混用），其像素坐标系与
DLC 推理坐标系可能相差 90°/180°。这种错误**不报错**——ROI 照样能点、指标照样
能算，但几何整体错位，分区停留/嗅探/SPI 全部失真，且下游没有任何迹象能提示。

## 两道闸门

1. **文字方向核查（人工）**：参考图上叠加方位文字，看一眼标签是否正立。
   自动旋转的错误在标签上一眼可见。本脚本把正确/旋转两种合成参考帧并排
   画进输出图，供对照建立直觉。
2. **互相关核查（自动）**：把"ROI 叠加图"（用标定坐标在 DLC 坐标系渲染）
   与参考帧中位数合成图做归一化互相关（NCC）。同一坐标系下得分高；
   参考帧被旋转过则得分显著掉到阈值之下——闸门 FAIL。

## 运行模式

* **合成演示模式（默认，开箱即跑）**：生成一张含不对称亮度梯度、刺激杯与
  方位文字的合成三箱图，叠加随机游走的"动物"造出 24 帧视频帧，取中位数
  合成参考帧；再生成旋转 180° 的变体（模拟自动旋转 bug），分别与叠加图
  算 NCC 并给出 PASS/FAIL。不含任何真实视频。
* **真实数据模式**：把 DATA_ROOT 指向你的数据根目录，其中
  `ref_frames/` 放某组参考帧（来自推理所用同一批视频）、
  `overlays/` 放同组叠加图（pick_roi.py --overlay-dir 的产物）。

用法：
    python rotcheck_demo.py
依赖：numpy matplotlib（真实数据模式另需 pillow 读图）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== 用户只需改这两行 ======
DATA_ROOT = r"./data"                   # 指向你的数据根目录（不存在则走内置合成演示）
OUTPUT_DIR = r"./rotcheck_demo_output"  # 演示图输出目录
# ==============================

N_FRAMES = 24          # 合成参考帧用到的帧数（真实流程建议 20~50 帧）
NCC_THRESHOLD = 0.30   # NCC 闸门经验阈值：低于即 FAIL（按机位/画面自行校准）
SEED = 2026


# --------------------------------------------------------------------------
def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """归一化互相关（即去均值的 Pearson 相关），两图同形状。"""
    a = a.astype(float).ravel()
    b = b.astype(float).ravel()
    a -= a.mean()
    b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d > 0 else 0.0


def load_gray(path: str) -> np.ndarray:
    from matplotlib.image import imread
    img = imread(path)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return img


# --------------------------------------------------------------------------
# 合成演示
# --------------------------------------------------------------------------
ARENA_W, ARENA_H = 600, 300
DIVIDERS = (200, 400)
CUPS = {"NS (left)": (70, 150), "SOC (right)": (530, 150)}
CUP_R = 28


def render_arena(annotate: bool = False) -> np.ndarray:
    """渲染一张"DLC 坐标系下的标准画面"：左亮右暗梯度 + 双杯 + 隔板 +
    方位文字。不对称元素（梯度、文字）是互相关能察觉旋转的前提。"""
    yy, xx = np.mgrid[0:ARENA_H, 0:ARENA_W]
    rng = np.random.default_rng(SEED)
    img = 0.35 + 0.40 * (1.0 - xx / ARENA_W)            # 左亮右暗梯度
    img += rng.normal(0, 0.02, img.shape)               # 轻微纹理噪声
    for xd in DIVIDERS:
        img[:, xd - 1:xd + 1] *= 0.45                   # 隔板暗线
    for (_lab, (cx, cy)) in CUPS.items():
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= CUP_R ** 2
        img[m] *= 0.35                                  # 杯口暗盘
    img[[0, -1], :] *= 0.5
    img[:, [0, -1]] *= 0.5
    if annotate:
        # 用 matplotlib 渲染文字后取回 buffer，叠加到画面上
        # （dpi=100 与上面 figsize 的换算一致，buffer 尺寸 = ARENA_H x ARENA_W）
        fig = plt.figure(figsize=(ARENA_W / 100, ARENA_H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        for lab, (cx, cy) in CUPS.items():
            ax.text(cx, cy - CUP_R - 8, lab, color="yellow", fontsize=9,
                    ha="center", fontweight="bold")
        ax.text(14, 24, "ORIENT >>", color="red", fontsize=10,
                fontweight="bold")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:,:,:3].mean(axis=2)
        plt.close(fig)
        img = 0.55 * img + 0.45 * buf / 255.0
    return np.clip(img, 0, 1)


def make_frames(base: np.ndarray, n: int = N_FRAMES) -> list[np.ndarray]:
    """合成"视频帧"：标准画面 + 逐帧噪声 + 一只随机游动的灰色动物团。
    动物在动、场景静止，中位数合成即可把动物平均掉、留下干净参考帧——
    这正是真实流程里参考帧取中位数合成图的原因。"""
    rng = np.random.default_rng(SEED + 1)
    h, w = base.shape
    pos = np.array([w / 2.0, h / 2.0])
    frames = []
    for _ in range(n):
        pos = np.clip(pos + rng.normal(0, 25, 2), [30, 30], [w - 30, h - 30])
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-(((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2) / (2 * 18 ** 2)))
        frame = base + rng.normal(0, 0.03, base.shape) - 0.25 * blob
        frames.append(np.clip(frame, 0, 1))
    return frames


def run_synthetic(out_png: str) -> None:
    base = render_arena(annotate=True)
    composite = np.median(np.stack(make_frames(base)), axis=0)   # 正确方向参考帧
    composite_rot = np.rot90(composite, 2)                       # 模拟自动旋转 180°

    # 叠加图：真实流程里是"参考帧 + 用标定坐标画上的 ROI"。这里直接用
    # 标准画面（含杯位与文字）充当同一坐标系下的比对基准。
    score_ok = ncc(composite, base)
    score_rot = ncc(composite_rot, base)

    print("== 朝向核查闸门（合成演示）==")
    print(f"闸门2 互相关（NCC，阈值 {NCC_THRESHOLD:.2f}）：")
    print(f"  参考帧(正确方向) vs 叠加图 : NCC = {score_ok:+.3f}  "
          f"-> {'PASS' if score_ok >= NCC_THRESHOLD else 'FAIL'}")
    print(f"  参考帧(旋转180°) vs 叠加图 : NCC = {score_rot:+.3f}  "
          f"-> {'PASS' if score_rot >= NCC_THRESHOLD else 'FAIL'}")
    print("闸门1 文字方向（人工）：见输出图左右两栏——右栏标签倒立即为旋转错误，"
          "该错误闸门2同样能拦住。")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [(composite, f"Reference frame (correct orientation)\nNCC={score_ok:+.3f}  PASS"),
              (composite_rot, f"Reference frame (rotated 180°, simulated auto-rotate bug)\n"
                              f"NCC={score_rot:+.3f}  FAIL"),
              (base, "ROI overlay (DLC coordinate reference)")]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("[DEMO] Orientation check gate — synthetic illustration",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n演示图已写出：{out_png}")

    if score_rot >= NCC_THRESHOLD:
        print("[!] 演示自检失败：旋转变体竟通过闸门，阈值需上调或画面对称度过高。")
        return 1
    return 0


# --------------------------------------------------------------------------
# 真实数据模式
# --------------------------------------------------------------------------
def run_real(ref_dir: str, ov_dir: str) -> int:
    refs = sorted(f for f in os.listdir(ref_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif")))
    ovs = sorted(f for f in os.listdir(ov_dir)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif")))
    if not refs or not ovs:
        print(f"[!] {ref_dir}（{len(refs)} 张）或 {ov_dir}（{len(ovs)} 张）为空")
        return 1
    composite = np.median(np.stack([load_gray(os.path.join(ref_dir, f))
                                    for f in refs]), axis=0)
    print(f"参考帧合成：{len(refs)} 张中位数 -> "
          f"{composite.shape[1]}x{composite.shape[0]}")
    print(f"== 朝向核查（NCC 阈值 {NCC_THRESHOLD:.2f}）==")
    nfail = 0
    for f in ovs:
        ov = load_gray(os.path.join(ov_dir, f))
        if ov.shape != composite.shape:
            print(f"  ! {f}: 尺寸 {ov.shape} 与参考帧 {composite.shape} 不一致，"
                  f"先解决坐标系/分辨率问题")
            nfail += 1
            continue
        s = ncc(composite, ov)
        verdict = "PASS" if s >= NCC_THRESHOLD else "FAIL  <== 疑似旋转/错位"
        nfail += int(s < NCC_THRESHOLD)
        print(f"  {f:<40} NCC = {s:+.3f}  {verdict}")
    if nfail:
        print(f"\n{nfail} 张未过闸门：先核查参考帧导出是否被自动旋转，"
              f"再核查叠加图渲染坐标系。这类错误不报错但会让 ROI 全盘错位。")
        return 1
    print("\n全部通过。")
    return 0


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ref_dir = os.path.join(DATA_ROOT, "ref_frames")
    ov_dir = os.path.join(DATA_ROOT, "overlays")
    if os.path.isdir(ref_dir) and os.path.isdir(ov_dir):
        print(f"[i] 检测到 {DATA_ROOT} 下的 ref_frames/ 与 overlays/，走真实数据模式")
        return run_real(ref_dir, ov_dir)
    print(f"[i] 未找到 {ref_dir}，走内置合成演示模式\n")
    return run_synthetic(os.path.join(OUTPUT_DIR, "rotcheck_demo.png"))


if __name__ == "__main__":
    sys.exit(main())
