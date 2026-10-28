#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-4.0
"""从 DLC 追踪结果计算三箱社交测试的行为学指标。

## 输出指标

  各区停留时间       左/中/右三箱，秒（用身体中心判定）
  嗅探时间           鼻尖进入杯区判定圈的累计时间，秒
  SPI                (T_social - T_object) / (T_social + T_object)
  穿梭次数           箱室间切换次数（带滞回，抑制边界抖动造成的虚假计数）
  运动距离/速度      排除运动能力受损这一替代解释

## 三个容易出错的地方，都在实现里显式处理了

1. **帧数换算成秒必须用每个视频自己的 fps。** 同一批视频常测得多个不同的
   fps（设备时钟漂移，且每个文件夹第一个视频往往偏离）。按统一常数换算会让
   时长和所有时间类指标系统性偏掉。

2. **分析时间窗要从数据里找，不能固定裁头尾。** 实测同一批视频时长可明显
   偏离标称值，且头部含放鼠、实验员手入画等操作段。这里用 likelihood 的滚动
   合格率找入场时刻，再往后取标称时长。

3. **穿梭次数必须带滞回。** 动物在隔板附近来回小幅移动时，裸阈值判定会产生几十
   次虚假穿梭。这里要求越过分界线并深入相邻箱室一定距离才计一次。

## 社交侧约定

Phase 3 默认 right=social（右侧陌生鼠、左侧新颖物体）。这只是本仓库的缺省
约定，**务必与你的实际实验核对**；做了左右平衡的实验必须逐只指定，否则 SPI
会整体翻符号——这是最致命的静默错误，所以脚本会把所用约定原样写进输出。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

BODYPARTS = ["nose", "left_ear", "right_ear", "tail_base"]

# 质心默认只用三个头部点，故意不含 tail_base。
#
# 大样本实测：
#   nose / left_ear / right_ear 三个点的可用率接近 100%，tail_base 常仅
#   60~70%（likelihood > 0.6 的帧占比；tail_base 多数帧仍合格，只是俯视下
#   弓身/靠壁/钻入杯下时被遮挡的长尾很厚。）
#
# body_center() 是"逐帧只对通过 pcutoff 的点求平均"，看似稳健，实际埋了一个
# 系统性偏差：tail_base 在时可用时不可用，质心就在"头三点中心"与"含尾根的
# 体中心"之间来回跳，凭空制造位移，把路程算高；而弓身/遮挡的频率本身是可能
# 随组别不同的（例如刻板行为/靠壁更多的组别遮挡也更频繁），于是这个伪位移会
# 被当成组间差异。固定成接近全可用的三个头部点，组成不再随行为漂移，代价是
# 它是"头部质心"而非体中心——报告里如实这么写即可。tail_base 仍保留在
# BODYPARTS 里供体轴/朝向类指标使用。
CENTER_BODYPARTS = ["nose", "left_ear", "right_ear"]


# --------------------------------------------------------------------------
def load_tracks(path: str) -> pd.DataFrame:
    if path.lower().endswith((".h5", ".hdf5")):
        df = pd.read_hdf(path)
    else:
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    while df.columns.nlevels > 2:
        df.columns = df.columns.droplevel(0)
    return df


def xy_lk(df: pd.DataFrame, bp: str):
    return (df[(bp, "x")].to_numpy(float),
            df[(bp, "y")].to_numpy(float),
            df[(bp, "likelihood")].to_numpy(float))


def body_center(df: pd.DataFrame, pcutoff: float, bps: list[str]):
    have = list(dict.fromkeys(df.columns.get_level_values(0)))
    use = [b for b in bps if b in have]
    xs = np.column_stack([df[(b, "x")].to_numpy(float) for b in use])
    ys = np.column_stack([df[(b, "y")].to_numpy(float) for b in use])
    lk = np.column_stack([df[(b, "likelihood")].to_numpy(float) for b in use])
    ok = np.isfinite(xs) & np.isfinite(ys) & (lk > pcutoff)
    n = ok.sum(axis=1)
    cx = np.where(n > 0, np.nansum(np.where(ok, xs, np.nan), axis=1) / np.maximum(n, 1), np.nan)
    cy = np.where(n > 0, np.nansum(np.where(ok, ys, np.nan), axis=1) / np.maximum(n, 1), np.nan)
    return cx, cy, n


def rolling_frac(valid: np.ndarray, win: int) -> np.ndarray:
    if win < 1:
        return valid.astype(float)
    ker = np.ones(win, dtype=float) / win
    return np.convolve(valid.astype(float), ker, mode="same")


def detect_window(valid: np.ndarray, fps: float, nominal_s: float,
                  frac_thresh: float = 0.8, sustain_s: float = 20.0,
                  smooth_s: float = 5.0) -> dict:
    """用 likelihood 滚动合格率找入场时刻，再往后取标称时长。

    放鼠前动物不在场或被手遮挡，合格率低且不稳；放好后合格率持续走高。
    要求"持续 sustain_s 秒达标"而不是"首次达标"，以免被操作过程中的瞬时
    高置信度骗到。
    """
    n = len(valid)
    win = max(int(round(smooth_s * fps)), 1)
    f = rolling_frac(valid, win)
    need = max(int(round(sustain_s * fps)), 1)
    good = f >= frac_thresh
    start = None
    run = 0
    for i, g in enumerate(good):
        run = run + 1 if g else 0
        if run >= need:
            start = i - need + 1
            break
    method = "likelihood_onset"
    if start is None:                       # 从未持续达标：退化为从头开始
        start = 0
        method = "fallback_from_zero"
    want = int(round(nominal_s * fps))
    end = min(start + want, n)
    return {"start_frame": int(start), "end_frame": int(end),
            "n_frames": int(end - start),
            "start_s": round(start / fps, 2),
            "duration_s": round((end - start) / fps, 2),
            "truncated": bool(end - start < want),
            "method": method,
            "mean_valid_frac_in_window": round(float(f[start:end].mean()), 4)
            if end > start else 0.0}


# --------------------------------------------------------------------------
def locate_cups(df: pd.DataFrame, pcutoff: float, arena: dict, div: dict,
                sl: slice, nbin: int = 42, wall_margin: float = 0.18) -> dict:
    """【仅供诊断，不可用于正式分析】从鼻尖密度峰猜杯位。

    这条路**在方法学上是循环论证的**，正式分析必须改用图像检测或人工指定的杯位。

    为什么循环：本函数假设"动物会反复嗅探杯子，所以鼻尖密度在杯周聚集"。但
    嗅探量本身就是我们要测的因变量。对回避社交的动物（正是最需要准确测量的那
    一类），它根本不靠近杯子，密度峰就落不到杯上——于是方法在"有偏好"的动物上
    好用、在"无偏好"的动物上失效，系统性地把结果推向"检出偏好"。

    实测证据（在一批确认无社交偏好的动物上）：
      - 不剔除贴边带：杯位被判到竞技场角落与门洞旁（趋边性 + 门洞流量的密度
        远超嗅探），SPI 算出 +0.67 的假阳性；而同一只鼠的分区停留显示它其实
        回避右侧箱室（占比仅 0.188），自相矛盾
      - 剔除贴边带后：左箱室内部仅剩数百个鼻点、右箱室不足两百个，无法定位

    所以保留本函数只为诊断用途（例如快速看一眼密度分布），返回结果会被标记
    unreliable，正式指标计算请通过 cups_override 传入图像检测/人工指定的杯位。
    """
    nx, ny, nl = xy_lk(df, "nose")
    m = np.isfinite(nx) & np.isfinite(ny) & (nl > pcutoff)
    idx = np.zeros(len(nx), dtype=bool)
    idx[sl] = True
    m &= idx
    ah = arena["y1"] - arena["y0"]
    my = wall_margin * ah
    out = {}
    for side, (xa, xb) in (("left", (arena["x0"], div["xd1"])),
                           ("right", (div["xd2"], arena["x1"]))):
        cw = xb - xa
        mx = wall_margin * cw
        # 只在箱室内部搜索：两侧各让开 wall_margin，避开贴边带与门洞
        sxa, sxb = xa + mx, xb - mx
        sya, syb = arena["y0"] + my, arena["y1"] - my
        sel = m & (nx >= sxa) & (nx <= sxb) & (ny >= sya) & (ny <= syb)
        if sel.sum() < 200:
            out[side] = None
            continue
        H, xe, ye = np.histogram2d(nx[sel], ny[sel], bins=nbin,
                                   range=[[sxa, sxb], [sya, syb]])
        # 轻度平滑后取峰，避免单格噪声
        k = np.ones((3, 3)) / 9.0
        Hs = np.zeros_like(H)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                i0, i1 = max(i - 1, 0), min(i + 2, H.shape[0])
                j0, j1 = max(j - 1, 0), min(j + 2, H.shape[1])
                Hs[i, j] = H[i0:i1, j0:j1].mean()
        ij = np.unravel_index(np.argmax(Hs), Hs.shape)
        out[side] = {
            "cx": float(0.5 * (xe[ij[0]] + xe[ij[0] + 1])),
            "cy": float(0.5 * (ye[ij[1]] + ye[ij[1] + 1])),
            "peak_frac": float(Hs[ij] / max(Hs.sum(), 1e-9)),
            "n_nose_pts": int(sel.sum()),
            "search_box": [round(sxa, 1), round(sxb, 1), round(sya, 1), round(syb, 1)],
            "method": f"nose_density_peak(wall_margin={wall_margin})",
            "unreliable": True,
            "why_unreliable": "由被测行为反推刺激位置，属循环论证；仅供诊断",
        }
    return out


def count_entries(x: np.ndarray, xd1: float, xd2: float,
                  hysteresis: float) -> dict:
    """带滞回的箱室切换计数。

    裸阈值判定会把隔板附近的小幅来回记成几十次穿梭。这里要求越线后再深入
    hysteresis 像素才确认换箱，显著抑制虚假计数。
    """
    def zone_of(v):
        if v < xd1 - hysteresis:
            return 0
        if v > xd2 + hysteresis:
            return 2
        if xd1 + hysteresis < v < xd2 - hysteresis:
            return 1
        return -1                       # 缓冲带内：不改变当前判定

    cur = None
    seq = []
    for v in x:
        if not np.isfinite(v):
            continue
        z = zone_of(v)
        if z == -1:
            continue
        if cur is None:
            cur = z
            seq.append(z)
        elif z != cur:
            cur = z
            seq.append(z)
    trans = max(len(seq) - 1, 0)
    return {"entries_total": int(trans), "zone_sequence_len": len(seq),
            "hysteresis_px": round(hysteresis, 1)}


def compute(df: pd.DataFrame, fps: float, arena: dict, div: dict,
            pcutoff: float, nominal_s: float, social_side: str,
            sniff_radius: float | None, cups_override: dict | None,
            center_bps: list[str] | None = None) -> dict:
    center_bps = center_bps or CENTER_BODYPARTS
    cx, cy, nvalid = body_center(df, pcutoff, center_bps)
    # 入场时刻判定沿用"全部体点"的合格率：实验员的手入画、鼠未落地时，
    # 尾根同样测不到，把它一起算进来能让 onset 判得更干脆。
    _, _, nvalid_all = body_center(df, pcutoff, BODYPARTS)
    valid = nvalid_all > 0
    win = detect_window(valid, fps, nominal_s)
    sl = slice(win["start_frame"], win["end_frame"])

    # 杯位必须来自独立于被测行为的来源（图像检测 / 人工指定）。
    # 没有传入时才退化到轨迹推断，且结果全程带 unreliable 标记。
    if cups_override:
        cups = cups_override
        cups_source = "external(image_or_manual)"
    else:
        cups = locate_cups(df, pcutoff, arena, div, sl)
        cups_source = "track_inferred_UNRELIABLE"
    aw = arena["x1"] - arena["x0"]
    # 嗅探判定半径的优先级：
    #   1) 命令行 --sniff-radius 显式指定
    #   2) 杯位标定里带的 r_sniff（pick_roi.py 里人工调过并肉眼确认的那个圈）
    #   3) 竞技场宽度的固定比例
    # 第 2 条必须在第 3 条之前：人工在图上把圈调到贴合杯子，结果却被公式算的
    # 半径悄悄顶掉，是"你批准的不是实际参与计算的几何"——pick_roi 的文档明确
    # 承诺过不会这样。
    r_default = sniff_radius if sniff_radius else 0.075 * aw

    def radius_for(side: str) -> tuple[float, str]:
        if sniff_radius:
            return sniff_radius, "cli"
        c = cups.get(side) or {}
        if c.get("r_sniff"):
            return float(c["r_sniff"]), "picked"
        return 0.075 * aw, "arena_frac_0.075"

    r = r_default

    bx, by = cx[sl], cy[sl]
    ok = np.isfinite(bx) & np.isfinite(by)
    n_ok = int(ok.sum())
    dt = 1.0 / fps

    zl = int((ok & (bx < div["xd1"])).sum())
    zc = int((ok & (bx >= div["xd1"]) & (bx <= div["xd2"])).sum())
    zr = int((ok & (bx > div["xd2"])).sum())

    nx, ny, nl = xy_lk(df, "nose")
    nx, ny, nl = nx[sl], ny[sl], nl[sl]
    nose_ok = np.isfinite(nx) & np.isfinite(ny) & (nl > pcutoff)

    sniff = {}
    r_used = {}
    for side in ("left", "right"):
        c = cups.get(side)
        if not c:
            sniff[side] = None
            continue
        rs, rsrc = radius_for(side)
        r_used[side] = {"r_px": round(rs, 1), "source": rsrc}
        d = np.hypot(nx - c["cx"], ny - c["cy"])
        inz = nose_ok & (d <= rs)
        sniff[side] = {"frames": int(inz.sum()), "seconds": round(inz.sum() * dt, 2)}
    # 左右半径必须一致，否则嗅探时长之比（SPI 的分子分母）被几何差异污染
    if len(r_used) == 2 and abs(r_used["left"]["r_px"] - r_used["right"]["r_px"]) > 0.5:
        raise SystemExit(
            f"[!] 左右嗅探半径不一致（左 {r_used['left']['r_px']}px / "
            f"右 {r_used['right']['r_px']}px）。SPI 是两侧嗅探时长的比值，"
            f"半径不等会直接制造偏向，必须用同一个半径。"
            f"请在 pick_roi.py 里把两侧调成同一半径，或用 --sniff-radius 统一指定。")

    ts = tobj = None
    if sniff.get("left") and sniff.get("right"):
        if social_side == "right":
            ts, tobj = sniff["right"]["seconds"], sniff["left"]["seconds"]
        else:
            ts, tobj = sniff["left"]["seconds"], sniff["right"]["seconds"]
    spi = None
    if ts is not None and (ts + tobj) > 0:
        spi = round((ts - tobj) / (ts + tobj), 4)

    ent = count_entries(bx, div["xd1"], div["xd2"], 0.02 * aw)

    # 运动量：逐帧位移求和。用中心而非鼻尖，鼻尖抖动会高估路程。
    dx = np.diff(bx); dy = np.diff(by)
    step = np.hypot(dx, dy)
    good = np.isfinite(step)
    # 剔除跳变：单帧位移超过竞技场宽度 8% 视为误检跳点
    good &= step < 0.08 * aw
    dist = float(step[good].sum())
    dur = max(n_ok * dt, 1e-9)

    # 每个体点的可用率一并写出：tail_base 天然偏低（俯视遮挡），这是判断
    # "某个视频是否真的追踪失败"的依据——不能只看聚合数字。
    avail = {}
    for b in BODYPARTS:
        if b in list(dict.fromkeys(df.columns.get_level_values(0))):
            lk = df[(b, "likelihood")].to_numpy(float)[sl]
            avail[b] = round(100.0 * float((lk > pcutoff).mean()), 1)

    return {
        "fps": round(fps, 4),
        "cups_source": cups_source,
        "sniff_metrics_trustworthy": cups_source == "external(image_or_manual)",
        "center_bodyparts": list(center_bps),
        "bodypart_availability_pct": avail,
        "window": win,
        "n_frames_used": n_ok,
        "valid_frac_in_window": round(n_ok / max(win["n_frames"], 1), 4),
        "time_left_s": round(zl * dt, 2),
        "time_center_s": round(zc * dt, 2),
        "time_right_s": round(zr * dt, 2),
        "frac_left": round(zl / max(n_ok, 1), 4),
        "frac_center": round(zc / max(n_ok, 1), 4),
        "frac_right": round(zr / max(n_ok, 1), 4),
        "sniff_radius_px": (r_used["left"]["r_px"] if r_used
                            else round(r_default, 1)),
        "sniff_radius_source": (r_used["left"]["source"] if r_used else "unused"),
        "sniff_left_s": sniff["left"]["seconds"] if sniff.get("left") else None,
        "sniff_right_s": sniff["right"]["seconds"] if sniff.get("right") else None,
        "social_side": social_side,
        "T_social_s": ts, "T_object_s": tobj, "SPI": spi,
        "entries_total": ent["entries_total"],
        "entry_hysteresis_px": ent["hysteresis_px"],
        "distance_px": round(dist, 1),
        "speed_px_per_s": round(dist / dur, 2),
        "cups": cups,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="计算 3CST 行为学指标")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--roi", required=True, help="pick_roi.py 的 json")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="不给则从 roi json 或必须显式提供；务必用该视频自己的 fps")
    ap.add_argument("--pcutoff", type=float, default=0.6)
    ap.add_argument("--nominal-seconds", type=float, default=600.0)
    ap.add_argument("--social-side", choices=["left", "right"], default="right",
                    help="Phase 3 中陌生鼠所在侧。默认 right 只是常见约定的缺省值，"
                         "务必核实你的实验约定：搞反会让 SPI 整体翻符号，而且下游"
                         "没有任何迹象能提示这个错误")
    ap.add_argument("--sniff-radius", type=float, default=0.0,
                    help="嗅探判定半径（像素）。0=按竞技场宽度 7.5% 自动取")
    ap.add_argument("--cups", default="",
                    help='杯位 json，形如 {"left":{"cx":..,"cy":..},'
                         '"right":{"cx":..,"cy":..}}。'
                         '来自图像检测或人工指定。**嗅探时间与 SPI 必须依赖它**：'
                         '从动物自身轨迹反推杯位是循环论证，会系统性偏向"检出偏好"')
    ap.add_argument("--center-bodyparts", default=",".join(CENTER_BODYPARTS),
                    help="用于算质心的体点，逗号分隔。默认只用接近全可用的三个"
                         "头部点，故意排除 tail_base（实测可用率常仅 60~70%%，"
                         "时有时无会让质心在头中心与体中心间跳动，凭空制造位移）")
    ap.add_argument("--roi-key", default="",
                    help="当 --roi 是 pick_roi.py 的多组标定文件时，指定组名。"
                         "该组的杯位与 social_side 会一并采用")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    center_bps = [b.strip() for b in args.center_bodyparts.split(",") if b.strip()]

    if not os.path.isfile(args.tracks):
        sys.exit(f"[!] 找不到 {args.tracks}")
    with open(args.roi, encoding="utf-8") as fh:
        roi = json.load(fh)

    # pick_roi.py 的产物是"按组名索引"的多组标定，一份文件里放着所有组；
    # 也兼容单组、arena/dividers 直接在顶层的 json。
    # 这里两种都直接吃，不写中间转换脚本——转换层是最容易悄悄搞错又看不出来的
    # 环节（pick_roi 的文档里就是这么写的，这里必须一致）。
    social_side = args.social_side
    cups_override = None
    if "arena" not in roi:
        if not args.roi_key:
            sys.exit(f"[!] {args.roi} 是按组索引的多组标定（含 "
                     f"{len(roi)} 组：{', '.join(sorted(roi)[:6])}...），"
                     f"必须用 --roi-key 指定用哪一组")
        if args.roi_key not in roi:
            sys.exit(f"[!] --roi-key {args.roi_key} 不在 {args.roi} 里。"
                     f"可选：{', '.join(sorted(roi))}")
        entry = roi[args.roi_key]
        arena, div = entry["arena"], entry["dividers"]
        # 杯位与社交侧同源于这份人工标定，默认一并采用；显式 --cups 可覆盖
        if entry.get("cups"):
            cups_override = entry["cups"]
        if entry.get("social_side"):
            social_side = entry["social_side"]
        if entry.get("warnings"):
            print(f"[!] 该组标定带 {len(entry['warnings'])} 条告警：")
            for msg in entry["warnings"]:
                print(f"    ! {msg}")
    else:
        arena, div = roi["arena"], roi["dividers"]

    fps = args.fps
    if fps <= 0:
        sys.exit("[!] 必须提供 --fps（每个视频用它自己的 fps；"
                 "同一批视频的 fps 常有浮动，用常数会算错时长）")

    if args.cups:
        with open(args.cups, encoding="utf-8") as fh:
            cups_override = json.load(fh)

    df = load_tracks(args.tracks)
    res = compute(df, fps, arena, div, args.pcutoff, args.nominal_seconds,
                  social_side,
                  args.sniff_radius if args.sniff_radius > 0 else None,
                  cups_override, center_bps)
    res["tracks"] = os.path.basename(args.tracks)

    w = res["window"]
    print(f"视频 {res['tracks']}   fps={res['fps']}   总帧 {len(df)}")
    print(f"体点可用率(lk>{args.pcutoff}) " +
          "  ".join(f"{k}={v}%" for k, v in res["bodypart_availability_pct"].items()))
    print(f"质心体点 = {','.join(res['center_bodyparts'])}")
    print(f"分析窗 帧[{w['start_frame']}..{w['end_frame']}]  "
          f"起始 {w['start_s']}s  时长 {w['duration_s']}s  "
          f"方法={w['method']}" + ("  [被截断]" if w["truncated"] else ""))
    print(f"  窗内合格率 {res['valid_frac_in_window']:.3f}   可用帧 {res['n_frames_used']}")
    print()
    print(f"停留时间  左 {res['time_left_s']:7.1f}s  中 {res['time_center_s']:7.1f}s  "
          f"右 {res['time_right_s']:7.1f}s")
    print(f"停留占比  左 {res['frac_left']:.3f}    中 {res['frac_center']:.3f}    "
          f"右 {res['frac_right']:.3f}")
    print(f"嗅探(半径{res['sniff_radius_px']}px, 来源={res['sniff_radius_source']})"
          f"  左 {res['sniff_left_s']}s  右 {res['sniff_right_s']}s")
    print(f"社交侧={res['social_side']}  T_social={res['T_social_s']}s  "
          f"T_object={res['T_object_s']}s  SPI={res['SPI']}")
    print(f"穿梭 {res['entries_total']} 次（滞回 {res['entry_hysteresis_px']}px）")
    print(f"路程 {res['distance_px']}px   平均速度 {res['speed_px_per_s']}px/s")
    for side in ("left", "right"):
        c = res["cups"].get(side)
        if not c:
            print(f"  杯-{side}: 未定位（嗅探与 SPI 不可用）")
            continue
        # peak_frac/n_nose_pts/method 只存在于"从轨迹反推"那条路径。人工标定
        # 传进来的杯位没有这些键，这里必须用 .get——原先直接下标，导致外部杯位
        # 一律在算完之后、写 --out 之前抛 KeyError，等于正式跑批时才会炸。
        extra = ""
        if "peak_frac" in c:
            extra = (f" 峰占比{c['peak_frac']:.4f} 鼻点{c.get('n_nose_pts', '?')}"
                     f" {c.get('method', '')}")
        elif c.get("source"):
            extra = f" 来源={c['source']}"
        if c.get("r_sniff"):
            extra += f" r_sniff={c['r_sniff']:.0f}"
        print(f"  杯-{side}: ({c['cx']:.0f},{c['cy']:.0f}){extra}")
    print(f"杯位来源 = {res['cups_source']}")
    if not res["sniff_metrics_trustworthy"]:
        print("  !! 杯位是从轨迹反推的，属循环论证：嗅探时间与 SPI 不可用于结论。")
        print("     请用 --cups 传入图像检测或人工指定的杯位后重算。")
        print("     （分区停留时间、穿梭次数、运动量不依赖杯位，仍然可用）")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        print(f"\n写出 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
