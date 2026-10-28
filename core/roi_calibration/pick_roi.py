#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-4.0
"""在参考图上人工标定完整 ROI：竞技场 + 两道隔板 + 两个杯位。

## 为什么把三件事合到一个工具里手动点

三件事各有看似可行的自动推断，但都有方法学陷阱：

* **杯位** —— 从轨迹反推（鼻点密度峰）是**循环论证**：嗅探量正是因变量，
  用它定刺激位置，再用刺激位置算嗅探量。实测该做法会把杯位落到角落或门洞旁，
  SPI 出现大幅假阳性，而同一只动物的分区停留显示它其实回避那一侧。
* **隔板** —— 用"穿越点 y 离散度"推断，指标本身有信息量（隔板处穿越 y 的
  离散度远高于全场），但它依赖动物穿门足够多次，短视频或少动的个体会失效。
* **竞技场边界** —— 从图像自动检测对"背景与箱底灰度接近"的场景不稳定；
  退而用轨迹包络则取决于动物去过哪里，少动的个体会把场地测小。

现在有了**跨全场数十帧的中位数合成图**：杯子和隔板静止所以清晰，动物在动被
平均掉。三样东西在图上都一目了然，直接量比任何推断都准，而且**独立于被测行为**，
这一点对杯位是方法学上的硬要求。

## 坐标空间（弄错则全盘偏移且下游无感）

参考图必须来自**推理所用的同一批视频、同一坐标系**。若推理前对视频做过
缩放/补边/旋转，参考帧必须取自处理后的那批视频，不能用原生分辨率的帧。
图像工具的"自动旋转"也会悄悄改变坐标空间——建议在标定前先过一遍
`../orientation_check/rotcheck_demo.py` 的朝向核查闸门。

## 操作

每张图按三步走，标题栏会提示当前该点什么。

  第 1 步 竞技场   点内侧地面的两个对角（左上、右下）
  第 2 步 隔板     依次点两道隔板上的任意位置（只取 x）
  第 3 步 杯位     点左杯圆心，再点右杯圆心；上/下方向键调判定圈半径

  Tab        在三步之间切换（可回头改）
  u          撤销当前步骤的上一次点击
  r          清空当前步骤
  Enter      三步齐全才接受，进入下一张
  s          跳过本张
  q          保存并退出（下次运行自动续做）

放大用 matplotlib 自带工具栏的放大镜，缩放不影响坐标。

用法：
    python pick_roi.py --frames-dir ./ref_frames --out roi_picks.json \\
        --overlay-dir ./overlay
"""
from __future__ import annotations

import argparse
import json
import os
import sys

STAGES = ("arena", "dividers", "cups")
STAGE_CN = {"arena": "第1步 竞技场：点内侧地面两个对角",
            "dividers": "第2步 隔板：依次点两道隔板",
            "cups": "第3步 杯位：先左杯后右杯（上/下键调半径）"}


def checks(rec: dict) -> list[str]:
    """标定完成后的自洽性检查。

    这些不是吹毛求疵：杯位左右搞反会让 SPI 整体翻符号，而下游没有任何迹象能
    提示这个错误——所以必须在录入的这一刻就拦住。
    """
    w = rec["arena"]
    d = rec["dividers"]
    c = rec["cups"]
    out = []
    aw = w["x1"] - w["x0"]
    ah = w["y1"] - w["y0"]
    if aw <= 0 or ah <= 0:
        out.append("竞技场宽或高为非正数")
        return out
    ar = aw / float(ah)
    if not (1.1 <= ar <= 3.0):
        out.append(f"竞技场长宽比 {ar:.2f} 不像三箱俯视（预期约 1.4~1.6）")
    if not (w["x0"] < d["xd1"] < d["xd2"] < w["x1"]):
        out.append("两道隔板未落在竞技场内、或左右顺序不对")
    else:
        seg = [d["xd1"] - w["x0"], d["xd2"] - d["xd1"], w["x1"] - d["xd2"]]
        if min(seg) < 0.12 * aw:
            out.append(f"三箱宽度悬殊：{[round(s) for s in seg]}px，最窄不足场宽 12%")
    for side, expect in (("left", "左"), ("right", "右")):
        cc = c.get(side)
        if not cc:
            out.append(f"{expect}杯未标"); continue
        if not (w["y0"] - 5 <= cc["cy"] <= w["y1"] + 5):
            out.append(f"{expect}杯 y={cc['cy']:.0f} 落在竞技场上下界之外")
    cl, cr = c.get("left"), c.get("right")
    if cl and cr:
        if not cl["cx"] < d["xd1"]:
            out.append("左杯不在左箱内（cx 应小于 xd1）")
        if not cr["cx"] > d["xd2"]:
            out.append("右杯不在右箱内（cx 应大于 xd2）")
        if cl["cx"] >= cr["cx"]:
            out.append("左杯 cx 不小于右杯 cx —— 左右点反了")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="人工标定竞技场/隔板/杯位")
    ap.add_argument("--frames-dir", required=True,
                    help="参考图目录（必须来自推理所用的同一批视频）")
    ap.add_argument("--out", required=True, help="输出 json")
    ap.add_argument("--overlay-dir", default="",
                    help="同时写出叠加图，用于事后核对")
    ap.add_argument("--sniff-margin", type=float, default=0.6,
                    help="嗅探判定圈相对杯半径的外扩比例，默认 0.6")
    ap.add_argument("--social-side", choices=["left", "right"], default="right",
                    help="Phase 3 陌生鼠所在侧，默认 right 仅为常见约定的缺省值。"
                         "务必与实际实验约定核对：搞反会让 SPI 整体翻符号且下游"
                         "无感；做了左右平衡的实验需逐只指定")
    ap.add_argument("--pattern", default=".png,.jpg")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except Exception as exc:                       # noqa: BLE001
        sys.exit(f"[!] 需要可交互的 matplotlib（TkAgg）：{exc}")
    try:
        import cv2
    except ImportError:
        sys.exit("[!] 需要 opencv-python")

    exts = tuple(e if e.startswith(".") else "." + e
                 for e in args.pattern.split(","))
    files = sorted(os.path.join(args.frames_dir, f)
                   for f in os.listdir(args.frames_dir)
                   if f.lower().endswith(exts))
    if not files:
        sys.exit(f"[!] {args.frames_dir} 下没有 {exts} 图像")

    result = {}
    if os.path.isfile(args.out):
        with open(args.out, encoding="utf-8") as fh:
            result = json.load(fh)
        print(f"[i] 续做：{args.out} 已有 {len(result)} 组，"
              f"要全部重来请先删掉该文件")

    todo = [f for f in files
            if os.path.splitext(os.path.basename(f))[0] not in result]
    print(f"待标定 {len(todo)} / 共 {len(files)}")
    if not todo:
        print("全部已完成。")
        return 0
    print("提示：放大用工具栏放大镜；Tab 切换步骤；u 撤销；Enter 接受\n")

    quit_all = {"v": False}
    last_radius = {"v": 0.0}
    for v in result.values():                      # 续做时接着上次的半径
        cl = (v.get("cups") or {}).get("left")
        if cl and cl.get("r_equiv"):
            last_radius["v"] = float(cl["r_equiv"])

    for path in todo:
        if quit_all["v"]:
            break
        key = os.path.splitext(os.path.basename(path))[0]
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"  读不出，跳过：{key}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        pts = {"arena": [], "dividers": [], "cups": []}
        # 半径沿用上一张的取值作为起点：多数组机位相同，省去反复调整；
        # 更重要的是，圈应当贴合**杯子这个实物**——杯子在每一组都是同一个物件，
        # 所以只要每张都按杯子大小调，跨组的判定圈就是同一个物理距离，
        # 即便像素尺度因机位高低略有差异也成立。
        radius = {"v": last_radius["v"] if last_radius["v"] else 0.045 * w}
        stage = {"i": 0}
        action = {"v": None}

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(rgb)
        arts = []

        def clear_arts():
            while arts:
                a = arts.pop()
                try:
                    a.remove()
                except Exception:                  # noqa: BLE001
                    pass

        def complete() -> bool:
            return (len(pts["arena"]) == 2 and len(pts["dividers"]) == 2
                    and len(pts["cups"]) == 2)

        def redraw():
            clear_arts()
            if len(pts["arena"]) >= 1:
                for (x, y) in pts["arena"]:
                    arts.append(ax.plot([x], [y], "+", color="lime",
                                        ms=14, mew=2)[0])
            if len(pts["arena"]) == 2:
                (x0, y0), (x1, y1) = pts["arena"]
                x0, x1 = min(x0, x1), max(x0, x1)
                y0, y1 = min(y0, y1), max(y0, y1)
                arts.append(ax.add_patch(Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fill=False,
                    ec="lime", lw=2.0)))
            for (x, _y) in pts["dividers"]:
                arts.append(ax.axvline(x, color="yellow", lw=1.8, ls="--"))
            for i, (x, y) in enumerate(pts["cups"]):
                col = "tab:cyan" if i == 0 else "tab:orange"
                arts.append(ax.add_patch(Circle((x, y), radius["v"],
                                                fill=False, ec=col, lw=2.0,
                                                ls="--")))
                arts.append(ax.add_patch(Circle(
                    (x, y), radius["v"] * (1.0 + args.sniff_margin),
                    fill=False, ec=col, lw=2.2)))
                arts.append(ax.text(x, y - radius["v"] * (1.0 + args.sniff_margin) - 10,
                                    f"{'left' if i == 0 else 'right'} "
                                    f"({x:.0f},{y:.0f})",
                                    color=col, fontsize=9, ha="center",
                                    bbox=dict(fc="black", alpha=0.5, pad=1.5)))
            st = STAGES[stage["i"]]
            done = "  [三步齐全，Enter 接受]" if complete() else ""
            ax.set_title(f"{key}    {STAGE_CN[st]}"
                         f"   已点 {len(pts[st])}"
                         f"   r={radius['v']:.0f}px{done}\n"
                         f"Tab=换步骤  u=撤销  r=清空本步  "
                         f"Enter=接受  s=跳过  q=保存退出",
                         fontsize=10)
            fig.canvas.draw_idle()

        def on_click(ev):
            # 工具栏处于放大/平移模式时不收点，否则框选会被当成标点
            if fig.canvas.toolbar is not None and fig.canvas.toolbar.mode:
                return
            if ev.inaxes is not ax or ev.xdata is None:
                return
            st = STAGES[stage["i"]]
            cap = 2
            if len(pts[st]) >= cap:
                pts[st].pop(0)
            pts[st].append((float(ev.xdata), float(ev.ydata)))
            if len(pts[st]) == cap and stage["i"] < len(STAGES) - 1:
                stage["i"] += 1                    # 自动进入下一步
            redraw()

        def on_key(ev):
            if ev.key == "tab":
                stage["i"] = (stage["i"] + 1) % len(STAGES)
                redraw()
            elif ev.key == "u":
                st = STAGES[stage["i"]]
                if pts[st]:
                    pts[st].pop()
                redraw()
            elif ev.key == "r":
                pts[STAGES[stage["i"]]].clear()
                redraw()
            elif ev.key == "up":
                radius["v"] *= 1.08
                redraw()
            elif ev.key == "down":
                radius["v"] /= 1.08
                redraw()
            elif ev.key == "enter":
                if complete():
                    action["v"] = "accept"
                    plt.close(fig)
                else:
                    print("  还没点完（竞技场2点 / 隔板2点 / 杯位2点），"
                          "用 Tab 切到缺的那步")
            elif ev.key == "s":
                action["v"] = "skip"
                plt.close(fig)
            elif ev.key == "q":
                action["v"] = "quit"
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        redraw()
        plt.show()

        act = action["v"]
        if act == "quit":
            quit_all["v"] = True
            continue
        if act != "accept":
            print(f"  跳过 {key}")
            continue

        (ax0, ay0), (ax1, ay1) = pts["arena"]
        xs = sorted(p[0] for p in pts["dividers"])
        cups_sorted = sorted(pts["cups"], key=lambda p: p[0])
        rec = {
            "image": os.path.basename(path),
            "image_w": int(w), "image_h": int(h),
            "arena": {"x0": round(min(ax0, ax1), 1), "x1": round(max(ax0, ax1), 1),
                      "y0": round(min(ay0, ay1), 1), "y1": round(max(ay0, ay1), 1)},
            "dividers": {"xd1": round(xs[0], 1), "xd2": round(xs[1], 1)},
            "cups": {},
            "sniff_margin": args.sniff_margin,
            "social_side": args.social_side,
            "source": "manual_pick_on_median_reference",
        }
        for side, (cx, cy) in zip(("left", "right"), cups_sorted):
            rec["cups"][side] = {
                "cx": round(cx, 1), "cy": round(cy, 1),
                "r_equiv": round(radius["v"], 1),
                "r_sniff": round(radius["v"] * (1.0 + args.sniff_margin), 1),
                "source": "manual",
            }
        warns = checks(rec)
        rec["warnings"] = warns
        result[key] = rec
        last_radius["v"] = radius["v"]

        a = rec["arena"]
        print(f"  {key}: 场[{a['x0']:.0f}..{a['x1']:.0f}]x"
              f"[{a['y0']:.0f}..{a['y1']:.0f}]  "
              f"隔板 {rec['dividers']['xd1']:.0f}/{rec['dividers']['xd2']:.0f}  "
              f"杯 L({rec['cups']['left']['cx']:.0f},{rec['cups']['left']['cy']:.0f})"
              f" R({rec['cups']['right']['cx']:.0f},{rec['cups']['right']['cy']:.0f})"
              f"  r={radius['v']:.0f}")
        for msg in warns:
            print(f"    ! {msg}")
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)

        if args.overlay_dir:
            os.makedirs(args.overlay_dir, exist_ok=True)
            ov = bgr.copy()
            cv2.rectangle(ov, (int(a["x0"]), int(a["y0"])),
                          (int(a["x1"]), int(a["y1"])), (0, 255, 0), 2)
            for xd in (rec["dividers"]["xd1"], rec["dividers"]["xd2"]):
                cv2.line(ov, (int(xd), int(a["y0"])), (int(xd), int(a["y1"])),
                         (0, 255, 255), 2)
            for side, col in (("left", (255, 255, 0)), ("right", (0, 165, 255))):
                cc = rec["cups"][side]
                cv2.circle(ov, (int(cc["cx"]), int(cc["cy"])),
                           int(cc["r_equiv"]), col, 2)
                cv2.circle(ov, (int(cc["cx"]), int(cc["cy"])),
                           int(cc["r_sniff"]), col, 2)
            cv2.imwrite(os.path.join(args.overlay_dir, key + "_roi.png"), ov)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"\n写出 {args.out}（{len(result)} / {len(files)} 组已标定）")
    missing = [os.path.splitext(os.path.basename(f))[0] for f in files
               if os.path.splitext(os.path.basename(f))[0] not in result]
    if missing:
        print(f"仍未标定：{missing}")
        print("这些组的嗅探时间与 SPI 无法计算。")
    nwarn = sum(1 for v in result.values() if v.get("warnings"))
    if nwarn:
        print(f"有告警的组 {nwarn} 个，请回看上面的 ! 行；"
              f"左右杯点反会让 SPI 整体翻符号且下游无感。")
    else:
        print("全部通过自洽性检查（隔板在场内、左右杯分居两侧、三箱宽度均衡）。")

    # 跨组一致性：判定圈大小直接决定嗅探时长，进而决定 SPI。实测同一条轨迹把
    # 半径放大一倍，SPI 可以从 -0.54 变到 -0.30。所以圈相对杯子/场地的
    # 大小必须各组一致，否则组间差异里混进了几何差异。
    if len(result) >= 3:
        import statistics
        ratios = {}
        for k, v in result.items():
            a = v["arena"]
            aw = a["x1"] - a["x0"]
            cl = (v.get("cups") or {}).get("left")
            if aw > 0 and cl:
                ratios[k] = cl["r_equiv"] / aw
        if ratios:
            med = statistics.median(ratios.values())
            print()
            print(f"跨组一致性（杯半径 / 场宽，中位数 {med:.4f}）：")
            odd = []
            for k in sorted(ratios):
                dev = ratios[k] / med - 1.0
                flag = "  <== 偏离超 15%" if abs(dev) > 0.15 else ""
                print(f"  {k:<16} {ratios[k]:.4f}  ({dev:+.1%}){flag}")
                if abs(dev) > 0.15:
                    odd.append(k)
            if odd:
                print(f"  ! {len(odd)} 组偏离较大：{odd}")
                print("    两种可能，都要处理：(a) 圈没按杯子实际大小调；"
                      "(b) 该组机位高度/焦距不同，场地像素尺度不同，"
                      "绝对像素指标不可跨组直接比。")
            else:
                print("  各组一致，判定圈几何不会污染组间比较。")

    print(f"\n约定：social_side = {args.social_side}"
          f"（务必与实际实验约定核对，搞反会让 SPI 整体翻符号）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
