#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-4.0
"""批量把 DLC 推理产物算成 3CST 指标长表。

把三样东西对齐后逐视频调用 compute_3cst_metrics.compute()：

    session_manifest.csv   谁是谁（组 / 动物 / 阶段 / 该视频自己的 fps）
    roi_picks.json         竞技场 / 隔板 / 杯位（core/roi_calibration/pick_roi.py 的输出）
    analysis_out/**/*.h5   轨迹

## 输入约定

manifest 为 UTF-8 CSV，每行一个视频，至少包含列：
`file`（视频文件名，用于对齐 h5 主干）、`day`（批次/天）、`group_code`（组名，
统一大写）、`phase`（1/2/3）、`fps`（该视频自己的帧率）。
可选列：`group_display`、`include_in_main`、`subject_id`、`mouse_index`、`phase_name`。

## 为什么阶段必须在这一层把关

3CST 的三个阶段通常是同一只鼠连续做完的（务必向实验执行方确认顺序）：

    phase 1 Habituation  两个空杯
    phase 2 Pre-test     两个相同物体
    phase 3 Test         一侧陌生鼠 / 另一侧新颖物

「鼻尖落在左/右杯判定圈内的时长」这个量三个阶段都算得出来，但**只有 phase 3
的左右之比才是社交偏好指数（SPI）**。拿 phase 1（两个空杯）算出来的比值是随机
侧偏，数值范围、分布形态都和真 SPI 差不多，**不会报错、不会触发任何校验**，
只会安静地毁掉结论。所以这里的做法是：

* 两侧嗅探时长照常输出（phase 2 的物体探索、phase 1 的杯子探索本身有用）
* 统一输出 `side_index = (右 - 左)/(右 + 左)`，这只是个几何量
* **`SPI` 字段只在 phase == 3 时填**，其余阶段留空，并写明 `spi_withheld_reason`

phase 2 的 `side_index` 是重要对照：两侧是相同物体，本应接近 0。若某组 phase 2
就已明显偏向一侧，则该组 phase 3 的 SPI 里混着位置偏好，不能直接解读为社交偏好。

## 为什么不用 subprocess 逐个调 compute_3cst_metrics.py

上百次进程启动 + 每次重新 import pandas，纯浪费；而且 CLI 那一层会把
「左右半径不一致」这类致命错误变成一个非零退出码，容易被循环吞掉。直接 import
并调用 compute()，异常会原样抛出来把整批停住——这正是想要的行为。

用法（在有 pandas/numpy 的环境里跑）：
    python run_batch_metrics.py \\
        --manifest session_manifest.csv \\
        --roi roi_picks.json \\
        --h5-root ./analysis_out \\
        --out-csv metrics_long.csv \\
        --qc-csv  metrics_qc.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_3cst_metrics as m3            # noqa: E402

TEST_PHASE = 3


def index_h5(root: str) -> dict:
    """视频名主干 -> h5 路径。

    DLC 的输出文件名是 `<视频主干>DLC_<模型标签>.h5`，且 destfolder 会镜像
    源目录结构（目录名可能带空格、大小写不一致）。所以这里递归找，按主干建
    索引，不去猜模型标签——标签会随 shuffle/snapshot 变，写死一定会在换
    snapshot 后失配。
    """
    out = {}
    dup = []
    for p in glob.glob(os.path.join(root, "**", "*.h5"), recursive=True):
        base = os.path.basename(p)
        if "_full" in base or base.endswith("_meta.h5"):
            continue
        i = base.find("DLC")
        stem = base[:i] if i > 0 else os.path.splitext(base)[0]
        if stem in out:
            dup.append((stem, out[stem], p))
        out[stem] = p
    for stem, a, b in dup:
        print(f"[!] 同一个视频有多个 h5，取后者：{stem}\n      {a}\n      {b}")
    return out


def norm_roi_keys(roi: dict) -> dict:
    """大写 -> 原 key。

    参考图的 tag 来自目录名拼接，而目录名大小写常不一致（同一组在不同批次
    目录下可能一处小写一处大写）；manifest 里的 group_code 已统一大写。
    不做这层归一化就会整组匹配不上，该组所有视频的杯位全部缺失。
    """
    return {k.upper(): k for k in roi}


def main() -> int:
    ap = argparse.ArgumentParser(description="批量计算 3CST 指标")
    ap.add_argument("--manifest", required=True, help="视频清单 CSV（见文末输入约定）")
    ap.add_argument("--roi", required=True, help="pick_roi.py 的输出")
    ap.add_argument("--h5-root", required=True, help="DLC 推理产物根目录（递归找 h5）")
    ap.add_argument("--out-csv", required=True, help="指标长表")
    ap.add_argument("--qc-csv", default="", help="体点可用率等 QC 表")
    ap.add_argument("--json-dir", default="", help="逐视频完整 json（可选）")
    ap.add_argument("--pcutoff", type=float, default=0.6)
    ap.add_argument("--nominal-seconds", type=float, default=600.0)
    ap.add_argument("--test-phase", type=int, default=TEST_PHASE,
                    help="哪个阶段是 Test（只有它输出 SPI），默认 3")
    args = ap.parse_args()

    for p in (args.manifest, args.roi):
        if not os.path.isfile(p):
            sys.exit(f"[!] 找不到 {p}")

    with open(args.roi, encoding="utf-8") as fh:
        roi_all = json.load(fh)
    roi_lut = norm_roi_keys(roi_all)

    with open(args.manifest, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("[!] manifest 为空")

    h5idx = index_h5(args.h5_root)
    print(f"manifest {len(rows)} 条 | 找到 h5 {len(h5idx)} 个 | "
          f"ROI 标定 {len(roi_all)} 组")

    if args.json_dir:
        os.makedirs(args.json_dir, exist_ok=True)

    out_rows, qc_rows = [], []
    n_ok = n_noh5 = n_noroi = n_err = 0
    warn_roi = set()

    for r in rows:
        stem = os.path.splitext(r["file"])[0]
        day = r.get("day", "")
        code = (r.get("group_code") or "").upper()
        tag_up = f"{day.replace(' ', '')}_{code}".upper()
        phase = int(r.get("phase") or 0)

        h5 = h5idx.get(stem)
        if not h5:
            n_noh5 += 1
            continue

        rk = roi_lut.get(tag_up)
        if rk is None:
            if tag_up not in warn_roi:
                print(f"[!] 没有 ROI 标定：{tag_up}（该组所有视频的嗅探/SPI 都算不了）")
                warn_roi.add(tag_up)
            n_noroi += 1
            continue
        entry = roi_all[rk]

        fps = float(r.get("fps") or 0)
        if fps <= 0:
            print(f"[!] {stem} 的 fps 缺失，跳过（同批视频的 fps 常有浮动，"
                  f"不能拿常数顶替）")
            n_err += 1
            continue

        try:
            df = m3.load_tracks(h5)
            res = m3.compute(
                df, fps, entry["arena"], entry["dividers"],
                args.pcutoff, args.nominal_seconds,
                entry.get("social_side", "right"),
                None,                       # 半径走标定里的 r_sniff
                entry.get("cups"),
                m3.CENTER_BODYPARTS,
            )
        except SystemExit as exc:           # 例如左右半径不一致，属致命
            print(f"[FATAL] {stem}: {exc}")
            raise
        except Exception:                   # noqa: BLE001
            n_err += 1
            print(f"[!] {stem} 计算失败：")
            traceback.print_exc()
            continue

        sl, sr = res.get("sniff_left_s"), res.get("sniff_right_s")
        side_index = None
        if sl is not None and sr is not None and (sl + sr) > 0:
            side_index = round((sr - sl) / (sr + sl), 4)

        is_test = (phase == args.test_phase)
        spi = res.get("SPI") if is_test else None
        reason = "" if is_test else (
            f"phase={phase} 不是 Test（{args.test_phase}）阶段，"
            f"两侧不是社交刺激 vs 物体，比值不构成 SPI")

        rec = {
            "file": r["file"],
            "day": day,
            "group_code": code,
            "group_display": r.get("group_display", code),
            "include_in_main": r.get("include_in_main", ""),
            "subject_id": r.get("subject_id", ""),
            "mouse_index": r.get("mouse_index", ""),
            "phase": phase,
            "phase_name": r.get("phase_name", ""),
            "roi_key": rk,
            "fps": res["fps"],
            "frames_total": len(df),
            "window_start_s": res["window"]["start_s"],
            "window_duration_s": res["window"]["duration_s"],
            "window_truncated": int(bool(res["window"]["truncated"])),
            "valid_frac_in_window": res["valid_frac_in_window"],
            "n_frames_used": res["n_frames_used"],
            "time_left_s": res["time_left_s"],
            "time_center_s": res["time_center_s"],
            "time_right_s": res["time_right_s"],
            "frac_left": res["frac_left"],
            "frac_center": res["frac_center"],
            "frac_right": res["frac_right"],
            "sniff_radius_px": res["sniff_radius_px"],
            "sniff_radius_source": res["sniff_radius_source"],
            "sniff_left_s": sl,
            "sniff_right_s": sr,
            "side_index": side_index,
            "social_side": res["social_side"],
            "T_social_s": res["T_social_s"] if is_test else None,
            "T_object_s": res["T_object_s"] if is_test else None,
            "SPI": spi,
            "spi_withheld_reason": reason,
            "entries_total": res["entries_total"],
            "distance_px": res["distance_px"],
            "speed_px_per_s": res["speed_px_per_s"],
            "cups_source": res["cups_source"],
            "sniff_metrics_trustworthy": int(bool(res["sniff_metrics_trustworthy"])),
            "center_bodyparts": ",".join(res["center_bodyparts"]),
        }
        out_rows.append(rec)

        qc = {"file": r["file"], "day": day, "group_code": code, "phase": phase,
              "subject_id": r.get("subject_id", ""),
              "frames_total": len(df),
              "valid_frac_in_window": res["valid_frac_in_window"]}
        qc.update({f"avail_{k}": v
                   for k, v in res["bodypart_availability_pct"].items()})
        qc_rows.append(qc)

        if args.json_dir:
            res["file"] = r["file"]
            res["phase"] = phase
            res["subject_id"] = r.get("subject_id", "")
            with open(os.path.join(args.json_dir, stem + ".json"),
                      "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=2)

        n_ok += 1
        if n_ok % 10 == 0:
            print(f"  ... {n_ok} 个已算完")

    if not out_rows:
        sys.exit("[!] 一个都没算出来。先确认 h5 是否存在、ROI 标定是否覆盖各组。")

    with open(args.out_csv, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"\n写出 {args.out_csv}：{len(out_rows)} 条")

    if args.qc_csv:
        with open(args.qc_csv, "w", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(qc_rows[0].keys()))
            wr.writeheader()
            wr.writerows(qc_rows)
        print(f"写出 {args.qc_csv}：{len(qc_rows)} 条")

    print(f"\n成功 {n_ok} | 缺 h5 {n_noh5} | 缺 ROI {n_noroi} | 出错 {n_err}")

    # ---------------- 汇总，让错误在这里就暴露出来 ----------------
    df = pd.DataFrame(out_rows)

    print()
    print("=== 各阶段计数（应为 1:1:1）===")
    print(df.groupby(["day", "phase"]).size().to_string())

    test = df[df["phase"] == args.test_phase]
    print()
    print(f"=== Test 阶段（phase {args.test_phase}）SPI 概览 ===")
    if test.empty:
        print("  没有 Test 阶段数据")
    else:
        g = test.groupby(["day", "group_code"])["SPI"].agg(["count", "mean", "std",
                                                           "min", "max"])
        print(g.round(4).to_string())

    ph2 = df[df["phase"] == 2]
    if not ph2.empty:
        print()
        print("=== Pre-test（phase 2）侧偏对照：两侧是相同物体，应接近 0 ===")
        g2 = ph2.groupby(["day", "group_code"])["side_index"].agg(["count", "mean",
                                                                  "std"])
        print(g2.round(4).to_string())
        bad = g2[g2["mean"].abs() > 0.2]
        if not bad.empty:
            print("  ! 以下组 phase 2 已有明显侧偏（|mean side_index| > 0.2），")
            print("    其 phase 3 的 SPI 混有位置偏好，解读时必须说明：")
            print(bad.round(4).to_string())

    ph1 = df[df["phase"] == 1]
    if not ph1.empty:
        print()
        print("=== Habituation（phase 1）活动量基线：路程 px ===")
        print(ph1.groupby(["day", "group_code"])["distance_px"]
              .agg(["count", "mean", "std"]).round(1).to_string())

    print()
    print("=== 数据质量红旗 ===")
    flags = 0
    untrust = df[df["sniff_metrics_trustworthy"] == 0]
    if not untrust.empty:
        flags += 1
        print(f"  ! {len(untrust)} 个视频的杯位不是外部指定的 —— 嗅探与 SPI 不可用")
    lowv = df[df["valid_frac_in_window"] < 0.8]
    if not lowv.empty:
        flags += 1
        print(f"  ! {len(lowv)} 个视频窗内合格率 < 0.8：")
        print(lowv[["file", "day", "group_code", "phase",
                    "valid_frac_in_window"]].to_string(index=False))
    trunc = df[df["window_truncated"] == 1]
    if not trunc.empty:
        print(f"  i {len(trunc)} 个视频的分析窗被截断（视频短于标称时长），属正常")
    rs = df["sniff_radius_px"].nunique()
    if rs > 1:
        flags += 1
        print(f"  ! 嗅探半径在不同视频间有 {rs} 种取值。跨组比较要求半径一致，"
              f"否则组间差异里混着几何差异。取值：{sorted(df['sniff_radius_px'].unique())}")
    if flags == 0:
        print("  无")

    return 0


if __name__ == "__main__":
    sys.exit(main())
