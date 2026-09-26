"""评估 caudal_fin_profile：海鲈尾三点（P6/P7U/P7L）几何推导 vs 人工真值。

混合输出 = V1 链（微调权重）的体部点 + caudal_fin_profile 的尾点。
用法：
    python scripts/eval_caudal_profile.py --batch 海鲈3 \
        --weights runs/pose/fewshot_n10_crop/weights/best.pt [--preview]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

POINT_ORDER = ["P1", "P2", "P3", "P5", "P8", "P9", "P10", "P11", "P6", "P7U", "P7L"]
SIGANUS_16_FULL_TO_11 = {
    "P1_snout_tip": "P1", "P2_eye_front": "P2", "P3_operculum_posterior": "P3",
    "P5_caudal_base_midpoint": "P5", "P8_body_depth_dorsal": "P8",
    "P9_body_depth_ventral": "P9", "P10_peduncle_depth_dorsal": "P10",
    "P11_peduncle_depth_ventral": "P11", "P6_caudal_fork_midpoint": "P6",
    "P7U_caudal_fin_upper_tip": "P7U", "P7L_caudal_fin_lower_tip": "P7L",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="海鲈3")
    parser.add_argument("--weights", default="runs/pose/fewshot_n10_crop/weights/best.pt")
    parser.add_argument("--pck-px", type=float, default=15.0)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    from siganusmorph.caudal_fin_profile import derive_caudal_points
    from siganusmorph.image_utils import load_image_file
    from siganusmorph.preannotation import preannotate_warped_image
    from siganusmorph.segmentation import segment_fish_from_blue_board

    ws = ROOT / "data" / "annotation_workspace" / args.batch
    weights = ROOT / args.weights
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted((ws / "annotations").glob("*.json"))
               if json.loads(p.read_text(encoding="utf-8")).get("status") == "done"]
    if not records:
        raise SystemExit("没有 status=done 的真值标注")

    rows = []
    for record in records:
        stem = record["image_stem"]
        image = load_image_file(ws / "warped" / record["warped_image"])
        truth = record["points"]
        mm_px = record.get("mm_per_pixel") or 0.1

        # 链（微调权重）：体部点
        chain = preannotate_warped_image(image, f"{stem}.png", project_root=ws / "_eval_run_root", model_path=weights)
        keypoints_full = chain.get("keypoints") or chain.get("warped_keypoints") or {}
        body_pts = {code: (float(kp[0]), float(kp[1]))
                    for full, code in SIGANUS_16_FULL_TO_11.items() if (kp := keypoints_full.get(full))}

        # caudal profile：尾三点（P5 用模型自己的体部点）
        body_mask, _bbox, _q = segment_fish_from_blue_board(image)
        caudal = derive_caudal_points(image, body_mask, body_pts["P5"])
        merged = dict(body_pts)
        for code in ("P6", "P7U", "P7L"):
            if caudal[code] is not None:
                merged[code] = caudal[code]
        print(f"{stem}: caudal quality={caudal['quality'].get('status')} "
              f"mask_edge={caudal['quality'].get('mask_right_edge')} notch={caudal['quality'].get('notch_depth_px')}")

        for code in POINT_ORDER:
            if code in merged and code in truth:
                err = float(np.hypot(merged[code][0] - truth[code][0], merged[code][1] - truth[code][1]))
                rows.append((code, err, err * mm_px))

        if args.preview:
            canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            for code, (x, y) in merged.items():
                cv2.circle(canvas, (int(x), int(y)), 10, (0, 140, 255), 3)
                cv2.putText(canvas, code, (int(x) + 12, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
            for code, (x, y) in truth.items():
                cv2.drawMarker(canvas, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 26, 3)
            out = ws / "eval_previews" / f"{stem}_caudal.png"
            out.parent.mkdir(exist_ok=True)
            ok, buf = cv2.imencode(".png", canvas)
            out.write_bytes(buf.tobytes())

    print(f"\n混合输出（体部=微调模型，尾点=caudal_fin_profile）· PCK@{args.pck_px:.0f}px")
    print(f"{'点':<5}{'平均px':>9}{'最大px':>9}{'平均mm':>9}{'PCK':>8}")
    summary: dict[str, list[tuple[float, float]]] = {}
    for code, err_px, err_mm in rows:
        summary.setdefault(code, []).append((err_px, err_mm))
    all_px, all_mm = [], []
    for code in POINT_ORDER:
        errs = summary.get(code, [])
        if not errs:
            continue
        px = [e[0] for e in errs]
        mm = [e[1] for e in errs]
        all_px.extend(px)
        all_mm.extend(mm)
        pck = sum(1 for e in px if e <= args.pck_px) / len(px)
        print(f"{code:<5}{sum(px)/len(px):>9.1f}{max(px):>9.1f}{sum(mm)/len(mm):>9.2f}{pck:>8.0%}")
    if all_px:
        pck_all = sum(1 for e in all_px if e <= args.pck_px) / len(all_px)
        print(f"{'ALL':<5}{sum(all_px)/len(all_px):>9.1f}{max(all_px):>9.1f}{sum(all_mm)/len(all_mm):>9.2f}{pck_all:>8.0%}")
    print("\n对照（此前测量）：零样本 ALL=24.24mm · N=10 混合(旧尾点) ALL=9.96mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
