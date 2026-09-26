"""在测试批次上评估：零样本 V1 链 vs 若干微调臂（裁剪域协议）。

对测试批次 annotation workspace 里的每张图（要求 status=done 的人工真值）：
  - zeroshot_v1：V1 完整链（分割→裁剪→YOLO→掩膜校正）16 点映射 11 点；
  - 各微调臂：与训练一致的裁剪域协议（分割→8% padding 裁剪→预测→偏移回全图）。

用法：
    python scripts/eval_on_test_batch.py --batch 海鲈3 \
        --arm n5=runs/pose/fewshot_n5_crop/weights/best.pt \
        --arm n10=runs/pose/fewshot_n10_crop/weights/best.pt \
        [--schema 16pt] [--pck-px 15] [--preview]
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
SCHEMA_16_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6", "P7U", "P7L", "P8", "P9", "P10", "P11", "C1", "C2", "C3", "C4"]


def zeroshot_v1_chain(image, model_path: Path, run_root: Path) -> dict[str, tuple[float, float]] | None:
    from siganusmorph.preannotation import preannotate_warped_image

    result = preannotate_warped_image(image, "eval.png", project_root=run_root, model_path=model_path)
    keypoints_full = result.get("keypoints") or result.get("warped_keypoints") or {}
    out = {code: (float(keypoints_full[full][0]), float(keypoints_full[full][1]))
           for full, code in SIGANUS_16_FULL_TO_11.items() if full in keypoints_full}
    return out or None


def crop_bbox_for(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """与训练一致的裁剪：分割 → 8% padding；失败返回 None（全图）。"""
    from siganusmorph.segmentation import padded_bbox, segment_fish_from_blue_board

    fish_mask, fish_bbox, quality = segment_fish_from_blue_board(rgb)
    height, width = rgb.shape[:2]
    if quality.get("segmentation_success", quality.get("success", False)) and fish_bbox is not None:
        x1, y1, x2, y2 = padded_bbox(fish_bbox, rgb.shape, padding_ratio=0.08)
        return max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))
    return None


def finetuned_crop_arm(weights: Path, image, schema: str):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    rgb = np.asarray(image)
    bbox = crop_bbox_for(rgb)
    ox, oy = (bbox[0], bbox[1]) if bbox else (0, 0)
    crop = rgb[oy:bbox[3], ox:bbox[2]] if bbox else rgb
    results = model.predict(crop, imgsz=960, conf=0.05, max_det=1, verbose=False)
    if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
        return None
    xy = results[0].keypoints.xy.cpu().numpy()[0]
    codes = SCHEMA_16_ORDER if len(xy) == len(SCHEMA_16_ORDER) else POINT_ORDER
    return {code: (float(p[0]) + ox, float(p[1]) + oy) for code, p in zip(codes, xy)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="测试批次名（annotation workspace 内需有人工真值）")
    parser.add_argument("--arm", action="append", default=[], help="微调臂，格式 name=weights.pt（可多次）")
    parser.add_argument("--schema", choices=["11pt", "16pt"], default="16pt")
    parser.add_argument("--pck-px", type=float, default=15.0)
    parser.add_argument("--preview", action="store_true", help="输出每个臂的标注叠加图")
    args = parser.parse_args()

    from siganusmorph.image_utils import load_image_file

    ws = ROOT / "data" / "annotation_workspace" / args.batch
    siganus_base = ROOT / "models" / "siganusmorph_yolopose_v0.3_corrected_real_5_15" / "preannotation_candidate.pt"
    arms: dict[str, Path | None] = {"zeroshot_v1": siganus_base}
    for spec in args.arm:
        name, _, weights = spec.partition("=")
        arms[name.strip()] = Path(weights.strip())
    arm_names = list(arms)

    records = []
    for path in sorted((ws / "annotations").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "done":
            records.append(record)
    if not records:
        raise SystemExit(f"批次 {args.batch} 没有 status=done 的人工真值标注")

    rows = []  # (code, err_px, err_mm, arm)
    for record in records:
        stem = record["image_stem"]
        warped_path = ws / "warped" / record["warped_image"]
        image = load_image_file(warped_path)
        truth = record["points"]
        mm_px = record.get("mm_per_pixel") or 0.1
        preds: dict[str, dict] = {}
        for name, weights in arms.items():
            try:
                if name.startswith("chain:"):
                    # 混合臂：V1 完整链（分割→裁剪→预测→掩膜校正→尾几何），换用微调权重
                    preds[name] = zeroshot_v1_chain(image, weights, ws / "_eval_run_root") or {}
                elif name == "zeroshot_v1":
                    preds[name] = zeroshot_v1_chain(image, weights, ws / "_eval_run_root") or {}
                else:
                    preds[name] = finetuned_crop_arm(weights, image, args.schema) or {}
            except Exception as exc:  # noqa: BLE001
                print(f"[{stem}] {name} 预测失败: {exc!r}")
                preds[name] = {}
            for code in POINT_ORDER:
                if code in preds[name] and code in truth:
                    err = float(np.hypot(preds[name][code][0] - truth[code][0],
                                         preds[name][code][1] - truth[code][1]))
                    rows.append((code, err, err * mm_px, name))
        if args.preview:
            preview_dir = ws / "eval_previews"
            preview_dir.mkdir(exist_ok=True)
            canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            colors = {"zeroshot_v1": (255, 0, 0), **{n: (0, 140, 255) for n in arms if n != "zeroshot_v1"}}
            for name, pts in preds.items():
                for code, (x, y) in pts.items():
                    cv2.circle(canvas, (int(x), int(y)), 10, colors.get(name, (0, 255, 0)), 3)
            for code, (x, y) in truth.items():
                cv2.drawMarker(canvas, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 26, 3)
            ok, buf = cv2.imencode(".png", canvas)
            (preview_dir / f"{stem}_eval.png").write_bytes(buf.tobytes())

    print(f"\n测试批次: {args.batch} · {len(records)} 张真值 · PCK 阈值 {args.pck_px:.0f}px")
    print(f"{'臂':<16}{'点':<5}{'平均px':>9}{'最大px':>9}{'平均mm':>9}{'PCK':>8}")
    summary: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for code, err_px, err_mm, name in rows:
        summary.setdefault((name, code), []).append((err_px, err_mm))
    for name in arm_names:
        all_px, all_mm = [], []
        for code in POINT_ORDER:
            errs = summary.get((name, code), [])
            if not errs:
                continue
            px = [e[0] for e in errs]
            mm = [e[1] for e in errs]
            all_px.extend(px)
            all_mm.extend(mm)
            pck = sum(1 for e in px if e <= args.pck_px) / len(px)
            print(f"{name:<16}{code:<5}{sum(px)/len(px):>9.1f}{max(px):>9.1f}{sum(mm)/len(mm):>9.2f}{pck:>8.0%}")
        if all_px:
            pck_all = sum(1 for e in all_px if e <= args.pck_px) / len(all_px)
            print(f"{name:<16}{'ALL':<5}{sum(all_px)/len(all_px):>9.1f}{max(all_px)/1:>9.1f}{sum(all_mm)/len(all_mm):>9.2f}{pck_all:>8.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
