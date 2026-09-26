"""少样本模型评估：微调模型 vs Siganus V1 零样本，在验证集上对比 11 点误差。

用法：
    python scripts/eval_fewshot_predictions.py --dataset data/fewshot_datasets/海鲈2_n3_s0 \
        --weights runs/pose/fewshot_n3/weights/best.pt [--imgsz 960]

指标：逐点平均/最大像素误差与毫米误差（mm/px 取自标注 JSON）、PCK@阈值。
基线：models/siganusmorph_yolopose_v0.3_corrected_real_5_15/preannotation_candidate.pt
（16 点零样本，按 code 映射到 11 点）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

POINT_ORDER = ["P1", "P2", "P3", "P5", "P8", "P9", "P10", "P11", "P6", "P7U", "P7L"]
SIGANUS_16_TO_11 = {
    "P1_snout_tip": "P1", "P2_eye_front": "P2", "P3_operculum_posterior": "P3",
    "P5_caudal_base_midpoint": "P5", "P8_body_depth_dorsal": "P8",
    "P9_body_depth_ventral": "P9", "P10_peduncle_depth_dorsal": "P10",
    "P11_peduncle_depth_ventral": "P11", "P6_caudal_fork_midpoint": "P6",
    "P7U_caudal_fin_upper_tip": "P7U", "P7L_caudal_fin_lower_tip": "P7L",
}
SIGANUS_CODES = ["P1", "P2", "P3", "P4", "P5", "P6", "P7U", "P7L", "P8", "P9", "P10", "P11", "C1", "C2", "C3", "C4"]


def predict_siganus_v1_chain(weights: Path, image_path: Path, run_root: Path) -> dict[str, tuple[float, float]] | None:
    """V1 完整链路（分割→裁剪→YOLO→掩膜校正），16 点复合格式 → 11 点码。"""
    import cv2
    from siganusmorph.image_utils import load_image_file
    from siganusmorph.preannotation import preannotate_warped_image

    image = load_image_file(image_path)
    result = preannotate_warped_image(image, image_path.stem, project_root=run_root, model_path=weights)
    keypoints_full = result.get("keypoints") or result.get("warped_keypoints") or {}
    out = {}
    for full, code in SIGANUS_16_TO_11.items():
        if full in keypoints_full:
            out[code] = (float(keypoints_full[full][0]), float(keypoints_full[full][1]))
    return out or None


def predict_keypoints(weights: Path, image_path: Path, imgsz: int) -> dict[str, tuple[float, float]] | None:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.predict(str(image_path), imgsz=imgsz, conf=0.05, max_det=1, verbose=False)
    if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
        return None
    r = results[0]
    xy = r.keypoints.xy.cpu().numpy()[0]
    n = len(SIGANUS_CODES) if len(xy) == len(SIGANUS_CODES) else len(POINT_ORDER)
    codes = SIGANUS_CODES if n == len(SIGANUS_CODES) else POINT_ORDER
    return {code: (float(p[0]), float(p[1])) for code, p in zip(codes, xy)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="build_fewshot_dataset.py 的输出目录")
    parser.add_argument("--weights", required=True, help="微调后的 best.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--pck-px", type=float, default=15.0, help="PCK 阈值（像素，≈1.5mm @0.1mm/px）")
    parser.add_argument("--split", choices=["val", "train", "both"], default="val")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    manifest = json.loads((dataset / "split_manifest.json").read_text(encoding="utf-8"))
    ws = ROOT / "data" / "annotation_workspace" / manifest["batch"]
    weights = Path(args.weights)
    siganus_base = ROOT / "models" / "siganusmorph_yolopose_v0.3_corrected_real_5_15" / "preannotation_candidate.pt"

    rows = []  # (code, err_px, err_mm, model)
    for split in ("val", "train"):
        if args.split != "both" and split != args.split:
            continue
        for stem in manifest["assignment"][split]:
            record = json.loads((ws / "annotations" / f"{stem}.json").read_text(encoding="utf-8"))
            mm_px = record.get("mm_per_pixel") or 0.1
            truth = record["points"]
            image_path = dataset / "images" / split / f"{stem}.png"

            for label, weights_path in (("finetuned", weights), ("zeroshot_v1", siganus_base)):
                if label == "zeroshot_v1":
                    pred = predict_siganus_v1_chain(siganus_base, image_path, ROOT / "data" / "eval_run_root")
                else:
                    pred = predict_keypoints(weights_path, image_path, args.imgsz)
                if pred is None:
                    print(f"[{split}] {stem} [{label}]: 未检出鱼体")
                    continue
                for code in POINT_ORDER:
                    if code not in truth or code not in pred:  # P6 缺席 / 16 点模型缺 C 点映射
                        continue
                    err = float(np.hypot(pred[code][0] - truth[code][0], pred[code][1] - truth[code][1]))
                    rows.append((code, err, err * mm_px, label))

    print(f"\n{'模型':<14}{'点':<5}{'平均px':>9}{'最大px':>9}{'平均mm':>9}{'PCK':>8}")
    summary: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for code, err_px, err_mm, label in rows:
        summary.setdefault((label, code), []).append((err_px, err_mm))
    for label in ("finetuned", "zeroshot_v1"):
        all_px, all_mm = [], []
        for code in POINT_ORDER:
            errs = summary.get((label, code), [])
            if not errs:
                continue
            px = [e[0] for e in errs]
            mm = [e[1] for e in errs]
            all_px.extend(px)
            all_mm.extend(mm)
            pck = sum(1 for e in px if e <= args.pck_px) / len(px)
            print(f"{label:<14}{code:<5}{sum(px)/len(px):>9.1f}{max(px):>9.1f}{sum(mm)/len(mm):>9.2f}{pck:>8.0%}")
        if all_px:
            pck_all = sum(1 for e in all_px if e <= args.pck_px) / len(all_px)
            print(f"{label:<14}{'ALL':<5}{sum(all_px)/len(all_px):>9.1f}{max(all_px):>9.1f}{sum(all_mm)/len(all_mm):>9.2f}{pck_all:>8.0%}")
    print(f"\nPCK 阈值 {args.pck_px:.0f}px；误差为预测点与人工标注的欧氏距离。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
