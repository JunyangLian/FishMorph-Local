"""FishMorph zero-shot 试测：用 Siganus V1 流水线跑非蓝子鱼照片（当前：海鲈）。

只观察、不接入系统。对指定目录的每张照片：
  1. load_image_file（RGB）；
  2. ArUco 校准板检测 → 命中则透视校正并得到 mm_per_pixel；
  3. preannotate_warped_image 全链路（分割→YOLO 16 点→掩膜校正→尾部几何→QC）；
  4. 有比例尺则计算形态参数，无则只输出像素坐标；
  5. draw_keypoints_and_measurements 生成标注预览图。

用法：
    python tests/zeroshot_trial.py --dir "photo/海鲈"
输出：data/zeroshot_trials/<目录名>/（标注预览 + summary.json + 控制台表格）
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="照片目录（相对项目根或绝对路径）")
    args = parser.parse_args()

    from siganusmorph.aruco_utils import detect_aruco_markers, draw_detected_aruco, select_board_config_for_markers, warp_board_by_aruco
    from siganusmorph.image_utils import load_image_file
    from siganusmorph.measurements import calculate_measurements
    from siganusmorph.preannotation import preannotate_warped_image
    from siganusmorph.realworld_review import full_to_short_keypoints
    from siganusmorph.visualization import draw_keypoints_and_measurements

    src_dir = Path(args.dir)
    if not src_dir.is_absolute():
        src_dir = ROOT / src_dir
    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"目录中没有图片: {src_dir}")

    trial_dir = ROOT / "data" / "zeroshot_trials" / src_dir.name
    trial_dir.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / "models" / "siganusmorph_yolopose_v0.3_corrected_real_5_15" / "preannotation_candidate.pt"

    print(f"zero-shot 试测：{src_dir} -> {trial_dir}")
    header = f"{'image':<28} {'board':<14} {'mm/px':<8} {'seg':<5} {'kp':<5} {'QC':<6} {'SL(mm)':<9} {'TL(mm)':<9}"
    print(header)
    print("-" * len(header))
    summary = []
    for path in images:
        record: dict = {"image": path.name}
        try:
            image = load_image_file(path)

            # 1) 校准板检测
            detection = detect_aruco_markers(image)
            markers = detection.get("markers", {})
            record["aruco_ids"] = sorted(int(i) for i in markers)
            board_label = f"{len(markers)} markers"
            warped, mm_per_pixel = None, None
            if markers:
                try:
                    board_config = select_board_config_for_markers(markers)
                    warped, _transform, info = warp_board_by_aruco(image, markers, board_config)
                    mm_per_pixel = float(info.get("mm_per_pixel", 0.0)) or None
                    board_label += f"->warp"
                except Exception as warp_error:  # 板型不匹配时退回原图
                    board_label += f" (warp fail: {type(warp_error).__name__})"
            record["mm_per_pixel"] = mm_per_pixel
            target = warped if warped is not None else image

            # 2) 全链路预标注（写盘重定向到试测目录，避免污染 results/）
            run_root = trial_dir / "_run_root"
            result = preannotate_warped_image(target, path.stem, project_root=run_root, model_path=model_path)
            keypoints_full = result.get("keypoints") or result.get("warped_keypoints") or {}
            short_keypoints = full_to_short_keypoints(keypoints_full)
            record["keypoints_px"] = {k: list(v) for k, v in keypoints_full.items()}
            record["segmentation_success"] = result.get("segmentation_success")
            record["needs_review"] = result.get("needs_review")
            record["review_reason"] = result.get("review_reason")

            # 3) 测量（仅有比例尺时毫米值才有物理意义）
            sl = tl = None
            if mm_per_pixel and len(short_keypoints) >= 16:
                measurements = calculate_measurements(short_keypoints, mm_per_pixel=mm_per_pixel, axis_mode_selected="auto")
                sl = measurements.get("SL_final_mm")
                tl = measurements.get("TL_final_mm")
                record["measurements_mm"] = {k: v for k, v in measurements.items() if isinstance(v, (int, float))}

            # 4) 标注预览（draw_keypoints_and_measurements 使用短键名 + 测量值）
            draw_measurements = record.get("measurements_mm")
            preview = draw_keypoints_and_measurements(target, short_keypoints, draw_measurements, image_name=path.stem)
            preview_path = trial_dir / f"{path.stem}_annotated.png"
            ok, buf = cv2.imencode(".png", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            if not ok:
                raise IOError("imencode preview failed")
            preview_path.write_bytes(buf.tobytes())

            seg_flag = "OK" if result.get("segmentation_success") else "FAIL"
            qc_flag = "review" if result.get("needs_review") else "ok"
            print(
                f"{path.stem:<28.28} {board_label:<14.14} {mm_per_pixel if mm_per_pixel else '-':<8.5} "
                f"{seg_flag:<5} {len(keypoints_full):<5} {qc_flag:<6} "
                f"{sl if sl is not None else '-':<9.2} {tl if tl is not None else '-':<9.2}"
            )
        except Exception as exc:  # noqa: BLE001
            record["error"] = repr(exc)
            traceback.print_exc()
            print(f"{path.stem:<28.28} ERROR: {exc!r}")
        summary.append(record)

    (trial_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n汇总已写入 {trial_dir / 'summary.json'}；标注预览图同目录 *_annotated.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
