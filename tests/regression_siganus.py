"""Siganus V1 迁移回归测试（迁移保真 A/B + 历史对照）。

背景：FishMorph Local 的流水线代码与权重自原项目字节级复制。
本脚本验证"执行级"迁移保真，并对照原项目 v0.6.4 e2e 历史记录。

用法（在项目根目录）：
    python tests/regression_siganus.py run --side D      # 用 FishMorph 副本代码跑
    python tests/regression_siganus.py run --side E      # 用原项目代码跑（只读执行，写盘重定向到本项目）
    python tests/regression_siganus.py compare           # A/B 对比（迁移保真门禁，应为 0 差异）
    python tests/regression_siganus.py historical        # 与 v0.6.4 e2e 历史值对照（信息性）

安全约束：
- E: 侧只读取代码与图像；导入前设置 sys.dont_write_bytecode 以免在原项目生成 __pycache__；
  预标注产物全部写入本项目 data/regression_outputs/。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PROJECT = Path("E:/1_yanjiusheng/SiganusMorph Local")
E2E_CSV = ORIGINAL_PROJECT / "results" / "e2e_regression_v0.6.4" / "test_measurements.csv"

IMAGES = ["real_089", "real_076", "real_042", "real_054", "real_028", "real_063"]
MM_PER_PIXEL = 0.1  # e2e 基线采用 board_mm_per_pixel=0.1
OUTPUT_ROOT = ROOT / "data" / "regression_outputs"

# Tier-2 对照的核心直线测量列（信息性，不受轴模式选择影响）
HISTORICAL_CORE_COLUMNS = [
    "SL_straight_mm",
    "TL_straight_mm",
    "body_depth_mm",
    "head_length_straight_mm",
    "snout_length_straight_mm",
    "caudal_peduncle_length_straight_mm",
    "caudal_peduncle_depth_mm",
]


def _require_dont_write_bytecode() -> None:
    sys.dont_write_bytecode = True


def load_pipeline(side: str):
    """按 side 导入对应代码位置的 siganusmorph 包。"""
    if side == "D":
        pkg_root = ROOT
    elif side == "E":
        pkg_root = ORIGINAL_PROJECT
        if not pkg_root.exists():
            raise SystemExit(f"原项目不存在: {pkg_root}")
    else:
        raise SystemExit(f"未知 side: {side}")
    _require_dont_write_bytecode()
    sys.path.insert(0, str(pkg_root))
    import siganusmorph  # noqa: E402

    from siganusmorph.measurements import calculate_measurements  # noqa: E402
    from siganusmorph.preannotation import preannotate_warped_image  # noqa: E402
    from siganusmorph.realworld_review import full_to_short_keypoints  # noqa: E402

    return siganusmorph, preannotate_warped_image, full_to_short_keypoints, calculate_measurements


def flatten(obj, prefix: str = "") -> dict:
    flat: dict = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            flat.update(flatten(value, f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            flat.update(flatten(value, f"{prefix}[{index}]"))
    else:
        flat[prefix] = obj
    return flat


def cmd_run(side: str) -> int:
    siganusmorph, preannotate_warped_image, full_to_short_keypoints, calculate_measurements = load_pipeline(side)
    from siganusmorph.image_utils import load_image_file  # noqa: E402  (PIL→RGB，与项目/e2e 输入约定一致)

    print(f"[{side}] siganusmorph 来自: {Path(siganusmorph.__file__).resolve().parent}")

    images_dir = ROOT / "data" / "regression_images"
    model_path = ROOT / "models" / "siganusmorph_yolopose_v0.3_corrected_real_5_15" / "preannotation_candidate.pt"

    # E: 侧用本项目内的假 project_root，确保 results/ 写盘全部落在 D:
    if side == "E":
        run_root = OUTPUT_ROOT / "_e_side_project_root"
    else:
        run_root = ROOT
    run_root.mkdir(parents=True, exist_ok=True)

    side_dir = OUTPUT_ROOT / side
    side_dir.mkdir(parents=True, exist_ok=True)

    for stem in IMAGES:
        image_path = images_dir / f"{stem}_warped.png"
        image = load_image_file(image_path)
        result = preannotate_warped_image(
            image,
            f"{stem}_warped.png",
            project_root=run_root,
            model_path=model_path,
        )
        keypoints_full = result.get("keypoints") or result.get("warped_keypoints") or {}
        short_keypoints = full_to_short_keypoints(keypoints_full)
        measurements = calculate_measurements(short_keypoints, mm_per_pixel=MM_PER_PIXEL, axis_mode_selected="auto")

        payload = {
            "side": side,
            "image": stem,
            "mm_per_pixel": MM_PER_PIXEL,
            "keypoints_full": {k: list(v) for k, v in keypoints_full.items()},
            "measurements": measurements,
            "segmentation_success": result.get("segmentation_success"),
            "qc": {
                k: result.get(k)
                for k in ("needs_review", "review_reason", "preannotation_reliability_level")
                if k in result
            },
        }
        out_path = side_dir / f"{stem}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[{side}] {stem}: {len(keypoints_full)} 关键点, SL={measurements.get('SL_final_mm')} mm, TL={measurements.get('TL_final_mm')} mm -> {out_path.name}")
    return 0


def _numeric_diff(a, b) -> float | None:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return None
    return abs(fa - fb)


def cmd_compare() -> int:
    worst = 0.0
    worst_where = ""
    total_fields = 0
    mismatched_non_numeric = []
    all_ok = True
    for stem in IMAGES:
        path_a = OUTPUT_ROOT / "E" / f"{stem}.json"
        path_b = OUTPUT_ROOT / "D" / f"{stem}.json"
        if not path_a.exists() or not path_b.exists():
            print(f"FAIL {stem}: 缺少输出文件 ({path_a.exists()=}, {path_b.exists()=})")
            all_ok = False
            continue
        data_a = flatten(json.loads(path_a.read_text(encoding="utf-8")))
        data_b = flatten(json.loads(path_b.read_text(encoding="utf-8")))
        keys = sorted(set(data_a) | set(data_b))
        image_worst, image_where = 0.0, ""
        for key in keys:
            if key in ("image", "side") or key.endswith(("/image", "/side")):
                continue
            va, vb = data_a.get(key), data_b.get(key)
            diff = _numeric_diff(va, vb)
            if diff is None:
                if json.dumps(va, default=str) != json.dumps(vb, default=str):
                    mismatched_non_numeric.append(f"{stem}:{key}")
                    all_ok = False
                continue
            total_fields += 1
            if diff > image_worst:
                image_worst, image_where = diff, key
            if diff > worst:
                worst, worst_where = diff, f"{stem}:{key}"
        status = "PASS" if image_worst <= 1e-9 else "FAIL"
        if image_worst > 1e-9:
            all_ok = False
        print(f"[{status}] {stem}: 数值字段最大差异 {image_worst:.3g}" + (f" @ {image_where}" if image_where else ""))
    print(f"\n数值字段总数 {total_fields}; 全局最大差异 {worst:.3g}" + (f" @ {worst_where}" if worst_where else ""))
    if mismatched_non_numeric:
        print(f"非数值字段不一致 {len(mismatched_non_numeric)} 处: {mismatched_non_numeric[:5]}")
    print("\n==== 迁移保真结论 ====")
    print("PASS: E: 原代码与 D: 副本在相同输入下输出完全一致" if all_ok and worst <= 1e-9 and not mismatched_non_numeric else "FAIL: 存在差异，需排查")
    return 0 if all_ok and worst <= 1e-9 and not mismatched_non_numeric else 1


def cmd_historical() -> int:
    import csv

    if not E2E_CSV.exists():
        raise SystemExit(f"找不到 e2e 基线 CSV: {E2E_CSV}")
    rows = {}
    with open(E2E_CSV, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[Path(row["image_name"]).stem] = row

    print(f"{'image':<10} {'column':<38} {'v0.6.4 e2e':>12} {'now':>12} {'delta':>10}")
    print("-" * 86)
    summary: dict[str, list[float]] = {}
    for stem in IMAGES:
        row = rows.get(stem)
        data = json.loads((OUTPUT_ROOT / "D" / f"{stem}.json").read_text(encoding="utf-8"))
        measurements = flatten(data["measurements"])
        if row is None:
            print(f"{stem:<10} (e2e 无此图)")
            continue
        for column in HISTORICAL_CORE_COLUMNS:
            old = row.get(column, "")
            new = measurements.get(column)
            diff = _numeric_diff(old, new)
            if diff is None:
                continue
            summary[column] = summary.get(column, []) + [diff]
            print(f"{stem:<10} {column:<38} {float(old):>12.3f} {float(new):>12.3f} {diff:>10.3f}")
        # 关键点坐标偏差（同 warp 坐标系）
        kp_diffs = []
        for key, coords in data["keypoints_full"].items():
            for axis, value in zip(("x", "y"), coords):
                old = row.get(f"{key}_{axis}", "")
                diff = _numeric_diff(old, value)
                if diff is not None:
                    kp_diffs.append(diff)
        if kp_diffs:
            print(f"{stem:<10} {'keypoints (max |delta| px)':<38} {'':>12} {'':>12} {max(kp_diffs):>10.3f}")
        print()
    print("==== 历史对照汇总（|平均偏差| mm，v0.6.4 → 当前 V1 含流水线演进）====")
    for column, diffs in summary.items():
        print(f"{column:<40} max={max(diffs):.3f}  mean={sum(diffs)/len(diffs):.3f}")
    print("\n说明：历史值为 v0.6.4 时期记录，期间存在 v0.6.5–v0.6.9 测量定义与尾部几何修正，")
    print("本对照仅作连续性/合理性参考；迁移保真以 compare 子命令的 A/B 零差异为准。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "compare", "historical"])
    parser.add_argument("--side", choices=["D", "E"], default="D")
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args.side)
    if args.command == "compare":
        return cmd_compare()
    return cmd_historical()


if __name__ == "__main__":
    raise SystemExit(main())
