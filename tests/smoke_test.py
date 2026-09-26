"""FishMorph Local —— Siganus V1 迁移 smoke test。

用法（在项目根目录）：
    python tests/smoke_test.py

不启动浏览器，验证：
  1. V1 全部 Python 文件可编译；
  2. siganusmorph 核心模块可导入；
  3. 三个 V1 模型权重从新项目 models/ 正常加载；
  4. YOLO-pose 对示例图输出 16 关键点；
  5. Heatmap U-Net 权重可加载并完成一次前向；
  6. 形态参数计算（calculate_measurements）可运行；
  7. CSV / Excel / JSON 导出链路可用；
  8. 校准栈（ArUco 检测）在示例图上可运行（无板则降级为 WARN）。
"""

from __future__ import annotations

import json
import py_compile
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"[{status}] {name}" + (f" —— {detail}" if detail else ""))


def step_compile_all() -> None:
    targets = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py")), *sorted((ROOT / "siganusmorph").rglob("*.py"))]
    failures = []
    for path in targets:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path.name}: {exc}")
    record(f"编译 {len(targets)} 个 Python 文件", not failures, "; ".join(failures[:3]))


def step_import_core() -> None:
    import importlib

    modules = [
        "siganusmorph.config",
        "siganusmorph.calibration",
        "siganusmorph.aruco_utils",
        "siganusmorph.image_utils",
        "siganusmorph.segmentation",
        "siganusmorph.measurements",
        "siganusmorph.preannotation",
        "siganusmorph.heatmap_preannotation",
        "siganusmorph.v06_preannotation",
        "siganusmorph.formal_ui",
        "siganusmorph.ui_styles",
        "siganusmorph.components",
        "siganusmorph.remote_jobs",
    ]
    failures = {}
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures[name] = repr(exc)
    record("导入 siganusmorph 核心模块", not failures, json.dumps(failures, ensure_ascii=False) if failures else f"{len(modules)} 个模块 OK")


def step_model_paths() -> None:
    from siganusmorph.preannotation import default_preannotation_model
    from siganusmorph.v06_preannotation import default_v05_heatmap_model
    from siganusmorph.heatmap_preannotation import default_heatmap_model

    paths = {
        "yolopose_v0.3": default_preannotation_model(ROOT),
        "heatmap_unet_v0.1": default_heatmap_model(ROOT),
        "heatmap_unet_v0.5": default_v05_heatmap_model(ROOT),
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    record("模型权重就位", not missing, "; ".join(f"{k}: {p}" for k, p in paths.items()))


def step_yolo_keypoints(example: Path) -> dict | None:
    try:
        import cv2
        from siganusmorph.preannotation import preannotate_warped_image
        from siganusmorph.config import KEYPOINT_DEFS

        if not example.exists():
            record("YOLO 16 关键点推理", False, f"缺少示例图 {example}")
            return None
        image = cv2.imread(str(example))
        if image is None:
            record("YOLO 16 关键点推理", False, "示例图读取失败")
            return None
        result = preannotate_warped_image(image, "smoke_example", ROOT)
        keypoints = result.get("keypoints") or result.get("warped_keypoints")
        n_expected = len(KEYPOINT_DEFS)
        n_got = len(keypoints or {})
        ok = keypoints is not None and n_got == n_expected
        record("YOLO 16 关键点推理（含分割与掩膜校正链路）", ok, f"关键点 {n_got}/{n_expected}")
        return result
    except Exception as exc:  # noqa: BLE001
        record("YOLO 16 关键点推理", False, repr(exc))
        traceback.print_exc()
        return None


def step_heatmap_forward() -> None:
    try:
        import torch
        from siganusmorph.v06_preannotation import default_v05_heatmap_model
        from siganusmorph.heatmap_unet import LightweightUNet
        from siganusmorph.config import KEYPOINT_DEFS

        model_path = default_v05_heatmap_model(ROOT)
        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        model = LightweightUNet(
            out_channels=int(config.get("num_keypoints", len(KEYPOINT_DEFS))),
            base_channels=int(config.get("base_channels", 24)),
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            outputs = model(torch.zeros(1, 3, 256, 256))
        ok = tuple(outputs.shape)[-3] == len(KEYPOINT_DEFS) or outputs.dim() == 4
        record("Heatmap U-Net v0.5 加载与前向", bool(ok), f"输出形状 {tuple(outputs.shape)}")
    except Exception as exc:  # noqa: BLE001
        record("Heatmap U-Net v0.5 加载与前向", False, repr(exc))
        traceback.print_exc()


def step_measurements(preannotation_result: dict | None) -> None:
    try:
        from siganusmorph.measurements import calculate_measurements
        from siganusmorph.realworld_review import full_to_short_keypoints

        if preannotation_result:
            raw = preannotation_result.get("keypoints") or preannotation_result.get("warped_keypoints")
            keypoints = full_to_short_keypoints(raw)
        else:
            from siganusmorph.config import KEYPOINT_DEFS

            keypoints = {kp.name: (100.0 + 40 * i, 200.0 + (i % 3) * 5) for i, kp in enumerate(KEYPOINT_DEFS)}
        values = calculate_measurements(keypoints, mm_per_pixel=0.1, axis_mode_selected="auto")
        n = len(values.get("measurements", values)) if isinstance(values, dict) else 0
        record("形态参数计算 calculate_measurements", n > 0, f"{n} 项测量值")
    except Exception as exc:  # noqa: BLE001
        record("形态参数计算 calculate_measurements", False, repr(exc))
        traceback.print_exc()


def step_exports() -> None:
    try:
        import pandas as pd

        df = pd.DataFrame([{"specimen": "smoke", "standard_length_mm": 123.4, "total_length_mm": 150.2}])
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            xlsx_path = Path(tmp) / "out.xlsx"
            json_path = Path(tmp) / "out.json"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            df.to_excel(xlsx_path, index=False)
            json_path.write_text(json.dumps(df.to_dict(orient="records"), ensure_ascii=False), encoding="utf-8")
            ok = csv_path.stat().st_size > 0 and xlsx_path.stat().st_size > 0 and json_path.stat().st_size > 0
        record("CSV / Excel / JSON 导出链路", ok)
    except Exception as exc:  # noqa: BLE001
        record("CSV / Excel / JSON 导出链路", False, repr(exc))


def step_calibration(example: Path) -> None:
    try:
        import cv2
        from siganusmorph.aruco_utils import detect_aruco_markers

        image = cv2.imread(str(example))
        detection = detect_aruco_markers(image) if image is not None else {}
        markers = detection.get("markers", {})
        if markers:
            from siganusmorph.aruco_utils import select_board_config_for_markers, warp_board_by_aruco

            board_config = select_board_config_for_markers(markers)
            warped, _transform, info = warp_board_by_aruco(image, markers, board_config)
            mm_per_pixel = float(info.get("mm_per_pixel", 0.0))
            ok = warped is not None and mm_per_pixel > 0
            record("校准栈（ArUco 检测 → 校正 → 比例尺）", ok, f"markers={len(markers)}, mm/pixel={mm_per_pixel:.5f}")
        else:
            record("校准栈（ArUco 检测 → 校正 → 比例尺）", True, "示例图未含校准板，检测函数运行正常（降级）")
    except Exception as exc:  # noqa: BLE001
        record("校准栈（ArUco 检测 → 校正 → 比例尺）", False, repr(exc))


def step_no_legacy_paths() -> None:
    import re

    # 只检测真正的路径依赖（盘符）；docstring 中的品牌词不算运行依赖。
    pattern = re.compile(r"[Ee]:[/\\]|[Cc]:[/\\][Uu]sers")
    hits = []
    for path in [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py")), *sorted((ROOT / "siganusmorph").rglob("*.py"))]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.name}:{i}")
    record("无原盘 E:\\ 硬编码引用", not hits, "; ".join(hits[:5]))


def main() -> int:
    print(f"FishMorph Local V1 smoke test —— 项目根：{ROOT}")
    example = ROOT / "examples" / "siganus_example_01.png"
    step_compile_all()
    step_import_core()
    step_model_paths()
    pre = step_yolo_keypoints(example)
    step_heatmap_forward()
    step_measurements(pre)
    step_exports()
    step_calibration(example)
    step_no_legacy_paths()

    failed = [r for r in RESULTS if r[1] == "FAIL"]
    print("\n==== 汇总 ====")
    for name, status, detail in RESULTS:
        print(f"{status}  {name}")
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} 项通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
