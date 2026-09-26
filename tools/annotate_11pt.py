"""FishMorph 11 点标注工具（schema v2：8 universal core + 3 caudal-fin）。

独立的 Streamlit 小应用，不改动 V1 主程序。对 photo/<批次>/ 下的照片：
  1. 首次进入自动准备工作区：ArUco 透视校正（存 warped 图）+ V1 预标注 16 点
     映射为 11 点草案（P4/C1-C4 丢弃）；
  2. 逐图点击修正 11 个点，P6 可标记"不存在"（微叉/截形尾）；
  3. 每次点击即保存 JSON 到 data/annotation_workspace/<批次>/annotations/。

启动：
    python -m streamlit run tools/annotate_11pt.py --server.port 8504
或双击项目根目录 annotate_11pt.bat。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

import cv2
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from siganusmorph.components import image_clicker

# --- 11 点 schema 定义 -----------------------------------------------------------

SCHEMA_ID = "fishmorph_11pt_v1"

POINTS: list[dict] = [
    {"code": "P1", "name": "吻端", "hint": "鱼吻部最前端（主轴起点）。不要点嘴唇阴影或鼻孔。"},
    {"code": "P2", "name": "眼眶前缘", "hint": "眼眶最靠近吻端的一侧边缘。不要点瞳孔中心或反光点。"},
    {"code": "P3", "name": "鳃盖后缘", "hint": "鳃盖骨后缘最明显的位置。"},
    {"code": "P5", "name": "尾鳍基部中点", "hint": "鱼身与尾鳍交界处的中点（标准长终点）。不要点尾鳍叶尖。"},
    {"code": "P8", "name": "最大体高上点", "hint": "躯干最大体高处的背侧轮廓。不含背鳍。"},
    {"code": "P9", "name": "最大体高下点", "hint": "与 P8 对应的腹侧轮廓。不含腹鳍/臀鳍，尽量与 P8 垂直于局部中轴。"},
    {"code": "P10", "name": "尾柄最窄上点", "hint": "尾柄最窄截面的背侧轮廓。不含尾鳍。"},
    {"code": "P11", "name": "尾柄最窄下点", "hint": "尾柄最窄截面的腹侧轮廓。"},
    {"code": "P6", "name": "尾鳍分叉点（可选）", "hint": "上下叶分叉凹陷的中间点。微叉/截形尾可勾选“P6 不存在”。"},
    {"code": "P7U", "name": "尾鳍上叶末端", "hint": "尾鳍上叶可见最远端。"},
    {"code": "P7L", "name": "尾鳍下叶末端", "hint": "尾鳍下叶可见最远端。"},
]
POINT_ORDER = [p["code"] for p in POINTS]
HINTS = {p["code"]: p["hint"] for p in POINTS}
LABELS = {p["code"]: f'{p["code"]} {p["name"]}' for p in POINTS}

# V1 16 点预标注（复合格式 "P1_snout_tip"）→ 11 点代码
PREFILL_MAP = {
    "P1_snout_tip": "P1",
    "P2_eye_front": "P2",
    "P3_operculum_posterior": "P3",
    "P5_caudal_base_midpoint": "P5",
    "P8_body_depth_dorsal": "P8",
    "P9_body_depth_ventral": "P9",
    "P10_peduncle_depth_dorsal": "P10",
    "P11_peduncle_depth_ventral": "P11",
    "P6_caudal_fork_midpoint": "P6",
    "P7U_caudal_fin_upper_tip": "P7U",
    "P7L_caudal_fin_lower_tip": "P7L",
}

DISPLAY_WIDTH = 1180
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def imwrite_unicode(path: Path, bgr) -> None:
    """cv2.imwrite 对非 ASCII 路径不可靠（GBK 代码页侥幸），改用 imencode 落盘。"""
    ok, buf = cv2.imencode(path.suffix, bgr)
    if not ok:
        raise IOError(f"imencode failed: {path}")
    path.write_bytes(buf.tobytes())


def workspace_for(batch: str) -> Path:
    ws = ROOT / "data" / "annotation_workspace" / batch
    (ws / "warped").mkdir(parents=True, exist_ok=True)
    (ws / "annotations").mkdir(parents=True, exist_ok=True)
    return ws


def prep_workspace(batch: str, ws: Path) -> None:
    """透视校正 + 预标注，产出 warped 图与 11 点草案 JSON。"""
    from siganusmorph.aruco_utils import detect_aruco_markers, select_board_config_for_markers, warp_board_by_aruco
    from siganusmorph.image_utils import load_image_file
    from siganusmorph.preannotation import preannotate_warped_image

    model_path = ROOT / "models" / "siganusmorph_yolopose_v0.3_corrected_real_5_15" / "preannotation_candidate.pt"
    sources = sorted(p for p in (ROOT / "photo" / batch).iterdir() if p.suffix.lower() in IMG_EXTS)
    if not sources:
        st.error(f"photo/{batch} 下没有图片")
        return

    progress = st.progress(0.0, text="准备工作区：透视校正 + 预标注…")
    for i, src in enumerate(sources):
        stem = src.stem
        meta: dict = {"source": str(src), "mm_per_pixel": None, "space": "raw"}
        image = load_image_file(src)
        markers = detect_aruco_markers(image).get("markers", {})
        target = image
        if markers:
            try:
                board_config = select_board_config_for_markers(markers)
                warped, _t, info = warp_board_by_aruco(image, markers, board_config)
                mm = float(info.get("mm_per_pixel", 0.0)) or None
                if mm:
                    target = warped
                    meta["mm_per_pixel"] = mm
                    meta["space"] = "warped_board"
            except Exception as exc:  # noqa: BLE001
                meta["warp_error"] = repr(exc)

        warped_path = ws / "warped" / f"{stem}_warped.png"
        imwrite_unicode(warped_path, cv2.cvtColor(target, cv2.COLOR_RGB2BGR))

        result = preannotate_warped_image(target, f"{stem}_warped.png", project_root=ws / "_run_root", model_path=model_path)
        keypoints_full = result.get("keypoints") or result.get("warped_keypoints") or {}
        draft = {code: [float(keypoints_full[full][0]), float(keypoints_full[full][1])]
                 for full, code in PREFILL_MAP.items() if full in keypoints_full}
        record = {
            "schema": SCHEMA_ID,
            "image_stem": stem,
            "source": str(src),
            "warped_image": f"{stem}_warped.png",
            "coordinate_space": meta["space"],
            "mm_per_pixel": meta.get("mm_per_pixel"),
            "warped_size": [int(target.shape[1]), int(target.shape[0])],
            "points": draft,
            "p6_present": "P6" in draft,
            "status": "draft",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "notes": "",
            "warp_error": meta.get("warp_error"),
        }
        (ws / "annotations" / f"{stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        progress.progress((i + 1) / len(sources), text=f"准备完成 {i + 1}/{len(sources)}：{stem}")


def save_annotation(ws: Path, record: dict) -> None:
    record["updated_at"] = datetime.now().isoformat(timespec="seconds")
    (ws / "annotations" / f'{record["image_stem"]}.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def draw_overlay(warped_path: Path, points: dict, selected: str | None) -> Image.Image:
    image = Image.open(warped_path).convert("RGB")
    scale = DISPLAY_WIDTH / image.width
    display = image.resize((DISPLAY_WIDTH, round(image.height * scale)))
    draw = ImageDraw.Draw(display)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    for code, xy in points.items():
        x, y = xy[0] * scale, xy[1] * scale
        color = (255, 215, 0) if code == selected else (255, 255, 255)
        r = 6
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
        if code == selected:
            draw.line((x - 14, y, x + 14, y), fill=(255, 215, 0), width=2)
            draw.line((x, y - 14, x, y + 14), fill=(255, 215, 0), width=2)
        draw.text((x + 9, y - 24), code, fill=(255, 215, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return display


def main() -> None:
    st.set_page_config(page_title="FishMorph 11 点标注", layout="wide")
    st.title("FishMorph 11 点标注工具（schema v2）")
    st.caption("8 universal core + 3 caudal-fin · 先在左侧选中点，再点击画面放置 · 每次点击自动保存")

    photo_root = ROOT / "photo"
    batches = sorted(d.name for d in photo_root.iterdir() if d.is_dir()) if photo_root.exists() else []
    if not batches:
        st.error("photo/ 下没有批次目录")
        return
    batch = st.sidebar.selectbox("批次", batches, index=len(batches) - 1)
    ws = workspace_for(batch)

    if not (ws / "prep_done.json").exists():
        with st.spinner("首次进入：生成透视校正图与预标注草案（约 1–3 分钟）…"):
            prep_workspace(batch, ws)
        (ws / "prep_done.json").write_text(datetime.now().isoformat(), encoding="utf-8")

    records = {p.stem: json.loads(p.read_text(encoding="utf-8"))
               for p in sorted((ws / "annotations").glob("*.json"))}
    stems = sorted(records)
    if not stems:
        st.warning("工作区为空，请刷新页面重新准备。")
        return
    done = [s for s in stems if records[s].get("status") == "done"]
    st.sidebar.success(f"进度：{len(done)}/{len(stems)} 张完成")
    for s in stems:
        icon = "✅" if records[s].get("status") == "done" else "✏️"
        st.sidebar.markdown(f"{icon} {s}")

    default_index = min(len(stems) - 1, stems.index(done[-1]) + 1) if done else 0
    stem = st.sidebar.selectbox("选择图片", stems, index=default_index)
    record = records[stem]
    points: dict = record.setdefault("points", {})

    if "sel_point" not in st.session_state or st.session_state["sel_point"] not in POINT_ORDER:
        st.session_state["sel_point"] = next((c for c in POINT_ORDER if c not in points), POINT_ORDER[0])
    selected = st.sidebar.radio("当前编辑点", POINT_ORDER,
                                format_func=lambda c: LABELS[c],
                                index=POINT_ORDER.index(st.session_state["sel_point"]))
    st.session_state["sel_point"] = selected

    st.markdown(f"### {LABELS[selected]}")
    st.info(HINTS[selected])

    if selected == "P6":
        p6_present = st.checkbox("P6 分叉点存在（微叉/截形尾可取消勾选）", value=record.get("p6_present", True))
        if p6_present != record.get("p6_present", True):
            record["p6_present"] = p6_present
            if not p6_present:
                points.pop("P6", None)
            save_annotation(ws, record)

    warped_path = ws / "warped" / record["warped_image"]
    warped_width = Image.open(warped_path).width
    display_img = draw_overlay(warped_path, points, selected)
    value = image_clicker(display_img, key=f"clicker_{stem}")
    if value and value.get("x") is not None:
        click_key = (round(float(value["x"]), 1), round(float(value["y"]), 1))
        if click_key != st.session_state.get("last_click"):
            st.session_state["last_click"] = click_key
            scale = warped_width / display_img.width
            points[selected] = [round(click_key[0] * scale, 2), round(click_key[1] * scale, 2)]
            if selected == "P6":
                record["p6_present"] = True
            save_annotation(ws, record)
            nxt = POINT_ORDER.index(selected) + 1
            st.session_state["sel_point"] = POINT_ORDER[min(nxt, len(POINT_ORDER) - 1)]
            st.rerun()

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("↩️ 删除当前点"):
            points.pop(selected, None)
            save_annotation(ws, record)
            st.rerun()
    with col2:
        if st.button("✅ 本图完成/取消完成"):
            record["status"] = "draft" if record.get("status") == "done" else "done"
            save_annotation(ws, record)
            st.rerun()
    with col3:
        notes = st.text_input("备注", value=record.get("notes", ""))
        if notes != record.get("notes", ""):
            record["notes"] = notes
            save_annotation(ws, record)

    p6_active = record.get("p6_present", True)
    missing = [c for c in POINT_ORDER if c not in points and not (c == "P6" and not p6_active)]
    st.caption(
        f'已放置 {len(points)}/{"11" if p6_active else "10"}（P6 {"存在" if p6_active else "标记为不存在"}）'
        + (f" · 未放置：{'、'.join(missing)}" if missing else " · 全部就绪")
    )


if __name__ == "__main__":
    main()
