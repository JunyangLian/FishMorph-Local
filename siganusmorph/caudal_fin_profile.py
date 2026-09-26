"""Caudal-fin profile：从掩膜几何推导尾鳍三点（P6/P7U/P7L）。

这是 FishMorph 第一个 per-species 模块：关键点回归对尾叶尖不可靠
（底座模型的 P7U/P7L 通道退化，且半透明鳍膜与板颜色难分），
因此尾点改由"分割掩膜 + 几何规则"推导，参数按物种配置。

海鲈（AI 图）profile 的像素依据（2026-09-26 实测）：
  - 鳍组织：hue 101–106，V 63–152（半透明鳍膜透出暗蓝）
  - 板面：  hue 99–100，V 179–212（亮蓝）
  → 按"蓝窗内、V 低于亮度阈值"区分鳍与板。

流程（输入：校正图 RGB + 常规蓝板分割的体掩膜 + 已知 P5）：
  1. 尾区 = P5.x 往前 5% 体宽的锚线 → 板内右缘（覆盖整个尾鳍，含分叉）；
  2. 尾区鳍组织 = 体掩膜 ∪（蓝窗内但 V < board_value_max 的暗部）；
  3. 合并掩膜的轮廓：右段上极值 = P7U，下极值 = P7L；
  4. P6 = P7U/P7L 之间轮廓上距 P5 最近的点（凹陷最深处）；
  5. QC：P6 凹陷深度不足 / 掩膜不过 P5 时降级标记。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

SEA_BASS_PROFILE = {
    "species_id": "seabass",
    "hue_min": 82,
    "hue_max": 132,
    "sat_min": 35,
    "board_value_min": 170,     # V ≥ 此值视为亮蓝板面
    "region_back_margin": 0.05,  # 尾区锚线：P5.x 前退 5% 体宽
    "lobe_band_px": 60,          # 判定极值时的右缘带宽度（按图像素，可按 mm/px 换算）
    "notch_min_depth_px": 30,    # P6 凹陷最低深度
}


def derive_caudal_points(
    warped_rgb: np.ndarray,
    body_mask: np.ndarray,
    p5: tuple[float, float],
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """返回 {"P6": (x,y) | None, "P7U": ..., "P7L": ..., "quality": {...}}，坐标为全图像素。"""
    prof = {**SEA_BASS_PROFILE, **(profile or {})}
    height, width = warped_rgb.shape[:2]
    hsv = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # 1) 尾区：P5 锚线 → 图右缘
    margin = int(prof["region_back_margin"] * width)
    tx0 = max(0, int(p5[0] - margin))
    tx1 = width

    # 2) 尾区鳍组织：体掩膜 ∪（蓝窗内的暗部 = 半透明鳍/暗鳍）
    fin_dark = ((h >= prof["hue_min"]) & (h <= prof["hue_max"]) &
                (s >= prof["sat_min"]) & (v < prof["board_value_min"])).astype(np.uint8) * 255
    region_body = np.zeros_like(body_mask)
    region_body[:, tx0:tx1] = body_mask[:, tx0:tx1]
    region_fin = np.zeros_like(body_mask)
    region_fin[:, tx0:tx1] = fin_dark[:, tx0:tx1]
    combined = ((region_body > 0) | (region_fin > 0)).astype(np.uint8) * 255
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=2)
    combined = _fill_holes_u8(combined)
    # 保留与鱼体相连的部分（剔除远处噪声）
    n, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    keep = np.zeros_like(combined)
    body_seed_y, body_seed_x = int(p5[1]), min(width - 1, int(p5[0]))
    body_label = labels[body_seed_y, body_seed_x] if body_seed_y < height else 0
    if body_label > 0:
        keep[labels == body_label] = 255
    combined = keep

    ys, xs = np.where(combined > 0)
    quality: dict[str, Any] = {"mask_right_edge": int(xs.max()) if len(xs) else None,
                               "region_x0": tx0}
    if len(xs) == 0:
        return {"P6": None, "P7U": None, "P7L": None, "quality": {**quality, "status": "empty_tail_mask"}}

    # 3) 轮廓右段上下极值 = P7U / P7L
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"P6": None, "P7U": None, "P7L": None, "quality": {**quality, "status": "no_contour"}}
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2)  # (x, y)
    right_band = max(20, prof["lobe_band_px"])
    right_pts = pts[pts[:, 0] >= pts[:, 0].max() - right_band]
    p7u_idx = int(np.argmin(right_pts[:, 1]))
    p7l_idx = int(np.argmax(right_pts[:, 1]))
    p7u = (float(right_pts[p7u_idx, 0]), float(right_pts[p7u_idx, 1]))
    p7l = (float(right_pts[p7l_idx, 0]), float(right_pts[p7l_idx, 1]))

    # 4) P6 = 两叶之间的轮廓点中距 P5 最近者
    band_pts = pts[(pts[:, 0] >= min(p7u[0], p7l[0]) - 10) | (pts[:, 0] >= p5[0])]
    lobe_y_min, lobe_y_max = min(p7u[1], p7l[1]), max(p7u[1], p7l[1])
    between = band_pts[(band_pts[:, 1] > lobe_y_min + 5) & (band_pts[:, 1] < lobe_y_max - 5)]
    p6 = None
    notch_depth = None
    if len(between):
        dists = np.hypot(between[:, 0] - p5[0], between[:, 1] - p5[1])
        idx = int(np.argmin(dists))
        p6 = (float(between[idx, 0]), float(between[idx, 1]))
        # 凹陷深度 = 叶尖连线中点到 P6 的距离（在尾轴方向的近似）
        mid = ((p7u[0] + p7l[0]) / 2, (p7u[1] + p7l[1]) / 2)
        notch_depth = float(np.hypot(mid[0] - p6[0], mid[1] - p6[1]))
    if p6 is not None and notch_depth is not None and notch_depth < prof["notch_min_depth_px"]:
        quality["p6_note"] = f"notch_shallow_{notch_depth:.0f}px"
    quality["notch_depth_px"] = notch_depth
    quality["status"] = "ok"
    return {"P6": p6, "P7U": p7u, "P7L": p7l, "quality": quality}


def _fill_holes_u8(mask: np.ndarray) -> np.ndarray:
    ff = mask.copy()
    h, w = mask.shape
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(mask, holes)
