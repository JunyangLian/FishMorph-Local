"""构建 FishMorph 少样本 YOLO-pose 训练集（11 点 schema v2 / 16 点兼容）。

从 data/annotation_workspace/<batch>/annotations/*.json（status=done）生成
YOLO-pose 格式数据集：images/{train,val} + labels/{train,val} + data.yaml。

用法示例：
    # 学习曲线（多批次合并，N=5，其余作 val）
    python scripts/build_fewshot_dataset.py --batch 海鲈 海鲈2 --train-n 5 --seed 0 --crop
    # 全量档（train=val=全部，检查点选择偏乐观，仅作最终模型）
    python scripts/build_fewshot_dataset.py --batch 海鲈 海鲈2 --train-n 10 --crop

--crop：训练域改为分割裁剪域（与 V1 训练/推理链一致）；分割失败的单图回退全图。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

POINT_ORDER = ["P1", "P2", "P3", "P5", "P8", "P9", "P10", "P11", "P6", "P7U", "P7L"]
SCHEMA_16_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6", "P7U", "P7L", "P8", "P9", "P10", "P11", "C1", "C2", "C3", "C4"]
BBOX_PAD_RATIO = 0.2  # 极端点（P1/P7U/P7L）定义 bbox 边界，padding 需覆盖增广位移，避免关键点被推出框外后静默剔除


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, nargs="+", help="标注工作区批次名（可多个）")
    parser.add_argument("--train-n", type=int, default=3, help="训练集图片数；val/test 依次向后切")
    parser.add_argument("--val-n", type=int, default=2, help="验证集图片数（检查点选择）")
    parser.add_argument("--test-n", type=int, default=0, help="测试集图片数（不参与训练与检查点选择）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schema", choices=["11pt", "16pt"], default="11pt",
                        help="11pt=schema v2；16pt=兼容 Siganus 权重微调（P4/C1-C4 置 v=0，不参与损失）")
    parser.add_argument("--crop", action="store_true", help="训练域改为分割裁剪域（与 V1 链一致）")
    parser.add_argument("--out", default=None, help="输出目录")
    args = parser.parse_args()

    point_order = SCHEMA_16_ORDER if args.schema == "16pt" else POINT_ORDER
    kpt_shape = [len(point_order), 3]

    records = []
    for batch in args.batch:
        ws = ROOT / "data" / "annotation_workspace" / batch
        for path in sorted((ws / "annotations").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") == "done":
                record["_batch"] = batch
                records.append(record)
    if not records:
        raise SystemExit("没有 status=done 的标注")
    if args.train_n + args.val_n + args.test_n > len(records):
        raise SystemExit(f"--train-n {args.train_n} + --val-n {args.val_n} + --test-n {args.test_n} 超过已完成标注数 {len(records)}")

    stems = [r["image_stem"] for r in records]
    rng = random.Random(args.seed)
    rng.shuffle(stems)
    # test 永远从序列末尾切（跨 train-n 臂保持同一测试集）；val 紧邻其前；其余为 train
    test_stems = set(stems[len(stems) - args.test_n:]) if args.test_n else set()
    val_stems = set(stems[len(stems) - args.test_n - args.val_n: len(stems) - args.test_n]) if args.val_n else set()
    train_stems = set(stems[: args.train_n])
    if args.test_n == 0 and args.train_n + args.val_n == len(records):
        print("警告：train+val 等于总图数（无独立测试集）")

    tag = "-".join(args.batch)
    out = Path(args.out) if args.out else ROOT / "data" / "fewshot_datasets" / (
        f"{tag}_{args.schema}_crop_n{args.train_n}_s{args.seed}"
        if args.crop else f"{tag}_{args.schema}_n{args.train_n}_s{args.seed}"
    )
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    if args.crop:
        from siganusmorph.image_utils import load_image_file
        from siganusmorph.segmentation import padded_bbox, segment_fish_from_blue_board

    assignment = {"train": [], "val": [], "test": []}
    crop_stats = {"crop": 0, "full": 0}
    for record in records:
        stem = record["image_stem"]
        split = ("train" if stem in train_stems else
                 "val" if stem in val_stems else
                 "test" if stem in test_stems else None)
        if split is None:
            continue
        file_key = f"{record['_batch']}_{stem}"  # 跨批次 stem 可能重名，加前缀防覆盖
        img_src = ROOT / "data" / "annotation_workspace" / record["_batch"] / "warped" / record["warped_image"]
        img_dst = out / "images" / split / f"{file_key}.png"

        if args.crop:
            # 分割裁剪域：与 V1 推理链相同（分割 → 8% padding 裁剪），失败回退全图
            rgb = np.asarray(load_image_file(img_src))
            fish_mask, fish_bbox, quality = segment_fish_from_blue_board(rgb)
            height, width = rgb.shape[:2]
            if quality.get("segmentation_success", quality.get("success", False)) and fish_bbox is not None:
                px1, py1, px2, py2 = padded_bbox(fish_bbox, rgb.shape, padding_ratio=0.08)
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(width, px2), min(height, py2)
            else:
                px1, py1, px2, py2 = 0, 0, width, height
            # 保证所有手工点都在裁剪内，否则回退全图
            pts = record["points"]
            if all(px1 <= xy[0] < px2 and py1 <= xy[1] < py2 for xy in pts.values()):
                crop_stats["crop"] += 1
            else:
                px1, py1, px2, py2 = 0, 0, width, height
                crop_stats["full"] += 1
            crop = rgb[py1:py2, px1:px2]
            ok, buf = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2BGR))
            if not ok:
                raise IOError(f"imencode failed for {img_dst}")
            img_dst.write_bytes(buf.tobytes())
            width, height = crop.shape[1], crop.shape[0]
            shifted = {code: [xy[0] - px1, xy[1] - py1] for code, xy in pts.items()}
        else:
            shutil.copyfile(img_src, img_dst)
            width, height = record["warped_size"]
            shifted = record["points"]

        pts = shifted
        xs = [xy[0] for xy in pts.values()]
        ys = [xy[1] for xy in pts.values()]
        pad_x = (max(xs) - min(xs)) * BBOX_PAD_RATIO
        pad_y = (max(ys) - min(ys)) * BBOX_PAD_RATIO
        x0 = max(0.0, min(xs) - pad_x)
        y0 = max(0.0, min(ys) - pad_y)
        x1 = min(float(width), max(xs) + pad_x)
        y1 = min(float(height), max(ys) + pad_y)
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        bw, bh = (x1 - x0) / width, (y1 - y0) / height

        kpts = []
        for code in point_order:
            if code in pts:
                kx, ky = pts[code]
                kpts.extend([kx / width, ky / height, 2])
            else:  # 11pt：P6 缺席；16pt：P4/C1-C4 无手工标注，置 v=0 不参与损失
                kpts.extend([0.0, 0.0, 0])
        label = "0 " + " ".join(f"{v:.6f}" for v in (cx, cy, bw, bh)) + " " + " ".join(f"{v:.6f}" for v in kpts)
        (out / "labels" / split / f"{file_key}.txt").write_text(label + "\n", encoding="utf-8")
        assignment[split].append(f"{record['_batch']}/{stem}")

    if not assignment["val"]:  # 无独立 val 时复制 train，保证 ultralytics 校验可用
        for src in sorted((out / "images" / "train").iterdir()):
            shutil.copyfile(src, out / "images" / "val" / src.name)
        for src in sorted((out / "labels" / "train").iterdir()):
            shutil.copyfile(src, out / "labels" / "val" / src.name)
        assignment["val"] = list(assignment["train"])

    data_yaml = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n"
        f"kpt_shape: [{kpt_shape[0]}, {kpt_shape[1]}]\n"
        "names:\n  0: fish\n"
    )
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"数据集输出: {out}")
    if args.crop:
        print(f"裁剪域统计: {crop_stats['crop']} 张裁剪 / {crop_stats['full']} 张全图回退")
    print(f"schema: {args.schema} · kpt_shape: {kpt_shape}（点序: {' '.join(point_order)}）")
    print(f"train ({len(assignment['train'])}): {', '.join(assignment['train'])}")
    print(f"val   ({len(assignment['val'])}): {', '.join(assignment['val'])}")
    print(f"test  ({len(assignment['test'])}): {', '.join(assignment['test'])}")
    (out / "split_manifest.json").write_text(
        json.dumps({"batch": args.batch, "seed": args.seed, "train_n": args.train_n,
                    "val_n": args.val_n, "test_n": args.test_n,
                    "schema": args.schema, "crop": args.crop,
                    "assignment": assignment, "point_order": point_order},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
