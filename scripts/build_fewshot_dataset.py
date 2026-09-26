"""构建 FishMorph 少样本 YOLO-pose 训练集（11 点 schema v2）。

从 data/annotation_workspace/<batch>/annotations/*.json（status=done）生成
YOLO-pose 格式数据集：images/{train,val} + labels/{train,val} + data.yaml。

用法：
    python scripts/build_fewshot_dataset.py --batch 海鲈2 --train-n 3 --seed 0
学习曲线实验：固定 --seed，改变 --train-n（3/5/10…），val 集始终为剩余的同一批图。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

POINT_ORDER = ["P1", "P2", "P3", "P5", "P8", "P9", "P10", "P11", "P6", "P7U", "P7L"]
SCHEMA_16_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6", "P7U", "P7L", "P8", "P9", "P10", "P11", "C1", "C2", "C3", "C4"]
KPT_SHAPE = [len(POINT_ORDER), 3]  # (x, y, v)
BBOX_PAD_RATIO = 0.12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, help="标注工作区批次名（data/annotation_workspace/<batch>）")
    parser.add_argument("--train-n", type=int, default=3, help="训练集图片数，其余作 val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schema", choices=["11pt", "16pt"], default="11pt",
                        help="11pt=schema v2；16pt=兼容 Siganus 权重微调（P4/C1-C4 置 v=0，不参与损失）")
    parser.add_argument("--out", default=None, help="输出目录（默认 data/fewshot_datasets/<batch>_n<train-n>）")
    args = parser.parse_args()

    point_order = SCHEMA_16_ORDER if args.schema == "16pt" else POINT_ORDER
    kpt_shape = [len(point_order), 3]

    ws = ROOT / "data" / "annotation_workspace" / args.batch
    records = []
    for path in sorted((ws / "annotations").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "done":
            records.append(record)
    if not records:
        raise SystemExit("没有 status=done 的标注")
    if args.train_n >= len(records):
        raise SystemExit(f"--train-n={args.train_n} 必须小于已完成标注数 {len(records)}（至少留 1 张 val）")

    stems = [r["image_stem"] for r in records]
    rng = random.Random(args.seed)
    rng.shuffle(stems)
    train_stems = set(stems[: args.train_n])

    out = Path(args.out) if args.out else ROOT / "data" / "fewshot_datasets" / f"{args.batch}_{args.schema}_n{args.train_n}_s{args.seed}"
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    assignment = {"train": [], "val": []}
    for record in records:
        stem = record["image_stem"]
        split = "train" if stem in train_stems else "val"
        img_src = ws / "warped" / record["warped_image"]
        img_dst = out / "images" / split / f"{stem}.png"
        shutil.copyfile(img_src, img_dst)

        width, height = record["warped_size"]
        pts = record["points"]
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
        (out / "labels" / split / f"{stem}.txt").write_text(label + "\n", encoding="utf-8")
        assignment[split].append(stem)

    data_yaml = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n"
        f"kpt_shape: [{kpt_shape[0]}, {kpt_shape[1]}]\n"
        "names:\n  0: fish\n"
    )
    (out / "data.yaml").write_text(data_yaml, encoding="utf-8")

    print(f"数据集输出: {out}")
    print(f"schema: {args.schema} · kpt_shape: {kpt_shape}（点序: {' '.join(point_order)}）")
    print(f"train ({len(assignment['train'])}): {', '.join(assignment['train'])}")
    print(f"val   ({len(assignment['val'])}): {', '.join(assignment['val'])}")
    (out / "split_manifest.json").write_text(
        json.dumps({"batch": args.batch, "seed": args.seed, "train_n": args.train_n,
                    "schema": args.schema, "assignment": assignment, "point_order": point_order},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
