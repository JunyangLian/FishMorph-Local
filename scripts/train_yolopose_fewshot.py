"""少样本 YOLO-pose 训练（11 点 schema v2）。

用法：
    python scripts/train_yolopose_fewshot.py --data data/fewshot_datasets/海鲈2_n3_s0/data.yaml \
        --epochs 60 --imgsz 960 --name fewshot_n3

底座默认 models/pretrained/yolo11n-pose.pt（COCO 预训练，kpt 头按 data.yaml 重建）。
产物位于 runs/pose/<name>/weights/best.pt。
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base", default="models/pretrained/yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default=None, help="0 / cpu，默认自动")
    parser.add_argument("--name", default="fewshot")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not Path(args.base).exists():
        raise SystemExit(f"底座权重不存在: {args.base}（原项目根目录有 yolo11n-pose.pt，复制到 models/pretrained/）")

    from ultralytics import YOLO

    if args.device is None:
        import torch

        args.device = 0 if torch.cuda.is_available() else "cpu"
    print(f"device={args.device} base={args.base} data={args.data}")

    model = YOLO(str(Path(args.base).resolve()))
    model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        seed=args.seed,
        workers=0,  # Windows + 小数据集避免 dataloader 多进程问题
        exist_ok=True,
    )
    weights_dir = ROOT / "runs" / "pose" / args.name / "weights"
    best = weights_dir / "best.pt"
    print(f"\n训练完成，最优权重: {best if best.exists() else '(未生成，检查上方日志)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
