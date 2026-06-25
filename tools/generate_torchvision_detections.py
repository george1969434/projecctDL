import argparse
import os
from pathlib import Path
import cv2
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.transforms.functional import to_tensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--output_file", required=True)
    # все боксы ниже этого порога отбрасываем
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    image_dir = Path(args.sequence_dir) / "img1"
    image_files = sorted(image_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    # берем модель Faster R-CNN, обученную на COCO.
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.to(device)
    model.eval()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for i, image_path in enumerate(image_files, start=1):
                if i % 25 == 0 or i == 1 or i == len(image_files):
                    print(f"Processed {i}/{len(image_files)} frames: {image_path.name}", flush=True)
                frame_id = int(image_path.stem)
                image_bgr = cv2.imread(str(image_path))
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                # картинку переводим tensor в RGB
                image_tensor = to_tensor(image_rgb).to(device)
                result = model([image_tensor])[0]
                boxes = result["boxes"].detach().cpu()
                scores = result["scores"].detach().cpu()
                labels = result["labels"].detach().cpu()
                for box, score, label in zip(boxes, scores, labels):
                    # оставляем только людей
                    if int(label.item()) != 1:
                        continue
                    confidence = float(score.item())
                    if confidence < args.conf:
                        continue
                    x1, y1, x2, y2 = box.numpy()
                    w = x2 - x1
                    h = y2 - y1
                    f.write(
                        f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                        f"{confidence:.6f},-1,-1,-1\n"
                    )
if __name__ == "__main__":
    main()