import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchreid
from PIL import Image
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", required=True)
    parser.add_argument("--detection_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model_name", default="osnet_x0_25")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_detections(path):
    if path.lower().endswith(".npy"):
        return np.load(path)[:,:10]
    return np.loadtxt(path, delimiter=",")


def crop_person(image, x, y, w, h):
    height, width = image.shape[:2]

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width, int(round(x + w)))
    y2 = min(height, int(round(y + h)))

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def main():
    args = parse_args()

    device = torch.device(args.device)
    detections = load_detections(args.detection_file)

    model = torchreid.models.build_model(
        name=args.model_name,
        num_classes=1000,
        pretrained=True
    )
    model.eval()
    model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_dir = Path(args.sequence_dir) / "img1"
    rows = []

    with torch.no_grad():
        for i, row in enumerate(detections, start=1):
            frame_id = int(row[0])
            x, y, w, h = row[2:6]

            image_path = image_dir / f"{frame_id:06d}.jpg"
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            crop = crop_person(image, x, y, w, h)
            if crop is None:
                continue

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = Image.fromarray(crop)
            tensor = preprocess(crop).unsqueeze(0).to(device)

            feature = model(tensor)
            feature = torch.nn.functional.normalize(feature, p=2, dim=1)
            feature = feature.cpu().numpy().reshape(-1)

            rows.append(np.concatenate([row[:10], feature]))

            if i % 100 == 0:
                print(f"Processed {i}/{len(detections)} detections", flush=True)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.save(args.output_file, np.asarray(rows, dtype=np.float32))


if __name__ == "__main__":
    main()