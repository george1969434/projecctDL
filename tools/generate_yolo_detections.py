import argparse
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    # папка с одной MOT-последовательностью
    parser.add_argument("--sequence_dir", required=True)
    # файл, куда сохраним новые yolo-детекции в MOT-формате
    parser.add_argument("--output_file", required=True)
    # yolo-модель
    parser.add_argument("--model", default="yolov8n.pt")
    # все bbox-ы ниже этого порога отбрасываются
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)

    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = Path(args.sequence_dir) / "img1"
    # сортируем кадры по номеру, чтобы они обрабатывались в правильном порядке
    image_files = sorted(image_dir.glob("*.jpg"), key=lambda p: int(p.stem))

    # загружаем YOLO-модель
    model = YOLO(args.model)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for image_path in image_files:
            frame_id = int(image_path.stem)

            #читаем кадр
            image = cv2.imread(str(image_path))
            # запускаем YOLO на кадре
            results = model.predict(
                image,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False
            )[0]

            if results.boxes is None:
                continue

            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                if cls_id != 0:
                    continue
                # yolo возвращает bbox в формате x1, y1, x2, y2.
                # MOT/DeepSORT ждут формат x, y, width, height
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w = x2 - x1
                h = y2 - y1
                f.write(
                    f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                    f"{confidence:.6f},-1,-1,-1\n"
                )


if __name__ == "__main__":
    main()
