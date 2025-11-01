import os
import time
import argparse
from ultralytics import YOLO
import cv2

def list_images(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def main(args):
    # Use provided weights argument, otherwise default to model/exp/weights/best.pt
    default_weights = os.path.join(os.getcwd(), 'model', 'exp', 'weights', 'best.pt')
    weights = args.weights if args.weights else default_weights
    if not os.path.exists(weights):
        print(f'Weights not found: {weights}')
        return

    model = YOLO(weights)
    data_yaml = args.data if args.data else os.path.join(os.getcwd(), 'test_data', 'data.yaml')

    # Use built-in val for overall evaluation (if data.yaml exists)
    if os.path.exists(data_yaml):
        print(f'Starting overall evaluation (val) using: {data_yaml}')
        metrics = model.val(data=data_yaml)
        print('Evaluation results (ultralytics output):')
        print(metrics)
    else:
        print(f'data.yaml not found, skipping val: {data_yaml}')

    # Single-image inference timing (use test_data/images)
    images_folder = os.path.join(os.getcwd(), 'test_data', 'images')
    imgs = list_images(images_folder)
    if not imgs:
        print(f'No test images found at: {images_folder}')
        return

    times = []
    print(f'Running single-image inference timing on {len(imgs)} images...')
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            continue
        t0 = time.time()
        res = model.predict(source=p, conf=0.25, verbose=False)
        t1 = time.time()
        times.append(t1 - t0)
        # Display brief results
        r = res[0]
        labels = []
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls = int(box.cls.cpu().numpy()) if hasattr(box, 'cls') else None
                conf = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else None
                name = model.model.names[cls] if cls is not None and cls in model.model.names else str(cls)
                labels.append((name, conf))
        print(f'{os.path.basename(p)} -> {labels}')

    avg = sum(times) / len(times) if times else 0
    print(f'Average single-image inference time: {avg:.4f} sec')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test YOLOv8 model and measure inference time')
    parser.add_argument('--weights', type=str, help='Path to weights file (default: model/exp/weights/best.pt)')
    parser.add_argument('--data', type=str, help='Path to test data.yaml (default: test_data/data.yaml)')
    args = parser.parse_args()
    main(args)
