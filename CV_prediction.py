import os
import argparse
from ultralytics import YOLO

def predict_image(weights, image_path, conf_thresh=0.25, topk=10):
    model = YOLO(weights)
    results = model.predict(source=image_path, conf=conf_thresh, verbose=False)
    if not results:
        return []
    r = results[0]
    outputs = []
    if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls = int(box.cls.cpu().numpy()) if hasattr(box, 'cls') else None
            conf = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else None
            name = model.model.names[cls] if cls is not None and cls in model.model.names else str(cls)
            outputs.append((name, conf))
    # Sort by confidence and return topk
    outputs.sort(key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
    return outputs[:topk]

def main(args):
    default_weights = os.path.join(os.getcwd(), 'model', 'exp', 'weights', 'best.pt')
    weights = args.weights if args.weights else default_weights
    if not os.path.exists(weights):
        print(f'Weights file not found: {weights}')
        return
    if not os.path.exists(args.image):
        print(f'Image not found: {args.image}')
        return
    results = predict_image(weights, args.image, conf_thresh=args.conf, topk=args.topk)
    print('Prediction results (label, confidence):')
    for label, conf in results:
        print(f'{label}: {conf:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict single image using YOLOv8 model')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--weights', type=str, help='Path to weights (default: model/exp/weights/best.pt)')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--topk', type=int, default=10)
    args = parser.parse_args()
    main(args)
