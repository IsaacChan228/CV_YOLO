import os
import time
from ultralytics import YOLO
import cv2
import warnings

### Global configuration ###
# Set to True to use origin model weight for testing
USE_ORIGINAL = False

# Confidence threshold for single-image prediction
CONF_THRESH = 0.25
IOU_THRESHOLD = 0.5

# Folder locations
DATA_YAML = os.path.join(os.getcwd(), 'training_data', 'grocery', 'data.yaml')
TEST_DIR = os.path.join(os.getcwd(), 'training_data', 'grocery', 'test')
IMAGES_FOLDER = os.path.join(TEST_DIR, 'images')
LABELS_FOLDER = os.path.join(TEST_DIR, 'labels')


### Code ###

def list_images(folder):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def find_original_weight():
    """Find a .pt file in model_original and return the most recently modified one, or None."""
    backup_dir = os.path.join(os.getcwd(), 'model_original')
    if not os.path.isdir(backup_dir):
        print('model_original directory not found.')
        return None
    pts = [os.path.join(backup_dir, f) for f in os.listdir(backup_dir) if f.lower().endswith('.pt')]
    if not pts:
        print('No .pt files found in model_original directory.')
        return None
    # return the most recently modified .pt
    pts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    print(f'Found original weight: {pts[0]}')
    return pts[0]

def load_gt_boxes(label_path, img_w, img_h):
    """Load ground-truth boxes from YOLO txt label file (class x_center y_center w h normalized).
    Returns list of tuples (class_id, (x1,y1,x2,y2))."""
    if not os.path.exists(label_path):
        return []
    boxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                xc = float(parts[1]) * img_w
                yc = float(parts[2]) * img_h
                bw = float(parts[3]) * img_w
                bh = float(parts[4]) * img_h
                x1 = xc - bw / 2.0
                y1 = yc - bh / 2.0
                x2 = xc + bw / 2.0
                y2 = yc + bh / 2.0
                boxes.append((cls, (x1, y1, x2, y2)))
    except Exception:
        return []
    return boxes

def iou(boxA, boxB):
    """Compute IoU between two boxes in xyxy format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union

def main():
    # Use default weight at model/exp/weights/best.pt
    default_weights = os.path.join(os.getcwd(), 'model', 'exp', 'weights', 'best.pt')
    weights = None

    # If USE_ORIGINAL is True, try to use model_original
    if USE_ORIGINAL:
        orig = find_original_weight()
        if orig:
            weights = orig
            print(f'USE_ORIGINAL enabled: using original weights: {weights}')
        else:
            print('USE_ORIGINAL enabled but no .pt found in model_original. Falling back to default weights.')
            weights = default_weights
    elif weights is None:
        weights = default_weights

    if not os.path.exists(weights):
        print(f'Weights not found: {weights}')
        return

    try:
        model = YOLO(weights)
    except Exception as e:
        warnings.warn(f'Failed to load model from {weights}: {e}')
        return

    # Run dataset-level evaluation if data.yaml exists
    if os.path.exists(DATA_YAML):
        try:
            val_res = model.val(data=DATA_YAML, verbose=False, save=False)
            # Attempt to extract mAP@0.5 if available
            accuracy_pct = None
            if isinstance(val_res, dict):
                # common keys to check
                for k in ('map50', 'mAP50', 'mAP_50', 'map_50'):
                    if k in val_res:
                        accuracy_pct = float(val_res[k]) * 100.0
                        break
                # sometimes nested
                if accuracy_pct is None and 'metrics' in val_res and isinstance(val_res['metrics'], dict):
                    for k, v in val_res['metrics'].items():
                        if 'map' in k.lower() and v is not None:
                            accuracy_pct = float(v) * 100.0
                            break
            # If we couldn't extract, set to None and continue
        except Exception:
            accuracy_pct = None
    else:
        accuracy_pct = None

    # Single-image inference: compute per-image correctness by matching preds to GT labels
    imgs = list_images(IMAGES_FOLDER)
    if not imgs:
        print(f'No test images found at: {IMAGES_FOLDER}')
        return

    times = []
    correct_images = 0
    skipped = 0
    processed = 0

    # Counters for inaccuracy breakdown
    no_prediction = 0
    wrong_label = 0
    localization_error = 0
    no_overlap = 0

    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        t0 = time.time()
        try:
            res = model.predict(source=p, conf=CONF_THRESH, verbose=False, save=False)
        except Exception:
            skipped += 1
            continue
        t1 = time.time()
        times.append(t1 - t0)

        # extract predictions
        r = res[0]
        preds = []
        try:
            # r.boxes may be an iterable of box objects
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    try:
                        cls = int(box.cls.cpu().numpy()) if hasattr(box, 'cls') else None
                    except Exception:
                        cls = None
                    try:
                        # box.xyxy may be tensor
                        xy = None
                        if hasattr(box, 'xyxy'):
                            xy = box.xyxy.cpu().numpy()
                        elif hasattr(box, 'xyxyn'):
                            xy = box.xyxyn.cpu().numpy() * [w, h, w, h]
                        if xy is not None:
                            # xy could be shape (4,) or (1,4)
                            if xy.ndim == 2:
                                xy = xy[0]
                            x1, y1, x2, y2 = float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])
                            preds.append((cls, (x1, y1, x2, y2)))
                    except Exception:
                        continue
        except Exception:
            pass

        # load ground-truth boxes for this image
        base = os.path.splitext(os.path.basename(p))[0]
        label_path = os.path.join(LABELS_FOLDER, base + '.txt')
        gt_boxes = load_gt_boxes(label_path, w, h)

        # if no GT available, skip image from correctness calculation
        if not gt_boxes:
            skipped += 1
            continue

        processed += 1
        # determine if image is correct: exists a pred and gt with same class and IoU >= IOU_THRESHOLD
        image_correct = False
        for p_cls, p_box in preds:
            if p_cls is None:
                continue
            for gt_cls, gt_box in gt_boxes:
                if p_cls != gt_cls:
                    continue
                if iou(p_box, gt_box) >= IOU_THRESHOLD:
                    image_correct = True
                    break
            if image_correct:
                break
        if image_correct:
            correct_images += 1
        else:
            # classify reason for inaccuracy for this image
            if not preds:
                no_prediction += 1
            else:
                # compute max iou and check types of mismatches
                max_iou = 0.0
                best_pred_cls = None
                best_gt_cls = None
                same_class_any = False
                for p_cls, p_box in preds:
                    if p_cls is None:
                        continue
                    for gt_cls, gt_box in gt_boxes:
                        cur_iou = iou(p_box, gt_box)
                        if cur_iou > max_iou:
                            max_iou = cur_iou
                            best_pred_cls = p_cls
                            best_gt_cls = gt_cls
                        if p_cls == gt_cls:
                            same_class_any = True
                if max_iou >= IOU_THRESHOLD and best_pred_cls is not None and best_gt_cls is not None and best_pred_cls != best_gt_cls:
                    wrong_label += 1
                elif same_class_any:
                    # predictions with same class exist but IoU too low
                    localization_error += 1
                else:
                    # predictions exist but no overlap with GT
                    no_overlap += 1

    total_images = processed
    avg_time = sum(times) / len(times) if times else 0.0

    # Prepare concise summary lines
    lines = []
    lines.append('Test summary:')
    if total_images > 0:
        accuracy = (correct_images / total_images) * 100.0
        lines.append(f' Accuracy: {accuracy:.2f}%')
        lines.append(f' Correct/Total: {correct_images}/{total_images}')
    else:
        lines.append(' No labeled test images processed (all skipped).')
    lines.append(f' Skipped images (no GT or error): {skipped}')
    lines.append(f' Average inference time (per image): {avg_time:.4f} sec')

    # Add inaccuracy breakdown
    lines.append(' Inaccuracy breakdown:')
    lines.append(f' No prediction: {no_prediction}')
    lines.append(f' Wrong label (IoU>=threshold but class mismatch): {wrong_label}')
    lines.append(f' Localization error (same class but IoU < threshold): {localization_error}')
    lines.append(f' No overlap (predictions do not overlap GT): {no_overlap}')

    # Print to console
    for ln in lines:
        print(ln)

    # Also save summary to a text file next to the script
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, 'test_summary.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'Summary saved to: {out_path}')
    except Exception as e:
        warnings.warn(f'Failed to write summary file: {e}')


if __name__ == '__main__':
    main()
