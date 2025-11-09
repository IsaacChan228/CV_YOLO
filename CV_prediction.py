import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import csv
import time
import subprocess

########## Configuration ##########
INPUT_DIR = os.path.join(os.getcwd(), 'prediction', 'input')
OUTPUT_DIR = os.path.join(os.getcwd(), 'prediction', 'output')
WEIGHTS = os.path.join(os.getcwd(), 'model', 'exp', 'weights', 'best.pt')
CONF_THRESH =0.25
MAPPING_FILE = os.path.join(os.getcwd(), 'mapping.txt')
PREPROCESS_SIZE =640

# Ensure output exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

# Set to True to use camera to obtain a input image
USE_CAMERA = False
CAMERA_INDEX =0
CAMERA_TIMEOUT =5 # seconds to wait for camera

# Set to True to enable audio output
AUDIO_OUTPUT = False
AUDIO_SCRIPT = os.path.join(os.getcwd(), 'play_audio.py')

########## Code ##########

def load_mapping(path):
    mapping = {}
    if not os.path.exists(path):
        return mapping
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or ':' not in line:
                    continue
                lhs, rhs = line.split(':',1)
                try:
                    cls = int(lhs.strip())
                    name = rhs.strip()
                    mapping[cls] = name
                except Exception:
                    continue
    except Exception:
        pass
    return mapping


def preprocess_image(img, size=PREPROCESS_SIZE):
    """Convert image to a square canvas with white background and resize to (size,size).
    Returns the resized image (BGR numpy array).
    """
    if img is None:
        return None
    h, w = img.shape[:2]
    side = max(w, h)
    # create white canvas (BGR)
    canvas = np.full((side, side,3),255, dtype=np.uint8)
    # compute top-left corner to paste original image
    left = (side - w) //2
    top = (side - h) //2
    # place image onto canvas
    canvas[top:top + h, left:left + w] = img
    # resize to target size
    resized = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized


def choose_best(boxes, img_w, img_h):
    # boxes: list of tuples (cls, conf, (x1,y1,x2,y2))
    if not boxes:
        return None
    # find max confidence
    max_conf = max([b[1] for b in boxes])
    candidates = [b for b in boxes if abs(b[1] - max_conf) <1e-9]
    if len(candidates) ==1:
        return candidates[0]
    # tie-breaker: choose largest area
    best = None
    best_area = -1
    for cls, conf, (x1, y1, x2, y2) in candidates:
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh
        if area > best_area:
            best_area = area
            best = (cls, conf, (x1, y1, x2, y2))
    return best


def flush_input_and_capture(camera_index=0, timeout=CAMERA_TIMEOUT):
    """Remove files in INPUT_DIR and capture one image from the camera into INPUT_DIR.
    Returns path to saved image or None on failure.
    """
    # remove existing files
    for fname in os.listdir(INPUT_DIR):
        try:
            fp = os.path.join(INPUT_DIR, fname)
            if os.path.isfile(fp):
                os.remove(fp)
        except Exception:
            pass
    # open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Camera {camera_index} could not be opened.")
        return None
    # warm up and wait for a frame until timeout
    end_time = time.time() + timeout
    frame = None
    while time.time() < end_time:
        ret, frm = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame = frm
        break
    cap.release()
    if frame is None:
        print("Failed to capture image from camera.")
        return None
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_name = f'camera_{ts}.jpg'
    out_path = os.path.join(INPUT_DIR, out_name)
    try:
        cv2.imwrite(out_path, frame)
        print(f'Captured camera image to: {out_path}')
        return out_path
    except Exception as e:
        print(f'Failed to write captured image: {e}')
        return None


def process_image(model, mapping, image_path):
    orig = cv2.imread(image_path)
    if orig is None:
        return None
    # Preprocess to square
    proc = preprocess_image(orig, size=PREPROCESS_SIZE)
    h, w = proc.shape[:2]
    # Run prediction on preprocessed image (pass numpy array)
    res = model.predict(source=proc, conf=CONF_THRESH, verbose=False, save=False)
    if not res:
        return None
    r = res[0]
    boxes = []
    if hasattr(r, 'boxes') and r.boxes is not None:
        for box in r.boxes:
            try:
                cls = int(box.cls.cpu().numpy()) if hasattr(box, 'cls') else None
            except Exception:
                cls = None
            try:
                conf = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else None
            except Exception:
                conf = None
            # get xyxy in pixels relative to proc image
            xy = None
            try:
                if hasattr(box, 'xyxy'):
                    xy = box.xyxy.cpu().numpy()
                elif hasattr(box, 'xyxyn'):
                    xy = box.xyxyn.cpu().numpy() * [w, h, w, h]
                if xy is not None:
                    if xy.ndim ==2:
                        xy = xy[0]
                    x1, y1, x2, y2 = float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])
                    # clamp
                    x1 = max(0, min(w -1, x1))
                    x2 = max(0, min(w -1, x2))
                    y1 = max(0, min(h -1, y1))
                    y2 = max(0, min(h -1, y2))
                    boxes.append((cls, conf, (x1, y1, x2, y2)))
            except Exception:
                continue
    best = choose_best(boxes, w, h)
    if best is None:
        return None
    cls, conf, (x1, y1, x2, y2) = best
    name = mapping.get(cls, str(cls))
    # save annotated image (annotate proc image)
    annotated = proc.copy()
    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0),2)
    label_text = f"{name} {conf:.3f}"
    cv2.putText(annotated, label_text, (int(x1), int(max(0, y1 -10))), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,255,0),2)
    base = os.path.splitext(os.path.basename(image_path))[0]
    annotated_path = os.path.join(OUTPUT_DIR, f"{base}_boxed.jpg")
    cv2.imwrite(annotated_path, annotated)
    # save cropped image from proc image
    cx1, cy1, cx2, cy2 = map(int, (x1, y1, x2, y2))
    try:
        crop = proc[cy1:cy2, cx1:cx2]
        if crop.size ==0:
            crop = proc
    except Exception:
        crop = proc
    crop_path = os.path.join(OUTPUT_DIR, f"{base}_crop.jpg")
    cv2.imwrite(crop_path, crop)
    # result info
    result = {
        'file': os.path.basename(image_path),
        'class_id': cls,
        'class_name': name,
        'confidence': conf,
        'bbox': (x1, y1, x2, y2),
        'annotated': annotated_path,
        'crop': crop_path
    }
    return result


def main():
    mapping = load_mapping(MAPPING_FILE)
    if not os.path.exists(WEIGHTS):
        print(f"Weights not found: {WEIGHTS}")
        return
    model = YOLO(WEIGHTS)

    # If using camera, flush input dir and capture one image
    if USE_CAMERA:
        captured = flush_input_and_capture(camera_index=CAMERA_INDEX, timeout=CAMERA_TIMEOUT)
        if captured is None:
            print('Camera capture failed; proceeding with existing images in input dir.')

    images = [p for p in os.listdir(INPUT_DIR) if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not images:
        print(f"No input images in {INPUT_DIR}")
        return
    results = []
    for fname in images:
        path = os.path.join(INPUT_DIR, fname)
        print(f"Processing: {path}")
        res = process_image(model, mapping, path)
        if res is None:
            print(f" No detection for {fname}")
        else:
            print(f" Detected: {res['class_name']} ({res['confidence']:.3f})")
            results.append(res)
    if AUDIO_OUTPUT and results:
        for r in results:
            txt = f"{r['class_name']} {r['confidence']:.2f}"
            try:
                subprocess.Popen([sys.executable, AUDIO_SCRIPT, txt])
            except Exception as e:
                print(f"Failed to invoke audio script: {e}")
    # write result.txt
    out_txt = os.path.join(OUTPUT_DIR, 'result.txt')
    with open(out_txt, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        for r in results:
            x1, y1, x2, y2 = r['bbox']
            writer.writerow([r['file'], r['class_name'], f"{r['confidence']:.6f}", int(x1), int(y1), int(x2), int(y2)])
    print(f"Results saved to {out_txt}")


if __name__ == '__main__':
    main()
