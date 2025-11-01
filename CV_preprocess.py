import os
import sys
import glob
import numpy as np
import cv2

# Global configuration (replaces CLI arguments)
side_length = 640
jpeg_quality = 95
INPUT_DIR = 'Input'
OUTPUT_DIR = 'Output'


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_image_file(path):
    lower = path.lower()
    return lower.endswith((
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'
    ))


def process_with_cv2(in_path, out_path, size=680, quality=95):
    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Unable to read image: {in_path}")

    # Normalize image to BGR or BGRA
    h, w = img.shape[:2]
    side = max(h, w)

    # Prepare white background canvas
    if img.ndim ==2:
        # grayscale -> convert to BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        canvas =255 * np.ones((side, side,3), dtype=img_bgr.dtype)
        top = (side - h) //2
        left = (side - w) //2
        canvas[top:top + h, left:left + w] = img_bgr
    elif img.shape[2] ==4:
        # BGRA with alpha: composite over white background
        b, g, r, a = cv2.split(img)
        alpha = a.astype(float) /255.0
        # create white background
        canvas =255 * np.ones((side, side,3), dtype=img.dtype)
        top = (side - h) //2
        left = (side - w) //2
        # Composite per-channel
        roi_b = canvas[top:top + h, left:left + w,0].astype(float)
        roi_g = canvas[top:top + h, left:left + w,1].astype(float)
        roi_r = canvas[top:top + h, left:left + w,2].astype(float)
        canvas[top:top + h, left:left + w,0] = (b.astype(float) * alpha + roi_b * (1 - alpha)).astype(img.dtype)
        canvas[top:top + h, left:left + w,1] = (g.astype(float) * alpha + roi_g * (1 - alpha)).astype(img.dtype)
        canvas[top:top + h, left:left + w,2] = (r.astype(float) * alpha + roi_r * (1 - alpha)).astype(img.dtype)
    else:
        # BGR
        canvas =255 * np.ones((side, side,3), dtype=img.dtype)
        top = (side - h) //2
        left = (side - w) //2
        canvas[top:top + h, left:left + w] = img

    # Resize to target size
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    resized = cv2.resize(canvas, (size, size), interpolation=interp)

    cv2.imwrite(out_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def process_file(path, out_dir, size=680, quality=95):
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, base + '.jpg')
    try:
        process_with_cv2(path, out_path, size=size, quality=quality)
        print(f"Processed: {path} -> {out_path}")
    except Exception as e:
        print(f"Failed processing {path}: {e}")


def main():
    in_dir = os.path.abspath(INPUT_DIR)
    out_dir = os.path.abspath(OUTPUT_DIR)

    if not os.path.isdir(in_dir):
        print(f"Input directory does not exist: {in_dir}")
        sys.exit(1)

    ensure_dir(out_dir)

    # Collect image files
    candidates = glob.glob(os.path.join(in_dir, '*'))
    files = [p for p in candidates if os.path.isfile(p) and is_image_file(p)]

    if not files:
        print(f"No image files found in {in_dir}")
        return

    for path in files:
        process_file(path, out_dir, size=side_length, quality=jpeg_quality)


if __name__ == '__main__':
    main()