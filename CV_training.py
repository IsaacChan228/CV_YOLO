import os
import shutil
import argparse
from ultralytics import YOLO

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main(args):
    # Load pretrained model
    src_pretrained = args.pretrained if args.pretrained else 'yolov8n.pt'
    model = YOLO(src_pretrained)

    # Backup original weights
    backup_dir = os.path.join(os.getcwd(), 'model_original')
    ensure_dir(backup_dir)
    backup_path = os.path.join(backup_dir, os.path.basename(src_pretrained))
    if not os.path.exists(backup_path):
        shutil.copyfile(src_pretrained, backup_path)
        print(f'Original weights backed up to: {backup_path}')
    else:
        print(f'Backup already exists: {backup_path}')

    # Set training output path (results will be in model/<name>/weights/best.pt)
    project_dir = os.path.join(os.getcwd(), 'model')
    ensure_dir(project_dir)

    data_yaml = args.data if args.data else os.path.join(os.getcwd(), 'training data', 'data.yaml')
    print(f'Using data: {data_yaml}')

    # Run training
    model.train(data=data_yaml,
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                project=project_dir,
                name='exp')

    weights_best = os.path.join(project_dir, 'exp', 'weights', 'best.pt')
    weights_last = os.path.join(project_dir, 'exp', 'weights', 'last.pt')
    print('Training finished.')
    print(f'Best weights (if any): {weights_best}')
    print(f'Last weights (if any): {weights_last}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8n')
    parser.add_argument('--data', type=str, help='Path to data.yaml (default: training data/data.yaml)')
    parser.add_argument('--pretrained', type=str, help='Pretrained weights (default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()
    main(args)
