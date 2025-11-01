import os
import shutil
import warnings
from ultralytics import YOLO
import torch

##### Global configuration (edit these variables for different training setting) #####
# Training post-trained model
# If True, use previously trained model weights as starting point
# If False, use pretrained model weights "PRETRAINED" as starting point
EX_POST_TRAIN = False

# Pretrained model path
PRETRAINED = 'yolov8n.pt'
# Training data YAML file path
DATA_YAML = os.path.join(os.getcwd(), 'training_data/grocery', 'data.yaml')

# Training parameters and Image size
EPOCHS =50
BATCH =16
IMGSZ =640

# Maximum VRAM to allow this process to use (in GB). Set to None to disable cap.
MAX_VRAM_GB =8


##### Code #####

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def configure_cuda(max_vram_gb: float = None):
    """
    Configure CUDA usage and optionally cap per-process GPU memory to max_vram_gb.
    Returns the device string to use (e.g. 'cuda:0' or 'cpu').
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return "cpu"

    device_idx =0
    device = f"cuda:{device_idx}"
    try:
        props = torch.cuda.get_device_properties(device_idx)
        total_bytes = props.total_memory
        print(f"Detected CUDA device0: {props.name}, total VRAM = {total_bytes / (1024**3):.2f} GB")
    except Exception as e:
        warnings.warn(f"Unable to query CUDA device properties: {e}")
        return device

    if max_vram_gb is None:
        print("No VRAM cap requested. Using full available GPU memory.")
        return device

    max_bytes = int(max_vram_gb *1024**3)
    fraction = max_bytes / total_bytes

    # Keep fraction in (0,1]
    if fraction <=0:
        warnings.warn("Requested MAX_VRAM_GB is <=0. Falling back to default GPU usage.")
        return device
    if fraction >=1.0:
        print("Requested VRAM >= physical VRAM. Using full GPU memory.")
        return device

    # Try to set per-process memory fraction (PyTorch API)
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device_idx)
        torch.cuda.empty_cache()
        print(f"Set per-process GPU memory fraction to {fraction:.3f} (~{max_vram_gb} GB)")
    except Exception as e:
        # Fallback: inform user and suggest reducing batch size if OOM occurs
        warnings.warn(
            "torch.cuda.set_per_process_memory_fraction failed or is not available in this PyTorch build. "
            "Training will proceed but VRAM may not be capped. Consider reducing batch size or using a smaller model."
            f" (error: {e})"
        )

    return device


def main():
    # Prepare output/project directory
    project_dir = os.path.join(os.getcwd(), 'model')
    ensure_dir(project_dir)
    weights_dir = os.path.join(project_dir, 'exp', 'weights')

    # Determine which weights to use as starting point
    if EX_POST_TRAIN:
        # Prefer last.pt, then best.pt if available
        last_w = os.path.join(weights_dir, 'last.pt')
        best_w = os.path.join(weights_dir, 'best.pt')
        if os.path.exists(last_w):
            src_pretrained = last_w
            print(f'EX_POST_TRAIN enabled: using previous weights: {src_pretrained}')
        elif os.path.exists(best_w):
            src_pretrained = best_w
            print(f'EX_POST_TRAIN enabled: using previous best weights: {src_pretrained}')
        else:
            src_pretrained = PRETRAINED
            print(f'EX_POST_TRAIN enabled but no previous weights found in {weights_dir}. Falling back to PRETRAINED: {src_pretrained}')
    else:
        src_pretrained = PRETRAINED

    # Configure CUDA and optionally cap VRAM before loading model
    device = configure_cuda(MAX_VRAM_GB)

    # Load model
    model = YOLO(src_pretrained)
    try:
        if device.startswith("cuda"):
            # try to move model to GPU
            model.to(device)
            print(f"Moved model to {device}")
    except Exception as e:
        warnings.warn(f"Failed to move model to {device}: {e}. The model may still run on CPU or default device.")

    # Backup original weights only when not post-training
    if not EX_POST_TRAIN:
        backup_dir = os.path.join(os.getcwd(), 'model_original')
        ensure_dir(backup_dir)
        backup_path = os.path.join(backup_dir, os.path.basename(src_pretrained))

        # If backup exists, remove and replace with the current pretrained file
        try:
            if os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                    print(f'Removed existing backup: {backup_path}')
                except Exception as e:
                    warnings.warn(f'Failed to remove existing backup {backup_path}: {e}')
            # Attempt to copy the pretrained file to backup location
            shutil.copyfile(src_pretrained, backup_path)
            print(f'Original weights backed up to: {backup_path}')
        except Exception as e:
            print(f'Warning: failed to backup {src_pretrained} to {backup_path}: {e}')
    else:
        print('EX_POST_TRAIN is True: skipping backup of original weights')

    data_yaml = DATA_YAML
    print(f'Using data: {data_yaml}')

    # Run training. Pass device to training call as well (Ultralytics accepts device arg).
    train_kwargs = dict(
        data=data_yaml,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        project=project_dir,
        name='exp',
        exist_ok=True
    )
    # If a CUDA device was selected, provide it to train (Ultralytics accepts device="cuda:0" or device=0)
    if isinstance(device, str) and device.startswith("cuda"):
        train_kwargs["device"] = device

    model.train(**train_kwargs)

    weights_best = os.path.join(project_dir, 'exp', 'weights', 'best.pt')
    weights_last = os.path.join(project_dir, 'exp', 'weights', 'last.pt')
    print('Training finished.')
    print(f'Best weights (if any): {weights_best}')
    print(f'Last weights (if any): {weights_last}')


if __name__ == '__main__':
    main()
