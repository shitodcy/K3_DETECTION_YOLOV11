import torch
from ultralytics import YOLO
import sys
import subprocess

def verify_gpu():
    """
    Fungsi ini memeriksa ketersediaan GPU (CUDA) dan mencetak informasi perangkat.
    """
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print("=" * 50)
            print(f"✅ GPU DETECTED! Found {gpu_count} CUDA-enabled GPU(s).")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   - GPU {i}: {gpu_name}")
            print("=" * 50)
            return 0 #switch gpu 
        else:
            return None
    except Exception as e:
        print(f"An error occurred while checking for GPU: {e}")
        return None

PRETRAINED_MODEL_PATH = '/home/azunya/kuliah/sem5/magang/project/K3_DETECTION_YOLOV8/yolo11n.pt'
DATASET_CONFIG_PATH = '/home/azunya/Downloads/safety-helmet.v1-44v.yolov11/data.yaml'
RUN_NAME = '/home/azunya/kuliah/sem5/magang/project/K3_DETECTION_YOLOV8/runs/detect/newscript-yolov11X'

if __name__ == '__main__':
    device_to_use = verify_gpu()
    
    if device_to_use is None:
        print("=" * 50)
        print("❌ ERROR: No CUDA-enabled GPU detected.")
        print("   This script requires a GPU for training.")
        print("   Stopping execution.")
        print("=" * 50)
        sys.exit()
    
    try:
        print("Membuka jendela terminal baru untuk memonitor GPU dengan 'nvidia-smi -l 1'...")
        command = 'gnome-terminal -- /bin/sh -c "nvidia-smi -l 1; exec bash"'
        subprocess.Popen(command, shell=True)
    except FileNotFoundError:
        print("⚠️ Peringatan: 'gnome-terminal' tidak ditemukan.")
        print("   Silakan buka terminal baru secara manual dan jalankan 'nvidia-smi -l 1' untuk memonitor.")
    except Exception as e:
        print(f"Gagal membuka terminal monitor: {e}")
        
    model = YOLO(PRETRAINED_MODEL_PATH)

    print(f"\nStarting training from model: {PRETRAINED_MODEL_PATH}")
    print(f"Using dataset: {DATASET_CONFIG_PATH}\n")

    results = model.train(
        data=DATASET_CONFIG_PATH,
        epochs=50,
        imgsz=640,
        device=device_to_use,
        batch=4,
        workers=4,
        cache='disk',
        name=RUN_NAME,
        patience=15,
        dropout=0.25,
        weight_decay=0.0005,
        degrees=20,
        translate=0.1,
        scale=0.2,
        shear=5,
        perspective=0.001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )
    
    print(f"\nTraining complete! Model terbaik disimpan di direktori 'runs/detect/{RUN_NAME}'")
