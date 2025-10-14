import torch
from ultralytics import YOLO

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
            return True
        else:
            print("=" * 50)
            print("⚠️ WARNING: No CUDA-enabled GPU detected.")
            print("   Training will run on CPU, which will be significantly slower.")
            print("=" * 50)
            return False
    except Exception as e:
        print(f"An error occurred while checking for GPU: {e}")
        return False

# --- KONFIGURASI TRAINING KEDUA ---
PRETRAINED_MODEL_PATH = '/home/azunya/kuliah/sem5/magang/project/K3_DETECTION_YOLOV8/yolo11n.pt'
NEW_DATASET_CONFIG_PATH = '/home/azunya/kuliah/sem5/magang/project/K3_DETECTION_YOLOV8/dataset/Hard Helmet Detect v4.v3i.yolov8/data.yaml'
NEW_RUN_NAME = '/home/azunya/kuliah/sem5/magang/project/K3_DETECTION_YOLOV8/runs/detect/V11-1'

if __name__ == '__main__':
    verify_gpu()
    model = YOLO(PRETRAINED_MODEL_PATH)

    print(f"\nContinuing training from model: {PRETRAINED_MODEL_PATH}")
    print(f"Using new dataset: {NEW_DATASET_CONFIG_PATH}\n")
    
    results = model.train(
        data=NEW_DATASET_CONFIG_PATH,
        epochs=50,
        imgsz=640,
        device=0,
        batch=4,           # Aman untuk VRAM 6GB
        workers=4,         # Ideal untuk CPU Core i7 HX 
        cache='disk',      # Pilihan terbaik untuk RAM 16GB
        
        name=NEW_RUN_NAME
    )
    
    print(f"\nSecond training complete! New model saved in 'runs/detect/{NEW_RUN_NAME}'")
