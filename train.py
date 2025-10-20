import torch
from ultralytics import YOLO
import sys
import subprocess
import os
import shutil
import platform

# --- Warna ---
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def open_terminal_monitor(command_to_run):
    """
    Mencoba membuka terminal default OS untuk menjalankan perintah monitoring.
    Mendukung Windows dan berbagai distro Linux.
    """
    try:
        current_os = platform.system()
        
        if current_os == "Windows":
            print("OS Windows terdeteksi. Menggunakan 'start cmd /k'...")
            full_command = f'start cmd /k "{command_to_run}"'
            subprocess.Popen(full_command, shell=True)
            return True

        elif current_os == "Linux":
            print("OS Linux terdeteksi. Mencari terminal yang tersedia...")
            linux_command_string = f'{command_to_run}; exec bash'
            
            terminals = [
                ('kitty', '--'), ('x-terminal-emulator', '-e'),
                ('konsole', '-e'), ('xfce4-terminal', '-e'),
                ('terminator', '-e'), ('xterm', '-e')
            ]
            
            for term, flag in terminals:
                if shutil.which(term):
                    print(f"✅ Ditemukan: {term}. Menjalankan...")
                    if flag == '--':
                        cmd_list = [term, flag, '/bin/sh', '-c', linux_command_string]
                    else:
                        cmd_list = [term, flag, f'/bin/sh -c "{linux_command_string}"']
                    subprocess.Popen(cmd_list)
                    return True
            
            print(f"{bcolors.WARNING}⚠️ Tidak ada terminal (gnome, konsole, xfce4, dll) yang ditemukan di PATH.{bcolors.ENDC}")
            return False

        else:
            print(f"{bcolors.WARNING}OS {current_os} tidak didukung secara otomatis.{bcolors.ENDC}")
            return False

    except Exception as e:
        print(f"{bcolors.FAIL}Gagal membuka terminal monitor: {e}{bcolors.ENDC}")
        return False


def verify_gpu():
    """
    Fungsi ini memeriksa ketersediaan GPU (CUDA) dan mencetak informasi perangkat.
    """
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print("=" * 50)
            print(f"{bcolors.OKGREEN}✅ GPU DETECTED! Found {gpu_count} CUDA-enabled GPU(s).{bcolors.ENDC}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   - GPU {i}: {gpu_name}")
            print("=" * 50)
            return 0
        else:
            print("=" * 50)
            print(f"{bcolors.FAIL}❌ ERROR: No CUDA-enabled GPU detected.{bcolors.ENDC}")
            print("   This script requires a GPU for training.")
            print("   Stopping execution.")
            print("=" * 50)
            return None
    except Exception as e:
        print(f"An error occurred while checking for GPU: {e}")
        return None

# --- KONFIGURASI PATH ---
PRETRAINED_MODEL_PATH = '/home/azunya/kuliah/sem5/magang/project/K3_DETECTION_YOLOV8/runs/detect/paling stable/v1 pakai seri m v2-v4 pakai seri m/yolov11-m-V_run_4/weights/best.pt' # bisa menggunakan model.pt atau hasil training sebelumnya
DATASET_CONFIG_PATH = '/home/azunya/Downloads/ID-Cards Segmentation.v1i.yolov11/data.yaml' # diganti ke dataset yang akan digunakan
BASE_RUN_NAME = '/home/azunya/kuliah/sem5/magang/project/K3_DETECTION_YOLOV8/runs/detect/fineshyt-tuning/V' #bisa diganti dengan folder lain


# ==============================================================================
# --- PERUBAHAN 1: KONFIGURASI HYPERPARAMETERS TRAINING ---
# Semua argumen statis untuk model.train() dipusatkan di sini.
# Mudah untuk di-update tanpa menyentuh logika fungsi.
# ==============================================================================
TRAIN_HYPERPARAMS = {
    'epochs': 50,
    'imgsz': 640,
    'batch': 4,
    'workers': 4,
    'cache': 'disk',
    'patience': 15,
    'dropout': 0.25,
    'weight_decay': 0.0005,
    'degrees': 20,
    'translate': 0.1,
    'scale': 0.2,
    'shear': 5,
    'perspective': 0.001,
    'flipud': 0.0,
    'fliplr': 0.5,
    'mosaic': 1.0,
    'mixup': 0.1
}

# --- PERUBAHAN 2: FUNGSI---
def run_training_session(model_path_to_load, run_name_to_save, device_idx):
    """
    Menjalankan satu sesi training YOLO.
    Mengembalikan path ke model 'best.pt' jika berhasil.
    """
    try:
        model = YOLO(model_path_to_load)
        print(f"\n{bcolors.BOLD}Starting training from model: {model_path_to_load}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}Using dataset: {DATASET_CONFIG_PATH}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}Saving results to: {run_name_to_save}\n{bcolors.ENDC}")
        
        # 1. Siapkan argumen dinamis (yang berubah setiap run)
        dynamic_args = {
            'data': DATASET_CONFIG_PATH,
            'device': device_idx,
            'name': run_name_to_save
        }
        
        # 2. Gabungkan config statis dari TRAIN_HYPERPARAMS dan args dinamis
        # Tanda (**) akan "membongkar" dictionary
        train_config = {**TRAIN_HYPERPARAMS, **dynamic_args}
        
        # 3. Cetak config yang akan dipakai (informatif)
        print(f"{bcolors.OKBLUE}--- Menjalankan training dengan config: ---{bcolors.ENDC}")
        # Ubah print config agar lebih rapi per baris
        for key, value in train_config.items():
            print(f"   {key}: {value}")
        print(f"{bcolors.OKBLUE}-------------------------------------------{bcolors.ENDC}")
        
        # 4. Jalankan training HANYA dengan satu variabel config
        results = model.train(**train_config)

        print(f"\n{bcolors.OKGREEN}Training run complete! Model disimpan di: {results.save_dir}{bcolors.ENDC}")
        best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')

        if not os.path.exists(best_model_path):
            print(f"{bcolors.FAIL}❌ ERROR: Tidak dapat menemukan 'best.pt' di {results.save_dir}/weights/{bcolors.ENDC}")
            return None
        return best_model_path

    except Exception as e:
        print(f"{bcolors.FAIL}Terjadi error selama training run: {e}{bcolors.ENDC}")
        return None


if __name__ == '__main__':
    device_to_use = verify_gpu()
    
    if device_to_use is None:
        sys.exit()
    
    command_to_monitor = "nvidia-smi -l 1"
    print(f"Mencoba membuka jendela terminal baru untuk memonitor GPU dengan '{command_to_monitor}'...")
    
    if not open_terminal_monitor(command_to_monitor):
        print(f"{bcolors.WARNING}⚠️ Peringatan: Gagal membuka terminal secara otomatis.{bcolors.ENDC}")
        print(f"   Silakan buka terminal baru secara manual dan jalankan '{command_to_monitor}' untuk memonitor.")

    try:
        total_runs = int(input(f"{bcolors.OKCYAN}{bcolors.BOLD}Masukkan jumlah total training run yang diinginkan: {bcolors.ENDC}"))
        if total_runs <= 0:
            print(f"{bcolors.FAIL}Jumlah run harus lebih besar dari 0.{bcolors.ENDC}")
            sys.exit()
    except ValueError:
        print(f"{bcolors.FAIL}Input tidak valid. Harap masukkan angka.{bcolors.ENDC}")
        sys.exit()

    print(f"\n{bcolors.OKGREEN}OK! Akan menjalankan training sebanyak {total_runs} kali secara berurutan.{bcolors.ENDC}")

    current_model_path = PRETRAINED_MODEL_PATH 

    for i in range(total_runs):
        print(f"\n{bcolors.HEADER}{bcolors.BOLD}" + "=" * 60 + f"{bcolors.ENDC}")
        print(f"{bcolors.HEADER}{bcolors.BOLD}MEMULAI TRAINING RUN {i + 1} DARI {total_runs}{bcolors.ENDC}")
        print(f"{bcolors.HEADER}{bcolors.BOLD}" + "=" * 60 + f"{bcolors.ENDC}")
        
        current_run_name = f"{BASE_RUN_NAME}_run_{i + 1}"
        
        best_model_from_run = run_training_session(
            model_path_to_load=current_model_path,
            run_name_to_save=current_run_name,
            device_idx=device_to_use
        )
        
        if best_model_from_run is None:
            print(f"{bcolors.FAIL}❌ Training run {i+1} GAGAL. Menghentikan proses.{bcolors.ENDC}")
            sys.exit()
        
        current_model_path = best_model_from_run
        
        print(f"{bcolors.OKGREEN}✅ Selesai training run {i + 1}. Model terbaik disimpan di: {current_model_path}{bcolors.ENDC}")

    print("\n" + f"{bcolors.OKCYAN}{bcolors.BOLD}" * 20 + f"{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}SELURUH PROSES TRAINING BERURUTAN SELESAI!{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Model final (terbaik dari run terakhir) ada di: {current_model_path}{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}" * 20 + f"{bcolors.ENDC}")
