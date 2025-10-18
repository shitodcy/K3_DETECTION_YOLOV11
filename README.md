# Deteksi APD Real-time dengan YOLOv8

Proyek ini menggunakan **YOLOv8** untuk mendeteksi penggunaan **Alat Pelindung Diri (APD)** secara **real-time** melalui webcam.
Cocok untuk pemantauan keselamatan di **pabrik, gudang, dan area konstruksi.**

---

## Fitur Utama

* **Deteksi Real-time:** Analisis langsung dari webcam.
* **Multi-Platform:** Berjalan di Windows & Linux.
* **GPU Acceleration:** Dukungan CUDA untuk performa optimal.
* **Kelas yang Dideteksi:**

  * 👷 Person
  * ⛑️ Hardhat
  * 🦺 Vest
  * 😷 Mask
  * 🧤 Gloves

---

## Prasyarat

### Dataset
[Unduh Dataset](https://www.kaggle.com/datasets/shlokraval/ppe-dataset-yolov8/data)

### Umum

* Python **3.9+**
* Git
* NVIDIA GPU (disarankan ≥ 6 GB VRAM)

### Windows

1. **Driver NVIDIA** → [Unduh di sini](https://www.nvidia.com/Download/index.aspx)
2. **CUDA Toolkit 12.1** → [Unduh di sini](https://developer.nvidia.com/cuda-downloads)
3. **cuDNN** → [Unduh di sini](https://developer.nvidia.com/cudnn)

   > Ekstrak dan salin folder `bin`, `include`, dan `lib` ke direktori instalasi CUDA (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`).

### Linux (Arch & Turunannya)

```bash
sudo pacman -S nvidia-dkms nvidia-utils cuda cudnn
sudo reboot
nvidia-smi
```

---

## Instalasi

1. **Clone Repositori**

   ```bash
   git clone https://github.com/Magang-API/K3_DETECTION_YOLOV8
   cd K3_DETECTION_YOLOV8
   ```

2. **Buat Virtual Environment**

   ```bash
   python -m venv venv
   ```

   * Windows: `venv\Scripts\activate`
   * Linux: `source venv/bin/activate`

3. **Instal Dependensi**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

   > *Pastikan `requirements.txt` berisi `ultralytics` dan `opencv-python`.*

---

## Training Model

Jika ingin melatih ulang model dengan dataset sendiri:

```bash
python train.py
```

Model terbaik akan tersimpan di:

```
runs/detect/NAMA_TRAINING/weights/best.pt
```

---

## Deteksi Real-time

1. Buka file `predict_webcam.py`
2. Ubah variabel:

   ```python
   MODEL_PATH = 'runs/detect/yolov8n_ppe_custom4/weights/best.pt'
   ```
3. Jalankan deteksi:

   ```bash
   python predict_webcam.py
   ```

   Tekan **q** untuk keluar.
