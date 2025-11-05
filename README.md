# Deteksi PPE Real-time dengan YOLOe

Petunjuk instalasi ini disiapkan untuk menjalankan skrip `yoloe-4.py` pada sistem Arch Linux dengan GPU NVIDIA.

## 1. Prasyarat Sistem (Arch Linux)

Pastikan sistem Anda memiliki prasyarat berikut yang terinstal melalui `pacman`.

1.  **Driver NVIDIA & CUDA:**
    Pastikan Anda memiliki driver NVIDIA, *utils*, dan CUDA toolkit yang terinstal.
    ```bash
    sudo pacman -S nvidia-dkms nvidia-utils cuda
    ```
    *(Reboot setelah menginstal driver jika Anda belum melakukannya.)*

2.  **Python, Git, dan Venv:**
    Anda membutuhkan Python, manajer paket `pip`, `virtualenv` untuk isolasi proyek, dan `git` (diperlukan untuk menginstal *dependency* `clip`).
    ```bash
    sudo pacman -S python python-pip python-virtualenv git
    ```

## 2. Penyiapan Proyek Python

Langkah-langkah ini akan menyiapkan *virtual environment* Python Anda dan menginstal semua *library* yang diperlukan.

1.  **Buat & Aktifkan Virtual Environment:**
    Dari dalam direktori proyek Anda (tempat `yoloe-4.py` berada), jalankan:
    ```bash
    # Buat environment bernama 'venv'
    python -m venv venv
    
    # Aktifkan environment
    source venv/bin/activate
    ```
    *(Terminal Anda sekarang seharusnya diawali dengan `(venv)`)*

2.  **Instal PyTorch (GPU/CUDA):**
    Anda harus menginstal PyTorch terlebih dahulu dengan dukungan CUDA. Kunjungi [situs resmi PyTorch](https://pytorch.org/get-started/locally/) untuk mendapatkan perintah yang paling sesuai dengan versi CUDA Anda. Perintah umum untuk CUDA 12.1 adalah:
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

3.  **Instal Library AI & CV:**
    Sekarang instal `ultralytics` (untuk YOLOe), `opencv` (untuk *webcam*), dan `clip` (yang merupakan *dependency* untuk YOLOe).
    ```bash
    # Instal Ultralytics (YOLO) dan OpenCV
    pip install ultralytics opencv-python

    # Instal 'clip' dari GitHub (diperlukan oleh YOLOe)
    pip install "git+[https://github.com/ultralytics/CLIP.git](https://github.com/ultralytics/CLIP.git)"
    ```

## 3. Menjalankan Aplikasi

Setelah semua instalasi selesai, Anda dapat menjalankan skrip deteksi:

```bash
python3 yoloe-4.py
