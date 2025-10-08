import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import time

# Path ke model hasil training Anda
MODEL_PATH = 'file best atau last .pt'

# Sesuaikan daftar ini dengan kelas pelanggaran dari model Anda.
CLASSES_PELANGGARAN = ['no hat', 'no vest']

# Direktori untuk menyimpan bukti pelanggaran (akan dibuat otomatis)
OUTPUT_DIR = '/home/azunya/Documents/Yolo/hasil'

# Waktu jeda (dalam detik) sebelum menyimpan bukti baru untuk pelanggaran berikutnya
COOLDOWN_SECONDS = 5.0

# Buat direktori output jika belum ada
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Muat model yang sudah ditraining
model = YOLO(MODEL_PATH)

# Inisialisasi variabel untuk cooldown
last_capture_time = 0

# Buka koneksi ke webcam (2 untuk kamera eksternal Anda)
cap = cv2.VideoCapture(0)

# Periksa apakah webcam berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# --- PENGATURAN JENDELA DAN ZOOM ---
WINDOW_NAME = "YOLOv8 PPE Detection"
zoom_level = 1.0
ZOOM_SPEED = 0.1

# Buat jendela yang bisa diubah ukurannya
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# --- LOOP UTAMA ---
while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
    success, frame = cap.read()
    if not success:
        print("Gagal membaca frame dari kamera.")
        break

    frame = cv2.flip(frame, 1)

    # --- LOGIKA UNTUK ZOOM IN DAN ZOOM OUT ---
    h, w, _ = frame.shape
    
    if zoom_level < 1.0:
        new_w, new_h = int(w * zoom_level), int(h * zoom_level)
        shrunken_frame = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros_like(frame)
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = shrunken_frame
        processed_frame = canvas
    else:
        new_w, new_h = int(w / zoom_level), int(h / zoom_level)
        center_x, center_y = w // 2, h // 2
        x1, y1 = center_x - new_w // 2, center_y - new_h // 2
        x2, y2 = center_x + new_w // 2, center_y + new_h // 2
        cropped_frame = frame[y1:y2, x1:x2]
        processed_frame = cv2.resize(cropped_frame, (w, h))

    # Lakukan deteksi objek pada frame yang sudah diproses
    results = model(processed_frame, stream=True, verbose=False)

    annotated_frame = None  # Inisialisasi annotated_frame

    for r in results:
        annotated_frame = r.plot()
        
        # 1. Dapatkan semua nama kelas yang terdeteksi di frame ini
        detected_classes_indices = r.boxes.cls.cpu().numpy()
        detected_classes_names = [r.names[int(i)] for i in detected_classes_indices]
        
        # 2. Cek apakah ada kelas pelanggaran yang terdeteksi
        pelanggaran_ditemukan = any(item in detected_classes_names for item in CLASSES_PELANGGARAN)

        # 3. Jika ada pelanggaran DAN cooldown sudah selesai
        if pelanggaran_ditemukan and (time.time() - last_capture_time) > COOLDOWN_SECONDS:
            
            # Buat nama file yang unik berdasarkan tanggal dan waktu
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"pelanggaran_{timestamp}.jpg"
            file_path = os.path.join(OUTPUT_DIR, filename)
            
            # Simpan frame yang sudah dianotasi (ada kotaknya)
            cv2.imwrite(file_path, annotated_frame)
            
            print(f"✅ Pelanggaran terdeteksi! Bukti disimpan di: {file_path}")
            
            # Perbarui waktu terakhir capture untuk memulai cooldown lagi
            last_capture_time = time.time()
            
    # Tampilkan frame ke jendela (pastikan annotated_frame tidak None)
    if annotated_frame is not None:
        cv2.imshow(WINDOW_NAME, annotated_frame)
        
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        zoom_level = min(zoom_level + ZOOM_SPEED, 5.0)
    elif key == ord('-'):
        zoom_level = max(zoom_level - ZOOM_SPEED, 0.2)
        
print("Menutup program...")
cap.release()
cv2.destroyAllWindows()
