import cv2
import torch
# Kita mengimpor YOLOE secara spesifik dari ultralytics
from ultralytics import YOLOE 

# --- 1. Konfigurasi ---
# Kita gunakan 's' (small) untuk kecepatan.
# Model YOLOe adalah model segmentasi, tapi juga menghasilkan bounding box.
MODEL_ID = 'yoloe-11s-seg.pt' 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WEBCAM_INDEX = 2  # Sesuaikan dengan webcam Anda
CONF_THRESHOLD = 0.3 # Ambang batas kepercayaan

# --- 2. Muat Model ---
print(f"Loading YOLOe model '{MODEL_ID}' to {DEVICE}...")
# Kita panggil kelas YOLOE secara eksplisit
model = YOLOE(MODEL_ID)
model.to(DEVICE)

# --- 3. Atur "Vocabulary" (Objek yang akan dideteksi) ---
# Ini adalah pengganti TEXT_PROMPT
CLASSES = ["a safety helmet", "a safety boot"]
print(f"Setting model to detect: {CLASSES}")

# YOLOe memerlukan sintaks 'set_classes' yang spesifik
# Ini membuat "text embedding" sebelum loop agar lebih cepat
text_embeddings = model.get_text_pe(CLASSES)
model.set_classes(CLASSES, text_embeddings)

# --- 4. Buka Webcam ---
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open webcam at index {WEBCAM_INDEX}.")
    exit()

print("Webcam opened. Press 'q' to quit.")

# --- 5. Loop Utama (Tidak Perlu Threading) ---
# Model ini sangat cepat dan berjalan di GPU Anda, 
# jadi tidak perlu threading manual untuk anti-lag.
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # --- 6. Lakukan Deteksi ---
    # 'verbose=False' untuk mematikan log di terminal
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    # --- 7. Gambar Hasil Deteksi ---
    # Kita gunakan fungsi .plot() bawaan untuk menggambar
    # box dan label secara otomatis.
    annotated_frame = results[0].plot()

    # --- 8. Tampilkan Hasil ---
    cv2.imshow("YOLOe Detection (Press 'q' to quit)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 9. Bersihkan ---
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
print("Application closed.")
