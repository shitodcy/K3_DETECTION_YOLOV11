import cv2
import torch
from ultralytics import YOLOE 

# --- 1. Konfigurasi ---
MODEL_ID = 'yoloe-11s-seg.pt' 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WEBCAM_INDEX = 2  # Sesuaikan dengan webcam Anda
CONF_THRESHOLD = 0.3

# --- PERUBAHAN 1: Beri nama jendela dan buat resizable ---
# Ini akan memungkinkan Anda mengubah ukuran jendela.
# Pada beberapa OS, ini juga mengaktifkan tombol Maximize.
WINDOW_NAME = "YOLOe Detection (Close window or press 'q' to quit)"

# --- 2. Muat Model ---
print(f"Loading YOLOe model '{MODEL_ID}' to {DEVICE}...")
model = YOLOE(MODEL_ID)
model.to(DEVICE)

# --- 3. Atur "Vocabulary" (Objek yang akan dideteksi) ---
CLASSES = ["a safety helmet", "a black safety shoe"]
print(f"Setting model to detect: {CLASSES}")

text_embeddings = model.get_text_pe(CLASSES)
model.set_classes(CLASSES, text_embeddings)

# --- 4. Buka Webcam ---
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open webcam at index {WEBCAM_INDEX}.")
    exit()

# --- PERUBAHAN 2: Buat jendela SEBELUM loop ---
# Kita gunakan cv2.WINDOW_NORMAL agar resizable
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 
print("Webcam opened. Press 'q' or close the window to quit.")

# --- 5. Loop Utama (Tidak Perlu Threading) ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # --- 6. Lakukan Deteksi ---
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    # --- 7. Gambar Hasil Deteksi ---
    annotated_frame = results[0].plot()

    # --- 8. Tampilkan Hasil ---
    # Tampilkan gambar di jendela yang sudah kita buat
    cv2.imshow(WINDOW_NAME, annotated_frame)

    # --- PERUBAHAN 3: Logika untuk keluar ---
    key = cv2.waitKey(1) & 0xFF

    # Opsi 1: Keluar jika 'q' atau 'ESC' ditekan
    if key == ord('q') or key == 27:
        break

    # Opsi 2: Keluar jika tombol 'X' jendela diklik
    # Kita cek apakah properti jendela masih 'visible'
    # Jika tidak (misal < 1), berarti jendela ditutup
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

# --- 9. Bersihkan ---
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
print("Application closed.")
