import cv2
from ultralytics import YOLO

# Path ke model hasil training Anda
MODEL_PATH = 'runs/detect/yolov8n_ppe_custom4/weights/best.pt'

# Muat model yang sudah ditraining
model = YOLO(MODEL_PATH)

# Buka koneksi ke webcam (0 biasanya adalah webcam bawaan laptop)
cap = cv2.VideoCapture(0)

# Periksa apakah webcam berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# Loop untuk membaca frame dari webcam secara terus-menerus
while True:
    # Baca satu frame dari kamera
    success, frame = cap.read()

    if success:
        # Balik frame secara horizontal untuk menghilangkan efek mirror
        frame = cv2.flip(frame, 1) # <<-- TAMBAHKAN BARIS INI

        # Lakukan deteksi objek pada frame
        # stream=True direkomendasikan untuk video agar lebih efisien memori
        results = model(frame, stream=True, verbose=False)

        # Visualisasikan hasil deteksi pada frame
        for r in results:
            annotated_frame = r.plot()  # r.plot() akan menggambar box dan label secara otomatis

            # Tampilkan frame yang sudah dianotasi
            cv2.imshow("YOLOv8 PPE Detection", annotated_frame)

        # Hentikan loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Hentikan loop jika gagal membaca frame
        print("Gagal membaca frame dari kamera.")
        break
    
cap.release()
cv2.destroyAllWindows()