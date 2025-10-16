import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import time

MODEL_PATH = '/home/azunya/kuliah/sem5/magang/project/K3_DETECTION_YOLOV8/runs/detect/newscript-yolov11X4/weights/best.pt'
CLASSES_PELANGGARAN = ['no_helmet', 'no_savety_shoes']
BASE_OUTPUT_DIR = '/home/azunya/kuliah/sem5/magang/project/K3_DETECTION_YOLOV8/bukti_pelanggaran/new'

COOLDOWN_SECONDS_PER_ID = 10.0

def auto_zoom_frame(frame, box, target_zoom=2.0):
    """
    Memotong frame di sekitar bounding box untuk menciptakan efek digital zoom.
    Penting: Digital zoom akan mengurangi kualitas gambar.
    """
    x1, y1, x2, y2 = map(int, box)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    frame_h, frame_w, _ = frame.shape
    
    crop_w = int(frame_w / target_zoom)
    crop_h = int(frame_h / target_zoom)
    
    crop_x1 = max(0, center_x - crop_w // 2)
    crop_y1 = max(0, center_y - crop_h // 2)
    crop_x2 = min(frame_w, crop_x1 + crop_w)
    crop_y2 = min(frame_h, crop_y1 + crop_h)
    
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    try:
        zoomed_frame = cv2.resize(cropped_frame, (frame_w, frame_h))
    except cv2.error:
        return frame
        
    return zoomed_frame

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error memuat model: {e}\nPastikan path model '{MODEL_PATH}' sudah benar.")
    exit()

cooldown_tracker = {}
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

WINDOW_NAME = "Sistem Deteksi K3 (Bukti Tersimpan per Kategori)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

while cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
    success, frame = cap.read()
    if not success:
        print("Gagal membaca frame dari kamera.")
        break

    frame = cv2.flip(frame, 1)
    
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    annotated_frame = results[0].plot()

    if results[0].boxes.id is not None:
        tracker_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.cpu().tolist()
        
        for track_id, cls_idx in zip(tracker_ids, class_indices):
            class_name = model.names[int(cls_idx)]

            if class_name in CLASSES_PELANGGARAN:
                current_time = time.time()
                last_capture_time = cooldown_tracker.get(track_id, 0)

                if (current_time - last_capture_time) > COOLDOWN_SECONDS_PER_ID:
                    
                    
                    # path folder kategori pelanggaran
                    category_folder_path = os.path.join(BASE_OUTPUT_DIR, class_name)
                    
                    # Buat folder kategori jika belum ada
                    os.makedirs(category_folder_path, exist_ok=True)
                    
                    # Buat nama file dan path lengkapnya di dalam folder kategori
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pelanggaran_ID-{track_id}_{timestamp}.jpg"
                    file_path = os.path.join(category_folder_path, filename)
                    
                    cv2.imwrite(file_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    print(f"✅ PELANGGARAN [ID: {track_id}, Kategori: {class_name}]. Bukti disimpan di folder '{class_name}'.")
                    cooldown_tracker[track_id] = current_time

    cv2.imshow(WINDOW_NAME, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Menutup program...")
cap.release()
cv2.destroyAllWindows()
