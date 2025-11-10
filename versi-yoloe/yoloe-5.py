import cv2
import torch
from ultralytics import YOLOE 
import time
import os
import datetime # --- TAMBAHAN: Dibutuhkan untuk tanggal/waktu ---

# --- Fungsi Helper IoU ---
def calculate_iou(box1, box2):
    # ... (Fungsi IoU Anda tetap sama) ...
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou
# -------------------------


# --- 1. Konfigurasi ---
MODEL_ID = 'yoloe-11s-seg.pt' 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WEBCAM_INDEX = 2
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.01

WINDOW_NAME = "YOLOe PPE Detection (Tekan 'f' untuk FPS, 'q' untuk keluar)"

# --- Konfigurasi Penyimpanan Pelanggaran ---
SAVE_DIR = "/home/ovensave/Documents/K3_DETECTION_YOLOV8/hasil"
SAVE_COOLDOWN_SECONDS = 5.0
# ----------------------------------------------------

# --- 2. Muat Model ---
print(f"Loading YOLOe model '{MODEL_ID}' to {DEVICE}...")
model = YOLOE(MODEL_ID)
model.to(DEVICE)

# Pastikan folder 'hasil' ada
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Folder penyimpanan pelanggaran diatur ke: {SAVE_DIR}")

# --- 3. Atur "Vocabulary" (Objek yang akan dideteksi) ---
CLASSES = ["person", "a safety helmet", "a safety boot", "an id card"]
print(f"Setting model to detect: {CLASSES}")

text_embeddings = model.get_text_pe(CLASSES)
model.set_classes(CLASSES, text_embeddings)

# --- 4. Buka Webcam ---
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open webcam at index {WEBCAM_INDEX}.")
    exit()

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 
print("Webcam opened. Tekan 'f' untuk toggle FPS, 'q' atau tutup jendela untuk keluar.")

# --- Inisialisasi variabel ---
prev_time = 0
fps = 0
show_fps = True
last_save_time = 0
# ------------------------------------------

# --- 5. Loop Utama ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    annotated_frame = frame.copy()
    current_time = time.time() # Waktu saat ini untuk FPS & Cooldown

    # --- 6. Lakukan Deteksi ---
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    # --- 7. Logika Asosiasi (Post-processing) ---
    person_boxes = []
    ppe_items = [] # (box, label, conf)
    
    # --- TAMBAHAN: Daftar unik pelanggaran yang terdeteksi di frame ini ---
    frame_violation_types = set()
    # ------------------------------------------------------------------

    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD: continue
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            if "person" in label.lower():
                person_boxes.append(box.xyxy[0].tolist())
            else:
                ppe_items.append((box.xyxy[0].tolist(), label, conf))

    # --- 8. Menggambar Hasil & Cek Pelanggaran ---

    # Gambar PPE (helm/sepatu) yang tidak terasosiasi dulu
    for ppe_box, label, conf in ppe_items:
        is_associated = False
        for p_box in person_boxes:
            if calculate_iou(ppe_box, p_box) > IOU_THRESHOLD:
                is_associated = True
                break
        
        # Hanya gambar jika TIDAK terasosiasi (warna abu-abu)
        if not is_associated:
            x1, y1, x2, y2 = map(int, ppe_box)
            color = (128, 128, 128)
            label_text = f"{label}: {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Gambar Person & PPE yang terasosiasi
    for p_box in person_boxes:
        x1, y1, x2, y2 = map(int, p_box)
        
        has_helmet = False
        has_boot = False
        has_id_card = False
        
        # Cek helm dan sepatu yang terasosiasi
        for ppe_box, label, conf in ppe_items:
            # Hanya proses PPE yang tumpang tindih
            if calculate_iou(p_box, ppe_box) > IOU_THRESHOLD:
                x_ppe, y_ppe, x2_ppe, y2_ppe = map(int, ppe_box)
                
                if "helmet" in label:
                    has_helmet = True
                    # Gambar helm (hijau)
                    cv2.rectangle(annotated_frame, (x_ppe, y_ppe), (x2_ppe, y2_ppe), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{label}: {conf:.2f}", (x_ppe, y_ppe - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                elif "boot" in label or "shoe" in label:
                    has_boot = True
                    # Gambar sepatu (biru)
                    cv2.rectangle(annotated_frame, (x_ppe, y_ppe), (x2_ppe, y2_ppe), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, f"{label}: {conf:.2f}", (x_ppe, y_ppe - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                elif "id card" in label:
                    has_id_card = True
                    color_id = (0, 255, 255)
                    # Gambar id card (kuning)
                    cv2.rectangle(annotated_frame, (x_ppe, y_ppe), (x2_ppe, y2_ppe), color_id, 2)
                    cv2.putText(annotated_frame, f"{label}: {conf:.2f}", (x_ppe, y_ppe - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_id, 2)

        # Tentukan status pelanggaran & gambar kotak person
        person_violations = []
        if not has_helmet:
            person_violations.append("No_Helmet")
            frame_violation_types.add("No_Helmet")
        
        if not has_boot:
            person_violations.append("No_Boot")
            frame_violation_types.add("No_Boot")

        if not has_id_card:
            person_violations.append("No_ID_Card")
            frame_violation_types.add("No_ID_Card")
        
        if person_violations:
            # Person melanggar (Merah)
            color = (0, 0, 255)
            label_text = f"Person ({', '.join(person_violations)})"
        else:
            # Person aman (Hijau)
            color = (0, 255, 0)
            label_text = "Person (Aman)"
            
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # --- 9. Logika Simpan Gambar Pelanggaran ---
    if frame_violation_types and (current_time - last_save_time) > SAVE_COOLDOWN_SECONDS:
        
        # Dapatkan info tanggal dan waktu
        now = datetime.datetime.now()
        date_str = now.strftime('%Y-%m-%d') # Folder untuk tanggal
        time_str = now.strftime('%Y%m%d_%H%M%S') # Nama file
        filename = f"violation_{time_str}.jpg"

        # Simpan satu gambar untuk setiap jenis pelanggaran yang terdeteksi
        for violation_type in frame_violation_types:
            # Buat path folder dinamis: .../hasil/No_Helmet/2025-11-05/
            target_dir = os.path.join(SAVE_DIR, violation_type, date_str)
            os.makedirs(target_dir, exist_ok=True)
            
            save_path = os.path.join(target_dir, filename)
            cv2.imwrite(save_path, annotated_frame) # Simpan frame yang sudah dianotasi
            print(f"PELANGGARAN DISIMPAN: {save_path}")
        
        last_save_time = current_time # Setel ulang cooldown

    # --- 10. Hitung dan Tampilkan FPS ---
    time_diff = current_time - prev_time
    prev_time = current_time
    
    if time_diff > 0:
        fps = 1 / time_diff

    if show_fps:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # --- 11. Tampilkan Hasil ---
    cv2.imshow(WINDOW_NAME, annotated_frame)

    # --- 12. Logika untuk keluar ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("Tombol 'q' ditekan, keluar...")
        break
    if key == ord('f'):
        show_fps = not show_fps

    # --- PERBAIKAN: Tangani error saat jendela ditutup manual (klik 'X') ---
    try:
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            print("Jendela ditutup manual, keluar...")
            break
    except cv2.error:
        # Ini normal terjadi jika jendela ditutup paksa
        # Anggap ini sebagai sinyal keluar yang valid
        print("Jendela ditutup, menangani cv2.error...")
        break
    # ------------------------------------------------------------------

# --- 13. Bersihkan ---
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
print("Application closed.")
