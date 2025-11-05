import cv2
import torch
from ultralytics import YOLOE 
import time
import os  # --- TAMBAHAN: Dibutuhkan untuk membuat folder dan menyimpan file ---

# --- Fungsi Helper IoU ---
def calculate_iou(box1, box2):
    """
    Menghitung Intersection over Union (IoU) antara dua bounding box.
    Format box: [x1, y1, x2, y2]
    """
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
WEBCAM_INDEX = 2  # Sesuaikan dengan webcam Anda
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.01 # Seberapa besar tumpang tindih untuk dianggap "terhubung"

WINDOW_NAME = "YOLOe PPE Detection (Tekan 'f' untuk FPS, 'q' untuk keluar)"

# --- TAMBAHAN: Konfigurasi Penyimpanan Pelanggaran ---
SAVE_DIR = "/home/azunya/kuliah/sem5/magang/project/yoloe_helmet_detector/hasil"
SAVE_COOLDOWN_SECONDS = 5.0 # Hanya simpan 1 gambar setiap 5 detik
# ----------------------------------------------------

# --- 2. Muat Model ---
print(f"Loading YOLOe model '{MODEL_ID}' to {DEVICE}...")
model = YOLOE(MODEL_ID)
model.to(DEVICE)

# --- TAMBAHAN: Buat folder 'hasil' jika belum ada ---
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Folder penyimpanan pelanggaran diatur ke: {SAVE_DIR}")
# ----------------------------------------------------

# --- 3. Atur "Vocabulary" (Objek yang akan dideteksi) ---
CLASSES = ["person", "a safety helmet", "a safety boot"]
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

# --- Inisialisasi variabel FPS dan Pelanggaran ---
prev_time = 0
fps = 0
show_fps = True  # --- TAMBAHAN: Variabel untuk toggle FPS ---
last_save_time = 0 # --- TAMBAHAN: Variabel untuk cooldown simpan ---
# ------------------------------------------

# --- 5. Loop Utama ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Buat salinan frame untuk digambar
    annotated_frame = frame.copy()

    # --- 6. Lakukan Deteksi (HANYA SATU KALI) ---
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    # --- 7. Logika Asosiasi (Post-processing) ---
    person_boxes = []
    ppe_items = [] # (box, label, conf)
    violation_detected_in_frame = False # Flag untuk frame ini

    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue
                
            cls_id = int(box.cls[0])
            label = model.names[cls_id] # Mendapatkan label teks
            
            if "person" in label.lower():
                person_boxes.append(box.xyxy[0].tolist()) # [x1, y1, x2, y2]
            else:
                ppe_items.append((box.xyxy[0].tolist(), label, conf))

    # --- 8. Menggambar Hasil & Cek Pelanggaran ---

    # Gambar PPE (helm/sepatu)
    for ppe_box, label, conf in ppe_items:
        is_associated = False
        for p_box in person_boxes:
            if calculate_iou(ppe_box, p_box) > IOU_THRESHOLD:
                is_associated = True
                break
        
        x1, y1, x2, y2 = map(int, ppe_box)
        if is_associated:
            color = (0, 255, 0) if "helmet" in label else (255, 0, 0)
        else:
            color = (128, 128, 128)
        
        label_text = f"{label}: {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Gambar Person & Cek Pelanggaran Helm
    for p_box in person_boxes:
        x1, y1, x2, y2 = map(int, p_box)
        
        has_helmet = False
        # Cek apakah person ini terasosiasi dengan helm
        for ppe_box, label, _ in ppe_items:
            if "helmet" in label and calculate_iou(p_box, ppe_box) > IOU_THRESHOLD:
                has_helmet = True
                break
        
        if has_helmet:
            # Person aman (memakai helm)
            color = (0, 255, 0) # Hijau
            label_text = "Person (Aman)"
        else:
            # Person melanggar (tidak memakai helm)
            color = (0, 0, 255) # Merah
            label_text = "Person (PELANGGARAN)"
            violation_detected_in_frame = True # Tandai frame ini untuk disimpan
            
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # --- 9. Logika Simpan Gambar Pelanggaran ---
    current_time = time.time()
    if violation_detected_in_frame and (current_time - last_save_time) > SAVE_COOLDOWN_SECONDS:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(SAVE_DIR, f"violation_{timestamp}.jpg")
        cv2.imwrite(filename, annotated_frame) # Simpan frame yang sudah dianotasi
        print(f"PELANGGARAN TERDETEKSI! Gambar disimpan ke: {filename}")
        last_save_time = current_time # Setel ulang cooldown

    # --- 10. Hitung dan Tampilkan FPS ---
    time_diff = current_time - prev_time
    prev_time = current_time
    
    if time_diff > 0:
        fps = 1 / time_diff

    # --- TAMBAHAN: Hanya tampilkan FPS jika 'show_fps' = True ---
    if show_fps:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # --------------------------------------------------------

    # --- 11. Tampilkan Hasil ---
    cv2.imshow(WINDOW_NAME, annotated_frame)

    # --- 12. Logika untuk keluar ---
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == 27: # Keluar dengan 'q' atau 'ESC'
        break
        
    if key == ord('f'): # --- TAMBAHAN: Toggle FPS ---
        show_fps = not show_fps
    
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1: # Keluar dengan tombol 'X'
        break

# --- 13. Bersihkan ---
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
print("Application closed.")
