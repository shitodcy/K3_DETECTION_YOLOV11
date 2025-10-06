from ultralytics import YOLO

# Muat model pre-trained YOLOv8n. 'n' adalah versi nano, paling ringan dan cepat.
# Cocok untuk memulai dan memastikan semua berjalan lancar.
model = YOLO('yolov8n.pt')

# Mulai proses training
if __name__ == '__main__':
    results = model.train(
        data='archive/data.yaml',   # Path ke file konfigurasi dataset Anda
        epochs=50,               # Jumlah epoch (berapa kali model melihat seluruh dataset)
        imgsz=640,               # Ukuran gambar input diubah menjadi 640x640
        batch=8,                # Jumlah gambar yang diproses dalam satu waktu. Sesuaikan dengan VRAM Anda.
        name='yolov8n_ppe_custom' # Nama folder untuk menyimpan hasil training
    )