import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- Konfigurasi ---
IMG_SIZE = 640
IMAGE_DIR = "gambar"  # Ganti ke folder gambar kamu
MODEL_PATH = "pose/train/weights/best.pt"  # Ganti ke path model kamu

# Load model
model = YOLO(MODEL_PATH)

# --- Fungsi untuk menggambar prediksi ---
# Buat warna berbeda per class
COLORS = {
    0: (0, 255, 0),   # class 0 → hijau
    1: (255, 0, 0),   # class 1 → biru
    2: (0, 0, 255),   # class 2 → merah
}

def draw_prediction(image, result, label="Model"):
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls.cpu().item()) if box.cls is not None else -1
            color = COLORS.get(cls, (255, 255, 255))  # default putih
            xyxy = box.xyxy.cpu().numpy().astype(int)[0]
            conf = box.conf.cpu().item()
            cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(image, f"{label} {conf:.2f} (cls {cls})", (xyxy[0], xyxy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if hasattr(result, 'keypoints') and result.keypoints is not None:
        for kps in result.keypoints.data:
            for x, y, conf in kps.cpu().numpy():
                if conf > 0.5:
                    cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)  # bisa dibuat per class juga
    return image



# --- Loop gambar ---
for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Failed to load: {filename}")
        continue

    # Resize (letterbox style)
    h0, w0 = image.shape[:2]
    scale = IMG_SIZE / max(h0, w0)
    resized = cv2.resize(image, (int(w0 * scale), int(h0 * scale)))
    padded = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    resized = padded

    # Prediksi dan visualisasi
    result = model.predict(
        source=resized,
        imgsz=IMG_SIZE,
        conf=0.16,        # ⬅️ threshold bounding box
        iou=0.45,         # ⬅️ threshold IOU untuk NMS
        verbose=False,
        device='cpu'
    )[0]

    pred_img = draw_prediction(resized.copy(), result, "Predicted")

    # Tampilkan
    combined_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(combined_rgb)
    plt.title(f"Prediction — {filename}", fontsize=14)
    plt.axis("off")
    plt.show()
