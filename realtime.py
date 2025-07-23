import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np
import math

CLASS_LABELS = {
    0: "Bad",
    1: "Good"
}

COLORS = {
    0: (0, 255, 0),  # Bad → hijau
    1: (255, 0, 0),  # Good → biru
}

KEYPOINT_CONNECTIONS = [(0, 1), (1, 2)]  

@st.cache_resource
def load_model():
    return YOLO("pose2/train2/weights/best.pt")  

model = load_model()

st.title("Deteksi Pose & Tracking dengan Sudut (3 Keypoints)")
source = st.radio("Pilih sumber input:", ["Webcam", "Upload Video"])

def calculate_angle(a, b, c):
    if None in (a, b, c):
        return None
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_pose_with_label(frame, keypoints_obj, label, box):
    color = COLORS.get(label, (255, 255, 255))

    try:
        keypoints = keypoints_obj.xy[0].cpu().numpy()
        confs = keypoints_obj.conf[0].cpu().numpy()
    except Exception as e:
        print("Keypoint error:", e)
        return frame

    pts = []
    for i, (x, y) in enumerate(keypoints):
        if confs[i] > 0.5:
            pt = (int(x), int(y))
            pts.append(pt)
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        else:
            pts.append(None)

    for i, j in KEYPOINT_CONNECTIONS:
        if i < len(pts) and j < len(pts):
            if pts[i] and pts[j]:
                cv2.line(frame, pts[i], pts[j], color, 2)
    if len(pts) >= 3 and all(pts[k] for k in [0, 1, 2]):
        angle = calculate_angle(pts[0], pts[1], pts[2])
        if angle is not None:
            pos = pts[1]
            cv2.putText(frame, f"{int(angle)}°", (pos[0] + 5, pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if box is not None:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, CLASS_LABELS.get(label, "Unknown"), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def infer_and_display(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=640, conf=0.5, save=False)

        for result in results:
            boxes = result.boxes
            kpts = result.keypoints
            if boxes is not None and kpts is not None:
                for box, kp in zip(boxes, kpts):
                    label = int(box.cls.cpu().item())
                    frame = draw_pose_with_label(frame, kp, label, box)

        stframe.image(frame, channels="BGR", use_container_width=True)

    cap.release()

if source == "Webcam":
    run = st.checkbox("Mulai Webcam")
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, imgsz=640, conf=0.5, save=False)

            for result in results:
                boxes = result.boxes
                kpts = result.keypoints
                if boxes is not None and kpts is not None:
                    for box, kp in zip(boxes, kpts):
                        label = int(box.cls.cpu().item())
                        frame = draw_pose_with_label(frame, kp, label, box)

            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()

else:
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        infer_and_display(tfile.name)
        os.unlink(tfile.name)
