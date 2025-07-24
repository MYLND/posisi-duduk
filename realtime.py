import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np
import math
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Page configuration
st.set_page_config(
    page_title="Deteksi dan Klasifikasi Pose",
    page_icon="🤸‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CLASS_LABELS = {
    0: "Postur Buruk",
    1: "Postur Baik"
}

COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
}

KEYPOINT_CONNECTIONS = [(0, 1), (1, 2)]

# Header
st.markdown('<h1 class="main-header">Deteksi dan Klasifikasi Pose</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analisis postur tubuh dengan deteksi pose bertenaga AI menggunakan YOLO v8</p>', unsafe_allow_html=True)

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# Sidebar Configuration
with st.sidebar:
    st.header("Konfigurasi")
    
    # Model settings
    st.subheader("Pengaturan Deteksi")
    confidence_threshold = st.slider("Ambang Batas Kepercayaan", 0.1, 1.0, 0.5, 0.05)
    image_size = st.selectbox("Ukuran Gambar", [320, 640, 1280], index=1)
    
    # Display settings
    st.subheader("Opsi Tampilan")
    show_keypoints = st.checkbox("Tampilkan Titik Kunci", value=True)
    show_connections = st.checkbox("Tampilkan Koneksi", value=True)
    show_angles = st.checkbox("Tampilkan Sudut", value=True)
    show_confidence = st.checkbox("Tampilkan Kepercayaan", value=True)
    
    # Advanced settings
    with st.expander("Pengaturan Lanjutan"):
        keypoint_threshold = st.slider("Kepercayaan Titik Kunci", 0.1, 1.0, 0.5, 0.05)
        line_thickness = st.slider("Ketebalan Garis", 1, 5, 2)
        text_scale = st.slider("Skala Teks", 0.3, 1.0, 0.6, 0.1)

# Model loading
@st.cache_resource
def load_model():
    model_path = "model/best.pt"
    
    if not os.path.exists(model_path):
        st.error("File model tidak ditemukan: " + model_path)
        st.info("Pastikan file model ada di direktori yang benar")
        return None
    
    with st.spinner("Memuat model YOLO..."):
        return YOLO(model_path)

# Load model
model = load_model()

if model is None:
    st.stop()

st.sidebar.success("Model berhasil dimuat!")

def calculate_angle(a, b, c):
    if None in (a, b, c):
        return None
    
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_pose_with_label(frame, keypoints_obj, label, box, conf_score):
    color = COLORS.get(label, (255, 255, 255))
    label_text = CLASS_LABELS.get(label, "Unknown")

    keypoints = keypoints_obj.xy[0].cpu().numpy()
    confs = keypoints_obj.conf[0].cpu().numpy()

    # Draw keypoints
    pts = []
    for i, (x, y) in enumerate(keypoints):
        if i < len(confs) and confs[i] > keypoint_threshold:
            pt = (int(x), int(y))
            pts.append(pt)
            
            if show_keypoints:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                cv2.circle(frame, pt, 6, (255, 255, 255), 2)
        else:
            pts.append(None)

    # Draw connections
    if show_connections:
        for i, j in KEYPOINT_CONNECTIONS:
            if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
                cv2.line(frame, pts[i], pts[j], color, line_thickness)

    # Calculate and display angle
    if show_angles and len(pts) >= 3 and all(pts[k] for k in [0, 1, 2]):
        angle = calculate_angle(pts[0], pts[1], pts[2])
        if angle is not None:
            pos = pts[1]
            angle_text = f"{int(angle)}°"
            
            (text_width, text_height), _ = cv2.getTextSize(
                angle_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2
            )
            cv2.rectangle(
                frame, 
                (pos[0] + 5, pos[1] - text_height - 15), 
                (pos[0] + text_width + 10, pos[1] - 5), 
                (0, 0, 0), 
                -1
            )
            
            cv2.putText(
                frame, angle_text, 
                (pos[0] + 8, pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), 2
            )

    # Draw bounding box and label
    if box is not None:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        display_text = label_text
        if show_confidence:
            display_text += f" ({conf_score:.2f})"
        
        # Background for label
        (text_width, text_height), _ = cv2.getTextSize(
            display_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2
        )
        cv2.rectangle(
            frame, (x1, y1 - text_height - 10), 
            (x1 + text_width + 10, y1), color, -1
        )
        
        # Label text
        cv2.putText(
            frame, display_text, (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 2
        )

    return frame

def process_frame_detection(frame):
    results = model.predict(frame, imgsz=image_size, conf=confidence_threshold, save=False, verbose=False)

    detection_count = 0
    pose_results = []

    for result in results:
        boxes = result.boxes
        kpts = result.keypoints
        
        if boxes is not None and kpts is not None:
            for box, kp in zip(boxes, kpts):
                label = int(box.cls.cpu().item())
                conf_score = float(box.conf.cpu().item())
                
                frame = draw_pose_with_label(frame, kp, label, box, conf_score)
                
                detection_count += 1
                pose_results.append({
                    'label': CLASS_LABELS.get(label, 'Unknown'),
                    'confidence': conf_score,
                    'bbox': box.xyxy[0].cpu().numpy().tolist()
                })

    return frame, detection_count, pose_results

# WebRTC Video Transformer Class
class PoseDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.detection_count = 0
        self.good_posture_count = 0
        self.bad_posture_count = 0
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process frame with pose detection
        processed_img, detection_count, pose_results = process_frame_detection(img)
        
        # Update statistics
        self.frame_count += 1
        self.detection_count = detection_count
        
        # Count posture types
        for result in pose_results:
            if result['label'] == 'Postur Baik':
                self.good_posture_count += 1
            else:
                self.bad_posture_count += 1
        
        return processed_img

def process_image(image):
    if isinstance(image, Image.Image):
        image_array = np.array(image)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            frame = image_array
    else:
        frame = image
    
    processed_frame, detection_count, pose_results = process_frame_detection(frame)
    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    
    return processed_rgb, detection_count, pose_results

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Tidak dapat membuka file video")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Display video info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("FPS", fps)
    with col2:
        st.metric("Total Frame", total_frames)
    with col3:
        st.metric("Durasi", f"{duration:.1f}s")
    
    # Create placeholders
    video_placeholder = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Statistics
    frame_count = 0
    total_detections = 0
    good_posture_count = 0
    bad_posture_count = 0
    
    # Process video frames
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, detection_count, pose_results = process_frame_detection(frame)
        
        # Update statistics
        total_detections += detection_count
        for result in pose_results:
            if result['label'] == 'Postur Baik':
                good_posture_count += 1
            else:
                bad_posture_count += 1
        
        # Display processed frame
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update progress
        frame_count += 1
        progress = frame_count / total_frames if total_frames > 0 else 0
        progress_bar.progress(progress)
        
        # Update status
        elapsed_time = time.time() - start_time
        processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        status_text.text(f"Memproses frame {frame_count}/{total_frames} | {processing_fps:.1f} FPS")
    
    cap.release()
    
    # Final statistics
    st.success("Pemrosesan video selesai!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Deteksi", total_detections)
    with col2:
        st.metric("Postur Baik", good_posture_count)
    with col3:
        st.metric("Postur Buruk", bad_posture_count)
    with col4:
        accuracy = (good_posture_count / (good_posture_count + bad_posture_count)) * 100 if (good_posture_count + bad_posture_count) > 0 else 0
        st.metric("Persentase Postur Baik", f"{accuracy:.1f}%")

# Main Interface
st.markdown("---")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Unggah Gambar", "Webcam Real-time", "Unggah Video"])

# Tab 1: Image Upload
with tab1:
    st.subheader("Unggah Gambar untuk Deteksi Pose")
    
    uploaded_image = st.file_uploader(
        "Pilih file gambar",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Unggah gambar yang berisi orang untuk deteksi dan klasifikasi pose"
    )
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Gambar Asli**")
            st.image(image, use_container_width=True)
            
            # Image info
            st.markdown(f"""
            <div class="info-box">
                <strong>Informasi Gambar:</strong><br>
                Ukuran: {image.size[0]} x {image.size[1]} piksel<br>
                Mode: {image.mode}<br>
                Format: {image.format}<br>
                Ukuran file: {uploaded_image.size / 1024:.1f} KB
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("Analisis Pose", type="primary"):
                with st.spinner("Menganalisis pose..."):
                    processed_image, detection_count, pose_results = process_image(image)
                
                st.markdown("**Hasil Pemrosesan**")
                st.image(processed_image, use_container_width=True)
                
                # Results summary
                if detection_count > 0:
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>Analisis Selesai!</strong><br>
                        Pose terdeteksi: {detection_count}<br>
                        Pemrosesan berhasil
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed results
                    with st.expander("Hasil Detail"):
                        for i, result in enumerate(pose_results, 1):
                            st.write(f"**Orang {i}:**")
                            st.write(f"- Klasifikasi: {result['label']}")
                            st.write(f"- Kepercayaan: {result['confidence']:.2%}")
                            st.write("---")
                else:
                    st.warning("Tidak ada pose yang terdeteksi dalam gambar. Coba sesuaikan ambang batas kepercayaan.")

# Tab 2: Real-time Webcam with WebRTC
with tab2:
    st.subheader("Deteksi Pose Webcam Real-time")
    
    # Instructions
    st.markdown("""
    <div class="info-box">
        <strong>Petunjuk Webcam WebRTC:</strong><br>
        1. Klik "START" untuk memulai streaming webcam<br>
        2. Izinkan akses kamera ketika diminta oleh browser<br>
        3. Posisikan diri Anda di depan kamera<br>
        4. AI akan menganalisis postur Anda secara real-time<br>
        5. Klik "STOP" untuk mengakhiri sesi
    </div>
    """, unsafe_allow_html=True)
    
    # WebRTC Streamer
    webrtc_ctx = webrtc_streamer(
        key="pose-detection",
        video_transformer_factory=PoseDetectionTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=True,
    )
    
    # Real-time statistics
    if webrtc_ctx.video_transformer:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Jumlah Frame", webrtc_ctx.video_transformer.frame_count)
        with col2:
            st.metric("Deteksi Saat Ini", webrtc_ctx.video_transformer.detection_count)
        with col3:
            st.metric("Total Postur Baik", webrtc_ctx.video_transformer.good_posture_count)
        with col4:
            st.metric("Total Postur Buruk", webrtc_ctx.video_transformer.bad_posture_count)
        
        # Session statistics
        total_postures = webrtc_ctx.video_transformer.good_posture_count + webrtc_ctx.video_transformer.bad_posture_count
        if total_postures > 0:
            good_percentage = (webrtc_ctx.video_transformer.good_posture_count / total_postures) * 100
            
            st.markdown(f"""
            <div class="success-box">
                <strong>Ringkasan Sesi:</strong><br>
                Tingkat Postur Baik: {good_percentage:.1f}%<br>
                Total Frame Diproses: {webrtc_ctx.video_transformer.frame_count}<br>
                Total Deteksi Postur: {total_postures}
            </div>
            """, unsafe_allow_html=True)

# Tab 3: Video Upload
with tab3:
    st.subheader("Unggah Video untuk Deteksi Pose")
    
    uploaded_video = st.file_uploader(
        "Pilih file video",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Unggah file video untuk analisis deteksi pose secara batch"
    )
    
    if uploaded_video is not None:
        # Video info
        st.markdown(f"""
        <div class="info-box">
            <strong>Informasi Video:</strong><br>
            Nama file: {uploaded_video.name}<br>
            Ukuran file: {uploaded_video.size / (1024*1024):.2f} MB<br>
            Tipe: {uploaded_video.type}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Proses Video", type="primary"):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                temp_video_path = tfile.name
            
            process_video(temp_video_path)
            
            # Clean up temporary file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)

# Tips Section
st.markdown("---")
st.subheader("Tips untuk Deteksi Pose yang Lebih Baik")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Pengaturan Kamera**
    - Pastikan pencahayaan yang baik
    - Posisikan kamera setinggi mata
    - Jaga jarak 1-2 meter
    - Hindari latar belakang yang sibuk
    """)

with col2:
    st.markdown("""
    **Tips Deteksi**
    - Duduk tegak untuk deteksi yang lebih baik
    - Kenakan pakaian dengan warna kontras
    - Hindari pakaian longgar/kebesaran
    - Tetap berada dalam frame kamera
    """)

with col3:
    st.markdown("""
    **Pengaturan**
    - Turunkan kepercayaan untuk sensitivitas
    - Sesuaikan ukuran gambar untuk performa
    - Toggle opsi tampilan sesuai kebutuhan
    - Periksa pengaturan lanjutan
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;'>
    <h4 style='color: #2c3e50; margin-bottom: 1rem;'>Sistem Deteksi Pose Bertenaga AI</h4>
    <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;'>
        <div><strong>Teknologi:</strong> YOLO v8 + OpenCV + Streamlit + WebRTC</div>
        <div><strong>Model:</strong> Klasifikasi pose hasil pelatihan khusus</div>
        <div><strong>Fitur:</strong> Dukungan webcam real-time</div>
    </div>
    <p style='margin-top: 1rem; color: #7f8c8d; font-style: italic;'>
        Analisis postur manusia dengan teknologi AI mutakhir
    </p>
</div>
""", unsafe_allow_html=True)
