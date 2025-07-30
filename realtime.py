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
import threading

# Page configuration
st.set_page_config(
    page_title="Deteksi dan Klasifikasi Pose",
    page_icon="🤸‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Responsive design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: clamp(1rem, 3vw, 1.2rem);
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin-bottom: 1rem;
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
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    /* Mobile responsive adjustments */
    @media (max-width: 768px) {
        .stColumn {
            padding: 0 0.5rem;
        }
        .metric-card {
            padding: 0.5rem;
            font-size: 0.9rem;
        }
    }
    
    /* Video container responsive */
    .video-container {
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .video-container video {
        width: 100% !important;
        height: auto !important;
        max-height: 600px;
        border-radius: 10px;
        border: 2px solid #4ECDC4;
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
st.markdown('<h1 class="main-header">Pose Estimation</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analisis postur tubuh dengan deteksi pose-estimation menggunakan YOLO v8</p>', unsafe_allow_html=True)

# WebRTC Configuration - Enhanced for cross-platform compatibility
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun.services.mozilla.com:3478"]},
        {"urls": ["stun:stun.stunprotocol.org:3478"]},
    ],
    "iceTransportPolicy": "all",
    "bundlePolicy": "balanced",
    "rtcpMuxPolicy": "require"
})

# Detect device type
def get_device_type():
    user_agent = st.context.headers.get("User-Agent", "").lower()
    if any(mobile in user_agent for mobile in ["mobile", "android", "iphone", "ipad"]):
        return "mobile"
    return "desktop"

device_type = get_device_type()

# Sidebar Configuration
with st.sidebar:
    st.header("Konfigurasi")
    
    # Device info
    st.info(f"Device: {device_type.title()}")
    
    # Model settings
    st.subheader("Pengaturan Deteksi")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    
    # Adaptive image size based on device
    if device_type == "mobile":
        image_size_options = [320, 480, 640]
        default_index = 1
    else:
        image_size_options = [320, 640, 1280]
        default_index = 1
    
    image_size = st.selectbox("Ukuran Gambar", image_size_options, index=default_index)
    
    # Display settings
    st.subheader("Opsi Tampilan")
    show_keypoints = st.checkbox("Tampilkan Keypoints", value=True)
    show_connections = st.checkbox("Tampilkan Koneksi", value=True)
    show_angles = st.checkbox("Tampilkan Sudut", value=True)
    show_confidence = st.checkbox("Tampilkan Confidence", value=True)
    
    # Advanced settings
    with st.expander("Pengaturan Lanjutan"):
        keypoint_threshold = st.slider("Keypoint Threshold", 0.1, 1.0, 0.5, 0.05)
        line_thickness = st.slider("Ketebalan Garis", 1, 5, 2)
        text_scale = st.slider("Skala Teks", 0.3, 1.0, 0.6, 0.1)
        
        # WebRTC settings
        st.subheader("Pengaturan WebRTC")
        if device_type == "mobile":
            video_width = st.selectbox("Lebar Video", [320, 480, 640], index=1)
            video_height = st.selectbox("Tinggi Video", [240, 360, 480], index=1)
            frame_rate = st.selectbox("Frame Rate", [15, 20, 24], index=1)
        else:
            video_width = st.selectbox("Lebar Video", [640, 800, 1280], index=0)
            video_height = st.selectbox("Tinggi Video", [480, 600, 720], index=0)
            frame_rate = st.selectbox("Frame Rate", [24, 30, 60], index=1)

# Model loading with error handling
@st.cache_resource
def load_model():
    model_paths = [
        "pose2/train2/weights/best.pt",
        "best.pt",
        "models/best.pt",
        "weights/best.pt"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                with st.spinner(f"Memuat model dari {model_path}..."):
                    model = YOLO(model_path)
                    return model, model_path
            except Exception as e:
                st.error(f"Error loading model from {model_path}: {str(e)}")
                continue
    
    # If no model found, show helpful message
    st.error("File model tidak ditemukan!")
    st.markdown("""
    <div class="error-box">
        <strong>Model tidak ditemukan!</strong><br>
        Pastikan salah satu file berikut ada:<br>
        • pose2/train2/weights/best.pt<br>
        • best.pt<br>
        • models/best.pt<br>
        • weights/best.pt<br><br>
        Download model YOLO pose estimation dan letakkan di salah satu path di atas.
    </div>
    """, unsafe_allow_html=True)
    
    return None, None

# Load model
model, model_path = load_model()

if model is None:
    st.stop()

st.sidebar.success(f"Model berhasil dimuat dari: {model_path}")

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    if None in (a, b, c):
        return None
    
    try:
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    except:
        return None

def draw_pose_with_label(frame, keypoints_obj, label, box, conf_score):
    """Draw pose keypoints and label on frame"""
    color = COLORS.get(label, (255, 255, 255))
    label_text = CLASS_LABELS.get(label, "Unknown")

    try:
        keypoints = keypoints_obj.xy[0].cpu().numpy()
        confs = keypoints_obj.conf[0].cpu().numpy()
    except:
        return frame

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
            
            try:
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
            except:
                pass

    # Draw bounding box and label
    if box is not None:
        try:
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
        except:
            pass

    return frame

def process_frame_detection(frame):
    """Process frame for pose detection"""
    try:
        results = model.predict(frame, imgsz=image_size, conf=confidence_threshold, save=False, verbose=False)

        detection_count = 0
        pose_results = []

        for result in results:
            boxes = result.boxes
            kpts = result.keypoints
            
            if boxes is not None and kpts is not None:
                for box, kp in zip(boxes, kpts):
                    try:
                        label = int(box.cls.cpu().item())
                        conf_score = float(box.conf.cpu().item())
                        
                        frame = draw_pose_with_label(frame, kp, label, box, conf_score)
                        
                        detection_count += 1
                        pose_results.append({
                            'label': CLASS_LABELS.get(label, 'Unknown'),
                            'confidence': conf_score,
                            'bbox': box.xyxy[0].cpu().numpy().tolist()
                        })
                    except Exception as e:
                        continue

        return frame, detection_count, pose_results
    except Exception as e:
        st.error(f"Error in frame processing: {str(e)}")
        return frame, 0, []

# WebRTC Video Transformer Class - Enhanced with error handling
class PoseDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.detection_count = 0
        self.good_posture_count = 0
        self.bad_posture_count = 0
        self.lock = threading.Lock()
    
    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Process frame with pose detection
            processed_img, detection_count, pose_results = process_frame_detection(img)
            
            # Update statistics with thread safety
            with self.lock:
                self.frame_count += 1
                self.detection_count = detection_count
                
                # Count posture types
                for result in pose_results:
                    if result['label'] == 'Postur Baik':
                        self.good_posture_count += 1
                    else:
                        self.bad_posture_count += 1
            
            return processed_img
        except Exception as e:
            # Return original frame if processing fails
            return frame.to_ndarray(format="bgr24")

def process_image(image):
    """Process uploaded image"""
    try:
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
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return np.array(image), 0, []

def process_video(video_path):
    """Process uploaded video"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Tidak dapat membuka file video")
        return
    
    try:
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Display video info
        cols = st.columns(3)
        with cols[0]:
            st.metric("FPS", fps)
        with cols[1]:
            st.metric("Total Frame", total_frames)
        with cols[2]:
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
            
            # Display processed frame (every 5th frame for performance)
            if frame_count % 5 == 0:
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
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Deteksi", total_detections)
        with cols[1]:
            st.metric("Postur Baik", good_posture_count)
        with cols[2]:
            st.metric("Postur Buruk", bad_posture_count)
        with cols[3]:
            accuracy = (good_posture_count / (good_posture_count + bad_posture_count)) * 100 if (good_posture_count + bad_posture_count) > 0 else 0
            st.metric("Persentase Postur Baik", f"{accuracy:.1f}%")
            
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    finally:
        cap.release()

# Main Interface
st.markdown("---")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📷 Upload Gambar", "📹 Webcam Real-time", "🎬 Upload Video"])

# Tab 1: Image Upload
with tab1:
    st.subheader("Upload Gambar untuk Deteksi Pose")
    
    uploaded_image = st.file_uploader(
        "Pilih file gambar",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload gambar yang berisi orang untuk deteksi dan klasifikasi pose"
    )
    
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            
            # Responsive columns
            if device_type == "mobile":
                # Stack vertically on mobile
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
                
                if st.button("Analisis Pose", type="primary", use_container_width=True):
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
                        with st.expander("Detail Hasil"):
                            for i, result in enumerate(pose_results, 1):
                                st.write(f"**Orang {i}:**")
                                st.write(f"- Klasifikasi: {result['label']}")
                                st.write(f"- Confidence: {result['confidence']:.2%}")
                                st.write("---")
                    else:
                        st.warning("Tidak ada pose yang terdeteksi dalam gambar. Coba sesuaikan confidence threshold.")
            else:
                # Side by side on desktop
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
                            with st.expander("Detail Hasil"):
                                for i, result in enumerate(pose_results, 1):
                                    st.write(f"**Orang {i}:**")
                                    st.write(f"- Klasifikasi: {result['label']}")
                                    st.write(f"- Confidence: {result['confidence']:.2%}")
                                    st.write("---")
                        else:
                            st.warning("Tidak ada pose yang terdeteksi dalam gambar. Coba sesuaikan confidence threshold.")
        
        except Exception as e:
            st.error(f"Error memproses gambar: {str(e)}")

# Tab 2: Real-time Webcam with WebRTC
with tab2:
    st.subheader("Deteksi Pose Webcam Real-time")
    
    # Device-specific instructions
    if device_type == "mobile":
        st.markdown("""
        <div class="info-box">
            <strong>Petunjuk Webcam Mobile:</strong><br>
            1. Pastikan browser memiliki izin akses kamera<br>
            2. Posisikan device dalam mode portrait atau landscape<br>
            3. Klik "START" untuk memulai deteksi<br>
            4. Posisikan diri dalam frame kamera<br>
            5. AI akan menganalisis postur secara real-time
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>Petunjuk Webcam Desktop:</strong><br>
            1. Klik "START" untuk memulai streaming webcam<br>
            2. Izinkan akses kamera ketika diminta oleh browser<br>
            3. Posisikan diri di depan kamera<br>
            4. AI akan menganalisis postur Anda secara real-time<br>
            5. Klik "STOP" untuk mengakhiri sesi
        </div>
        """, unsafe_allow_html=True)
    
    # WebRTC Streamer with device-specific settings
    try:
        webrtc_ctx = webrtc_streamer(
            key="pose-detection",
            video_transformer_factory=PoseDetectionTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": video_width, "min": 320, "max": 1920},
                    "height": {"ideal": video_height, "min": 240, "max": 1080},
                    "frameRate": {"ideal": frame_rate, "min": 10, "max": 60},
                    "facingMode": "user" if device_type == "mobile" else "environment"
                },
                "audio": False
            },
            async_processing=True,
        )
        
        # Real-time statistics
        if webrtc_ctx.video_transformer:
            # Responsive layout for statistics
            if device_type == "mobile":
                # Stack metrics vertically on mobile
                st.metric("Jumlah Frame", webrtc_ctx.video_transformer.frame_count)
                st.metric("Deteksi Saat Ini", webrtc_ctx.video_transformer.detection_count)
                st.metric("Total Postur Baik", webrtc_ctx.video_transformer.good_posture_count)
                st.metric("Total Postur Buruk", webrtc_ctx.video_transformer.bad_posture_count)
            else:
                # Side by side on desktop
                cols = st.columns(4)
                
                with cols[0]:
                    st.metric("Jumlah Frame", webrtc_ctx.video_transformer.frame_count)
                with cols[1]:
                    st.metric("Deteksi Saat Ini", webrtc_ctx.video_transformer.detection_count)
                with cols[2]:
                    st.metric("Total Postur Baik", webrtc_ctx.video_transformer.good_posture_count)
                with cols[3]:
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
    
    except Exception as e:
        st.error(f"Error mengakses webcam: {str(e)}")
        st.markdown("""
        <div class="warning-box">
            <strong>Troubleshooting:</strong><br>
            1. Pastikan browser mendukung WebRTC<br>
            2. Periksa izin akses kamera<br>
            3. Tutup aplikasi lain yang menggunakan kamera<br>
            4. Refresh halaman dan coba lagi<br>
            5. Gunakan browser Chrome/Firefox untuk kompatibilitas terbaik
        </div>
        """, unsafe_allow_html=True)

# Tab 3: Video Upload
with tab3:
    st.subheader("Upload Video untuk Deteksi Pose")
    
    uploaded_video = st.file_uploader(
        "Pilih file video",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload file video untuk analisis deteksi pose secara batch"
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
        
        if st.button("Proses Video", type="primary", use_container_width=True):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                temp_video_path = tfile.name
            
            try:
                process_video(temp_video_path)
            except Exception as e:
                st.error(f"Error memproses video: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)

# Tips Section
st.markdown("---")
st.subheader("Tips untuk Deteksi Pose yang Lebih Baik")

if device_type == "mobile":
    # Stack tips vertically on mobile
    st.markdown("""
    **Pengaturan Kamera Mobile**
    - Pastikan pencahayaan yang baik
    - Pegang device dengan stabil
    - Gunakan mode landscape untuk area lebih luas
    - Jaga jarak 1.5-2 meter dari kamera
    - Hindari latar belakang yang rumit
    """)
    
    st.markdown("""
    **Tips Deteksi Mobile**
    - Duduk tegak untuk deteksi yang lebih baik
    - Kenakan pakaian dengan warna kontras
    - Hindari pakaian longgar/kebesaran
    - Tetap berada dalam frame kamera
    - Gunakan tripod atau penyangga jika tersedia
    """)
    
    st.markdown("""
    **Pengaturan Optimal Mobile**
    - Gunakan resolusi 480x360 untuk performa terbaik
    - Turunkan confidence threshold jika deteksi kurang sensitif
    - Aktifkan mode hemat baterai jika diperlukan
    - Tutup aplikasi lain untuk mengurangi lag
    """)
else:
    # Side by side layout for desktop
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Pengaturan Kamera Desktop**
        - Pastikan pencahayaan yang baik
        - Posisikan kamera setinggi mata
        - Jaga jarak 1-2 meter
        - Hindari latar belakang yang rumit
        - Gunakan webcam dengan resolusi tinggi
        """)

    with col2:
        st.markdown("""
        **Tips Deteksi Desktop**
        - Duduk tegak untuk deteksi yang lebih baik
        - Kenakan pakaian dengan warna kontras
        - Hindari pakaian longgar/kebesaran
        - Tetap berada dalam frame kamera
        - Pastikan postur tubuh terlihat jelas
        """)

    with col3:
        st.markdown("""
        **Pengaturan Optimal Desktop**
        - Gunakan resolusi 640x480 atau lebih tinggi
        - Sesuaikan ukuran gambar untuk performa optimal
        - Toggle opsi tampilan sesuai kebutuhan
        - Periksa pengaturan lanjutan
        - Gunakan browser Chrome untuk performa terbaik
        """)

# Troubleshooting Section
st.markdown("---")
st.subheader("Troubleshooting")

with st.expander("Masalah Umum dan Solusi"):
    st.markdown("""
    **Webcam tidak berfungsi:**
    - Pastikan browser memiliki izin akses kamera
    - Tutup aplikasi lain yang menggunakan kamera
    - Coba refresh halaman
    - Gunakan browser Chrome atau Firefox
    - Periksa driver kamera (Windows/Linux)
    
    **Performa lambat:**
    - Turunkan resolusi video di pengaturan lanjutan
    - Kurangi frame rate
    - Tutup tab browser lain
    - Gunakan ukuran gambar yang lebih kecil (320px)
    - Nonaktifkan opsi tampilan yang tidak perlu
    
    **Deteksi tidak akurat:**
    - Sesuaikan confidence threshold
    - Pastikan pencahayaan yang cukup
    - Posisikan tubuh dengan jelas dalam frame
    - Hindari latar belakang yang kompleks
    - Gunakan pakaian dengan warna kontras
    
    **Model tidak ditemukan:**
    - Pastikan file model YOLO ada di direktori yang benar
    - Download model pose estimation YOLO v8
    - Letakkan file .pt di folder yang sesuai
    - Periksa path model di kode
    
    **Error pada mobile:**
    - Gunakan browser mobile yang mendukung WebRTC
    - Periksa koneksi internet
    - Tutup aplikasi lain untuk menghemat RAM
    - Coba mode landscape
    - Restart browser jika perlu
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>Pose Estimation App</strong> - Powered by YOLO v8 & Streamlit</p>
    <p>Kompatibel dengan Desktop, Laptop, Tablet, dan Mobile</p>
    <p>Untuk hasil terbaik, gunakan pencahayaan yang baik dan posisikan tubuh dengan jelas dalam frame</p>
</div>
""", unsafe_allow_html=True)
