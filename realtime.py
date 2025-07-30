import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np
import math
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Deteksi dan Klasifikasi Pose",
    page_icon="🤸‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced responsive design
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .error-box {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    /* Video container */
    .video-container {
        border: 3px solid #667eea;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .stColumn {
            padding: 0 0.25rem;
        }
        .metric-card {
            padding: 0.75rem;
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Constants
CLASS_LABELS = {
    0: "Postur Buruk",
    1: "Postur Baik"
}

COLORS = {
    0: (0, 0, 255),    # Red for bad posture
    1: (0, 255, 0),    # Green for good posture
}

KEYPOINT_CONNECTIONS = [(0, 1), (1, 2)]

# Header
st.markdown('<h1 class="main-header">🤸‍♂️ Pose Estimation AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analisis postur tubuh real-time dengan YOLO v8 Neural Network</p>', unsafe_allow_html=True)

# Enhanced WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun.services.mozilla.com:3478"]},
        {"urls": ["stun:stun.stunprotocol.org:3478"]},
    ],
    "iceTransportPolicy": "all",
    "bundlePolicy": "balanced",
    "rtcpMuxPolicy": "require",
    "iceCandidatePoolSize": 10
})

# Initialize session state
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_frames': 0,
        'total_detections': 0,
        'good_posture': 0,
        'bad_posture': 0
    }

# Detect device type
def get_device_type():
    try:
        user_agent = st.context.headers.get("User-Agent", "").lower()
        if any(mobile in user_agent for mobile in ["mobile", "android", "iphone", "ipad"]):
            return "mobile"
        return "desktop"
    except:
        return "desktop"

device_type = get_device_type()

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi")
    
    # Device info
    device_icon = "📱" if device_type == "mobile" else "💻"
    st.markdown(f"""
    <div class="info-box">
        {device_icon} <strong>Device:</strong> {device_type.title()}
    </div>
    """, unsafe_allow_html=True)
    
    # Model settings
    st.markdown("#### 🎯 Pengaturan Deteksi")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    
    # Adaptive settings based on device
    if device_type == "mobile":
        image_size_options = [320, 480, 640]
        default_size_idx = 0
        video_width_options = [320, 480, 640]
        video_height_options = [240, 360, 480]
        fps_options = [10, 15, 20, 24]
        default_fps_idx = 2
    else:
        image_size_options = [320, 640, 1280]
        default_size_idx = 1
        video_width_options = [480, 640, 800, 1280]
        video_height_options = [360, 480, 600, 720]
        fps_options = [15, 20, 24, 30]
        default_fps_idx = 3
    
    image_size = st.selectbox("Ukuran Model", image_size_options, index=default_size_idx)
    
    # WebRTC settings
    st.markdown("#### 📹 Pengaturan Video")
    video_width = st.selectbox("Lebar Video", video_width_options, index=1)
    video_height = st.selectbox("Tinggi Video", video_height_options, index=1)
    frame_rate = st.selectbox("Frame Rate", fps_options, index=default_fps_idx)
    
    # Display settings
    st.markdown("#### 🎨 Opsi Tampilan")
    show_keypoints = st.checkbox("Tampilkan Keypoints", value=True)
    show_connections = st.checkbox("Tampilkan Koneksi", value=True)
    show_angles = st.checkbox("Tampilkan Sudut", value=True)
    show_confidence = st.checkbox("Tampilkan Confidence", value=True)
    show_fps = st.checkbox("Tampilkan FPS", value=True)
    
    # Advanced settings
    with st.expander("🔧 Pengaturan Lanjutan"):
        keypoint_threshold = st.slider("Keypoint Threshold", 0.1, 1.0, 0.5, 0.05)
        line_thickness = st.slider("Ketebalan Garis", 1, 8, 3)
        text_scale = st.slider("Skala Teks", 0.3, 1.5, 0.7, 0.1)
        processing_interval = st.slider("Interval Processing (ms)", 33, 200, 50, 17)
        
        # Performance settings
        st.markdown("**Optimasi Performa:**")
        use_gpu = st.checkbox("Gunakan GPU (jika tersedia)", value=True)
        async_processing = st.checkbox("Async Processing", value=True)

# Model loading with enhanced error handling
@st.cache_resource
def load_model():
    model_paths = [
        "pose2/train2/weights/best.pt",
        "best.pt",
        "models/best.pt",
        "weights/best.pt",
        "yolo_pose.pt"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                with st.spinner(f"🔄 Memuat model dari {model_path}..."):
                    # Try to use GPU if available
                    device = 'cuda' if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
                    model = YOLO(model_path)
                    model.to(device)
                    return model, model_path, device
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {str(e)}")
                continue
    
    # Try to download a pretrained model if none found
    try:
        with st.spinner("📥 Mengunduh model YOLO pose..."):
            model = YOLO('yolov8n-pose.pt')  # Download pretrained pose model
            return model, 'yolov8n-pose.pt (downloaded)', 'cpu'
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
    
    return None, None, None

# Load model
model, model_path, device = load_model()

if model is None:
    st.markdown("""
    <div class="error-box">
        <h3>❌ Model tidak ditemukan!</h3>
        <p>Pastikan salah satu file berikut ada:</p>
        <ul>
            <li>pose2/train2/weights/best.pt</li>
            <li>best.pt</li>
            <li>models/best.pt</li>
            <li>weights/best.pt</li>
        </ul>
        <p><strong>Atau:</strong> Aplikasi akan mencoba mengunduh model YOLO pose otomatis.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.sidebar.markdown(f"""
<div class="success-box">
    ✅ <strong>Model Ready!</strong><br>
    📁 Path: {model_path}<br>
    💻 Device: {device.upper()}<br>
    🔧 Size: {image_size}px
</div>
""", unsafe_allow_html=True)

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    if None in (a, b, c):
        return None
    
    try:
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        
        dot_product = np.dot(ba, bc)
        norms = np.linalg.norm(ba) * np.linalg.norm(bc)
        
        if norms == 0:
            return None
            
        cosine_angle = dot_product / norms
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    except Exception as e:
        logger.warning(f"Error calculating angle: {e}")
        return None

def draw_pose_with_label(frame, keypoints_obj, label, box, conf_score):
    """Enhanced pose drawing with better visualization"""
    try:
        color = COLORS.get(label, (255, 255, 255))
        label_text = CLASS_LABELS.get(label, "Unknown")

        # Extract keypoints
        keypoints = keypoints_obj.xy[0].cpu().numpy()
        confs = keypoints_obj.conf[0].cpu().numpy()

        # Draw keypoints with enhanced visualization
        pts = []
        for i, (x, y) in enumerate(keypoints):
            if i < len(confs) and confs[i] > keypoint_threshold:
                pt = (int(x), int(y))
                pts.append(pt)
                
                if show_keypoints:
                    # Draw keypoint with outline
                    cv2.circle(frame, pt, 8, (0, 0, 0), -1)  # Black outline
                    cv2.circle(frame, pt, 6, color, -1)      # Colored center
                    cv2.circle(frame, pt, 8, (255, 255, 255), 2)  # White border
            else:
                pts.append(None)

        # Draw connections with gradient effect
        if show_connections:
            for i, j in KEYPOINT_CONNECTIONS:
                if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
                    cv2.line(frame, pts[i], pts[j], color, line_thickness + 2, cv2.LINE_AA)
                    cv2.line(frame, pts[i], pts[j], (255, 255, 255), line_thickness, cv2.LINE_AA)

        # Calculate and display angle with better styling
        if show_angles and len(pts) >= 3 and all(pts[k] for k in [0, 1, 2]):
            angle = calculate_angle(pts[0], pts[1], pts[2])
            if angle is not None:
                pos = pts[1]
                angle_text = f"{int(angle)}°"
                
                # Background for angle text
                (text_width, text_height), baseline = cv2.getTextSize(
                    angle_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2
                )
                
                # Draw background rectangle
                bg_x1 = pos[0] + 10
                bg_y1 = pos[1] - text_height - 10
                bg_x2 = pos[0] + text_width + 20
                bg_y2 = pos[1] + 5
                
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)
                
                # Draw angle text
                cv2.putText(
                    frame, angle_text, 
                    (pos[0] + 15, pos[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), 2, cv2.LINE_AA
                )

        # Draw enhanced bounding box and label
        if box is not None:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Draw bounding box with rounded corners effect
            thickness = 3
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness + 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced label
            display_text = label_text
            if show_confidence:
                display_text += f" ({conf_score:.1%})"
            
            # Label background with gradient effect
            (text_width, text_height), _ = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_DUPLEX, text_scale, 2
            )
            
            label_y1 = y1 - text_height - 20
            label_y2 = y1 - 5
            label_x2 = x1 + text_width + 20
            
            # Background
            cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), (255, 255, 255), 2)
            
            # Label text
            cv2.putText(
                frame, display_text, (x1 + 10, y1 - 10),
                cv2.FONT_HERSHEY_DUPLEX, text_scale, (255, 255, 255), 2, cv2.LINE_AA
            )

        return frame
    except Exception as e:
        logger.error(f"Error in draw_pose_with_label: {e}")
        return frame

def process_frame_detection(frame, fps_counter=None):
    """Optimized frame processing"""
    try:
        start_time = time.time()
        
        # Run YOLO inference
        results = model.predict(
            frame, 
            imgsz=image_size, 
            conf=confidence_threshold, 
            save=False, 
            verbose=False,
            device=device
        )

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
                        logger.warning(f"Error processing detection: {e}")
                        continue

        # Draw FPS if enabled
        if show_fps and fps_counter:
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            fps_text = f"FPS: {fps:.1f}"
            
            cv2.putText(
                frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
            )

        return frame, detection_count, pose_results
    except Exception as e:
        logger.error(f"Error in process_frame_detection: {e}")
        return frame, 0, []

# Enhanced WebRTC Video Transformer
class PoseDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.detection_count = 0
        self.good_posture_count = 0
        self.bad_posture_count = 0
        self.fps_counter = time.time()
        self.lock = threading.Lock()
        self.last_process_time = 0
        
    def transform(self, frame):
        try:
            # Convert frame
            img = frame.to_ndarray(format="bgr24")
            
            # Throttle processing based on interval
            current_time = time.time()
            if (current_time - self.last_process_time) * 1000 < processing_interval:
                return img
            
            self.last_process_time = current_time
            
            # Process frame
            processed_img, detection_count, pose_results = process_frame_detection(img, self.fps_counter)
            
            # Update statistics thread-safely
            with self.lock:
                self.frame_count += 1
                self.detection_count = detection_count
                
                # Update session state
                st.session_state.stats['total_frames'] = self.frame_count
                st.session_state.stats['total_detections'] += detection_count
                
                # Count posture types
                for result in pose_results:
                    if result['label'] == 'Postur Baik':
                        self.good_posture_count += 1
                        st.session_state.stats['good_posture'] += 1
                    else:
                        self.bad_posture_count += 1
                        st.session_state.stats['bad_posture'] += 1
            
            return processed_img
        except Exception as e:
            logger.error(f"Error in transformer: {e}")
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
        st.error(f"❌ Error processing image: {str(e)}")
        return np.array(image), 0, []

def process_video(video_path):
    """Process uploaded video with progress tracking"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Tidak dapat membuka file video")
        return
    
    try:
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Display video info
        cols = st.columns(4)
        with cols[0]:
            st.metric("📊 FPS", fps)
        with cols[1]:
            st.metric("🎞️ Total Frame", total_frames)
        with cols[2]:
            st.metric("⏱️ Durasi", f"{duration:.1f}s")
        with cols[3]:
            st.metric("📐 Resolusi", f"{width}x{height}")
        
        # Create placeholders
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_placeholder = st.empty()
        
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
            
            # Display processed frame (every nth frame for performance)
            skip_frames = max(1, fps // 10)  # Show ~10 FPS
            if frame_count % skip_frames == 0:
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update progress and stats
            frame_count += 1
            progress = frame_count / total_frames if total_frames > 0 else 0
            progress_bar.progress(progress)
            
            # Real-time stats
            elapsed_time = time.time() - start_time
            processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            status_text.markdown(f"""
            **⚡ Processing:** Frame {frame_count:,}/{total_frames:,} | 
            **🔥 Speed:** {processing_fps:.1f} FPS | 
            **⏰ ETA:** {((total_frames - frame_count) / processing_fps / 60):.1f}m
            """)
            
            # Update stats every 30 frames
            if frame_count % 30 == 0:
                cols = st.columns(4)
                with cols[0]:
                    st.metric("🎯 Deteksi", total_detections)
                with cols[1]:
                    st.metric("✅ Postur Baik", good_posture_count)
                with cols[2]:
                    st.metric("❌ Postur Buruk", bad_posture_count)
                with cols[3]:
                    accuracy = (good_posture_count / (good_posture_count + bad_posture_count)) * 100 if (good_posture_count + bad_posture_count) > 0 else 0
                    st.metric("📈 Akurasi", f"{accuracy:.1f}%")
        
        cap.release()
        
        # Final results
        st.markdown("""
        <div class="success-box">
            <h3>🎉 Pemrosesan Video Selesai!</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Final statistics
        cols = st.columns(4)
        with cols[0]:
            st.metric("🎯 Total Deteksi", total_detections)
        with cols[1]:
            st.metric("✅ Postur Baik", good_posture_count)
        with cols[2]:
            st.metric("❌ Postur Buruk", bad_posture_count)
        with cols[3]:
            final_accuracy = (good_posture_count / (good_posture_count + bad_posture_count)) * 100 if (good_posture_count + bad_posture_count) > 0 else 0
            st.metric("📈 Tingkat Postur Baik", f"{final_accuracy:.1f}%")
            
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        logger.error(f"Video processing error: {e}")
    finally:
        cap.release()

# Main Interface
st.markdown("---")

# Create tabs with enhanced design
tab1, tab2, tab3 = st.tabs(["📷 Upload Gambar", "📹 Webcam Real-time", "🎬 Upload Video"])

# Tab 1: Image Upload
with tab1:
    st.markdown("### 📷 Upload Gambar untuk Deteksi Pose")
    
    uploaded_image = st.file_uploader(
        "Pilih file gambar (JPG, PNG, BMP, TIFF)",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload gambar yang berisi orang untuk deteksi dan klasifikasi pose"
    )
    
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            
            # Responsive layout
            if device_type == "mobile":
                # Mobile layout - vertical stack
                st.markdown("#### 🖼️ Gambar Asli")
                st.image(image, use_container_width=True)
                
                # Image info
                st.markdown(f"""
                <div class="info-box">
                    <h4>📊 Informasi Gambar</h4>
                    <p><strong>Ukuran:</strong> {image.size[0]} x {image.size[1]} piksel</p>
                    <p><strong>Mode:</strong> {image.mode}</p>
                    <p><strong>Format:</strong> {image.format}</p>
                    <p><strong>Size:</strong> {uploaded_image.size / 1024:.1f} KB</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔍 Analisis Pose", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI sedang menganalisis pose..."):
                        processed_image, detection_count, pose_results = process_image(image)
                    
                    st.markdown("#### 🎯 Hasil Pemrosesan")
                    st.image(processed_image, use_container_width=True)
                    
                    # Results
                    if detection_count > 0:
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>🎉 Analisis Berhasil!</h4>
                            <p><strong>Pose terdeteksi:</strong> {detection_count}</p>
                            <p><strong>Status:</strong> Pemrosesan selesai</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Detailed results
                        with st.expander("📋 Detail Hasil Analisis"):
                            for i, result in enumerate(pose_results, 1):
                                posture_icon = "✅" if result['label'] == 'Postur Baik' else "❌"
                                st.markdown(f"""
                                **{posture_icon} Orang {i}:**
                                - **Klasifikasi:** {result['label']}
                                - **Confidence:** {result['confidence']:.2%}
                                - **Koordinat:** {[int(x) for x in result['bbox']]}
                                """)
                                st.markdown("---")
                    else:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Tidak Ada Pose Terdeteksi</h4>
                            <p>Coba sesuaikan confidence threshold di sidebar atau gunakan gambar dengan pose yang lebih jelas.</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                # Desktop layout - side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🖼️ Gambar Asli")
                    st.image(image, use_container_width=True)
                    
                    # Image info
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>📊 Informasi Gambar</h4>
                        <p><strong>Ukuran:</strong> {image.size[0]} x {image.size[1]} piksel</p>
                        <p><strong>Mode:</strong> {image.mode}</p>
                        <p><strong>Format:</strong> {image.format}</p>
                        <p><strong>Size:</strong> {uploaded_image.size / 1024:.1f} KB</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("🔍 Analisis Pose", type="primary", use_container_width=True):
                        with st.spinner("🤖 AI sedang menganalisis pose..."):
                            processed_image, detection_count, pose_results = process_image(image)
                        
                        st.markdown("#### 🎯 Hasil Pemrosesan")
                        st.image(processed_image, use_container_width=True)
                        
                        # Results
                        if detection_count > 0:
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>🎉 Analisis Berhasil!</h4>
                                <p><strong>Pose terdeteksi:</strong> {detection_count}</p>
                                <p><strong>Status:</strong> Pemrosesan selesai</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Detailed results
                            with st.expander("📋 Detail Hasil Analisis"):
                                for i, result in enumerate(pose_results, 1):
                                    posture_icon = "✅" if result['label'] == 'Postur Baik' else "❌"
                                    st.markdown(f"""
                                    **{posture_icon} Orang {i}:**
                                    - **Klasifikasi:** {result['label']}
                                    - **Confidence:** {result['confidence']:.2%}
                                    - **Koordinat:** {[int(x) for x in result['bbox']]}
                                    """)
                                    st.markdown("---")
                        else:
                            st.markdown("""
                            <div class="warning-box">
                                <h4>⚠️ Tidak Ada Pose Terdeteksi</h4>
                                <p>Coba sesuaikan confidence threshold di sidebar atau gunakan gambar dengan pose yang lebih jelas.</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ Error Memproses Gambar</h4>
                <p>{str(e)}</p>
            </div>
            """, unsafe_allow_html=True)

# Tab 2: Real-time Webcam with Enhanced WebRTC
with tab2:
    st.markdown("### 📹 Deteksi Pose Webcam Real-time")
    
    # Enhanced instructions based on device
    if device_type == "mobile":
        st.markdown("""
        <div class="info-box">
            <h4>📱 Petunjuk Webcam Mobile</h4>
            <p><strong>1.</strong> Pastikan browser memiliki izin akses kamera</p>
            <p><strong>2.</strong> Gunakan mode landscape untuk area deteksi lebih luas</p>
            <p><strong>3.</strong> Klik tombol <strong>START</strong> untuk memulai</p>
            <p><strong>4.</strong> Posisikan diri dalam frame dengan pencahayaan baik</p>
            <p><strong>5.</strong> AI akan menganalisis postur secara real-time</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>💻 Petunjuk Webcam Desktop</h4>
            <p><strong>1.</strong> Klik tombol <strong>START</strong> untuk memulai streaming</p>
            <p><strong>2.</strong> Izinkan akses kamera ketika diminta browser</p>
            <p><strong>3.</strong> Posisikan diri di depan kamera (jarak 1-2 meter)</p>
            <p><strong>4.</strong> Pastikan pencahayaan yang cukup</p>
            <p><strong>5.</strong> AI akan menganalisis postur secara real-time</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Reset stats button
    if st.button("🔄 Reset Statistik", help="Reset semua statistik sesi"):
        st.session_state.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'good_posture': 0,
            'bad_posture': 0
        }
        st.success("✅ Statistik direset!")
    
    # WebRTC Streamer with enhanced configuration
    try:
        # Determine camera facing mode
        facing_mode = "user" if device_type == "mobile" else {"exact": "environment"}
        
        webrtc_ctx = webrtc_streamer(
            key="pose-detection-enhanced",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=PoseDetectionTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": video_width, "min": 320, "max": 1920},
                    "height": {"ideal": video_height, "min": 240, "max": 1080},
                    "frameRate": {"ideal": frame_rate, "min": 10, "max": 60},
                    "facingMode": facing_mode
                },
                "audio": False
            },
            async_processing=async_processing,
            video_html_attrs={
                "style": {"width": "100%", "margin": "0 auto", "border-radius": "15px"},
                "controls": False,
                "autoPlay": True,
            }
        )
        
        # Real-time statistics display
        if webrtc_ctx.video_transformer:
            st.markdown("### 📊 Statistik Real-time")
            
            # Create metrics layout
            if device_type == "mobile":
                # Mobile: 2x2 grid
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{webrtc_ctx.video_transformer.frame_count:,}</h3>
                        <p>🎞️ Total Frame</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{webrtc_ctx.video_transformer.good_posture_count:,}</h3>
                        <p>✅ Postur Baik</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{webrtc_ctx.video_transformer.detection_count}</h3>
                        <p>🎯 Deteksi Saat Ini</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{webrtc_ctx.video_transformer.bad_posture_count:,}</h3>
                        <p>❌ Postur Buruk</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Desktop: 4 columns
                cols = st.columns(4)
                
                with cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{webrtc_ctx.video_transformer.frame_count:,}</h3>
                        <p>🎞️ Total Frame</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{webrtc_ctx.video_transformer.detection_count}</h3>
                        <p>🎯 Deteksi Saat Ini</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{webrtc_ctx.video_transformer.good_posture_count:,}</h3>
                        <p>✅ Postur Baik</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{webrtc_ctx.video_transformer.bad_posture_count:,}</h3>
                        <p>❌ Postur Buruk</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Session summary
            total_postures = webrtc_ctx.video_transformer.good_posture_count + webrtc_ctx.video_transformer.bad_posture_count
            if total_postures > 0:
                good_percentage = (webrtc_ctx.video_transformer.good_posture_count / total_postures) * 100
                
                # Progress bar for posture quality
                st.markdown("#### 📈 Kualitas Postur Sesi")
                st.progress(good_percentage / 100)
                
                st.markdown(f"""
                <div class="success-box">
                    <h4>📊 Ringkasan Sesi Real-time</h4>
                    <p><strong>🎯 Tingkat Postur Baik:</strong> {good_percentage:.1f}%</p>
                    <p><strong>🎞️ Total Frame Diproses:</strong> {webrtc_ctx.video_transformer.frame_count:,}</p>
                    <p><strong>👥 Total Deteksi Postur:</strong> {total_postures:,}</p>
                    <p><strong>⚡ Status:</strong> {"🟢 Excellent" if good_percentage >= 80 else "🟡 Good" if good_percentage >= 60 else "🔴 Needs Improvement"}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Live recommendations
            if webrtc_ctx.video_transformer.detection_count > 0:
                st.markdown("#### 💡 Saran Real-time")
                if webrtc_ctx.video_transformer.bad_posture_count > webrtc_ctx.video_transformer.good_posture_count:
                    st.markdown("""
                    <div class="warning-box">
                        <p><strong>⚠️ Perbaiki Postur:</strong></p>
                        <p>• Tegakkan punggung</p>
                        <p>• Sejajarkan bahu</p>
                        <p>• Angkat dagu sedikit</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        <p><strong>✅ Postur Bagus!</strong></p>
                        <p>Pertahankan posisi ini</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Error Mengakses Webcam</h4>
            <p><strong>Error:</strong> {str(e)}</p>
            <h5>🔧 Troubleshooting:</h5>
            <p><strong>1.</strong> Pastikan browser mendukung WebRTC (Chrome/Firefox)</p>
            <p><strong>2.</strong> Periksa izin akses kamera di browser</p>
            <p><strong>3.</strong> Tutup aplikasi lain yang menggunakan kamera</p>
            <p><strong>4.</strong> Refresh halaman dan coba lagi</p>
            <p><strong>5.</strong> Coba gunakan browser lain</p>
        </div>
        """, unsafe_allow_html=True)

# Tab 3: Video Upload with Enhanced Processing
with tab3:
    st.markdown("### 🎬 Upload Video untuk Deteksi Pose")
    
    uploaded_video = st.file_uploader(
        "Pilih file video (MP4, AVI, MOV, MKV, WMV)",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload file video untuk analisis deteksi pose secara batch"
    )
    
    if uploaded_video is not None:
        # Enhanced video info
        file_size_mb = uploaded_video.size / (1024*1024)
        st.markdown(f"""
        <div class="info-box">
            <h4>🎬 Informasi Video</h4>
            <p><strong>📁 Nama:</strong> {uploaded_video.name}</p>
            <p><strong>💾 Ukuran:</strong> {file_size_mb:.2f} MB</p>
            <p><strong>🏷️ Tipe:</strong> {uploaded_video.type}</p>
            <p><strong>⚡ Estimasi waktu:</strong> ~{file_size_mb * 2:.0f} detik</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            skip_frames = st.selectbox(
                "Skip Frame (untuk mempercepat)", 
                [1, 2, 3, 5, 10], 
                index=1,
                help="Skip frame untuk mempercepat processing (1 = semua frame)"
            )
        with col2:
            show_preview = st.checkbox("Tampilkan Preview", value=True, help="Tampilkan video preview saat processing")
        
        if st.button("🚀 Proses Video", type="primary", use_container_width=True):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                temp_video_path = tfile.name
            
            try:
                with st.spinner("🎬 Memproses video..."):
                    process_video(temp_video_path)
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Error Memproses Video</h4>
                    <p><strong>Error:</strong> {str(e)}</p>
                    <p>Coba dengan file video yang lebih kecil atau format yang berbeda.</p>
                </div>
                """, unsafe_allow_html=True)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)

# Enhanced Tips Section
st.markdown("---")
st.markdown("### 💡 Tips untuk Deteksi Pose yang Optimal")

if device_type == "mobile":
    # Mobile tips - stacked layout
    st.markdown("""
    <div class="info-box">
        <h4>📱 Tips Mobile Optimal</h4>
        <p><strong>📹 Kamera:</strong></p>
        <p>• Gunakan mode landscape untuk area deteksi lebih luas</p>
        <p>• Pastikan pencahayaan yang baik (hindari backlighting)</p>
        <p>• Jaga jarak 1.5-2 meter dari device</p>
        <p>• Gunakan tripod atau penyangga untuk stabilitas</p>
        
        <p><strong>🎯 Deteksi:</strong></p>
        <p>• Kenakan pakaian dengan warna kontras dari background</p>
        <p>• Hindari pakaian terlalu longgar</p>
        <p>• Posisikan seluruh tubuh dalam frame</p>
        <p>• Latar belakang yang simpel akan lebih baik</p>
        
        <p><strong>⚡ Performa:</strong></p>
        <p>• Tutup aplikasi lain untuk menghemat RAM</p>
        <p>• Gunakan WiFi yang stabil</p>
        <p>• Kurangi resolusi jika lag</p>
        <p>• Aktifkan mode hemat baterai jika diperlukan</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Desktop tips - 3 column layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>💻 Setup Desktop</h4>
            <p><strong>📹 Kamera:</strong></p>
            <p>• Posisikan webcam setinggi mata</p>
            <p>• Jaga jarak 1-2 meter</p>
            <p>• Pencahayaan dari depan</p>
            <p>• Resolusi tinggi untuk akurasi</p>
            
            <p><strong>💻 Browser:</strong></p>
            <p>• Chrome/Firefox terbaru</p>
            <p>• Izinkan akses kamera</p>
            <p>• Tutup tab lain</p>
            <p>• Hardware acceleration ON</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Optimasi Deteksi</h4>
            <p><strong>👤 Postur:</strong></p>
            <p>• Duduk/berdiri tegak</p>
            <p>• Bahu sejajar</p>
            <p>• Kepala tegak</p>
            <p>• Seluruh tubuh dalam frame</p>
            
            <p><strong>👕 Pakaian:</strong></p>
            <p>• Warna kontras dengan background</p>
            <p>• Hindari pakaian terlalu longgar</p>
            <p>• Tidak ada aksesoris menutupi</p>
            <p>• Pola sederhana lebih baik</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>⚙️ Settings Optimal</h4>
            <p><strong>🔧 Model:</strong></p>
            <p>• Confidence: 0.5-0.7</p>
            <p>• Image size: 640px</p>
            <p>• Keypoint threshold: 0.5</p>
            <p>• GPU jika tersedia</p>
            
            <p><strong>📹 Video:</strong></p>
            <p>• 640x480 atau 800x600</p>
            <p>• 24-30 FPS</p>
            <p>• Stable internet</p>
            <p>• Processing interval: 50ms</p>
        </div>
        """, unsafe_allow_html=True)

# Advanced Troubleshooting
st.markdown("---")
st.markdown("### 🔧 Troubleshooting Lanjutan")

with st.expander("🚨 Masalah Umum dan Solusi Lengkap"):
    st.markdown("""
    ### 📹 **Masalah Webcam**
    
    **Problem: Webcam tidak muncul/blank**
    - ✅ Periksa izin browser: `chrome://settings/content/camera`
    - ✅ Restart browser setelah memberikan izin
    - ✅ Coba browser lain (Chrome/Firefox)
    - ✅ Update driver webcam (Windows: Device Manager)
    
    **Problem: Video lag/choppy**
    - ✅ Turunkan resolusi ke 480x360
    - ✅ Kurangi frame rate ke 15-20 FPS
    - ✅ Tingkatkan processing interval ke 100ms
    - ✅ Tutup aplikasi lain yang menggunakan kamera
    
    ### 🎯 **Masalah Deteksi**
    
    **Problem: Tidak ada deteksi pose**
    - ✅ Turunkan confidence threshold ke 0.3
    - ✅ Pastikan seluruh tubuh dalam frame
    - ✅ Periksa pencahayaan (tidak terlalu gelap/terang)
    - ✅ Coba posisi yang lebih jelas
    
    **Problem: Deteksi tidak akurat**
    - ✅ Gunakan background yang kontras
    - ✅ Hindari pakaian dengan pola kompleks
    - ✅ Pastikan tidak ada objek menghalangi
    - ✅ Tingkatkan keypoint threshold ke 0.6
    
    ### ⚡ **Masalah Performa**
    
    **Problem: FPS rendah**
    - ✅ Gunakan model size 320px
    - ✅ Nonaktifkan opsi visual yang tidak perlu
    - ✅ Aktifkan GPU jika tersedia
    - ✅ Tutup tab browser lain
    
    **Problem: Browser crash/freeze**
    - ✅ Refresh halaman
    - ✅ Clear browser cache
    - ✅ Restart browser
    - ✅ Coba incognito mode
    
    ### 📱 **Masalah Mobile**
    
    **Problem: Tidak bisa akses kamera mobile**
    - ✅ Pastikan menggunakan HTTPS
    - ✅ Gunakan Chrome Mobile atau Safari
    - ✅ Izinkan kamera di browser settings
    - ✅ Restart browser mobile
    
    **Problem: Performa lambat di mobile**
    - ✅ Gunakan resolusi 320x240
    - ✅ Frame rate maksimal 15 FPS
    - ✅ Tutup aplikasi lain
    - ✅ Gunakan WiFi yang stabil
    """)

# System Status
st.markdown("---")
st.markdown("### 📊 System Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>🤖 Model</h4>
        <p>✅ Loaded</p>
        <small>{device.upper()}</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>📱 Device</h4>
        <p>{device_type.title()}</p>
        <small>Auto-detected</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>⚙️ Config</h4>
        <p>{image_size}px</p>
        <small>Model size</small>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_sessions = st.session_state.stats['good_posture'] + st.session_state.stats['bad_posture']
    accuracy = (st.session_state.stats['good_posture'] / total_sessions * 100) if total_sessions > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <h4>📈 Session</h4>
        <p>{accuracy:.0f}%</p>
        <small>Accuracy</small>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-top: 2rem;">
    <h3>🤸‍♂️ Pose Estimation AI</h3>
    <p><strong>Powered by:</strong> YOLO v8 Neural Network • Streamlit • OpenCV</p>
    <p><strong>Compatible:</strong> 💻 Desktop • 📱 Mobile • 📟 Tablet</p>
    <p><strong>Best Results:</strong> Good lighting • Clear posture • Stable connection</p>
    <br>
    <p><em>Developed with ❤️ for better posture analysis</em></p>
</div>
""", unsafe_allow_html=True)
