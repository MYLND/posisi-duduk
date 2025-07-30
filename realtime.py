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
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
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
    0: (0, 255, 0),    # Green for good posture
    1: (255, 0, 0),    # Red for bad posture
}

KEYPOINT_CONNECTIONS = [(0, 1), (1, 2)]

# Header
st.markdown('<h1 class="main-header">Pose Estimation</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analisis postur tubuh dengan deteksi pose-estimation menggunakan YOLO v8</p>', unsafe_allow_html=True)

# Multiple WebRTC Configurations for different network conditions
RTC_CONFIGURATIONS = {
    "basic": RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }),
    
    "enhanced": RTCConfiguration({
        "iceServers": [
            # Google STUN servers
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            
            # Alternative STUN servers
            {"urls": ["stun:stun.stunprotocol.org:3478"]},
            {"urls": ["stun:stun.cloudflare.com:3478"]},
            {"urls": ["stun:stun.voiparound.com"]},
            {"urls": ["stun:stun.voipbuster.com"]},
        ],
        "iceCandidatePoolSize": 10,
        "iceTransportPolicy": "all",
    }),
    
    "turn_enabled": RTCConfiguration({
        "iceServers": [
            # STUN servers
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun.cloudflare.com:3478"]},
            
            # Free TURN servers for firewall bypass
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject", 
                "credential": "openrelayproject"
            },
            {
                "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ],
        "iceCandidatePoolSize": 10,
        "iceTransportPolicy": "all",
    })
}

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Konfigurasi")
    
    # Model settings
    st.subheader("Pengaturan Deteksi")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    image_size = st.selectbox("Ukuran Gambar", [320, 640, 1280], index=1)
    
    # Display settings
    st.subheader("Opsi Tampilan")
    show_keypoints = st.checkbox("Tampilkan Keypoints", value=True)
    show_connections = st.checkbox("Tampilkan Koneksi", value=True)
    show_angles = st.checkbox("Tampilkan Sudut", value=True)
    show_confidence = st.checkbox("Tampilkan Confidence", value=True)
    
    # WebRTC Configuration
    st.subheader("Konfigurasi WebRTC")
    rtc_config_option = st.selectbox(
        "Network Configuration:",
        ["basic", "enhanced", "turn_enabled"],
        index=1,
        help="""
        - basic: STUN servers dasar (cepat)
        - enhanced: Multiple STUN servers (recommended)
        - turn_enabled: Dengan TURN servers (untuk firewall bypass)
        """
    )
    
    # Advanced settings
    with st.expander("Pengaturan Lanjutan"):
        keypoint_threshold = st.slider("Keypoint Threshold", 0.1, 1.0, 0.5, 0.05)
        line_thickness = st.slider("Ketebalan Garis", 1, 5, 2)
        text_scale = st.slider("Skala Teks", 0.3, 1.0, 0.6, 0.1)
        
        # WebRTC advanced settings
        st.markdown("**WebRTC Settings:**")
        async_processing = st.checkbox("Async Processing", value=False, help="Enable untuk performa lebih baik, disable untuk stabilitas")
        max_fps = st.slider("Max Frame Rate", 10, 60, 30, 5)

# Model loading with error handling
@st.cache_resource
def load_model():
    model_path = "pose2/train2/weights/best.pt"
    
    if not os.path.exists(model_path):
        st.error(f"❌ File model tidak ditemukan: {model_path}")
        st.info("📁 Pastikan file model ada di direktori yang benar")
        
        # Show directory structure for debugging
        current_dir = os.getcwd()
        st.code(f"Current directory: {current_dir}")
        
        if os.path.exists("pose2"):
            st.info("✅ Folder 'pose2' ditemukan")
            if os.path.exists("pose2/train2"):
                st.info("✅ Folder 'pose2/train2' ditemukan")
                if os.path.exists("pose2/train2/weights"):
                    st.info("✅ Folder 'pose2/train2/weights' ditemukan")
                    weights_files = os.listdir("pose2/train2/weights")
                    st.write("Files in weights folder:", weights_files)
                else:
                    st.error("❌ Folder 'pose2/train2/weights' tidak ditemukan")
            else:
                st.error("❌ Folder 'pose2/train2' tidak ditemukan")
        else:
            st.error("❌ Folder 'pose2' tidak ditemukan")
        
        return None
    
    try:
        with st.spinner("🔄 Memuat model YOLO..."):
            model = YOLO(model_path)
            st.sidebar.success("✅ Model berhasil dimuat!")
            return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

if model is None:
    st.stop()

# Utility functions
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
    except Exception as e:
        return None

def draw_pose_with_label(frame, keypoints_obj, label, box, conf_score):
    """Enhanced pose drawing with better error handling"""
    try:
        color = COLORS.get(label, (255, 255, 255))
        label_text = CLASS_LABELS.get(label, "Unknown")

        keypoints = keypoints_obj.xy[0].cpu().numpy()
        confs = keypoints_obj.conf[0].cpu().numpy()

        # Draw keypoints
        pts = []
        for i, (x, y) in enumerate(keypoints):
            if i < len(confs) and confs[i] > keypoint_threshold and not (np.isnan(x) or np.isnan(y)):
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
                
                # Safe text size calculation
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
                except Exception:
                    pass  # Skip angle display if error

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
            except Exception:
                pass  # Skip bounding box if error

    except Exception as e:
        st.error(f"Error in draw_pose_with_label: {str(e)}")
    
    return frame

def process_frame_detection(frame):
    """Process frame with enhanced error handling"""
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
                        continue  # Skip this detection if error

        return frame, detection_count, pose_results
    
    except Exception as e:
        st.error(f"Error in process_frame_detection: {str(e)}")
        return frame, 0, []

# Enhanced WebRTC Video Transformer Class
class PoseDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.detection_count = 0
        self.good_posture_count = 0
        self.bad_posture_count = 0
        self.error_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
    
    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Process frame with pose detection
            processed_img, detection_count, pose_results = process_frame_detection(img)
            
            # Update statistics
            self.frame_count += 1
            self.detection_count = detection_count
            
            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:  # Update FPS every second
                self.fps = self.frame_count / (current_time - self.last_fps_time + 1e-6)
                self.last_fps_time = current_time
            
            # Count posture types
            for result in pose_results:
                if result['label'] == 'Postur Baik':
                    self.good_posture_count += 1
                else:
                    self.bad_posture_count += 1
            
            return processed_img
        
        except Exception as e:
            self.error_count += 1
            # Return original frame if processing fails
            return frame.to_ndarray(format="bgr24")

def process_image(image):
    """Process uploaded image with error handling"""
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
    """Process video with enhanced progress tracking and error handling"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ Tidak dapat membuka file video")
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
        error_count = 0
        
        # Process video frames
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
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
                
            except Exception as e:
                error_count += 1
                continue
            
            # Update progress
            frame_count += 1
            progress = frame_count / total_frames if total_frames > 0 else 0
            progress_bar.progress(progress)
            
            # Update status
            elapsed_time = time.time() - start_time
            processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            status_text.text(f"Memproses frame {frame_count}/{total_frames} | {processing_fps:.1f} FPS | Errors: {error_count}")
        
        cap.release()
        
        # Final statistics
        st.success("✅ Pemrosesan video selesai!")
        
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
        
        if error_count > 0:
            st.warning(f"⚠️ {error_count} frame gagal diproses")
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

# Network diagnostics function
def show_network_diagnostics():
    """Show network diagnostic information"""
    st.subheader("🔍 Network Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Browser Console Test:**")
        st.code("""
// Paste ini di Browser Console (F12):
navigator.mediaDevices.getUserMedia({video: true})
  .then(stream => {
    console.log('✅ Camera access OK');
    stream.getTracks().forEach(track => track.stop());
  })
  .catch(err => console.error('❌ Camera error:', err));
        """, language='javascript')
    
    with col2:
        st.markdown("**Quick Network Fixes:**")
        st.markdown("""
        1. **Test Hotspot**: Connect laptop ke hotspot HP
        2. **Disable VPN**: Matikan VPN sementara
        3. **Chrome Incognito**: Coba mode incognito
        4. **Firewall**: Add Python/Streamlit ke whitelist
        5. **DNS**: Ganti ke 8.8.8.8 dan 8.8.4.4
        """)

# Main Interface
st.markdown("---")

# Create tabs for different input methods
tab1, tab2, tab3, tab4 = st.tabs(["📷 Upload Gambar", "🎥 Webcam Real-time", "🎬 Upload Video", "🔧 Troubleshooting"])

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
                if st.button("🔍 Analisis Pose", type="primary"):
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
                        with st.expander("📊 Detail Hasil"):
                            for i, result in enumerate(pose_results, 1):
                                st.write(f"**Orang {i}:**")
                                st.write(f"- Klasifikasi: {result['label']}")
                                st.write(f"- Confidence: {result['confidence']:.2%}")
                                st.write("---")
                    else:
                        st.warning("⚠️ Tidak ada pose yang terdeteksi dalam gambar. Coba sesuaikan confidence threshold.")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

# Tab 2: Real-time Webcam with WebRTC
with tab2:
    st.subheader("Deteksi Pose Webcam Real-time")
    
    # Network configuration
    selected_rtc_config = RTC_CONFIGURATIONS[rtc_config_option]
    
    # Get server count safely
    try:
        server_count = len(RTC_CONFIGURATIONS[rtc_config_option].configuration['iceServers'])
    except:
        server_count = "Unknown"
    
    # Instructions with enhanced troubleshooting
    st.markdown(f"""
    <div class="info-box">
        <strong>Petunjuk Webcam WebRTC:</strong><br>
        1. Pastikan menggunakan Chrome/Firefox terbaru<br>
        2. Klik "START" untuk memulai streaming webcam<br>
        3. Izinkan akses kamera ketika diminta oleh browser<br>
        4. Posisikan diri Anda di depan kamera<br>
        5. AI akan menganalisis postur Anda secara real-time<br><br>
        <strong>Current Config:</strong> {rtc_config_option}<br>
        <strong>STUN Servers:</strong> {server_count} servers
    </div>
    """, unsafe_allow_html=True)
    
    # Connection troubleshooting
    with st.expander("🚨 Jika 'Connection taking longer than expected'"):
        st.markdown("""
        **Quick Fixes:**
        1. **Hotspot Test**: Connect laptop ke hotspot HP
        2. **Browser**: Coba Chrome Incognito mode
        3. **VPN**: Matikan VPN jika aktif
        4. **Firewall**: Add Streamlit ke Windows Firewall whitelist
        5. **Config**: Coba ubah "Network Configuration" di sidebar
        
        **Advanced Fixes:**
        - Ganti DNS ke 8.8.8.8 dan 8.8.4.4
        - Disable antivirus real-time protection sementara
        - Test dengan: `telnet stun.l.google.com 19302`
        """)
    
    # Error tracking
    if 'webrtc_errors' not in st.session_state:
        st.session_state.webrtc_errors = []
    
    try:
        # WebRTC Streamer with enhanced configuration
        webrtc_ctx = webrtc_streamer(
            key=f"pose-detection-{rtc_config_option}",
            video_transformer_factory=PoseDetectionTransformer,
            rtc_configuration=selected_rtc_config,
            media_stream_constraints={
                "video": {
                    "width": {"min": 320, "ideal": 640, "max": 1280},
                    "height": {"min": 240, "ideal": 480, "max": 720},
                    "frameRate": {"min": 10, "ideal": max_fps, "max": 60}
                },
                "audio": False
            },
            async_processing=async_processing,
            video_html_attrs={
                "style": {"width": "100%", "margin": "0 auto", "border": "2px solid #4ECDC4"},
                "controls": False,
                "autoPlay": True,
                "muted": True,
            },
        )
        
        # Connection status
        status_placeholder = st.empty()
        
        if webrtc_ctx.state.playing:
            status_placeholder.success("✅ Camera connected successfully!")
        elif webrtc_ctx.state.signalling:
            status_placeholder.info("🔄 Connecting to camera... Please wait.")
        else:
            status_placeholder.warning("⚠️ Click START to begin camera connection")
        
        # Real-time statistics
        if webrtc_ctx.video_transformer:
            st.markdown("### 📊 Real-time Statistics")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Frames", webrtc_ctx.video_transformer.frame_count)
            with col2:
                st.metric("Current Detections", webrtc_ctx.video_transformer.detection_count)
            with col3:
                st.metric("Good Posture", webrtc_ctx.video_transformer.good_posture_count)
            with col4:
                st.metric("Bad Posture", webrtc_ctx.video_transformer.bad_posture_count)
            with col5:
                st.metric("FPS", f"{webrtc_ctx.video_transformer.fps:.1f}")
            
            # Session statistics
            total_postures = webrtc_ctx.video_transformer.good_posture_count + webrtc_ctx.video_transformer.bad_posture_count
            if total_postures > 0:
                good_percentage = (webrtc_ctx.video_transformer.good_posture_count / total_postures) * 100
                
                st.markdown(f"""
                <div class="success-box">
                    <strong>Ringkasan Sesi:</strong><br>
                    Tingkat Postur Baik: {good_percentage:.1f}%<br>
                    Total Frame Diproses: {webrtc_ctx.video_transformer.frame_count}<br>
                    Total Deteksi Postur: {total_postures}<br>
                    Processing Errors: {webrtc_ctx.video_transformer.error_count}
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ WebRTC Error: {str(e)}")
        st.session_state.webrtc_errors.append(str(e))
        
        # Show error log
        if len(st.session_state.webrtc_errors) > 0:
            with st.expander("🐛 Error Log"):
                for i, error in enumerate(st.session_state.webrtc_errors[-3:], 1):
                    st.code(f"{i}. {error}")
        
        # Alternative solutions
        st.markdown("""
        <div class="warning-box">
            <strong>WebRTC gagal. Coba alternative:</strong><br>
            1. Refresh halaman dan coba lagi<br>
            2. Gunakan tab "Upload Gambar" untuk test model<br>
            3. Coba dengan smartphone (biasanya lebih stabil)<br>
            4. Gunakan script OpenCV local (lihat tab Troubleshooting)
        </div>
        """, unsafe_allow_html=True)

# Tab 3: Video Upload
with tab3:
    st.subheader("Upload Video untuk Deteksi Pose")
    
    uploaded_video = st.file_uploader(
        "Pilih file video",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'],
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
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            process_every_n_frames = st.slider("Process Every N Frames", 1, 10, 1, help="Skip frames untuk processing lebih cepat")
        with col2:
            max_frames = st.number_input("Max Frames to Process", 1, 10000, 1000, help="Batasi jumlah frame untuk video panjang")
        
        if st.button("🎬 Proses Video", type="primary"):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_video.read())
                    temp_video_path = tfile.name
                
                # Process video with options
                process_video(temp_video_path)
                
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                    
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                # Clean up on error
                if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)

# Tab 4: Troubleshooting
with tab4:
    st.subheader("🔧 Troubleshooting & Alternatives")
    
    # Network diagnostics
    show_network_diagnostics()
    
    st.markdown("---")
    
    # Alternative solutions
    st.subheader("🔄 Alternative Solutions")
    
    option_tabs = st.tabs(["Local OpenCV Script", "System Requirements", "Common Issues"])
    
    with option_tabs[0]:
        st.markdown("**Jika WebRTC tidak bekerja, gunakan script OpenCV terpisah:**")
        
        opencv_script = '''
# Simpan sebagai: local_pose_detection.py
import cv2
from ultralytics import YOLO
import numpy as np

def main():
    print("🔄 Loading YOLO model...")
    try:
        model = YOLO("pose2/train2/weights/best.pt")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("🔄 Opening webcam...")
    cap = cv2.VideoCapture(0)  # 0 untuk webcam default, coba 1, 2 jika tidak bekerja
    
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam")
        print("💡 Try changing camera index (0, 1, 2...)")
        return
    
    print("✅ Webcam opened successfully!")
    print("📹 Press 'q' to quit, 's' to save screenshot")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading frame")
            break
        
        try:
            # YOLO prediction
            results = model.predict(frame, conf=0.5, imgsz=640, verbose=False)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Add frame counter
            cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Pose Detection - Press Q to quit, S to save', annotated_frame)
            
            frame_count += 1
            
        except Exception as e:
            print(f"⚠️ Error processing frame {frame_count}: {e}")
            cv2.imshow('Pose Detection - Press Q to quit', frame)
            
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'pose_screenshot_{frame_count}.jpg', annotated_frame)
            print(f"📸 Screenshot saved: pose_screenshot_{frame_count}.jpg")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Webcam closed. Goodbye!")

if __name__ == "__main__":
    main()
        '''
        
        st.code(opencv_script, language='python')
        
        st.markdown("**Cara menjalankan:**")
        st.code("python local_pose_detection.py", language='bash')
        
        st.info("💡 Script ini akan berjalan tanpa WebRTC dan biasanya lebih stabil untuk laptop")
    
    with option_tabs[1]:
        st.markdown("### 📋 System Requirements")
        
        requirements = {
            "Python": "3.8+",
            "Browser": "Chrome 88+ / Firefox 85+",
            "Camera": "Working webcam device",
            "Network": "Internet connection for STUN servers",
            "OS": "Windows 10+ / macOS 10.14+ / Linux"
        }
        
        dependencies = {
            "streamlit": "Latest version",
            "streamlit-webrtc": "Latest version", 
            "ultralytics": "Latest version",
            "opencv-python": "4.5+",
            "numpy": "Latest version",
            "Pillow": "Latest version"
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**System Requirements:**")
            for req, ver in requirements.items():
                st.write(f"✓ **{req}**: {ver}")
        
        with col2:
            st.markdown("**Python Dependencies:**")
            for dep, ver in dependencies.items():
                st.write(f"✓ **{dep}**: {ver}")
        
        st.markdown("**Installation Commands:**")
        st.code("""
pip install streamlit streamlit-webrtc ultralytics opencv-python numpy Pillow
# Atau
pip install -r requirements.txt
        """, language='bash')
    
    with option_tabs[2]:
        st.markdown("### ❓ Common Issues & Solutions")
        
        issues = [
            {
                "issue": "Model tidak ditemukan (best.pt)",
                "solution": "Pastikan file pose2/train2/weights/best.pt ada dan path benar"
            },
            {
                "issue": "WebRTC connection timeout",
                "solution": "Coba hotspot HP, disable VPN, atau ubah network config di sidebar"
            },
            {
                "issue": "Kamera tidak terdeteksi",
                "solution": "Check browser permission, coba browser lain, atau restart browser"
            },
            {
                "issue": "Frame rate rendah",
                "solution": "Turunkan image size, disable beberapa opsi tampilan, atau gunakan GPU"
            },
            {
                "issue": "Error saat processing",
                "solution": "Update dependencies, check Python version, atau restart aplikasi"
            },
            {
                "issue": "HTTPS required error",
                "solution": "Deploy ke cloud atau setup local HTTPS certificate"
            }
        ]
        
        for i, item in enumerate(issues, 1):
            with st.expander(f"{i}. {item['issue']}"):
                st.write(f"**Solution:** {item['solution']}")

# Footer with tips
st.markdown("---")
st.subheader("💡 Tips untuk Deteksi Pose yang Lebih Baik")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🎥 Pengaturan Kamera**
    - Pastikan pencahayaan yang baik dan merata
    - Posisikan kamera setinggi mata
    - Jaga jarak 1-2 meter dari kamera
    - Hindari latar belakang yang rumit atau bergerak
    - Gunakan background yang kontras dengan pakaian
    """)

with col2:
    st.markdown("""
    **👤 Tips Deteksi**
    - Duduk tegak untuk deteksi postur yang akurat
    - Kenakan pakaian dengan warna kontras
    - Hindari pakaian longgar atau terlalu ketat
    - Tetap berada dalam frame kamera
    - Gerakan pelan untuk hasil yang lebih stabil
    """)

with col3:
    st.markdown("""
    **⚙️ Pengaturan Optimal**
    - Turunkan confidence threshold untuk sensitivitas tinggi
    - Sesuaikan ukuran gambar (640 recommended)
    - Toggle opsi tampilan sesuai kebutuhan
    - Gunakan enhanced network config
    - Monitor FPS dan error count
    """)

# Performance monitoring
if st.sidebar.button("📊 Show Performance Info"):
    st.sidebar.markdown("### Performance Info")
    st.sidebar.write(f"Python: {st.__version__}")
    st.sidebar.write(f"OpenCV: {cv2.__version__}")
    
    # Memory usage (basic)
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    st.sidebar.write(f"Memory: {memory_mb:.1f} MB")

# Version info
st.sidebar.markdown("---")
st.sidebar.markdown("**App Version:** 2.0 Enhanced")
st.sidebar.markdown("**Last Updated:** July 2025")
st.sidebar.markdown("**Status:** ✅ Ready")
