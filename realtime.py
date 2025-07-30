import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np
import math
from PIL import Image
import time
import logging

# Suppress warnings and set proper logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

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
    .camera-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CLASS_LABELS = {
    0: "Postur Buruk",
    1: "Postur Baik"
}

COLORS = {
    0: (0, 255, 0),  # Green for good posture
    1: (255, 0, 0),  # Red for bad posture
}

KEYPOINT_CONNECTIONS = [(0, 1), (1, 2)]

# Header
st.markdown('<h1 class="main-header">Pose Estimation</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analisis postur tubuh dengan deteksi pose-estimation menggunakan YOLO v8</p>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("Konfigurasi")
    
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
    
    # Advanced settings
    with st.expander("Pengaturan Lanjutan"):
        keypoint_threshold = st.slider("Keypoint Threshold", 0.1, 1.0, 0.5, 0.05)
        line_thickness = st.slider("Ketebalan Garis", 1, 5, 2)
        text_scale = st.slider("Skala Teks", 0.3, 1.0, 0.6, 0.1)

# Model loading function
@st.cache_resource
def load_model():
    """Load YOLO model with multiple fallback options"""
    model_paths = [
        "pose2/train2/weights/best.pt",
        "best.pt",
        "yolov8n-pose.pt"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                with st.spinner(f"Memuat model dari {model_path}..."):
                    # Set environment variables to avoid CUDA issues
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
                    model = YOLO(model_path)
                    return model, model_path
            except Exception as e:
                st.warning(f"Gagal memuat model dari {model_path}: {str(e)}")
                continue
    
    # Try to download default model
    try:
        st.info("Mengunduh model YOLO pose default...")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        model = YOLO("yolov8n-pose.pt")
        return model, "yolov8n-pose.pt (downloaded)"
    except Exception as e:
        st.error(f"Tidak dapat memuat model: {str(e)}")
        return None, None

# Load model
model_result = load_model()
if model_result[0] is None:
    st.error("❌ Tidak dapat memuat model YOLO. Pastikan koneksi internet tersedia.")
    st.stop()

model, model_path = model_result
st.sidebar.success(f"✅ Model berhasil dimuat: {model_path}")

def calculate_angle(a, b, c):
    """Calculate angle between three points safely"""
    if None in (a, b, c):
        return None
    
    try:
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return None
        
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    except:
        return None

def draw_pose_annotations(frame, keypoints_obj, label, box, conf_score):
    """Draw pose keypoints and annotations on frame"""
    try:
        color = COLORS.get(label, (255, 255, 255))
        label_text = CLASS_LABELS.get(label, "Unknown")

        # Extract keypoints safely
        if hasattr(keypoints_obj, 'xy') and keypoints_obj.xy is not None:
            keypoints = keypoints_obj.xy[0].cpu().numpy()
            confs = keypoints_obj.conf[0].cpu().numpy() if hasattr(keypoints_obj, 'conf') and keypoints_obj.conf is not None else None
        else:
            return frame

        # Process keypoints
        pts = []
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0 and (confs is None or (i < len(confs) and confs[i] > keypoint_threshold)):
                pt = (int(x), int(y))
                pts.append(pt)
                
                if show_keypoints:
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)
                    cv2.circle(frame, pt, 5, (255, 255, 255), 1)
            else:
                pts.append(None)

        # Draw connections
        if show_connections and len(pts) >= 2:
            for i, j in KEYPOINT_CONNECTIONS:
                if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
                    cv2.line(frame, pts[i], pts[j], color, line_thickness)

        # Draw angle
        if show_angles and len(pts) >= 3 and all(pts[k] for k in [0, 1, 2]):
            angle = calculate_angle(pts[0], pts[1], pts[2])
            if angle is not None:
                pos = pts[1]
                angle_text = f"{int(angle)}°"
                
                # Draw angle background
                (text_width, text_height), _ = cv2.getTextSize(
                    angle_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1
                )
                cv2.rectangle(
                    frame, 
                    (pos[0] + 5, pos[1] - text_height - 10), 
                    (pos[0] + text_width + 10, pos[1] - 5), 
                    (0, 0, 0), 
                    -1
                )
                
                cv2.putText(
                    frame, angle_text, 
                    (pos[0] + 8, pos[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), 1
                )

        # Draw bounding box and label
        if box is not None:
            try:
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label text
                display_text = label_text
                if show_confidence:
                    display_text += f" ({conf_score:.2f})"
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    display_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1
                )
                cv2.rectangle(
                    frame, (x1, y1 - text_height - 10), 
                    (x1 + text_width + 10, y1), color, -1
                )
                
                # Draw label text
                cv2.putText(
                    frame, display_text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 1
                )
            except:
                pass

    except Exception as e:
        print(f"Error in draw_pose_annotations: {e}")
    
    return frame

def process_frame_detection(frame):
    """Process frame for pose detection with error handling"""
    try:
        # Run prediction
        results = model.predict(
            frame, 
            imgsz=image_size, 
            conf=confidence_threshold, 
            save=False, 
            verbose=False,
            device='cpu'
        )

        detection_count = 0
        pose_results = []

        for result in results:
            if hasattr(result, 'boxes') and hasattr(result, 'keypoints'):
                boxes = result.boxes
                kpts = result.keypoints
                
                if boxes is not None and kpts is not None and len(boxes) > 0:
                    for box, kp in zip(boxes, kpts):
                        try:
                            label = int(box.cls.cpu().item())
                            conf_score = float(box.conf.cpu().item())
                            
                            frame = draw_pose_annotations(frame, kp, label, box, conf_score)
                            
                            detection_count += 1
                            pose_results.append({
                                'label': CLASS_LABELS.get(label, 'Unknown'),
                                'confidence': conf_score
                            })
                        except Exception as e:
                            continue

        return frame, detection_count, pose_results
    except Exception as e:
        print(f"Error in process_frame_detection: {e}")
        return frame, 0, []

def process_image(image):
    """Process uploaded image for pose detection"""
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
    """Process uploaded video for pose detection"""
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
            
            # Display frame every 10 frames for performance
            if frame_count % 10 == 0:
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
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

def check_camera_availability():
    """Check if camera is available"""
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                return True, "Camera available"
            else:
                return False, "Camera opened but cannot read frames"
        else:
            return False, "Cannot open camera"
    except Exception as e:
        return False, f"Camera check error: {str(e)}"
    finally:
        if cap is not None:
            cap.release()

def capture_single_frame():
    """Capture and process a single frame from webcam"""
    cap = None
    try:
        # First check if camera is available
        camera_available, camera_status = check_camera_availability()
        if not camera_available:
            return None, f"❌ Kamera tidak tersedia: {camera_status}"
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None, "❌ Tidak dapat mengakses kamera. Pastikan kamera tidak digunakan aplikasi lain."
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            return None, "❌ Tidak dapat mengambil gambar dari kamera"
        
        # Process frame
        processed_frame, detection_count, pose_results = process_frame_detection(frame)
        processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        return {
            'frame': processed_rgb,
            'detection_count': detection_count,
            'pose_results': pose_results
        }, None
        
    except Exception as e:
        return None, f"❌ Error mengakses kamera: {str(e)}"
    finally:
        if cap is not None:
            cap.release()

# Main Interface
st.markdown("---")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📷 Upload Gambar", "📹 Webcam Snapshot", "🎬 Upload Video"])

# Tab 1: Image Upload
with tab1:
    st.subheader("Upload Gambar untuk Deteksi Pose")
    
    uploaded_image = st.file_uploader(
        "Pilih file gambar",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload gambar yang berisi orang untuk deteksi dan klasifikasi pose"
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
                📏 Ukuran: {image.size[0]} x {image.size[1]} piksel<br>
                🎨 Mode: {image.mode}<br>
                📄 Format: {image.format}<br>
                💾 Ukuran file: {uploaded_image.size / 1024:.1f} KB
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
                        <strong>✅ Analisis Selesai!</strong><br>
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
                    st.warning("⚠️ Tidak ada pose yang terdeteksi. Coba sesuaikan confidence threshold.")

# Tab 2: Webcam Snapshot (Simplified)
with tab2:
    st.subheader("Deteksi Pose Webcam Snapshot")
    
    # Check camera availability first
    camera_available, camera_status = check_camera_availability()
    
    if not camera_available:
        st.error(f"🚫 Kamera Tidak Tersedia: {camera_status}")
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Kamera tidak dapat diakses</strong><br><br>
            <strong>Kemungkinan penyebab:</strong><br>
            • Aplikasi berjalan di environment cloud (Streamlit Cloud) yang tidak memiliki akses kamera<br>
            • Kamera sedang digunakan aplikasi lain<br>
            • Driver kamera belum terinstall<br>
            • Izin kamera ditolak<br><br>
            <strong>💡 Solusi:</strong><br>
            • Jalankan aplikasi di komputer lokal untuk mengakses kamera<br>
            • Gunakan fitur "Upload Gambar" sebagai alternatif<br>
            • Atau upload video untuk analisis batch
        </div>
        """, unsafe_allow_html=True)
        
        # Show demo functionality
        st.markdown("---")
        st.subheader("🎯 Demo Mode - Upload Gambar Sebagai Alternatif")
        
        demo_image = st.file_uploader(
            "Upload gambar untuk simulasi deteksi webcam",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            key="demo_webcam"
        )
        
        if demo_image is not None:
            image = Image.open(demo_image)
            
            col_demo1, col_demo2 = st.columns(2)
            
            with col_demo1:
                st.markdown("**📸 Gambar Input (Simulasi Webcam)**")
                st.image(image, use_container_width=True)
            
            with col_demo2:
                if st.button("🔍 Analisis Demo", type="primary", key="demo_analyze"):
                    with st.spinner("Menganalisis pose..."):
                        processed_image, detection_count, pose_results = process_image(image)
                    
                    st.markdown("**📊 Hasil Analisis**")
                    st.image(processed_image, use_container_width=True)
                    
                    if detection_count > 0:
                        good_count = sum(1 for r in pose_results if r['label'] == 'Postur Baik')
                        bad_count = sum(1 for r in pose_results if r['label'] == 'Postur Buruk')
                        
                        st.markdown(f"""
                        <div class="success-box">
                            <strong>📈 Hasil Demo:</strong><br>
                            🎯 Total Deteksi: {detection_count}<br>
                            ✅ Postur Baik: {good_count}<br>
                            ❌ Postur Buruk: {bad_count}
                        </div>
                        """, unsafe_allow_html=True)
    else:
        # Original webcam functionality
        st.success(f"✅ {camera_status}")
        
        # Instructions
        st.markdown("""
        <div class="info-box">
            <strong>📋 Petunjuk Webcam:</strong><br>
            1. Klik "Ambil Foto" untuk mengakses kamera<br>
            2. Posisikan diri Anda dengan baik<br>
            3. AI akan menganalisis postur dari foto yang diambil<br>
            4. Hasil akan ditampilkan setelah pemrosesan
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("📸 Ambil Foto dari Kamera", type="primary", use_container_width=True):
                with st.spinner("📹 Mengakses kamera dan mengambil foto..."):
                    result, error = capture_single_frame()
                
                if error:
                    st.error(error)
                    st.markdown("""
                    <div class="warning-box">
                        <strong>💡 Tips Mengatasi Masalah Kamera:</strong><br>
                        • Pastikan tidak ada aplikasi lain yang menggunakan kamera<br>
                        • Refresh halaman dan coba lagi<br>
                        • Periksa izin kamera di browser<br>
                        • Coba restart browser jika masalah berlanjut
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ Foto berhasil diambil dan diproses!")
                    
                    # Display results
                    col_img1, col_img2 = st.columns(2)
                    
                    with col_img1:
                        st.markdown("**📸 Hasil Capture & Analisis**")
                        st.image(result['frame'], use_container_width=True)
                    
                    with col_img2:
                        st.markdown("**📊 Statistik Deteksi**")
                        
                        # Show detection metrics
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>🎯 Deteksi: {result['detection_count']}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if result['detection_count'] > 0:
                            # Count posture types
                            good_count = sum(1 for r in result['pose_results'] if r['label'] == 'Postur Baik')
                            bad_count = sum(1 for r in result['pose_results'] if r['label'] == 'Postur Buruk')
                            
                            st.markdown(f"""
                            <div class="success-box">
                                <strong>📈 Hasil Analisis:</strong><br>
                                ✅ Postur Baik: {good_count}<br>
                                ❌ Postur Buruk: {bad_count}<br>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Detailed results
                            with st.expander("📋 Detail Lengkap"):
                                for i, pose_result in enumerate(result['pose_results'], 1):
                                    st.write(f"**Deteksi {i}:**")
                                    st.write(f"- Klasifikasi: {pose_result['label']}")
                                    st.write(f"- Confidence: {pose_result['confidence']:.2%}")
                                    st.write("---")
                        else:
                            st.warning("⚠️ Tidak ada pose yang terdeteksi dalam foto.")
                            st.info("💡 Coba sesuaikan posisi atau confidence threshold di sidebar.")

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
            <strong>📹 Informasi Video:</strong><br>
            📄 Nama file: {uploaded_video.name}<br>
            💾 Ukuran file: {uploaded_video.size / (1024*1024):.2f} MB<br>
            🎬 Tipe: {uploaded_video.type}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Proses Video", type="primary"):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                temp_video_path = tfile.name
            
            try:
                process_video(temp_video_path)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)

# Tips Section
st.markdown("---")
st.subheader("💡 Tips untuk Deteksi Pose yang Lebih Baik")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📷 Pengaturan Kamera**
    - ✅ Pastikan pencahayaan yang baik
    - ✅ Posisi kamera setinggi mata
    - ✅ Jaga jarak 1-2 meter
    - ✅ Hindari latar belakang rumit
    """)

with col2:
    st.markdown("""
    **🎯 Tips Deteksi**
    - ✅ Duduk tegak untuk hasil terbaik
    - ✅ Pakai pakaian kontras
    - ✅ Hindari pakaian longgar
    - ✅ Pastikan seluruh tubuh terlihat
    """)

with col3:
    st.markdown("""
    **⚙️ Pengaturan**
    - ✅ Turunkan threshold untuk sensitivitas tinggi
    - ✅ Sesuaikan ukuran gambar
    - ✅ Toggle opsi tampilan
    - ✅ Cek pengaturan lanjutan
    """)

# Footer
st.markdown("---")
st.markdown("""
<div class="info-box">
    <strong>ℹ️ Informasi Aplikasi:</strong><br>
    • Menggunakan YOLO v8 untuk deteksi pose<br>
    • Mendukung format gambar: JPG, PNG, BMP, TIFF<br>
    • Mendukung format video: MP4, AVI, MOV, MKV, WMV<br>
    • Webcam menggunakan snapshot mode untuk stabilitas<br>
    • Performa tergantung kualitas input dan pengaturan
</div>
""", unsafe_allow_html=True)
