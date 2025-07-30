import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np
import math
from PIL import Image
import time

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
    .camera-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #6c757d;
        text-align: center;
        margin: 2rem 0;
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
    
    # Advanced settings
    with st.expander("Pengaturan Lanjutan"):
        keypoint_threshold = st.slider("Keypoint Threshold", 0.1, 1.0, 0.5, 0.05)
        line_thickness = st.slider("Ketebalan Garis", 1, 5, 2)
        text_scale = st.slider("Skala Teks", 0.3, 1.0, 0.6, 0.1)

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

# Camera instruction function
def show_camera_instructions():
    """Show instructions for using local camera with OpenCV script"""
    st.markdown("""
    <div class="camera-box">
        <h3>📹 Real-time Camera Detection</h3>
        <p>Untuk deteksi real-time menggunakan webcam, jalankan script OpenCV terpisah:</p>
    </div>
    """, unsafe_allow_html=True)
    
    opencv_script = '''
# Simpan sebagai: realtime_pose_detection.py
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
    cap = cv2.VideoCapture(0)  # 0 untuk webcam default
    
    if not cap.isOpened():
        print("❌ Error: Cannot access webcam")
        print("💡 Try changing camera index (0, 1, 2...)")
        return
    
    print("✅ Webcam opened successfully!")
    print("📹 Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Press 'r' to reset counters")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    good_posture_count = 0
    bad_posture_count = 0
    
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
            
            # Count detections
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    label = int(box.cls.cpu().item())
                    if label == 1:  # Postur Baik
                        good_posture_count += 1
                    else:  # Postur Buruk
                        bad_posture_count += 1
            
            # Add statistics overlay
            stats_text = [
                f"Frame: {frame_count}",
                f"Good Posture: {good_posture_count}",
                f"Bad Posture: {bad_posture_count}",
                "Controls: Q=Quit, S=Save, R=Reset"
            ]
            
            for i, text in enumerate(stats_text):
                y_pos = 30 + (i * 30)
                cv2.putText(annotated_frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Pose Detection - Real-time', annotated_frame)
            
            frame_count += 1
            
        except Exception as e:
            print(f"⚠️ Error processing frame {frame_count}: {e}")
            cv2.imshow('Pose Detection - Real-time', frame)
            
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'pose_screenshot_{frame_count}.jpg'
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Screenshot saved: {filename}")
        elif key == ord('r'):
            good_posture_count = 0
            bad_posture_count = 0
            frame_count = 0
            print("🔄 Counters reset!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    total_postures = good_posture_count + bad_posture_count
    if total_postures > 0:
        accuracy = (good_posture_count / total_postures) * 100
        print(f"\\n📊 Final Statistics:")
        print(f"  Total Frames: {frame_count}")
        print(f"  Good Posture: {good_posture_count}")
        print(f"  Bad Posture: {bad_posture_count}")
        print(f"  Accuracy: {accuracy:.1f}%")
    
    print("👋 Webcam closed. Goodbye!")

if __name__ == "__main__":
    main()
    '''
    
    st.code(opencv_script, language='python')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Cara Menggunakan:**")
        st.markdown("""
        1. Copy script di atas ke file `realtime_pose_detection.py`
        2. Jalankan dengan: `python realtime_pose_detection.py`
        3. Tekan 'q' untuk keluar
        4. Tekan 's' untuk save screenshot
        5. Tekan 'r' untuk reset counter
        """)
    
    with col2:
        st.markdown("**⚡ Keuntungan:**")
        st.markdown("""
        - ✅ Tidak perlu WebRTC (lebih stabil)
        - ✅ Akses langsung ke webcam
        - ✅ Performance lebih baik
        - ✅ Real-time statistics
        - ✅ Screenshot & counter features
        """)

# Main Interface
st.markdown("---")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📷 Upload Gambar", "🎥 Camera Real-time", "🎬 Upload Video"])

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

# Tab 2: Camera Real-time (OpenCV Instructions)
with tab2:
    st.subheader("Deteksi Pose Real-time dengan Webcam")
    
    # Show camera instructions
    show_camera_instructions()
    
    # Alternative options
    st.markdown("---")
    st.subheader("🔄 Alternative Options")
    
    option_col1, option_col2 = st.columns(2)
    
    with option_col1:
        st.markdown("""
        <div class="info-box">
            <strong>🖥️ Local Development:</strong><br>
            Gunakan script OpenCV di atas untuk:
            <ul>
                <li>Performance terbaik</li>
                <li>Akses langsung webcam</li>
                <li>Real-time processing</li>
                <li>Tidak butuh internet</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with option_col2:
        st.markdown("""
        <div class="warning-box">
            <strong>🌐 Web Deployment:</strong><br>
            Untuk web app dengan webcam:
            <ul>
                <li>Butuh WebRTC (kompleks)</li>
                <li>Perlu HTTPS</li>
                <li>Network dependent</li>
                <li>Browser compatibility issues</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Troubleshooting section
    with st.expander("🔧 Troubleshooting Camera Issues"):
        st.markdown("""
        **Jika script OpenCV tidak bisa akses kamera:**
        
        1. **Ganti Camera Index:**
           ```python
           cap = cv2.VideoCapture(1)  # Coba 1, 2, 3...
           ```
        
        2. **Check Camera dengan Command:**
           ```bash
           # Windows
           devmgmt.msc  # Device Manager > Cameras
           
           # Linux
           ls /dev/video*
           
           # MacOS
           system_profiler SPCameraDataType
           ```
        
        3. **Permission Issues:**
           - Windows: Check Privacy Settings > Camera
           - macOS: System Preferences > Security & Privacy > Camera
           - Linux: Check user groups (video, camera)
        
        4. **Alternative Camera Access:**
           ```python
           # DirectShow (Windows)
           cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
           
           # V4L2 (Linux)
           cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
           ```
        """)

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

# Footer with tips and information
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
    - Monitor error count pada video processing
    - Test dengan gambar dulu sebelum video
    """)

# System Requirements
st.markdown("---")
st.subheader("📋 System Requirements")

req_col1, req_col2 = st.columns(2)

with req_col1:
    st.markdown("""
    **Minimum Requirements:**
    - Python 3.8+
    - RAM: 4GB (8GB recommended)
    - Storage: 2GB free space
    - CPU: Intel i5 atau setara
    - Webcam: Any USB/built-in camera
    """)

with req_col2:
    st.markdown("""
    **Dependencies:**
    ```bash
    pip install streamlit
    pip install ultralytics
    pip install opencv-python
    pip install numpy
    pip install Pillow
    ```
    """)

# Performance monitoring
if st.sidebar.button("📊 Show Performance Info"):
    st.sidebar.markdown("### Performance Info")
    st.sidebar.write(f"Streamlit: {st.__version__}")
    st.sidebar.write(f"OpenCV: {cv2.__version__}")
    
    # Memory usage
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        st.sidebar.write(f"Memory Usage: {memory_mb:.1f} MB")
    except ImportError:
        st.sidebar.write("Memory info: psutil not installed")

# Version info and status
st.sidebar.markdown("---")
st.sidebar.markdown("**App Version:** 3.0 No-WebRTC")
st.sidebar.markdown("**Last Updated:** July 2025")
st.sidebar.markdown("**Status:** ✅ Stable (No WebRTC)")

# Camera status indicator
st.sidebar.markdown("### 📹 Camera Options")
st.sidebar.info("🖥️ **Local Script**: Best performance")
st.sidebar.warning("🌐 **Web Camera**: Use WebRTC version")
st.sidebar.success("📷 **Upload**: Always available")

# Quick start guide
with st.sidebar.expander("🚀 Quick Start"):
    st.markdown("""
    **1. Upload Image**: Easiest way to test
    **2. Local Camera**: Best for real-time
    **3. Upload Video**: Batch processing
    
    **Model Path**: `pose2/train2/weights/best.pt`
    """)

# Footer note
st.markdown("""
---
<div style="text-align: center; color: #666; padding: 20px;">
    <strong>Pose Estimation App - No WebRTC Version</strong><br>
    Untuk real-time webcam, gunakan script OpenCV local yang disediakan di tab Camera Real-time.<br>
    Version ini lebih stabil dan tidak memerlukan konfigurasi network yang kompleks.
</div>
""", unsafe_allow_html=True)
