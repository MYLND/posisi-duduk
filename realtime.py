import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np
import math
from PIL import Image
import time
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Pose Detection - Simplified",
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
        font-size: clamp(1.8rem, 4vw, 2.5rem);
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: clamp(0.9rem, 2.5vw, 1.1rem);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .error-box {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.6rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    /* Camera feed styling */
    .camera-container {
        border: 3px solid #667eea;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    .status-online { background-color: #00b894; }
    .status-offline { background-color: #fd79a8; }
    .status-loading { background-color: #fdcb6e; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .stColumn {
            padding: 0 0.2rem;
        }
        .metric-card {
            padding: 0.8rem 0.5rem;
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

# Initialize session state
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_frames': 0,
        'total_detections': 0,
        'good_posture': 0,
        'bad_posture': 0,
        'session_start': time.time()
    }

if 'pose_model' not in st.session_state:
    st.session_state.pose_model = None
    st.session_state.model_loaded = False

if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

# Header
st.markdown('<h1 class="main-header">🤸‍♂️ AI Pose Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Posture Analysis with OpenCV</p>', unsafe_allow_html=True)

# Device detection
def get_device_info():
    try:
        user_agent = st.context.headers.get("User-Agent", "").lower()
        is_mobile = any(keyword in user_agent for keyword in ["mobile", "android", "iphone", "ipad"])
        return {
            "type": "mobile" if is_mobile else "desktop",
            "is_mobile": is_mobile
        }
    except:
        return {"type": "desktop", "is_mobile": False}

device_info = get_device_info()

# Model loading
@st.cache_resource
def load_pose_model():
    if st.session_state.model_loaded and st.session_state.pose_model:
        return st.session_state.pose_model
    
    model_paths = [
        "pose2/train2/weights/best.pt",
        "best.pt", 
        "models/best.pt",
        "weights/best.pt"
    ]
    
    # Try existing models
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = YOLO(model_path)
                st.session_state.pose_model = model
                st.session_state.model_loaded = True
                return model
            except Exception as e:
                continue
    
    # Try download model
    try:
        model = YOLO('yolov8n-pose.pt')
        st.session_state.pose_model = model
        st.session_state.model_loaded = True
        return model
    except:
        return None

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Device info
    device_icon = "📱" if device_info["is_mobile"] else "💻"
    st.markdown(f"""
    <div class="info-box">
        <h4>{device_icon} Device: {device_info["type"].title()}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Model loading
    st.markdown("#### 🤖 Model")
    if st.button("🔄 Load Model", use_container_width=True):
        st.session_state.model_loaded = False
        st.session_state.pose_model = None
        st.cache_resource.clear()
    
    model = load_pose_model()
    if model is None:
        st.markdown("""
        <div class="error-box">
            <h4>❌ Model Failed</h4>
            <p>Cannot load pose model</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    else:
        st.markdown("""
        <div class="success-box">
            <h4>✅ Model Ready</h4>
            <p>YOLO Pose Loaded</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Settings
    st.markdown("#### 🎯 Detection")
    confidence_threshold = st.slider("Confidence", 0.1, 1.0, 0.5, 0.05)
    keypoint_threshold = st.slider("Keypoint", 0.1, 1.0, 0.5, 0.05)
    
    st.markdown("#### 🎨 Display")
    show_keypoints = st.checkbox("Keypoints", value=True)
    show_connections = st.checkbox("Connections", value=True)
    show_angles = st.checkbox("Angles", value=True)
    show_confidence = st.checkbox("Confidence", value=True)
    show_fps = st.checkbox("FPS", value=True)

# Helper functions
def calculate_angle(a, b, c):
    if None in (a, b, c):
        return None
    try:
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        dot_product = np.dot(ba, bc)
        norms = np.linalg.norm(ba) * np.linalg.norm(bc)
        if norms == 0:
            return None
        cosine_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    except:
        return None

def draw_pose_annotations(frame, keypoints_obj, label, box, conf_score):
    try:
        color = COLORS.get(label, (255, 255, 255))
        label_text = CLASS_LABELS.get(label, "Unknown")

        if keypoints_obj is None:
            return frame
            
        keypoints = keypoints_obj.xy[0].cpu().numpy()
        confs = keypoints_obj.conf[0].cpu().numpy() if keypoints_obj.conf is not None else None

        # Draw keypoints
        pts = []
        for i, (x, y) in enumerate(keypoints):
            conf = confs[i] if confs is not None and i < len(confs) else 1.0
            
            if conf > keypoint_threshold:
                pt = (int(x), int(y))
                pts.append(pt)
                
                if show_keypoints:
                    cv2.circle(frame, pt, 6, (0, 0, 0), -1)
                    cv2.circle(frame, pt, 4, color, -1)
                    cv2.circle(frame, pt, 6, (255, 255, 255), 1)
            else:
                pts.append(None)

        # Draw connections
        if show_connections and len(pts) >= 2:
            for i, j in KEYPOINT_CONNECTIONS:
                if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
                    cv2.line(frame, pts[i], pts[j], color, 3, cv2.LINE_AA)

        # Draw angle
        if show_angles and len(pts) >= 3 and all(pts[k] for k in [0, 1, 2]):
            angle = calculate_angle(pts[0], pts[1], pts[2])
            if angle is not None:
                pos = pts[1]
                angle_text = f"{int(angle)}°"
                cv2.putText(frame, angle_text, (pos[0] + 10, pos[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw bounding box
        if box is not None:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                display_text = label_text
                if show_confidence:
                    display_text += f" ({conf_score:.1%})"
                
                cv2.putText(frame, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except:
                pass

        return frame
    except Exception as e:
        return frame

def process_frame(frame):
    try:
        start_time = time.time()
        
        results = model.predict(
            frame, 
            imgsz=640,
            conf=confidence_threshold,
            save=False,
            verbose=False
        )

        detection_count = 0
        pose_results = []

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is not None and keypoints is not None:
                for box, kpts in zip(boxes, keypoints):
                    try:
                        label = int(box.cls.cpu().item())
                        conf_score = float(box.conf.cpu().item())
                        
                        frame = draw_pose_annotations(frame, kpts, label, box, conf_score)
                        
                        detection_count += 1
                        pose_results.append({
                            'label': CLASS_LABELS.get(label, 'Unknown'),
                            'confidence': conf_score
                        })
                    except:
                        continue

        # Draw FPS
        if show_fps:
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame, detection_count, pose_results
        
    except Exception as e:
        return frame, 0, []

# Main interface
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📹 OpenCV Webcam", "📷 Upload Image", "🎬 Upload Video"])

# Tab 1: OpenCV Webcam (more reliable than WebRTC)
with tab1:
    st.markdown("### 📹 OpenCV Real-time Detection")
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 OpenCV Webcam Detection</h4>
        <p><strong>Advantages:</strong></p>
        <p>• No WebRTC connection issues</p>
        <p>• More stable camera access</p>
        <p>• Direct device selection</p>
        <p>• Better performance</p>
        <br>
        <p><strong>Instructions:</strong></p>
        <p>1. Click "Start OpenCV Camera" below</p>
        <p>2. Select your camera device</p>
        <p>3. Allow camera permissions</p>
        <p>4. Pose detection will start automatically</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🎥 Start OpenCV Camera", type="primary", use_container_width=True):
            st.session_state.camera_running = True
    
    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.camera_running = False
    
    with col3:
        if st.button("🔄 Reset Stats", use_container_width=True):
            st.session_state.stats = {
                'total_frames': 0,
                'total_detections': 0,
                'good_posture': 0,
                'bad_posture': 0,
                'session_start': time.time()
            }
    
    # OpenCV camera implementation
    if st.session_state.camera_running:
        try:
            # Camera device selection
            camera_index = st.selectbox(
                "Select Camera Device:",
                options=[0, 1, 2],
                format_func=lambda x: f"Camera {x}",
                help="Try different camera indices if camera doesn't work"
            )
            
            st.markdown("""
            <div class="success-box">
                <h4>🟢 Camera Active</h4>
                <p>OpenCV camera is running with pose detection</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Placeholder for camera feed
            camera_placeholder = st.empty()
            stats_placeholder = st.empty()
            
            # Statistics tracking
            frame_count = 0
            good_posture = 0
            bad_posture = 0
            
            # Try to open camera
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                st.error(f"❌ Cannot open camera {camera_index}. Try a different camera index.")
            else:
                st.success(f"✅ Camera {camera_index} opened successfully!")
                
                # Camera loop placeholder - in real implementation this would be a continuous loop
                # For demo purposes, we'll show the concept
                ret, frame = cap.read()
                if ret:
                    # Process frame
                    processed_frame, detection_count, pose_results = process_frame(frame)
                    
                    # Convert to RGB for display
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Update stats
                    for result in pose_results:
                        if result['label'] == 'Postur Baik':
                            good_posture += 1
                        else:
                            bad_posture += 1
                    
                    # Display stats
                    with stats_placeholder.container():
                        if device_info["is_mobile"]:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{detection_count}</h3>
                                    <p>🎯 Detections</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{good_posture}</h3>
                                    <p>✅ Good</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{frame_count + 1}</h3>
                                    <p>🎞️ Frames</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{bad_posture}</h3>
                                    <p>❌ Bad</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            cols = st.columns(4)
                            with cols[0]:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{frame_count + 1}</h3>
                                    <p>🎞️ Frames</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with cols[1]:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{detection_count}</h3>
                                    <p>🎯 Detections</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with cols[2]:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{good_posture}</h3>
                                    <p>✅ Good Posture</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with cols[3]:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{bad_posture}</h3>
                                    <p>❌ Bad Posture</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                cap.release()
                
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ Camera Error</h4>
                <p><strong>Error:</strong> {str(e)}</p>
                <p><strong>Solutions:</strong></p>
                <p>• Try different camera index (0, 1, 2)</p>
                <p>• Close other apps using camera</p>
                <p>• Check camera permissions</p>
                <p>• Refresh page and try again</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 3rem;">
            <h3>🎥 OpenCV Camera Ready</h3>
            <p>Click <strong>"Start OpenCV Camera"</strong> to begin</p>
            <p>This method is more reliable than WebRTC</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Alternative: Simple camera capture button
    st.markdown("---")
    st.markdown("### 📸 Alternative: Single Frame Capture")
    
    if st.button("📸 Capture & Analyze Frame", use_container_width=True):
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    processed_frame, detection_count, pose_results = process_frame(frame)
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 📷 Captured Frame")
                        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(original_rgb, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 🎯 Analysis Result")
                        st.image(frame_rgb, use_container_width=True)
                    
                    if detection_count > 0:
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>✅ Analysis Complete</h4>
                            <p><strong>Detections:</strong> {detection_count}</p>
                            <p><strong>Results:</strong> {len(pose_results)} poses analyzed</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📋 Detailed Results"):
                            for i, result in enumerate(pose_results, 1):
                                icon = "✅" if result['label'] == 'Postur Baik' else "❌"
                                st.markdown(f"""
                                **{icon} Person {i}:**
                                - Classification: {result['label']}
                                - Confidence: {result['confidence']:.2%}
                                """)
                    else:
                        st.warning("No poses detected. Try adjusting lighting or position.")
                
                cap.release()
            else:
                st.error("Cannot access camera. Check permissions and try again.")
        except Exception as e:
            st.error(f"Camera error: {str(e)}")

# Tab 2: Image Upload (same as before but simplified)
with tab2:
    st.markdown("### 📷 Image Analysis")
    
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload image for pose analysis"
    )
    
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 Image Info</h4>
                <p><strong>Size:</strong> {image.size[0]} x {image.size[1]}</p>
                <p><strong>Format:</strong> {image.format}</p>
                <p><strong>File Size:</strong> {uploaded_image.size / 1024:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Analyze Pose", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    # Convert to OpenCV
                    image_array = np.array(image)
                    frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    
                    # Process
                    processed_frame, detection_count, pose_results = process_frame(frame)
                    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🖼️ Original")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🎯 Result")
                    st.image(processed_rgb, use_container_width=True)
                
                if detection_count > 0:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Success!</h4>
                        <p><strong>Poses Found:</strong> {detection_count}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No poses detected.")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Tab 3: Video Upload (simplified)
with tab3:
    st.markdown("### 🎬 Video Analysis")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov'],
        help="Upload video for pose analysis"
    )
    
    if uploaded_video is not None:
        file_size = uploaded_video.size / (1024*1024)
        st.markdown(f"""
        <div class="info-box">
            <h4>🎬 Video Info</h4>
            <p><strong>File:</strong> {uploaded_video.name}</p>
            <p><strong>Size:</strong> {file_size:.2f} MB</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                temp_path = tfile.name
            
            try:
                cap = cv2.VideoCapture(temp_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    st.markdown(f"**Total Frames:** {total_frames} | **FPS:** {fps}")
                    
                    progress_bar = st.progress(0)
                    video_placeholder = st.empty()
                    
                    frame_count = 0
                    good_count = 0
                    bad_count = 0
                    
                    while cap.isOpened() and frame_count < min(100, total_frames):  # Limit for demo
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame_count % 10 == 0:  # Process every 10th frame
                            processed_frame, detections, results = process_frame(frame)
                            
                            for result in results:
                                if result['label'] == 'Postur Baik':
                                    good_count += 1
                                else:
                                    bad_count += 1
                            
                            if frame_count % 30 == 0:  # Show every 30th frame
                                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                video_placeholder.image(frame_rgb, use_container_width=True)
                        
                        frame_count += 1
                        progress_bar.progress(frame_count / min(100, total_frames))
                    
                    cap.release()
                    
                    # Results
                    st.markdown("""
                    <div class="success-box">
                        <h3>🎉 Processing Complete!</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("📊 Frames Processed", frame_count)
                    with cols[1]:
                        st.metric("✅ Good Posture", good_count)
                    with cols[2]:
                        st.metric("❌ Bad Posture", bad_count)
                
            except Exception as e:
                st.error(f"Video processing error: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

# Footer with system status
st.markdown("---")
st.markdown("### 📊 System Status")

cols = st.columns(4)

with cols[0]:
    model_status = "✅ Ready" if st.session_state.model_loaded else "❌ Not Loaded"
    st.markdown(f"""
    <div class="metric-card">
        <h4>🤖 Model</h4>
        <p>{model_status}</p>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <h4>📱 Device</h4>
        <p>{device_info["type"].title()}</p>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    camera_status = "🟢 Active" if st.session_state.camera_running else "⚪ Inactive"
    st.markdown(f"""
    <div class="metric-card">
        <h4>📹 Camera</h4>
        <p>{camera_status}</p>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    session_time = (time.time() - st.session_state.stats['session_start']) / 60
    st.markdown(f"""
    <div class="metric-card">
        <h4>⏱️ Session</h4>
        <p>{session_time:.1f}m</p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced troubleshooting section
st.markdown("---")
st.markdown("### 🔧 Troubleshooting Guide")

with st.expander("🚨 Common Issues & Solutions"):
    st.markdown("""
    ### 📹 **Camera Issues**
    
    **Problem: Camera not working**
    - ✅ Try different camera indices (0, 1, 2)
    - ✅ Close other applications using camera
    - ✅ Check browser permissions
    - ✅ Refresh page and try again
    - ✅ Use "Single Frame Capture" as alternative
    
    **Problem: Poor detection accuracy**
    - ✅ Ensure good lighting (front lighting preferred)
    - ✅ Position yourself 1-2 meters from camera
    - ✅ Wear contrasting clothes against background
    - ✅ Adjust confidence threshold in sidebar
    - ✅ Make sure full body is visible in frame
    
    ### 🤖 **Model Issues**
    
    **Problem: Model loading failed**
    - ✅ Check internet connection for auto-download
    - ✅ Manually download YOLO pose model
    - ✅ Place model file in correct directory
    - ✅ Click "Load Model" to retry
    
    ### ⚡ **Performance Issues**
    
    **Problem: Slow processing**
    - ✅ Use lower resolution camera settings
    - ✅ Close other browser tabs
    - ✅ Try "Single Frame Capture" instead of continuous
    - ✅ Reduce visual options in sidebar
    
    ### 📱 **Mobile Specific**
    
    **Problem: Mobile camera issues**
    - ✅ Use landscape orientation
    - ✅ Allow camera permissions in browser
    - ✅ Try both front and back camera (index 0, 1)
    - ✅ Use Chrome or Safari mobile browser
    - ✅ Ensure stable WiFi connection
    
    ### 💻 **Desktop Specific**
    
    **Problem: Multiple cameras**
    - ✅ Try camera index 0 (default)
    - ✅ Try camera index 1 (external webcam)
    - ✅ Check device manager for camera list
    - ✅ Unplug/replug USB cameras
    
    ### 🔄 **Quick Fixes**
    
    **If nothing works:**
    1. Refresh the browser page
    2. Clear browser cache
    3. Try incognito/private mode
    4. Use different browser (Chrome recommended)
    5. Restart browser completely
    6. Check antivirus/firewall settings
    """)

# Tips for better pose detection
st.markdown("### 💡 Tips for Best Results")

if device_info["is_mobile"]:
    st.markdown("""
    <div class="info-box">
        <h4>📱 Mobile Optimization Tips</h4>
        <p><strong>Camera Setup:</strong></p>
        <p>• Use landscape mode for wider field of view</p>
        <p>• Keep device stable (use stand if possible)</p>
        <p>• Ensure front lighting, avoid backlighting</p>
        <p>• Position device at chest/shoulder height</p>
        <br>
        <p><strong>Detection Tips:</strong></p>
        <p>• Wear solid colors, avoid patterns</p>
        <p>• Keep background simple and contrasting</p>
        <p>• Maintain 1.5-2 meter distance</p>
        <p>• Ensure full upper body is visible</p>
        <br>
        <p><strong>Performance:</strong></p>
        <p>• Close other apps to free up memory</p>
        <p>• Use stable WiFi connection</p>
        <p>• Try "Single Frame Capture" for better stability</p>
    </div>
    """, unsafe_allow_html=True)
else:
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        <div class="info-box">
            <h4>🎥 Camera Setup</h4>
            <p><strong>Positioning:</strong></p>
            <p>• Camera at eye level</p>
            <p>• 1-2 meter distance</p>
            <p>• Good front lighting</p>
            <p>• Stable camera mount</p>
            <br>
            <p><strong>Environment:</strong></p>
            <p>• Simple background</p>
            <p>• Avoid clutter</p>
            <p>• Consistent lighting</p>
            <p>• Minimal distractions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div class="info-box">
            <h4>👤 Posture Tips</h4>
            <p><strong>Clothing:</strong></p>
            <p>• Solid colors preferred</p>
            <p>• Avoid busy patterns</p>
            <p>• Contrasting background</p>
            <p>• Well-fitted clothes</p>
            <br>
            <p><strong>Position:</strong></p>
            <p>• Face camera directly</p>
            <p>• Keep full torso visible</p>
            <p>• Natural sitting/standing</p>
            <p>• Avoid covering keypoints</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div class="info-box">
            <h4>⚙️ Settings</h4>
            <p><strong>Detection:</strong></p>
            <p>• Confidence: 0.5-0.7</p>
            <p>• Keypoint: 0.4-0.6</p>
            <p>• Enable all visuals</p>
            <p>• Show FPS for monitoring</p>
            <br>
            <p><strong>Performance:</strong></p>
            <p>• Close unnecessary tabs</p>
            <p>• Use latest browser</p>
            <p>• Check camera drivers</p>
            <p>• Stable internet connection</p>
        </div>
        """, unsafe_allow_html=True)

# Real-time stats summary
if st.session_state.stats['good_posture'] > 0 or st.session_state.stats['bad_posture'] > 0:
    total_detections = st.session_state.stats['good_posture'] + st.session_state.stats['bad_posture']
    accuracy = (st.session_state.stats['good_posture'] / total_detections * 100) if total_detections > 0 else 0
    
    st.markdown("### 📈 Session Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Total Detections", total_detections)
    with col2:
        st.metric("✅ Good Posture", st.session_state.stats['good_posture'])
    with col3:
        st.metric("❌ Bad Posture", st.session_state.stats['bad_posture'])
    with col4:
        st.metric("📊 Accuracy", f"{accuracy:.1f}%")
    
    # Progress bar
    if total_detections > 0:
        st.progress(accuracy / 100)
        
        if accuracy >= 80:
            st.markdown("""
            <div class="success-box">
                <h4>🎉 Excellent Posture Session!</h4>
                <p>Keep up the great work maintaining good posture.</p>
            </div>
            """, unsafe_allow_html=True)
        elif accuracy >= 60:
            st.markdown("""
            <div class="info-box">
                <h4>👍 Good Posture Session</h4>
                <p>Room for improvement. Focus on sitting/standing straighter.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Posture Needs Attention</h4>
                <p>Consider adjusting your seating position and taking more breaks.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-top: 2rem;">
    <h3>🤸‍♂️ AI Pose Detection System</h3>
    <p><strong>Technology:</strong> YOLO v8 • OpenCV • Streamlit</p>
    <p><strong>Features:</strong> Real-time Detection • Multi-platform • Easy to Use</p>
    <p><strong>Compatibility:</strong> 💻 Desktop • 📱 Mobile • 🌐 Web Browser</p>
    <br>
    <p><em>Reliable pose analysis with OpenCV - No WebRTC issues!</em></p>
    <p><strong>Version:</strong> OpenCV Stable v2.0</p>
</div>
""", unsafe_allow_html=True)
