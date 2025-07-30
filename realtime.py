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
import asyncio
from typing import Optional, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Pose Detection - Real-time",
    page_icon="🤸‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced mobile support
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
        padding: 0.8rem;
        border-radius: 12px;
        margin: 0.3rem 0;
        text-align: center;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .error-box {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    
    /* Responsive video container */
    .video-container {
        width: 100%;
        max-width: 100%;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    /* Mobile optimizations */
    @media (max-width: 768px) {
        .stColumn {
            padding: 0 0.2rem;
        }
        .metric-card {
            padding: 0.6rem 0.4rem;
            font-size: 0.85rem;
        }
        .metric-card h3 {
            font-size: 1.2rem;
            margin: 0;
        }
        .metric-card p {
            font-size: 0.8rem;
            margin: 0.2rem 0 0 0;
        }
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-online { background-color: #00b894; }
    .status-offline { background-color: #fd79a8; }
    .status-loading { background-color: #fdcb6e; }
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

# Global model variable
if 'pose_model' not in st.session_state:
    st.session_state.pose_model = None
    st.session_state.model_loaded = False
    st.session_state.model_path = None
    st.session_state.device_info = None

# Initialize session state for statistics
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_frames': 0,
        'total_detections': 0,
        'good_posture': 0,
        'bad_posture': 0,
        'session_start': time.time()
    }

# Header
st.markdown('<h1 class="main-header">🤸‍♂️ AI Pose Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Posture Analysis with Auto Device Detection</p>', unsafe_allow_html=True)

# Enhanced WebRTC Configuration - More flexible constraints
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ],
    "iceTransportPolicy": "all",
    "bundlePolicy": "max-bundle",
    "rtcpMuxPolicy": "require"
})

# Device detection and camera constraints
def get_device_info():
    """Enhanced device detection"""
    try:
        user_agent = st.context.headers.get("User-Agent", "").lower()
        
        # Mobile detection
        mobile_keywords = ["mobile", "android", "iphone", "ipad", "tablet"]
        is_mobile = any(keyword in user_agent for keyword in mobile_keywords)
        
        # Browser detection
        if "chrome" in user_agent:
            browser = "chrome"
        elif "firefox" in user_agent:
            browser = "firefox"
        elif "safari" in user_agent and "chrome" not in user_agent:
            browser = "safari"
        else:
            browser = "other"
        
        # OS detection
        if "windows" in user_agent:
            os_type = "Windows"
        elif "mac" in user_agent:
            os_type = "macOS"
        elif "linux" in user_agent:
            os_type = "Linux"
        elif "android" in user_agent:
            os_type = "Android"
        elif "ios" in user_agent:
            os_type = "iOS"
        else:
            os_type = "Unknown"
        
        return {
            "type": "mobile" if is_mobile else "desktop",
            "browser": browser,
            "os": os_type,
            "is_mobile": is_mobile,
            "user_agent": user_agent[:100] + "..." if len(user_agent) > 100 else user_agent
        }
    except Exception as e:
        logger.warning(f"Device detection failed: {e}")
        return {
            "type": "desktop",
            "browser": "unknown",
            "os": "Unknown",
            "is_mobile": False,
            "user_agent": "Unknown"
        }

device_info = get_device_info()

# Auto-detect optimal camera constraints
def get_camera_constraints(device_type: str, quality: str = "balanced"):
    """Get optimal camera constraints based on device and quality setting"""
    
    constraints = {
        "audio": False,
        "video": {}
    }
    
    if device_type == "mobile":
        if quality == "low":
            constraints["video"] = {
                "width": {"ideal": 320, "min": 240, "max": 480},
                "height": {"ideal": 240, "min": 180, "max": 360},
                "frameRate": {"ideal": 15, "min": 10, "max": 20}
            }
        elif quality == "high":
            constraints["video"] = {
                "width": {"ideal": 640, "min": 320, "max": 1280},
                "height": {"ideal": 480, "min": 240, "max": 720},
                "frameRate": {"ideal": 24, "min": 15, "max": 30}
            }
        else:  # balanced
            constraints["video"] = {
                "width": {"ideal": 480, "min": 320, "max": 640},
                "height": {"ideal": 360, "min": 240, "max": 480},
                "frameRate": {"ideal": 20, "min": 15, "max": 24}
            }
    else:  # desktop
        if quality == "low":
            constraints["video"] = {
                "width": {"ideal": 480, "min": 320, "max": 640},
                "height": {"ideal": 360, "min": 240, "max": 480},
                "frameRate": {"ideal": 20, "min": 15, "max": 24}
            }
        elif quality == "high":
            constraints["video"] = {
                "width": {"ideal": 1280, "min": 640, "max": 1920},
                "height": {"ideal": 720, "min": 480, "max": 1080},
                "frameRate": {"ideal": 30, "min": 20, "max": 60}
            }
        else:  # balanced
            constraints["video"] = {
                "width": {"ideal": 640, "min": 480, "max": 800},
                "height": {"ideal": 480, "min": 360, "max": 600},
                "frameRate": {"ideal": 24, "min": 20, "max": 30}
            }
    
    # Add facing mode for mobile
    if device_type == "mobile":
        constraints["video"]["facingMode"] = "user"  # Front camera
    
    return constraints

# Model loading with caching and optimization
@st.cache_resource
def load_pose_model(force_reload: bool = False):
    """Load YOLO pose model with smart fallbacks"""
    if not force_reload and st.session_state.model_loaded and st.session_state.pose_model:
        return st.session_state.pose_model, st.session_state.model_path, st.session_state.device_info
    
    model_paths = [
        "pose2/train2/weights/best.pt",
        "best.pt", 
        "models/best.pt",
        "weights/best.pt",
        "pose_model.pt"
    ]
    
    # Try to load existing model
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                with st.spinner(f"🔄 Loading model: {model_path}"):
                    model = YOLO(model_path)
                    
                    # Auto-detect best device
                    device = "cpu"
                    try:
                        import torch
                        if torch.cuda.is_available():
                            device = "cuda"
                            model.to(device)
                    except:
                        pass
                    
                    device_details = {
                        "device": device,
                        "model_path": model_path,
                        "model_size": os.path.getsize(model_path) / (1024*1024) if os.path.exists(model_path) else 0
                    }
                    
                    # Cache in session state
                    st.session_state.pose_model = model
                    st.session_state.model_loaded = True
                    st.session_state.model_path = model_path
                    st.session_state.device_info = device_details
                    
                    return model, model_path, device_details
                    
            except Exception as e:
                logger.error(f"Failed to load {model_path}: {e}")
                continue
    
    # Try to download pretrained model
    try:
        with st.spinner("📥 Downloading YOLO pose model..."):
            model = YOLO('yolov8n-pose.pt')
            device = "cpu"
            
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    model.to(device)
            except:
                pass
            
            device_details = {
                "device": device,
                "model_path": "yolov8n-pose.pt (downloaded)",
                "model_size": 0
            }
            
            # Cache in session state
            st.session_state.pose_model = model
            st.session_state.model_loaded = True
            st.session_state.model_path = "yolov8n-pose.pt (downloaded)"
            st.session_state.device_info = device_details
            
            return model, "yolov8n-pose.pt (downloaded)", device_details
            
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
    
    return None, None, None

# Sidebar with enhanced controls
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    
    # Device information
    device_icon = "📱" if device_info["is_mobile"] else "💻"
    st.markdown(f"""
    <div class="info-box">
        <h4>{device_icon} Device Info</h4>
        <p><strong>Type:</strong> {device_info["type"].title()}</p>
        <p><strong>OS:</strong> {device_info["os"]}</p>
        <p><strong>Browser:</strong> {device_info["browser"].title()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model loading section
    st.markdown("#### 🤖 Model Settings")
    
    # Quality preset
    quality_options = {
        "low": "Low (Fast)",
        "balanced": "Balanced (Recommended)", 
        "high": "High (Accurate)"
    }
    
    quality_preset = st.selectbox(
        "Quality Preset",
        options=list(quality_options.keys()),
        format_func=lambda x: quality_options[x],
        index=1,  # Default to balanced
        help="Auto-optimizes all settings based on your device"
    )
    
    # Load model button
    if st.button("🔄 Load/Reload Model", use_container_width=True):
        st.session_state.model_loaded = False
        st.session_state.pose_model = None
        st.cache_resource.clear()
        st.rerun()
    
    # Model status
    if not st.session_state.model_loaded:
        model, model_path, device_details = load_pose_model()
        
        if model is None:
            st.markdown("""
            <div class="error-box">
                <h4>❌ Model Loading Failed</h4>
                <p>Unable to load pose detection model.</p>
                <p>Check internet connection for auto-download.</p>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        else:
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ Model Ready</h4>
                <p><strong>Path:</strong> {model_path}</p>
                <p><strong>Device:</strong> {device_details['device'].upper()}</p>
                <p><strong>Size:</strong> {device_details['model_size']:.1f} MB</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="success-box">
            <h4>✅ Model Loaded</h4>
            <p><strong>Status:</strong> Ready for inference</p>
            <p><strong>Device:</strong> {st.session_state.device_info['device'].upper()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
        keypoint_threshold = st.slider("Keypoint Threshold", 0.1, 1.0, 0.5, 0.05)
        
        # Visual options
        st.markdown("**Visual Options:**")
        show_keypoints = st.checkbox("Show Keypoints", value=True)
        show_connections = st.checkbox("Show Connections", value=True)
        show_angles = st.checkbox("Show Angles", value=True)
        show_confidence = st.checkbox("Show Confidence", value=True)
        show_fps = st.checkbox("Show FPS", value=True)
        
        # Performance settings
        st.markdown("**Performance:**")
        processing_skip = st.slider("Process Every N Frames", 1, 5, 1, 1, help="Skip frames to improve performance")
        line_thickness = st.slider("Line Thickness", 1, 5, 2)
        text_scale = st.slider("Text Scale", 0.3, 1.0, 0.6, 0.1)

# Enhanced pose drawing function
def draw_pose_annotations(frame, keypoints_obj, label, box, conf_score):
    """Enhanced pose drawing with better error handling"""
    try:
        color = COLORS.get(label, (255, 255, 255))
        label_text = CLASS_LABELS.get(label, "Unknown")

        # Extract keypoints safely
        if keypoints_obj is None or len(keypoints_obj.xy) == 0:
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
                    cv2.circle(frame, pt, 6, (0, 0, 0), -1)  # Black outline
                    cv2.circle(frame, pt, 4, color, -1)      # Colored center
                    cv2.circle(frame, pt, 6, (255, 255, 255), 1)  # White border
            else:
                pts.append(None)

        # Draw connections
        if show_connections and len(pts) >= 2:
            for i, j in KEYPOINT_CONNECTIONS:
                if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
                    cv2.line(frame, pts[i], pts[j], color, line_thickness, cv2.LINE_AA)

        # Calculate and display angle
        if show_angles and len(pts) >= 3 and all(pts[k] for k in [0, 1, 2]):
            angle = calculate_angle(pts[0], pts[1], pts[2])
            if angle is not None:
                pos = pts[1]
                angle_text = f"{int(angle)}°"
                
                # Background for angle
                (text_width, text_height), _ = cv2.getTextSize(
                    angle_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2
                )
                
                cv2.rectangle(frame, 
                    (pos[0] + 8, pos[1] - text_height - 8), 
                    (pos[0] + text_width + 16, pos[1] + 2), 
                    (0, 0, 0), -1)
                
                cv2.putText(frame, angle_text, 
                    (pos[0] + 12, pos[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), 2)

        # Draw bounding box and label
        if box is not None:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label
                display_text = label_text
                if show_confidence:
                    display_text += f" ({conf_score:.1%})"
                
                # Label background
                (text_width, text_height), _ = cv2.getTextSize(
                    display_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2
                )
                
                cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                    (x1 + text_width + 10, y1), color, -1)
                
                cv2.putText(frame, display_text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 2)
            except Exception as e:
                logger.warning(f"Error drawing bounding box: {e}")

        return frame
    except Exception as e:
        logger.error(f"Error in draw_pose_annotations: {e}")
        return frame

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
            
        cosine_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    except:
        return None

# Optimized frame processing
def process_frame_for_pose(frame, frame_count=0):
    """Optimized pose detection processing"""
    try:
        # Skip frames for performance
        if frame_count % processing_skip != 0:
            return frame, 0, []
        
        start_time = time.time()
        
        # Get model from session state
        model = st.session_state.pose_model
        if model is None:
            return frame, 0, []
        
        # Determine image size based on quality preset
        img_size = 320 if quality_preset == "low" else 640 if quality_preset == "balanced" else 1280
        
        # Run inference
        results = model.predict(
            frame, 
            imgsz=img_size,
            conf=confidence_threshold,
            save=False,
            verbose=False
        )

        detection_count = 0
        pose_results = []

        # Process results
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is not None and keypoints is not None:
                for box, kpts in zip(boxes, keypoints):
                    try:
                        label = int(box.cls.cpu().item())
                        conf_score = float(box.conf.cpu().item())
                        
                        # Draw annotations
                        frame = draw_pose_annotations(frame, kpts, label, box, conf_score)
                        
                        detection_count += 1
                        pose_results.append({
                            'label': CLASS_LABELS.get(label, 'Unknown'),
                            'confidence': conf_score
                        })
                    except Exception as e:
                        logger.warning(f"Error processing detection: {e}")
                        continue

        # Draw FPS
        if show_fps:
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame, detection_count, pose_results
        
    except Exception as e:
        logger.error(f"Error in pose processing: {e}")
        return frame, 0, []

# Enhanced WebRTC Video Transformer
class OptimizedPoseTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_count = 0
        self.detection_count = 0
        self.good_posture = 0
        self.bad_posture = 0
        self.lock = threading.Lock()
        self.last_update = time.time()
        
    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Process frame
            processed_img, detections, results = process_frame_for_pose(img, self.frame_count)
            
            # Update statistics thread-safely
            with self.lock:
                self.frame_count += 1
                self.detection_count = detections
                
                # Count posture types
                for result in results:
                    if result['label'] == 'Postur Baik':
                        self.good_posture += 1
                    else:
                        self.bad_posture += 1
                
                # Update session state periodically
                current_time = time.time()
                if current_time - self.last_update > 1.0:  # Update every second
                    st.session_state.stats.update({
                        'total_frames': self.frame_count,
                        'total_detections': self.frame_count,
                        'good_posture': self.good_posture,
                        'bad_posture': self.bad_posture
                    })
                    self.last_update = current_time
            
            return processed_img
            
        except Exception as e:
            logger.error(f"Transform error: {e}")
            return frame.to_ndarray(format="bgr24")

# Main interface
st.markdown("---")

# Check if model is loaded before showing interface
if not st.session_state.model_loaded:
    st.markdown("""
    <div class="warning-box">
        <h3>⏳ Loading Model...</h3>
        <p>Please wait while the pose detection model is being loaded.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Try to load model
    model, model_path, device_details = load_pose_model()
    if model is None:
        st.error("Failed to load model. Please check the sidebar for more information.")
        st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(["📹 Real-time Webcam", "📷 Upload Image", "🎬 Upload Video"])

# Tab 1: Enhanced Real-time Webcam
with tab1:
    st.markdown("### 📹 Real-time Pose Detection")
    
    # Auto-detect optimal settings
    camera_constraints = get_camera_constraints(device_info["type"], quality_preset)
    
    # Display current settings
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <h4>📋 Current Settings</h4>
            <p><strong>Quality:</strong> {quality_options[quality_preset]}</p>
            <p><strong>Resolution:</strong> {camera_constraints['video']['width']['ideal']}x{camera_constraints['video']['height']['ideal']}</p>
            <p><strong>FPS:</strong> {camera_constraints['video']['frameRate']['ideal']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Reset Statistics", use_container_width=True):
            st.session_state.stats = {
                'total_frames': 0,
                'total_detections': 0,
                'good_posture': 0,
                'bad_posture': 0,
                'session_start': time.time()
            }
            st.success("✅ Statistics reset!")
    
    # Instructions based on device
    if device_info["is_mobile"]:
        st.markdown("""
        <div class="info-box">
            <h4>📱 Mobile Instructions</h4>
            <p><strong>1.</strong> Allow camera access when prompted</p>
            <p><strong>2.</strong> Use landscape mode for better detection</p>
            <p><strong>3.</strong> Ensure good lighting</p>
            <p><strong>4.</strong> Keep device stable</p>
            <p><strong>5.</strong> Position yourself 1.5-2m from camera</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>💻 Desktop Instructions</h4>
            <p><strong>1.</strong> Click START to begin webcam stream</p>
            <p><strong>2.</strong> Allow camera access in browser</p>
            <p><strong>3.</strong> Position yourself 1-2m from camera</p>
            <p><strong>4.</strong> Ensure good lighting from front</p>
            <p><strong>5.</strong> Sit/stand with good posture for calibration</p>
        </div>
        """, unsafe_allow_html=True)
    
    # WebRTC Streamer with flexible constraints
    try:
        webrtc_ctx = webrtc_streamer(
            key=f"pose-detection-{quality_preset}",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=OptimizedPoseTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=camera_constraints,
            async_processing=True,
            video_html_attrs={
                "style": {"width": "100%", "border-radius": "12px"},
                "controls": False,
                "autoPlay": True,
            }
        )
        
        # Real-time statistics
        if webrtc_ctx.video_transformer:
            st.markdown("### 📊 Live Statistics")
            
            # Get current stats
            transformer = webrtc_ctx.video_transformer
            
            if device_info["is_mobile"]:
                # Mobile: 2x2 grid
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{transformer.frame_count:,}</h3>
                        <p>🎞️ Frames</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{transformer.good_posture:,}</h3>
                        <p>✅ Good Posture</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{transformer.detection_count}</h3>
                        <p>🎯 Current</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{transformer.bad_posture:,}</h3>
                        <p>❌ Bad Posture</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Desktop: 4 columns
                cols = st.columns(4)
                
                with cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{transformer.frame_count:,}</h3>
                        <p>🎞️ Total Frames</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{transformer.detection_count}</h3>
                        <p>🎯 Current Detections</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{transformer.good_posture:,}</h3>
                        <p>✅ Good Posture</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{transformer.bad_posture:,}</h3>
                        <p>❌ Bad Posture</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Session summary and progress
            total_postures = transformer.good_posture + transformer.bad_posture
            if total_postures > 0:
                good_percentage = (transformer.good_posture / total_postures) * 100
                
                st.markdown("#### 📈 Session Progress")
                progress_bar = st.progress(good_percentage / 100)
                
                # Status based on percentage
                if good_percentage >= 80:
                    status_color = "success-box"
                    status_icon = "🟢"
                    status_text = "Excellent Posture"
                elif good_percentage >= 60:
                    status_color = "info-box"
                    status_icon = "🟡"
                    status_text = "Good Posture"
                else:
                    status_color = "warning-box"
                    status_icon = "🔴"
                    status_text = "Needs Improvement"
                
                st.markdown(f"""
                <div class="{status_color}">
                    <h4>{status_icon} {status_text}</h4>
                    <p><strong>Posture Quality:</strong> {good_percentage:.1f}%</p>
                    <p><strong>Session Duration:</strong> {(time.time() - st.session_state.stats.get('session_start', time.time())) / 60:.1f} minutes</p>
                    <p><strong>Total Detections:</strong> {total_postures:,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Real-time recommendations
                st.markdown("#### 💡 Live Recommendations")
                if transformer.detection_count > 0:
                    recent_bad = transformer.bad_posture > transformer.good_posture
                    if recent_bad:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Posture Alert</h4>
                            <p><strong>Recommendations:</strong></p>
                            <p>• Straighten your back</p>
                            <p>• Align your shoulders</p>
                            <p>• Lift your chin slightly</p>
                            <p>• Take a deep breath and reset</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-box">
                            <h4>✅ Great Posture!</h4>
                            <p>Keep maintaining this excellent position.</p>
                            <p>Remember to take breaks every 30 minutes.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-box">
                        <h4>👀 Position Yourself</h4>
                        <p>Move into the camera view for pose detection.</p>
                        <p>Ensure good lighting and clear visibility.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Performance metrics
            if transformer.frame_count > 100:  # After some frames processed
                with st.expander("📊 Performance Metrics"):
                    avg_fps = transformer.frame_count / (time.time() - st.session_state.stats.get('session_start', time.time()))
                    detection_rate = (total_postures / transformer.frame_count * 100) if transformer.frame_count > 0 else 0
                    
                    perf_col1, perf_col2 = st.columns(2)
                    with perf_col1:
                        st.metric("Average FPS", f"{avg_fps:.1f}")
                        st.metric("Detection Rate", f"{detection_rate:.1f}%")
                    with perf_col2:
                        st.metric("Processing Quality", quality_options[quality_preset])
                        st.metric("Model Device", st.session_state.device_info['device'].upper())
    
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Camera Access Error</h4>
            <p><strong>Error:</strong> {str(e)}</p>
            <h5>🔧 Quick Fixes:</h5>
            <p><strong>Common Solutions:</strong></p>
            <p>• Refresh the page and try again</p>
            <p>• Check camera permissions in browser settings</p>
            <p>• Close other apps using the camera</p>
            <p>• Try a different browser (Chrome recommended)</p>
            <p>• Switch to lower quality preset</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Fallback options
        st.markdown("#### 🔄 Alternative Options")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry with Low Quality", use_container_width=True):
                st.session_state.quality_preset = "low"
                st.rerun()
        with col2:
            if st.button("📱 Switch to Image Upload", use_container_width=True):
                st.info("Please use the 'Upload Image' tab as an alternative.")

# Tab 2: Image Upload
with tab2:
    st.markdown("### 📷 Image Pose Analysis")
    
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image containing people for pose detection and classification"
    )
    
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            
            # Image info
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 Image Information</h4>
                <p><strong>Size:</strong> {image.size[0]} x {image.size[1]} pixels</p>
                <p><strong>Mode:</strong> {image.mode}</p>
                <p><strong>Format:</strong> {image.format}</p>
                <p><strong>File Size:</strong> {uploaded_image.size / 1024:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Process image
            if st.button("🔍 Analyze Pose", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing pose..."):
                    # Convert PIL to OpenCV format
                    image_array = np.array(image)
                    if len(image_array.shape) == 3:
                        frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    else:
                        frame = image_array
                    
                    # Process with pose detection
                    processed_frame, detection_count, pose_results = process_frame_for_pose(frame)
                    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🖼️ Original Image")
                    st.image(image, use_container_width=True)
                
                with col2:
                    st.markdown("#### 🎯 Analysis Result")
                    st.image(processed_rgb, use_container_width=True)
                
                # Results summary
                if detection_count > 0:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>🎉 Analysis Complete!</h4>
                        <p><strong>Poses Detected:</strong> {detection_count}</p>
                        <p><strong>Processing:</strong> Successful</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed results
                    with st.expander("📋 Detailed Results"):
                        for i, result in enumerate(pose_results, 1):
                            posture_icon = "✅" if result['label'] == 'Postur Baik' else "❌"
                            st.markdown(f"""
                            **{posture_icon} Person {i}:**
                            - **Classification:** {result['label']}
                            - **Confidence:** {result['confidence']:.2%}
                            """)
                            st.markdown("---")
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ No Poses Detected</h4>
                        <p>Try adjusting the confidence threshold in the sidebar or use an image with clearer poses.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ Image Processing Error</h4>
                <p>{str(e)}</p>
            </div>
            """, unsafe_allow_html=True)

# Tab 3: Video Upload
with tab3:
    st.markdown("### 🎬 Video Pose Analysis")
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload a video file for batch pose analysis"
    )
    
    if uploaded_video is not None:
        file_size_mb = uploaded_video.size / (1024*1024)
        st.markdown(f"""
        <div class="info-box">
            <h4>🎬 Video Information</h4>
            <p><strong>File:</strong> {uploaded_video.name}</p>
            <p><strong>Size:</strong> {file_size_mb:.2f} MB</p>
            <p><strong>Type:</strong> {uploaded_video.type}</p>
            <p><strong>Estimated Time:</strong> ~{file_size_mb * 3:.0f} seconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            skip_frames = st.selectbox(
                "Frame Skip (for speed)", 
                [1, 2, 3, 5, 10], 
                index=1,
                help="Process every Nth frame to speed up analysis"
            )
        with col2:
            show_preview = st.checkbox("Show Preview", value=True)
        
        if st.button("🚀 Process Video", type="primary", use_container_width=True):
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                temp_video_path = tfile.name
            
            try:
                # Process video
                cap = cv2.VideoCapture(temp_video_path)
                
                if not cap.isOpened():
                    st.error("❌ Cannot open video file")
                else:
                    # Get video properties
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps > 0 else 0
                    
                    # Display video info
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("📊 FPS", fps)
                    with cols[1]:
                        st.metric("🎞️ Frames", total_frames)
                    with cols[2]:
                        st.metric("⏱️ Duration", f"{duration:.1f}s")
                    
                    # Processing placeholders
                    video_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Statistics
                    frame_count = 0
                    total_detections = 0
                    good_posture_count = 0
                    bad_posture_count = 0
                    
                    start_time = time.time()
                    
                    # Process frames
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process every skip_frames
                        if frame_count % skip_frames == 0:
                            processed_frame, detection_count, pose_results = process_frame_for_pose(frame)
                            
                            # Update stats
                            total_detections += detection_count
                            for result in pose_results:
                                if result['label'] == 'Postur Baik':
                                    good_posture_count += 1
                                else:
                                    bad_posture_count += 1
                            
                            # Show preview
                            if show_preview and frame_count % (skip_frames * 5) == 0:
                                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                video_placeholder.image(frame_rgb, use_container_width=True)
                        
                        # Update progress
                        frame_count += 1
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        
                        # Status update
                        elapsed = time.time() - start_time
                        fps_processing = frame_count / elapsed if elapsed > 0 else 0
                        eta = (total_frames - frame_count) / fps_processing / 60 if fps_processing > 0 else 0
                        
                        status_text.text(f"Processing: {frame_count:,}/{total_frames:,} | Speed: {fps_processing:.1f} FPS | ETA: {eta:.1f}m")
                    
                    cap.release()
                    
                    # Final results
                    st.markdown("""
                    <div class="success-box">
                        <h3>🎉 Video Processing Complete!</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Statistics
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("🎯 Total Detections", total_detections)
                    with cols[1]:
                        st.metric("✅ Good Posture", good_posture_count)
                    with cols[2]:
                        st.metric("❌ Bad Posture", bad_posture_count)
                    with cols[3]:
                        accuracy = (good_posture_count / (good_posture_count + bad_posture_count)) * 100 if (good_posture_count + bad_posture_count) > 0 else 0
                        st.metric("📈 Posture Quality", f"{accuracy:.1f}%")
                        
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Video Processing Error</h4>
                    <p>{str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
            finally:
                # Cleanup
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)

# Footer with system info
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
    st.markdown(f"""
    <div class="metric-card">
        <h4>⚙️ Quality</h4>
        <p>{quality_options[quality_preset].split()[0]}</p>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    session_duration = (time.time() - st.session_state.stats.get('session_start', time.time())) / 60
    st.markdown(f"""
    <div class="metric-card">
        <h4>⏱️ Session</h4>
        <p>{session_duration:.1f}m</p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("""
<div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-top: 2rem;">
    <h3>🤸‍♂️ AI Pose Detection System</h3>
    <p><strong>Technology Stack:</strong> YOLO v8 • OpenCV • WebRTC • Streamlit</p>
    <p><strong>Auto-Detection:</strong> Device Type • Camera Settings • Quality Optimization</p>
    <p><strong>Cross-Platform:</strong> 💻 Desktop • 📱 Mobile • 📟 Tablet</p>
    <br>
    <p><em>Intelligent posture analysis with automatic optimization</em></p>
</div>
""", unsafe_allow_html=True)
