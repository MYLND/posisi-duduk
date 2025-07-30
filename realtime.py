import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np
import math
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode, VideoProcessorBase
import av
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Pose Detection - Fixed WebRTC",
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
    
    /* Status indicators */
    .status-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    .status-online { background-color: #00b894; }
    .status-offline { background-color: #fd79a8; }
    .status-connecting { background-color: #fdcb6e; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Video container */
    .video-frame {
        border: 3px solid #667eea;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Responsive */
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

if 'webrtc_running' not in st.session_state:
    st.session_state.webrtc_running = False

# Header
st.markdown('<h1 class="main-header">🤸‍♂️ AI Pose Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Fixed WebRTC Implementation - No Connection Issues</p>', unsafe_allow_html=True)

# Enhanced WebRTC Configuration - More reliable
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ],
    "iceTransportPolicy": "all",
    "bundlePolicy": "max-bundle",
})

# Device detection
def get_device_info():
    try:
        user_agent = st.context.headers.get("User-Agent", "").lower()
        is_mobile = any(keyword in user_agent for keyword in ["mobile", "android", "iphone", "ipad"])
        
        if "chrome" in user_agent:
            browser = "Chrome"
        elif "firefox" in user_agent:
            browser = "Firefox"
        elif "safari" in user_agent and "chrome" not in user_agent:
            browser = "Safari"
        else:
            browser = "Other"
            
        return {
            "type": "mobile" if is_mobile else "desktop",
            "is_mobile": is_mobile,
            "browser": browser
        }
    except:
        return {"type": "desktop", "is_mobile": False, "browser": "Unknown"}

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
    st.markdown("### ⚙️ Configuration")
    
    # Device info
    device_icon = "📱" if device_info["is_mobile"] else "💻"
    st.markdown(f"""
    <div class="info-box">
        <h4>{device_icon} Device Info</h4>
        <p><strong>Type:</strong> {device_info["type"].title()}</p>
        <p><strong>Browser:</strong> {device_info["browser"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model loading
    st.markdown("#### 🤖 Model Status")
    if st.button("🔄 Reload Model", use_container_width=True):
        st.session_state.model_loaded = False
        st.session_state.pose_model = None
        st.cache_resource.clear()
        st.rerun()
    
    model = load_pose_model()
    if model is None:
        st.markdown("""
        <div class="error-box">
            <h4>❌ Model Loading Failed</h4>
            <p>Cannot load pose detection model</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    else:
        st.markdown("""
        <div class="success-box">
            <h4>✅ Model Ready</h4>
            <p>YOLO Pose Model Loaded</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Settings
    st.markdown("#### 🎯 Detection Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    keypoint_threshold = st.slider("Keypoint Threshold", 0.1, 1.0, 0.5, 0.05)
    
    st.markdown("#### 🎨 Display Options")
    show_keypoints = st.checkbox("Show Keypoints", value=True)
    show_connections = st.checkbox("Show Connections", value=True)
    show_angles = st.checkbox("Show Angles", value=True)
    show_confidence = st.checkbox("Show Confidence", value=True)
    show_fps = st.checkbox("Show FPS", value=True)
    
    # Quality settings
    st.markdown("#### ⚙️ Quality Settings")
    quality_preset = st.selectbox(
        "Quality Preset",
        ["Low (320p)", "Medium (480p)", "High (720p)"],
        index=1
    )

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
                    cv2.circle(frame, pt, 8, (0, 0, 0), -1)  # Black outline
                    cv2.circle(frame, pt, 6, color, -1)      # Colored center
                    cv2.circle(frame, pt, 8, (255, 255, 255), 2)  # White border
            else:
                pts.append(None)

        # Draw connections
        if show_connections and len(pts) >= 2:
            for i, j in KEYPOINT_CONNECTIONS:
                if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
                    cv2.line(frame, pts[i], pts[j], color, 4, cv2.LINE_AA)

        # Draw angle
        if show_angles and len(pts) >= 3 and all(pts[k] for k in [0, 1, 2]):
            angle = calculate_angle(pts[0], pts[1], pts[2])
            if angle is not None:
                pos = pts[1]
                angle_text = f"{int(angle)}°"
                
                # Background for text
                (text_width, text_height), _ = cv2.getTextSize(
                    angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(frame, 
                    (pos[0] + 10, pos[1] - text_height - 10), 
                    (pos[0] + text_width + 20, pos[1] + 5), 
                    (0, 0, 0), -1)
                
                cv2.putText(frame, angle_text, (pos[0] + 15, pos[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw bounding box
        if box is not None:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Label
                display_text = label_text
                if show_confidence:
                    display_text += f" ({conf_score:.1%})"
                
                # Label background
                (text_width, text_height), _ = cv2.getTextSize(
                    display_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
                )
                
                cv2.rectangle(frame, (x1, y1 - text_height - 15), 
                    (x1 + text_width + 15, y1 - 5), color, -1)
                
                cv2.putText(frame, display_text, (x1 + 7, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass

        return frame
    except Exception as e:
        logger.error(f"Error in draw_pose_annotations: {e}")
        return frame

def process_frame_with_pose(frame):
    try:
        start_time = time.time()
        
        # Determine image size based on quality
        if "Low" in quality_preset:
            img_size = 320
        elif "High" in quality_preset:
            img_size = 640
        else:
            img_size = 480
        
        results = model.predict(
            frame, 
            imgsz=img_size,
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
                    except Exception as e:
                        logger.warning(f"Error processing detection: {e}")
                        continue

        # Draw FPS
        if show_fps:
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        return frame, detection_count, pose_results
        
    except Exception as e:
        logger.error(f"Error in pose processing: {e}")
        return frame, 0, []

# Enhanced WebRTC Video Processor
class StablePoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.detection_count = 0
        self.good_posture = 0
        self.bad_posture = 0
        self.lock = threading.Lock()
        self.last_update = time.time()
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Process every frame for better responsiveness
            processed_img, detections, results = process_frame_with_pose(img)
            
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
                if current_time - self.last_update > 2.0:  # Update every 2 seconds
                    st.session_state.stats.update({
                        'total_frames': self.frame_count,
                        'total_detections': self.frame_count,
                        'good_posture': self.good_posture,
                        'bad_posture': self.bad_posture
                    })
                    self.last_update = current_time
            
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Processor error: {e}")
            return frame

# Get optimal constraints based on device and quality
def get_media_constraints():
    if "Low" in quality_preset:
        width, height, fps = 320, 240, 15
    elif "High" in quality_preset:
        width, height, fps = 640, 480, 25
    else:  # Medium
        width, height, fps = 480, 360, 20
    
    constraints = {
        "video": {
            "width": width,
            "height": height,
            "frameRate": fps
        },
        "audio": False
    }
    
    # Add facing mode for mobile
    if device_info["is_mobile"]:
        constraints["video"]["facingMode"] = "user"
    
    return constraints

# Main interface
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📹 Live Camera", "📷 Upload Image", "🎬 Upload Video"])

# Tab 1: Fixed WebRTC Implementation
with tab1:
    st.markdown("### 📹 Fixed WebRTC Real-time Detection")
    
    # Instructions based on device
    if device_info["is_mobile"]:
        st.markdown("""
        <div class="info-box">
            <h4>📱 Mobile Instructions</h4>
            <p><strong>1.</strong> Click "START" button below</p>
            <p><strong>2.</strong> Allow camera access when prompted</p>
            <p><strong>3.</strong> Wait for video to appear (may take 5-10 seconds)</p>
            <p><strong>4.</strong> If video doesn't appear, try refreshing page</p>
            <p><strong>5.</strong> Use landscape mode for better detection</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>💻 Desktop Instructions</h4>
            <p><strong>1.</strong> Click "START" button below</p>
            <p><strong>2.</strong> Allow camera access in browser popup</p>
            <p><strong>3.</strong> Camera will start automatically - no device selection needed</p>
            <p><strong>4.</strong> Position yourself 1-2 meters from camera</p>
            <p><strong>5.</strong> Ensure good lighting for best results</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show current settings
    constraints = get_media_constraints()
    st.markdown(f"""
    <div class="success-box">
        <h4>⚙️ Current Settings</h4>
        <p><strong>Quality:</strong> {quality_preset}</p>
        <p><strong>Resolution:</strong> {constraints['video']['width']}x{constraints['video']['height']}</p>
        <p><strong>Frame Rate:</strong> {constraints['video']['frameRate']} FPS</p>
        <p><strong>Device:</strong> {device_info['type'].title()} ({device_info['browser']})</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Control buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 🎥 Camera Controls")
    with col2:
        if st.button("🔄 Reset Stats", use_container_width=True):
            st.session_state.stats = {
                'total_frames': 0,
                'total_detections': 0,
                'good_posture': 0,
                'bad_posture': 0,
                'session_start': time.time()
            }
            st.success("✅ Statistics reset!")
    
    # WebRTC Streamer with improved configuration
    try:
        # Use a unique key to avoid caching issues
        webrtc_key = "stable-pose-detector-v3"
        
        webrtc_ctx = webrtc_streamer(
            key=webrtc_key,
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=StablePoseProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=constraints,
            async_processing=True,
            video_html_attrs={
                "style": {
                    "width": "100%", 
                    "height": "auto",
                    "border-radius": "15px",
                    "box-shadow": "0 8px 25px rgba(0,0,0,0.1)"
                },
                "controls": False,
                "autoPlay": True,
                "muted": True,
            },
            sendback_audio=False
        )
        
        # Connection status indicator
        if webrtc_ctx.state.playing:
            st.markdown("""
            <div style="text-align: center; margin: 15px 0;">
                <span class="status-dot status-online"></span>
                <strong style="color: #00b894;">🟢 Camera Active - AI Processing Live Feed</strong>
            </div>
            """, unsafe_allow_html=True)
        elif webrtc_ctx.state.signalling:
            st.markdown("""
            <div style="text-align: center; margin: 15px 0;">
                <span class="status-dot status-connecting"></span>
                <strong style="color: #fdcb6e;">🟡 Connecting to Camera...</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; margin: 15px 0;">
                <span class="status-dot status-offline"></span>
                <strong style="color: #fd79a8;">🔴 Camera Disconnected</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time statistics - only show when active
        if webrtc_ctx.video_processor and webrtc_ctx.state.playing:
            st.markdown("### 📊 Live Statistics")
            
            processor = webrtc_ctx.video_processor
            
            if device_info["is_mobile"]:
                # Mobile: 2x2 grid
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{processor.frame_count:,}</h3>
                        <p>🎞️ Frames</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{processor.good_posture:,}</h3>
                        <p>✅ Good</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{processor.detection_count}</h3>
                        <p>🎯 Current</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{processor.bad_posture:,}</h3>
                        <p>❌ Bad</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # Desktop: 4 columns
                cols = st.columns(4)
                
                with cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{processor.frame_count:,}</h3>
                        <p>🎞️ Total Frames</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{processor.detection_count}</h3>
                        <p>🎯 Current Detections</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{processor.good_posture:,}</h3>
                        <p>✅ Good Posture</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{processor.bad_posture:,}</h3>
                        <p>❌ Bad Posture</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Session analysis
            total_postures = processor.good_posture + processor.bad_posture
            if total_postures > 0:
                good_percentage = (processor.good_posture / total_postures) * 100
                
                st.markdown("#### 📈 Session Analysis")
                st.progress(good_percentage / 100)
                
                # Status based on percentage
                if good_percentage >= 80:
                    status_box = "success-box"
                    status_icon = "🟢"
                    status_text = "Excellent Posture!"
                elif good_percentage >= 60:
                    status_box = "info-box"
                    status_icon = "🟡"
                    status_text = "Good Posture"
                else:
                    status_box = "warning-box"
                    status_icon = "🔴"
                    status_text = "Needs Improvement"
                
                st.markdown(f"""
                <div class="{status_box}">
                    <h4>{status_icon} {status_text}</h4>
                    <p><strong>Posture Quality:</strong> {good_percentage:.1f}%</p>
                    <p><strong>Session Duration:</strong> {(time.time() - st.session_state.stats.get('session_start', time.time())) / 60:.1f} minutes</p>
                    <p><strong>Total Detections:</strong> {total_postures:,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Live recommendations
                st.markdown("#### 💡 Live Recommendations")
                if processor.detection_count > 0:
                    if good_percentage < 60:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Posture Alert</h4>
                            <p><strong>Immediate Actions:</strong></p>
                            <p>• Sit up straight and align your spine</p>
                            <p>• Pull shoulders back and down</p>
                            <p>• Keep feet flat on the floor</p>
                            <p>• Take a deep breath and reset position</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-box">
                            <h4>✅ Excellent Work!</h4>
                            <p>You're maintaining great posture! Keep it up.</p>
                            <p>Remember to take movement breaks every 30 minutes.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-box">
                        <h4>👀 Position Yourself</h4>
                        <p>Move into the camera view for pose detection.</p>
                        <p>Ensure your upper body is clearly visible.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        elif webrtc_ctx.state.signalling:
            st.markdown("""
            <div class="warning-box" style="text-align: center;">
                <h4>🔄 Connecting to Camera...</h4>
                <p>Please wait while we establish connection.</p>
                <p>This may take a few seconds.</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <h3>📹 Ready to Start</h3>
                <p>Click the <strong>START</strong> button above to begin camera detection.</p>
                <p>Make sure to allow camera permissions when prompted.</p>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ WebRTC Connection Error</h4>
            <p><strong>Error:</strong> {str(e)}</p>
            <h5>🔧 Troubleshooting Steps:</h5>
            <p><strong>1. Browser Issues:</strong></p>
            <p>• Use Chrome or Firefox (recommended)</p>
            <p>• Update browser to latest version</p>
            <p>• Try incognito/private mode</p>
            <br>
            <p><strong>2. Permissions:</strong></p>
            <p>• Allow camera access when prompted</p>
            <p>• Check browser camera settings</p>
            <p>• Refresh page after allowing permissions</p>
            <br>
            <p><strong>3. Camera Access:</strong></p>
            <p>• Close other apps using camera</p>
            <p>• Try different quality preset (Low/Medium/High)</p>
            <p>• Restart browser if needed</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Alternative solutions
        st.markdown("#### 🔄 Alternative Solutions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Try Lower Quality", type="secondary", use_container_width=True):
                st.info("Switch to 'Low (320p)' in the sidebar and try again.")
        with col2:
            if st.button("📱 Mobile Browser Tips", type="secondary", use_container_width=True):
                st.info("For mobile: Use Chrome mobile, allow camera access, try landscape mode.")

# Tab 2: Image Upload (simplified and stable)
with tab2:
    st.markdown("### 📷 Image Pose Analysis")
    
    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image containing people for pose analysis"
    )
    
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            
            # Image information
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 Image Information</h4>
                <p><strong>Dimensions:</strong> {image.size[0]} x {image.size[1]} pixels</p>
                <p><strong>Format:</strong> {image.format}</p>
                <p><strong>Mode:</strong> {image.mode}</p>
                <p><strong>File Size:</strong> {uploaded_image.size / 1024:.1f} KB</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔍 Analyze Pose", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing the pose..."):
                    # Convert PIL to OpenCV format
                    image_array = np.array(image)
                    if len(image_array.shape) == 3:
                        frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    else:
                        frame = image_array
                    
                    # Process with pose detection
                    processed_frame, detection_count, pose_results = process_frame_with_pose(frame)
                    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display results side by side
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
                        <h4>🎉 Analysis Completed Successfully!</h4>
                        <p><strong>Poses Detected:</strong> {detection_count}</p>
                        <p><strong>Processing Status:</strong> Complete</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed results
                    with st.expander("📋 Detailed Analysis Results"):
                        for i, result in enumerate(pose_results, 1):
                            posture_icon = "✅" if result['label'] == 'Postur Baik' else "❌"
                            confidence_color = "green" if result['confidence'] > 0.7 else "orange" if result['confidence'] > 0.5 else "red"
                            
                            st.markdown(f"""
                            **{posture_icon} Person {i}:**
                            - **Classification:** {result['label']}
                            - **Confidence:** <span style="color: {confidence_color};">{result['confidence']:.2%}</span>
                            - **Quality:** {"High" if result['confidence'] > 0.7 else "Medium" if result['confidence'] > 0.5 else "Low"}
                            """, unsafe_allow_html=True)
                            st.markdown("---")
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ No Poses Detected</h4>
                        <p><strong>Possible reasons:</strong></p>
                        <p>• Person not clearly visible in image</p>
                        <p>• Image quality too low</p>
                        <p>• Pose is partially obscured</p>
                        <p>• Confidence threshold too high</p>
                        <br>
                        <p><strong>Try:</strong> Adjust confidence threshold in sidebar or use a clearer image.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ Image Processing Error</h4>
                <p><strong>Error:</strong> {str(e)}</p>
                <p>Please try with a different image or check the file format.</p>
            </div>
            """, unsafe_allow_html=True)

# Tab 3: Video Upload (enhanced with better progress tracking)
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
            <p><strong>Filename:</strong> {uploaded_video.name}</p>
            <p><strong>File Size:</strong> {file_size_mb:.2f} MB</p>
            <p><strong>Type:</strong> {uploaded_video.type}</p>
            <p><strong>Estimated Processing Time:</strong> ~{file_size_mb * 3:.0f} seconds</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            frame_skip = st.selectbox(
                "Frame Processing", 
                [1, 2, 3, 5, 10], 
                index=2,
                format_func=lambda x: f"Process every {x} frame{'s' if x > 1 else ''}",
                help="Skip frames to speed up processing"
            )
        with col2:
            show_preview = st.checkbox("Show Processing Preview", value=True, 
                help="Display video frames during processing")
        
        if st.button("🚀 Start Video Processing", type="primary", use_container_width=True):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                temp_video_path = tfile.name
            
            try:
                # Open video
                cap = cv2.VideoCapture(temp_video_path)
                
                if not cap.isOpened():
                    st.error("❌ Cannot open video file. Please try a different format.")
                else:
                    # Get video properties
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = total_frames / fps if fps > 0 else 0
                    
                    # Display video properties
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("📊 FPS", fps)
                    with cols[1]:
                        st.metric("🎞️ Total Frames", f"{total_frames:,}")
                    with cols[2]:
                        st.metric("⏱️ Duration", f"{duration:.1f}s")
                    with cols[3]:
                        st.metric("📐 Resolution", f"{width}x{height}")
                    
                    # Processing placeholders
                    if show_preview:
                        video_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    stats_container = st.container()
                    
                    # Processing statistics
                    frame_count = 0
                    processed_frames = 0
                    total_detections = 0
                    good_posture_count = 0
                    bad_posture_count = 0
                    
                    # Start processing
                    start_time = time.time()
                    
                    with stats_container:
                        stats_placeholder = st.empty()
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process frame based on skip setting
                        if frame_count % frame_skip == 0:
                            processed_frame, detection_count, pose_results = process_frame_with_pose(frame)
                            processed_frames += 1
                            
                            # Update statistics
                            total_detections += detection_count
                            for result in pose_results:
                                if result['label'] == 'Postur Baik':
                                    good_posture_count += 1
                                else:
                                    bad_posture_count += 1
                            
                            # Show preview every 5th processed frame
                            if show_preview and processed_frames % 5 == 0:
                                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                video_placeholder.image(frame_rgb, use_container_width=True)
                        
                        # Update progress
                        frame_count += 1
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        
                        # Update status
                        elapsed_time = time.time() - start_time
                        processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        eta_seconds = (total_frames - frame_count) / processing_fps if processing_fps > 0 else 0
                        eta_minutes = eta_seconds / 60
                        
                        status_text.markdown(f"""
                        **⚡ Processing Progress:** {frame_count:,}/{total_frames:,} frames ({progress*100:.1f}%) | 
                        **🔥 Speed:** {processing_fps:.1f} FPS | 
                        **⏰ ETA:** {eta_minutes:.1f}m | 
                        **🎯 Detections:** {total_detections}
                        """)
                        
                        # Update live statistics every 100 frames
                        if frame_count % 100 == 0:
                            with stats_placeholder.container():
                                current_accuracy = (good_posture_count / (good_posture_count + bad_posture_count)) * 100 if (good_posture_count + bad_posture_count) > 0 else 0
                                
                                stat_cols = st.columns(4)
                                with stat_cols[0]:
                                    st.metric("🎯 Detections", total_detections)
                                with stat_cols[1]:
                                    st.metric("✅ Good", good_posture_count)
                                with stat_cols[2]:
                                    st.metric("❌ Bad", bad_posture_count)
                                with stat_cols[3]:
                                    st.metric("📊 Quality", f"{current_accuracy:.1f}%")
                    
                    cap.release()
                    
                    # Final results
                    processing_time = time.time() - start_time
                    
                    st.markdown("""
                    <div class="success-box">
                        <h3>🎉 Video Processing Completed!</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Final statistics
                    final_accuracy = (good_posture_count / (good_posture_count + bad_posture_count)) * 100 if (good_posture_count + bad_posture_count) > 0 else 0
                    
                    cols = st.columns(5)
                    with cols[0]:
                        st.metric("🎯 Total Detections", total_detections)
                    with cols[1]:
                        st.metric("✅ Good Posture", good_posture_count)
                    with cols[2]:
                        st.metric("❌ Bad Posture", bad_posture_count)
                    with cols[3]:
                        st.metric("📈 Posture Quality", f"{final_accuracy:.1f}%")
                    with cols[4]:
                        st.metric("⏱️ Processing Time", f"{processing_time:.1f}s")
                    
                    # Quality assessment
                    if final_accuracy >= 80:
                        st.markdown("""
                        <div class="success-box">
                            <h4>🌟 Excellent Posture Throughout Video!</h4>
                            <p>The subject maintained very good posture for most of the video duration.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif final_accuracy >= 60:
                        st.markdown("""
                        <div class="info-box">
                            <h4>👍 Good Overall Posture</h4>
                            <p>Generally good posture with some room for improvement.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Posture Needs Attention</h4>
                            <p>The analysis shows significant posture issues that should be addressed.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Video Processing Error</h4>
                    <p><strong>Error:</strong> {str(e)}</p>
                    <p>Please try with a different video file or check the format compatibility.</p>
                </div>
                """, unsafe_allow_html=True)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)

# Footer with comprehensive system information
st.markdown("---")
st.markdown("### 📊 System Status & Information")

# System status cards
cols = st.columns(5)

with cols[0]:
    model_status = "✅ Ready" if st.session_state.model_loaded else "❌ Failed"
    st.markdown(f"""
    <div class="metric-card">
        <h4>🤖 AI Model</h4>
        <p>{model_status}</p>
        <small>YOLO v8</small>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <h4>📱 Device</h4>
        <p>{device_info["type"].title()}</p>
        <small>{device_info["browser"]}</small>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <h4>⚙️ Quality</h4>
        <p>{quality_preset.split()[0]}</p>
        <small>Current Setting</small>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    session_duration = (time.time() - st.session_state.stats.get('session_start', time.time())) / 60
    st.markdown(f"""
    <div class="metric-card">
        <h4>⏱️ Session</h4>
        <p>{session_duration:.1f}m</p>
        <small>Duration</small>
    </div>
    """, unsafe_allow_html=True)

with cols[4]:
    total_session_detections = st.session_state.stats['good_posture'] + st.session_state.stats['bad_posture']
    session_accuracy = (st.session_state.stats['good_posture'] / total_session_detections * 100) if total_session_detections > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <h4>📊 Accuracy</h4>
        <p>{session_accuracy:.0f}%</p>
        <small>Session Avg</small>
    </div>
    """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("""
<div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-top: 2rem;">
    <h3>🤸‍♂️ AI Pose Detection System - WebRTC Fixed</h3>
    <p><strong>🚀 Technology Stack:</strong> YOLO v8 Neural Network • OpenCV Computer Vision • WebRTC Real-time • Streamlit Framework</p>
    <p><strong>✨ Key Features:</strong> Real-time Analysis • Cross-platform • Responsive Design • Advanced Error Handling</p>
    <p><strong>🌐 Compatibility:</strong> 💻 Desktop (Chrome, Firefox) • 📱 Mobile (Chrome, Safari) • 📟 Tablet Devices</p>
    <br>
    <p><strong>🔧 Version:</strong> WebRTC Stabilized v3.0 - Connection Issues Resolved</p>
    <p><em>Professional posture analysis with reliable real-time processing</em></p>
</div>
""", unsafe_allow_html=True)
