# Perbaikan untuk WebRTC configuration

# 1. Tambahkan debugging dan error handling
def debug_webrtc():
    st.write("Debug Info:")
    st.write(f"User Agent: {st.session_state.get('user_agent', 'Unknown')}")
    st.write("Pastikan Anda menggunakan Chrome/Firefox terbaru")
    st.write("Pastikan mengakses via HTTPS atau localhost")

# 2. Konfigurasi RTC yang lebih robust
RTC_CONFIGURATION_IMPROVED = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
    ],
    "iceCandidatePoolSize": 10,
})

# 3. Media constraints yang lebih fleksibel
MEDIA_CONSTRAINTS_FLEXIBLE = {
    "video": {
        "width": {"min": 320, "ideal": 640, "max": 1280},
        "height": {"min": 240, "ideal": 480, "max": 720},
        "frameRate": {"min": 15, "ideal": 30, "max": 60}
    },
    "audio": False
}

# 4. Versi WebRTC streamer yang diperbaiki
with tab2:
    st.subheader("Deteksi Pose Webcam Real-time")
    
    # Debug information
    debug_webrtc()
    
    # Instructions dengan troubleshooting
    st.markdown("""
    <div class="info-box">
        <strong>Petunjuk Webcam WebRTC:</strong><br>
        1. Pastikan menggunakan Chrome/Firefox terbaru<br>
        2. Izinkan akses kamera ketika diminta<br>
        3. Jika tidak muncul, refresh halaman dan coba lagi<br>
        4. Periksa permission kamera di browser settings<br>
        5. Pastikan tidak ada aplikasi lain yang menggunakan kamera
    </div>
    """, unsafe_allow_html=True)
    
    # Troubleshooting section
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Jika kamera tidak muncul:**
        1. **Periksa URL**: Pastikan menggunakan `https://` atau `localhost`
        2. **Browser Permission**: Klik ikon kunci/kamera di address bar
        3. **Refresh**: Tekan F5 atau Ctrl+R untuk refresh halaman
        4. **Browser Compatibility**: Gunakan Chrome versi 88+ atau Firefox 85+
        5. **Antivirus**: Nonaktifkan sementara antivirus/firewall
        6. **Device Manager**: Pastikan driver kamera ter-install dengan benar
        """)
        
        if st.button("Test Kamera Browser"):
            st.markdown("""
            <div class="warning-box">
                Buka Console Browser (F12) dan jalankan:<br>
                <code>navigator.mediaDevices.getUserMedia({video: true})</code><br>
                Jika ada error, itulah penyebab masalah kamera.
            </div>
            """, unsafe_allow_html=True)
    
    # WebRTC Streamer dengan konfigurasi yang diperbaiki
    try:
        webrtc_ctx = webrtc_streamer(
            key="pose-detection-improved",
            video_transformer_factory=PoseDetectionTransformer,
            rtc_configuration=RTC_CONFIGURATION_IMPROVED,
            media_stream_constraints=MEDIA_CONSTRAINTS_FLEXIBLE,
            async_processing=False,  # Ubah ke False untuk stabilitas
            video_html_attrs={
                "style": {"width": "100%", "margin": "0 auto", "border": "2px solid #4ECDC4"},
                "controls": False,
                "autoPlay": True,
            },
        )
        
        # Status connection
        if webrtc_ctx.state.playing:
            st.success("✅ Kamera berhasil terhubung!")
        elif webrtc_ctx.state.signalling:
            st.info("🔄 Menghubungkan ke kamera...")
        else:
            st.warning("⚠️ Kamera belum terhubung. Klik START untuk memulai.")
            
    except Exception as e:
        st.error(f"Error WebRTC: {str(e)}")
        st.markdown("""
        <div class="warning-box">
            <strong>Jika terus bermasalah, coba alternative:</strong><br>
            1. Gunakan tab "Upload Gambar" sebagai alternative<br>
            2. Gunakan smartphone dengan browser Chrome<br>
            3. Update browser ke versi terbaru<br>
            4. Coba akses via IP address local (192.168.x.x)
        </div>
        """, unsafe_allow_html=True)

# 5. Alternative webcam using OpenCV (fallback)
def create_opencv_webcam_alternative():
    st.subheader("Alternative: OpenCV Webcam")
    st.markdown("""
    <div class="info-box">
        Jika WebRTC tidak bekerja, Anda bisa menggunakan alternative ini
        dengan menjalankan script terpisah menggunakan OpenCV.
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Script OpenCV Alternative"):
        opencv_code = '''
import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np

def run_opencv_webcam():
    # Load model
    model = YOLO("pose2/train2/weights/best.pt")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 untuk webcam default
    
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses webcam")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process with YOLO
        results = model.predict(frame, conf=0.5)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Display
        cv2.imshow('Pose Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_opencv_webcam()
        '''
        st.code(opencv_code, language='python')

# 6. Sistem requirement check
def check_system_requirements():
    st.markdown("### 🔍 System Requirements Check")
    
    requirements = {
        "Browser": "Chrome 88+ atau Firefox 85+",
        "Protocol": "HTTPS atau localhost",
        "Permissions": "Camera access allowed",
        "Network": "Stable internet connection",
        "Hardware": "Working webcam device"
    }
    
    for req, desc in requirements.items():
        st.write(f"✓ **{req}**: {desc}")

# Tambahkan di sidebar
with st.sidebar:
    st.markdown("---")
    if st.button("🔧 Check System"):
        check_system_requirements()
