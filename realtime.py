elif webrtc_ctx.state.signalling:
            st.markdown("""
            <div style="text-align: center; margin: 15px 0;">
                <span class="status-indicator status-connecting"></span>
                <strong style="color: #fdcb6e; font-size: 1.1em;">🟡 Connecting to Camera...</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; margin: 15px 0;">
                <span class="status-indicator status-offline"></span>
                <strong style="color: #fd79a8; font-size: 1.1em;">🔴 Camera Disconnected</strong>
                <br><small>Click START above to begin</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time statistics display
        if webrtc_ctx.video_transformer and webrtc_ctx.state.playing:
            st.markdown("### 📊 Live Statistics")
            
            transformer = webrtc_ctx.video_transformer
            
            # Update session stats from queue
            try:
                while not transformer.result_queue.empty():
                    stats_update = transformer.result_queue.get_nowait()
                    st.session_state.stats.update(stats_update)
            except queue.Empty:
                pass
            
            # Display metrics based on device type
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
                        <h3>{transformer.good_posture_count:,}</h3>
                        <p>✅ Good</p>
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
                        <h3>{transformer.bad_posture_count:,}</h3>
                        <p>❌ Bad</p>
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
                        <p>🎯 Live Detections</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{transformer.good_posture_count:,}</h3>
                        <p>✅ Good Posture</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with cols[3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{transformer.bad_posture_count:,}</h3>
                        <p>❌ Bad Posture</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Session analysis
            total_postures = transformer.good_posture_count + transformer.bad_posture_count
            if total_postures > 10:  # Show analysis after some data
                good_percentage = (transformer.good_posture_count / total_postures) * 100
                
                st.markdown("#### 📈 Session Analysis")
                
                # Progress bar with custom styling
                progress_color = "#00b894" if good_percentage >= 80 else "#fdcb6e" if good_percentage >= 60 else "#fd79a8"
                st.markdown(f"""
                <div style="background-color: #f0f2f6; border-radius: 10px; padding: 5px; margin: 10px 0;">
                    <div style="background-color: {progress_color}; width: {good_percentage}%; height: 20px; border-radius: 5px; transition: width 0.3s ease;"></div>
                </div>
                <p style="text-align: center; margin: 5px 0; font-weight: bold;">Posture Quality: {good_percentage:.1f}%</p>
                """, unsafe_allow_html=True)
                
                # Status assessment
                if good_percentage >= 80:
                    status_box = "success-box"
                    status_icon = "🌟"
                    status_text = "Excellent Posture!"
                    advice = "Outstanding work! You're maintaining excellent posture consistently."
                elif good_percentage >= 60:
                    status_box = "info-box"
                    status_icon = "👍"
                    status_text = "Good Posture"
                    advice = "Good job! Try to maintain this consistency and aim for even better."
                else:
                    status_box = "warning-box"
                    status_icon = "⚠️"
                    status_text = "Needs Improvement"
                    advice = "Focus on sitting up straighter and aligning your shoulders."
                
                session_duration = (time.time() - st.session_state.stats.get('session_start', time.time())) / 60
                
                st.markdown(f"""
                <div class="{status_box}">
                    <h4>{status_icon} {status_text}</h4>
                    <p><strong>Quality Score:</strong> {good_percentage:.1f}%</p>
                    <p><strong>Session Time:</strong> {session_duration:.1f} minutes</p>
                    <p><strong>Total Analyses:</strong> {total_postures:,}</p>
                    <p><strong>Advice:</strong> {advice}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Real-time recommendations
                st.markdown("#### 💡 Live Coaching")
                if transformer.detection_count > 0:
                    recent_ratio = transformer.bad_posture_count / max(transformer.good_posture_count, 1)
                    if recent_ratio > 1.5:  # More bad than good recently
                        st.markdown("""
                        <div class="warning-box">
                            <h4>📢 Posture Alert!</h4>
                            <p><strong>Immediate Actions:</strong></p>
                            <p>• Roll shoulders back and down</p>
                            <p>• Straighten your spine</p>
                            <p>• Adjust screen to eye level</p>
                            <p>• Take 3 deep breaths</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif good_percentage > 75:
                        st.markdown("""
                        <div class="success-box">
                            <h4>🎉 Keep It Up!</h4>
                            <p>Excellent posture detected! You're doing great.</p>
                            <p>Remember to take movement breaks every 30 minutes.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="info-box">
                            <h4>💪 Stay Focused</h4>
                            <p>You're on the right track! Keep working on maintaining good posture.</p>
                            <p>Small adjustments make a big difference over time.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-box">
                        <h4>👀 Position Check</h4>
                        <p>Move into the camera view for pose analysis.</p>
                        <p>Ensure your upper body is clearly visible and well-lit.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance metrics
                if transformer.frame_count > 300:  # After 300 frames
                    with st.expander("📊 Performance Metrics"):
                        avg_fps = transformer.frame_count / (time.time() - st.session_state.stats.get('session_start', time.time()))
                        detection_rate = (total_postures / transformer.frame_count * 100) if transformer.frame_count > 0 else 0
                        
                        perf_col1, perf_col2 = st.columns(2)
                        with perf_col1:
                            st.metric("⚡ Average FPS", f"{avg_fps:.1f}")
                            st.metric("🎯 Detection Rate", f"{detection_rate:.1f}%")
                        with perf_col2:
                            st.metric("📱 Device Type", device_info["type"].title())
                            st.metric("🌐 Browser", device_info["browser"])
        
        # Connection troubleshooting
        elif webrtc_ctx.state.signalling:
            st.markdown("""
            <div class="warning-box" style="text-align: center;">
                <h4>🔄 Establishing Connection...</h4>
                <p>Please wait while we connect to your camera.</p>
                <p>This usually takes 5-15 seconds.</p>
                <br>
                <p><strong>If connection takes too long:</strong></p>
                <p>• Check camera permissions</p>
                <p>• Try refreshing the page</p>
                <p>• Ensure stable internet connection</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # Not started yet
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h3>🚀 Ready to Start</h3>
                <p style="font-size: 1.1em;">Click the <strong>START</strong> button above to begin real-time pose detection.</p>
                <p>Make sure to allow camera permissions when prompted by your browser.</p>
                <br>
                <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 20px;">
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; min-width: 200px;">
                        <h4>🎯 What You'll Get</h4>
                        <p>• Real-time posture analysis</p>
                        <p>• Live feedback and coaching</p>
                        <p>• Detailed session statistics</p>
                        <p>• Performance tracking</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ WebRTC Setup Error</h4>
            <p><strong>Error Details:</strong> {str(e)}</p>
            
            <h5>🔧 Troubleshooting Guide:</h5>
            
            <p><strong>🌐 Browser Compatibility:</strong></p>
            <p>• <strong>Recommended:</strong> Chrome, Firefox, Safari, Edge</p>
            <p>• <strong>Current:</strong> {device_info["browser"]}</p>
            <p>• Update to the latest browser version</p>
            
            <p><strong>📹 Camera Permissions:</strong></p>
            <p>• Allow camera access when prompted</p>
            <p>• Check browser settings: chrome://settings/content/camera</p>
            <p>• Refresh page after granting permissions</p>
            
            <p><strong>🔌 Connection Issues:</strong></p>
            <p>• Ensure stable internet connection</p>
            <p>• Close other apps using camera</p>
            <p>• Try incognito/private browsing mode</p>
            <p>• Restart browser if problems persist</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick fix buttons
        st.markdown("#### 🚀 Quick Solutions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Retry Connection", type="primary", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("📱 Switch to Image Mode", type="secondary", use_container_width=True):
                st.info("Use the 'Image Analysis' tab as an alternative.")
        
        with col3:
            if st.button("💡 Show Browser Tips", type="secondary", use_container_width=True):
                st.info(f"For {device_info['browser']}: Enable camera permissions in browser settings and refresh the page.")

# Tab 2: Image Analysis
with tab2:
    st.markdown("### 📷 Static Image Pose Analysis")
    
    uploaded_image = st.file_uploader(
        "Upload an image for pose analysis",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
        help="Supported formats: JPG, PNG, BMP, TIFF, WebP"
    )
    
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            
            # Image metadata
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 Image Information</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                    <div><strong>Dimensions:</strong> {image.size[0]} × {image.size[1]} pixels</div>
                    <div><strong>Format:</strong> {image.format}</div>
                    <div><strong>Mode:</strong> {image.mode}</div>
                    <div><strong>File Size:</strong> {uploaded_image.size / 1024:.1f} KB</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Analysis button
            if st.button("🔍 Analyze Pose", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing the pose... This may take a few seconds."):
                    # Convert PIL to OpenCV format
                    image_array = np.array(image)
                    if len(image_array.shape) == 3:
                        frame = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    else:
                        frame = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
                    
                    # Process with pose detection
                    settings = st.session_state.detection_settings
                    processed_frame, detection_count, detections = process_frame_pose_detection(frame, settings)
                    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display results
                if device_info["is_mobile"]:
                    # Mobile: vertical layout
                    st.markdown("#### 🖼️ Original Image")
                    st.image(image, use_container_width=True)
                    
                    st.markdown("#### 🎯 Analysis Result")
                    st.image(processed_rgb, use_container_width=True)
                else:
                    # Desktop: side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🖼️ Original Image")
                        st.image(image, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 🎯 Analysis Result")
                        st.image(processed_rgb, use_container_width=True)
                
                # Results analysis
                if detection_count > 0:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>🎉 Analysis Completed Successfully!</h4>
                        <p><strong>Poses Detected:</strong> {detection_count}</p>
                        <p><strong>Processing Status:</strong> Complete</p>
                        <p><strong>Analysis Time:</strong> ~2-3 seconds</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed breakdown
                    with st.expander("📋 Detailed Analysis Results", expanded=True):
                        for i, detection in enumerate(detections, 1):
                            posture_icon = "✅" if detection.label == 'Postur Baik' else "❌"
                            confidence_level = "High" if detection.confidence > 0.75 else "Medium" if detection.confidence > 0.5 else "Low"
                            confidence_color = "#00b894" if detection.confidence > 0.75 else "#fdcb6e" if detection.confidence > 0.5 else "#fd79a8"
                            
                            st.markdown(f"""
                            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid {confidence_color};">
                                <h5>{posture_icon} Person {i}</h5>
                                <p><strong>Classification:</strong> {detection.label}</p>
                                <p><strong>Confidence:</strong> <span style="color: {confidence_color}; font-weight: bold;">{detection.confidence:.2%}</span> ({confidence_level})</p>
                                <p><strong>Bounding Box:</strong> [{', '.join([f'{x:.0f}' for x in detection.bbox])}]</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Summary statistics
                    good_count = sum(1 for d in detections if d.label == 'Postur Baik')
                    bad_count = detection_count - good_count
                    avg_confidence = sum(d.confidence for d in detections) / detection_count
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("🎯 Total Detected", detection_count)
                    with cols[1]:
                        st.metric("✅ Good Posture", good_count)
                    with cols[2]:
                        st.metric("❌ Bad Posture", bad_count)
                    with cols[3]:
                        st.metric("📊 Avg Confidence", f"{avg_confidence:.1%}")
                    
                    # Overall assessment
                    if good_count > bad_count:
                        st.balloons()
                        st.markdown("""
                        <div class="success-box">
                            <h4>🌟 Excellent Results!</h4>
                            <p>The majority of detected poses show good posture. Keep up the great work!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif good_count == bad_count:
                        st.markdown("""
                        <div class="info-box">
                            <h4>⚖️ Mixed Results</h4>
                            <p>Equal amounts of good and bad posture detected. There's room for improvement!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Posture Attention Needed</h4>
                            <p>Most detected poses show poor posture. Consider focusing on posture improvement.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>😔 No Poses Detected</h4>
                        <p><strong>Possible reasons:</strong></p>
                        <ul>
                            <li>Person not clearly visible in the image</li>
                            <li>Image resolution too low</li>
                            <li>Pose is partially obscured or unclear</li>
                            <li>Confidence threshold set too high</li>
                            <li>Lighting conditions are poor</li>
                        </ul>
                        <br>
                        <p><strong>💡 Suggestions:</strong></p>
                        <ul>
                            <li>Try lowering the confidence threshold in the sidebar</li>
                            <li>Use an image with better lighting</li>
                            <li>Ensure the person's upper body is clearly visible</li>
                            <li>Use a higher resolution image</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <h4>❌ Image Processing Error</h4>
                <p><strong>Error Details:</strong> {str(e)}</p>
                <p><strong>Troubleshooting:</strong></p>
                <ul>
                    <li>Ensure the image file is not corrupted</li>
                    <li>Try a different image format (JPG recommended)</li>
                    <li>Check if the image file size is reasonable (&lt; 10MB)</li>
                    <li>Refresh the page and try again</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Tab 3: Video Analysis
with tab3:
    st.markdown("### 🎬 Video Pose Analysis")
    
    uploaded_video = st.file_uploader(
        "Upload a video for batch pose analysis",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'],
        help="Supported formats: MP4, AVI, MOV, MKV, WMV, FLV"
    )
    
    if uploaded_video is not None:
        file_size_mb = uploaded_video.size / (1024*1024)
        
        st.markdown(f"""
        <div class="info-box">
            <h4>🎬 Video Information</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div><strong>Filename:</strong> {uploaded_video.name}</div>
                <div><strong>File Size:</strong> {file_size_mb:.2f} MB</div>
                <div><strong>Type:</strong> {uploaded_video.type}</div>
                <div><strong>Est. Processing:</strong> ~{file_size_mb * 2:.0f} seconds</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Processing configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            frame_skip = st.selectbox(
                "Frame Processing Interval", 
                [1, 2, 3, 5, 10, 15, 30], 
                index=3,
                format_func=lambda x: f"Every {x} frame{'s' if x > 1 else ''}",
                help="Process every Nth frame to balance speed vs accuracy"
            )
        
        with col2:
            max_frames = st.selectbox(
                "Maximum Frames to Process",
                [100, 300, 500, 1000, -1],
                index=2,
                format_func=lambda x: "All frames" if x == -1 else f"{x} frames",
                help="Limit processing for faster results"
            )
        
        with col3:
            show_preview = st.checkbox(
                "Show Processing Preview", 
                value=True, 
                help="Display frames during processing (may slow down processing)"
            )
        
        # Processing button
        if st.button("🚀 Start Video Analysis", type="primary", use_container_width=True):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                temp_video_path = tfile.name
            
            try:
                # Open video file
                cap = cv2.VideoCapture(temp_video_path)
                
                if not cap.isOpened():
                    st.error("❌ Cannot open video file. Please try a different format or file.")
                else:
                    # Get video properties
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = total_frames / fps if fps > 0 else 0
                    
                    # Limit frames if specified
                    frames_to_process = min(total_frames, max_frames) if max_frames > 0 else total_frames
                    
                    # Display video properties
                    st.markdown("#### 📊 Video Properties")
                    cols = st.columns(5)
                    with cols[0]:
                        st.metric("📊 FPS", fps)
                    with cols[1]:
                        st.metric("🎞️ Total Frames", f"{total_frames:,}")
                    with cols[2]:
                        st.metric("⏱️ Duration", f"{duration:.1f}s")
                    with cols[3]:
                        st.metric("📐 Resolution", f"{width}×{height}")
                    with cols[4]:
                        st.metric("🎯 Processing", f"{frames_to_process:,}")
                    
                    # Create processing interface
                    if show_preview:
                        video_placeholder = st.empty()
                    
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        live_stats = st.empty()
                    
                    # Processing variables
                    frame_count = 0
                    processed_frames = 0
                    total_detections = 0
                    good_posture_count = 0
                    bad_posture_count = 0
                    all_confidences = []
                    
                    # Start processing
                    start_time = time.time()
                    settings = st.session_state.detection_settings
                    
                    st.markdown("#### 🔄 Processing Status")
                    
                    while cap.isOpened() and frame_count < frames_to_process:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process frame based on skip interval
                        if frame_count % frame_skip == 0:
                            processed_frame, detection_count, detections = process_frame_pose_detection(frame, settings)
                            processed_frames += 1
                            
                            # Update statistics
                            total_detections += detection_count
                            for detection in detections:
                                all_confidences.append(detection.confidence)
                                if detection.label == 'Postur Baik':
                                    good_posture_count += 1
                                else:
                                    bad_posture_count += 1
                            
                            # Show preview periodically
                            if show_preview and processed_frames % 3 == 0:
                                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                video_placeholder.image(frame_rgb, 
                                    caption=f"Frame {frame_count:,} - Detections: {detection_count}",
                                    use_container_width=True)
                        
                        # Update progress
                        frame_count += 1
                        progress = frame_count / frames_to_process
                        progress_bar.progress(progress)
                        
                        # Update status
                        elapsed_time = time.time() - start_time
                        processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                        eta_seconds = (frames_to_process - frame_count) / processing_fps if processing_fps > 0 else 0
                        
                        status_text.markdown(f"""
                        **⚡ Progress:** {frame_count:,}/{frames_to_process:,} frames ({progress*100:.1f}%) | 
                        **🔥 Speed:** {processing_fps:.1f} FPS | 
                        **⏰ ETA:** {eta_seconds/60:.1f}m | 
                        **🎯 Total Detections:** {total_detections:,}
                        """)
                        
                        # Update live stats every 50 frames
                        if frame_count % 50 == 0 and total_detections > 0:
                            current_accuracy = (good_posture_count / (good_posture_count + bad_posture_count)) * 100
                            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
                            
                            with live_stats.container():
                                live_cols = st.columns(4)
                                with live_cols[0]:
                                    st.metric("🎯 Detections", total_detections)
                                with live_cols[1]:
                                    st.metric("✅ Good", good_posture_count) 
                                with live_cols[2]:
                                    st.metric("❌ Bad", bad_posture_count)
                                with live_cols[3]:
                                    st.metric("📊 Quality", f"{current_accuracy:.1f}%")
                    
                    cap.release()
                    processing_time = time.time() - start_time
                    
                    # Final results
                    st.markdown("---")
                    st.markdown("### 🎉 Analysis Complete!")
                    
                    if total_detections > 0:
                        final_accuracy = (good_posture_count / (good_posture_count + bad_posture_count)) * 100
                        avg_confidence = sum(all_confidences) / len(all_confidences)
                        detection_rate = (total_detections / processed_frames) * 100 if processed_frames > 0 else 0
                        
                        # Key metrics
                        st.markdown("#### 📊 Final Results")
                        cols = st.columns(6)
                        
                        with cols[0]:
                            st.metric("🎯 Total Detections", f"{total_detections:,}")
                        with cols[1]:
                            st.metric("✅ Good Posture", f"{good_posture_count:,}")
                        with cols[2]:
                            st.metric("❌ Bad Posture", f"{bad_posture_count:,}")
                        with cols[3]:
                            st.metric("📈 Posture Quality", f"{final_accuracy:.1f}%")
                        with cols[4]:
                            st.metric("🎯 Detection Rate", f"{detection_rate:.1f}%")
                        with cols[5]:
                            st.metric("⏱️ Processing Time", f"{processing_time:.1f}s")
                        
                        # Detailed analysis
                        st.markdown("#### 📈 Detailed Analysis")
                        
                        # Quality assessment with visual indicator
                        if final_accuracy >= 80:
                            quality_color = "#00b894"
                            quality_icon = "🌟"
                            quality_text = "Excellent"
                            recommendation = "Outstanding posture throughout the video! Keep up the excellent work."
                        elif final_accuracy >= 60:
                            quality_color = "#fdcb6e" 
                            quality_icon = "👍"
                            quality_text = "Good"
                            recommendation = "Good overall posture with room for improvement. Focus on consistency."
                        else:
                            quality_color = "#fd79a8"
                            quality_icon = "⚠️"
                            quality_text = "Needs Work"
                            recommendation = "Significant posture issues detected. Consider ergonomic improvements."
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {quality_color}20, {quality_color}10); 
                                    border: 2px solid {quality_color}; border-radius: 15px; padding: 20px; margin: 20px 0;">
                            <h4 style="color: {quality_color}; margin: 0;">{quality_icon} Overall Assessment: {quality_text}</h4>
                            <p style="margin: 10px 0 0 0; font-size: 1.1em;">{recommendation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Statistics breakdown
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### 📊 Processing Statistics")
                            st.write(f"**Frames Analyzed:** {processed_frames:,} / {total_frames:,}")
                            st.write(f"**Processing Interval:** Every {frame_skip} frame(s)")
                            st.write(f"**Average Confidence:** {avg_confidence:.2%}")
                            st.write(f"**Processing Speed:** {frame_count/processing_time:.1f} FPS")
                            st.write(f"**Coverage:** {(processed_frames/total_frames)*100:.1f}% of video")
                        
                        with col2:
                            st.markdown("##### 🎯 Detection Insights")
                            frames_with_detection = sum(1 for _ in range(processed_frames) if _ % frame_skip == 0)
                            avg_detections_per_frame = total_detections / processed_frames if processed_frames > 0 else 0
                            
                            st.write(f"**Detection Coverage:** {detection_rate:.1f}% of frames")
                            st.write(f"**Avg Detections/Frame:** {avg_detections_per_frame:.1f}")
                            st.write(f"**Good vs Bad Ratio:** {good_posture_count}:{bad_posture_count}")
                            st.write(f"**Confidence Range:** {min(all_confidences):.1%} - {max(all_confidences):.1%}")
                            st.write(f"**Video Duration:** {duration:.1f} seconds")
                        
                        # Confidence distribution
                        if len(all_confidences) > 10:
                            st.markdown("##### 📈 Confidence Distribution")
                            
                            # Simple confidence analysis
                            high_conf = sum(1 for c in all_confidences if c > 0.75)
                            med_conf = sum(1 for c in all_confidences if 0.5 <= c <= 0.75) 
                            low_conf = sum(1 for c in all_confidences if c < 0.5)
                            total_conf = len(all_confidences)
                            
                            conf_cols = st.columns(3)
                            with conf_cols[0]:
                                st.metric("🟢 High Confidence (>75%)", f"{high_conf} ({high_conf/total_conf*100:.1f}%)")
                            with conf_cols[1]:
                                st.metric("🟡 Medium Confidence (50-75%)", f"{med_conf} ({med_conf/total_conf*100:.1f}%)")
                            with conf_cols[2]:
                                st.metric("🟠 Low Confidence (<50%)", f"{low_conf} ({low_conf/total_conf*100:.1f}%)")
                        
                        # Performance summary
                        st.markdown("##### ⚡ Performance Summary")
                        performance_data = {
                            "Metric": ["Processing Speed", "Detection Accuracy", "Coverage", "Efficiency"],
                            "Value": [
                                f"{frame_count/processing_time:.1f} FPS",
                                f"{final_accuracy:.1f}%", 
                                f"{detection_rate:.1f}%",
                                f"{processed_frames/processing_time:.1f} frames/sec"
                            ],
                            "Rating": [
                                "🟢 Excellent" if frame_count/processing_time > 20 else "🟡 Good" if frame_count/processing_time > 10 else "🟠 Slow",
                                "🟢 Excellent" if final_accuracy > 80 else "🟡 Good" if final_accuracy > 60 else "🟠 Needs Work",
                                "🟢 Excellent" if detection_rate > 50 else "🟡 Good" if detection_rate > 25 else "🟠 Low",
                                "🟢 Efficient" if processed_frames/processing_time > 15 else "🟡 Moderate" if processed_frames/processing_time > 8 else "🟠 Slow"
                            ]
                        }
                        
                        for i, metric in enumerate(performance_data["Metric"]):
                            st.write(f"**{metric}:** {performance_data['Value'][i]} - {performance_data['Rating'][i]}")
                        
                        # Export option
                        st.markdown("##### 💾 Export Results")
                        results_summary = f"""
VIDEO ANALYSIS REPORT
=====================
Video: {uploaded_video.name}
Processed: {frame_count:,} frames in {processing_time:.1f}s
Detection Rate: {detection_rate:.1f}%

POSTURE ANALYSIS:
- Total Detections: {total_detections:,}
- Good Posture: {good_posture_count:,} ({good_posture_count/(good_posture_count+bad_posture_count)*100:.1f}%)
- Bad Posture: {bad_posture_count:,} ({bad_posture_count/(good_posture_count+bad_posture_count)*100:.1f}%)
- Average Confidence: {avg_confidence:.2%}

QUALITY ASSESSMENT: {quality_text}
RECOMMENDATION: {recommendation}
                        """
                        
                        st.download_button(
                            label="📄 Download Report",
                            data=results_summary,
                            file_name=f"pose_analysis_{uploaded_video.name.split('.')[0]}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                    else:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>😔 No Poses Detected in Video</h4>
                            <p><strong>Possible reasons:</strong></p>
                            <ul>
                                <li>People not clearly visible in most frames</li>
                                <li>Video quality too low for detection</li>
                                <li>Confidence threshold set too high</li>
                                <li>Poor lighting conditions throughout video</li>
                                <li>Camera angle doesn't show upper body clearly</li>
                            </ul>
                            <br>
                            <p><strong>💡 Suggestions for better results:</strong></p>
                            <ul>
                                <li>Lower the confidence threshold in sidebar settings</li>
                                <li>Use videos with clear upper body visibility</li>
                                <li>Ensure good lighting conditions</li>
                                <li>Try processing every frame (set interval to 1)</li>
                                <li>Use higher quality video files</li>
                                <li>Check that people face the camera</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <h4>❌ Video Processing Error</h4>
                    <p><strong>Error Details:</strong> {str(e)}</p>
                    <br>
                    <p><strong>🔧 Troubleshooting:</strong></p>
                    <ul>
                        <li><strong>File Format:</strong> Try converting to MP4 format</li>
                        <li><strong>File Size:</strong> Large files (&gt;100MB) may cause issues</li>
                        <li><strong>Codec:</strong> Ensure video uses standard codecs (H.264)</li>
                        <li><strong>Corruption:</strong> Re-download or re-export the video</li>
                        <li><strong>Memory:</strong> Try processing fewer frames or lower interval</li>
                    </ul>
                    <br>
                    <p><strong>💡 Quick Fixes:</strong></p>
                    <ul>
                        <li>Reduce "Maximum Frames to Process" to 300 or less</li>
                        <li>Increase "Frame Processing Interval" to 10 or higher</li>
                        <li>Try a different video file</li>
                        <li>Refresh the page and try again</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)

# Footer with comprehensive system information
st.markdown("---")
st.markdown("### 📊 System Information & Status")

# System status grid
cols = st.columns(6)

with cols[0]:
    model_status = "✅ Ready" if st.session_state.model_loaded else "❌ Failed"
    st.markdown(f"""
    <div class="metric-card">
        <h4>🤖 AI Model</h4>
        <p>{model_status}</p>
        <small>YOLO v8 Pose</small>
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
    webrtc_support = "✅ Full" if device_info["browser"] in ["Chrome", "Firefox"] else "⚠️ Limited" 
    st.markdown(f"""
    <div class="metric-card">
        <h4>🌐 WebRTC</h4>
        <p>{webrtc_support}</p>
        <small>Real-time Support</small>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    st.markdown(f"""
    <div class="metric-card">
        <h4>⚙️ Quality</h4>
        <p>{quality_mode.split()[0]}</p>
        <small>Current Setting</small>
    </div>
    """, unsafe_allow_html=True)

with cols[4]:
    session_duration = (time.time() - st.session_state.stats.get('session_start', time.time())) / 60
    st.markdown(f"""
    <div class="metric-card">
        <h4>⏱️ Session</h4>
        <p>{session_duration:.1f}m</p>
        <small>Active Time</small>
    </div>
    """, unsafe_allow_html=True)

with cols[5]:
    total_session_detections = st.session_state.stats['good_posture'] + st.session_state.stats['bad_posture']
    session_accuracy = (st.session_state.stats['good_posture'] / total_session_detections * 100) if total_session_detections > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <h4>📊 Session Avg</h4>
        <p>{session_accuracy:.0f}%</p>
        <small>Posture Quality</small>
    </div>
    """, unsafe_allow_html=True)

# Technical information
with st.expander("🔧 Technical Information"):
    st.markdown(f"""
    **Implementation:** Official streamlit-webrtc patterns  
    **Model:** YOLO v8 Pose Detection Neural Network  
    **Framework:** Streamlit {st.__version__}  
    **Video Processing:** OpenCV {cv2.__version__}  
    **Real-time Communication:** WebRTC with STUN servers  
    **Device Detection:** {device_info["user_agent"][:100]}...  
    **Current Settings:** Confidence={st.session_state.detection_settings['confidence_threshold']}, Keypoint={st.session_state.detection_settings['keypoint_threshold']}  
    **Quality Mode:** {quality_mode}  
    **Session Start:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state.stats['session_start']))}
    """)

# Enhanced footer with version information
st.markdown("""
<div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin-top: 2rem;">
    <h3>🤸‍♂️ AI Pose Detection System</h3>
    <p><strong>🚀 Built with Official streamlit-webrtc Implementation</strong></p>
    <p><strong>💻 Technology Stack:</strong> YOLO v8 • OpenCV • WebRTC • Streamlit • Python</p>
    <p><strong>✨ Features:</strong> Real-time Detection • Cross-platform • Responsive UI • Advanced Analytics</p>
    <p><strong>🌐 Platform Support:</strong> 💻 Desktop (Chrome, Firefox, Edge) • 📱 Mobile (Chrome, Safari) • 📟 Tablet</p>
    <br>
    <p><strong>🔧 Version:</strong> Official Implementation v4.0 - Following streamlit-webrtc Best Practices</p>
    <p><strong>📚 Based on:</strong> <a href="https://github.com/whitphx/streamlit-webrtc" style="color: #FFE66D;">github.com/whitphx/streamlit-webrtc</a></p>
    <p><em>Professional-grade pose analysis with reliable real-time processing</em></p>
</div>
""", unsafe_allow_html=True)
                import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np
import math
from PIL import Image
import time
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    VideoTransformerBase,
    ClientSettings,
)
import av
import threading
import queue
import logging
from typing import List, NamedTuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Pose Detection - Official Implementation",
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
    
    /* WebRTC video styling */
    video {
        border-radius: 15px !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1) !important;
        max-width: 100% !important;
        height: auto !important;
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
    .status-connecting { background-color: #fdcb6e; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .stColumn {
            padding: 0 0.25rem;
        }
        .metric-card {
            padding: 0.8rem 0.5rem;
            font-size: 0.9rem;
        }
        .metric-card h3 {
            font-size: 1.2rem;
            margin: 0.2rem 0;
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

# Result structure
class Detection(NamedTuple):
    label: str
    confidence: float
    bbox: List[float]

# Global settings following streamlit-webrtc best practices
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration=RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }),
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)

# Initialize session state
def init_session_state():
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
    
    if 'detection_settings' not in st.session_state:
        st.session_state.detection_settings = {
            'confidence_threshold': 0.5,
            'keypoint_threshold': 0.5,
            'show_keypoints': True,
            'show_connections': True,
            'show_angles': True,
            'show_confidence': True,
            'show_fps': True,
        }

init_session_state()

# Header
st.markdown('<h1 class="main-header">🤸‍♂️ AI Pose Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Official streamlit-webrtc Implementation</p>', unsafe_allow_html=True)

# Device detection
@st.cache_data
def detect_device():
    try:
        user_agent = st.context.headers.get("User-Agent", "").lower()
        is_mobile = any(keyword in user_agent for keyword in ["mobile", "android", "iphone", "ipad"])
        
        browser = "unknown"
        if "chrome" in user_agent and "edg" not in user_agent:
            browser = "Chrome"
        elif "firefox" in user_agent:
            browser = "Firefox"
        elif "safari" in user_agent and "chrome" not in user_agent:
            browser = "Safari"
        elif "edg" in user_agent:
            browser = "Edge"
            
        return {
            "type": "mobile" if is_mobile else "desktop",
            "is_mobile": is_mobile,
            "browser": browser,
            "user_agent": user_agent
        }
    except:
        return {"type": "desktop", "is_mobile": False, "browser": "Unknown", "user_agent": ""}

device_info = detect_device()

# Model loading with proper caching
@st.cache_resource
def load_pose_model():
    """Load YOLO pose model following streamlit best practices"""
    model_paths = [
        "pose2/train2/weights/best.pt",
        "best.pt", 
        "models/best.pt",
        "weights/best.pt",
        "yolo_pose.pt"
    ]
    
    # Try local models first
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading model from {model_path}")
                model = YOLO(model_path)
                return model, model_path
            except Exception as e:
                logger.error(f"Failed to load {model_path}: {e}")
                continue
    
    # Try downloading pre-trained model
    try:
        logger.info("Downloading YOLO pose model...")
        model = YOLO('yolov8n-pose.pt')
        return model, 'yolov8n-pose.pt (downloaded)'
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None, None

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Device information
    device_icon = "📱" if device_info["is_mobile"] else "💻"
    st.markdown(f"""
    <div class="info-box">
        <h4>{device_icon} Device Information</h4>
        <p><strong>Type:</strong> {device_info["type"].title()}</p>
        <p><strong>Browser:</strong> {device_info["browser"]}</p>
        <p><strong>WebRTC Support:</strong> {"✅ Yes" if device_info["browser"] in ["Chrome", "Firefox", "Safari", "Edge"] else "⚠️ Limited"}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model management
    st.markdown("#### 🤖 Model Management")
    
    if st.button("🔄 Reload Model", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.model_loaded = False
        st.session_state.pose_model = None
        st.rerun()
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading YOLO model..."):
            model, model_path = load_pose_model()
            if model is not None:
                st.session_state.pose_model = model
                st.session_state.model_loaded = True
                st.session_state.model_path = model_path
            else:
                st.error("Failed to load pose detection model")
                st.stop()
    
    # Model status
    if st.session_state.model_loaded:
        st.markdown(f"""
        <div class="success-box">
            <h4>✅ Model Ready</h4>
            <p><strong>Path:</strong> {st.session_state.model_path}</p>
            <p><strong>Status:</strong> Loaded Successfully</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detection settings
    st.markdown("#### 🎯 Detection Settings")
    
    st.session_state.detection_settings['confidence_threshold'] = st.slider(
        "Confidence Threshold", 0.1, 1.0, 
        st.session_state.detection_settings['confidence_threshold'], 0.05
    )
    
    st.session_state.detection_settings['keypoint_threshold'] = st.slider(
        "Keypoint Threshold", 0.1, 1.0, 
        st.session_state.detection_settings['keypoint_threshold'], 0.05
    )
    
    # Display settings
    st.markdown("#### 🎨 Display Settings")
    
    st.session_state.detection_settings['show_keypoints'] = st.checkbox(
        "Show Keypoints", st.session_state.detection_settings['show_keypoints']
    )
    
    st.session_state.detection_settings['show_connections'] = st.checkbox(
        "Show Connections", st.session_state.detection_settings['show_connections']
    )
    
    st.session_state.detection_settings['show_angles'] = st.checkbox(
        "Show Angles", st.session_state.detection_settings['show_angles']
    )
    
    st.session_state.detection_settings['show_confidence'] = st.checkbox(
        "Show Confidence", st.session_state.detection_settings['show_confidence']
    )
    
    st.session_state.detection_settings['show_fps'] = st.checkbox(
        "Show FPS", st.session_state.detection_settings['show_fps']
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        quality_mode = st.selectbox(
            "Quality Mode",
            ["Low (320p)", "Medium (480p)", "High (720p)"],
            index=1
        )
        
        line_thickness = st.slider("Line Thickness", 1, 8, 3)
        text_scale = st.slider("Text Scale", 0.3, 1.5, 0.7, 0.1)

# Helper functions
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
    except Exception as e:
        logger.warning(f"Error calculating angle: {e}")
        return None

def draw_pose_annotations(frame, keypoints_obj, label, box, conf_score, settings):
    """Draw pose annotations on frame"""
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
            
            if conf > settings['keypoint_threshold']:
                pt = (int(x), int(y))
                pts.append(pt)
                
                if settings['show_keypoints']:
                    # Enhanced keypoint visualization
                    cv2.circle(frame, pt, 8, (0, 0, 0), -1)  # Black outline
                    cv2.circle(frame, pt, 6, color, -1)      # Colored center
                    cv2.circle(frame, pt, 8, (255, 255, 255), 2)  # White border
            else:
                pts.append(None)

        # Draw connections
        if settings['show_connections'] and len(pts) >= 2:
            for i, j in KEYPOINT_CONNECTIONS:
                if i < len(pts) and j < len(pts) and pts[i] and pts[j]:
                    # Enhanced connection visualization
                    cv2.line(frame, pts[i], pts[j], (0, 0, 0), line_thickness + 2, cv2.LINE_AA)  # Black outline
                    cv2.line(frame, pts[i], pts[j], color, line_thickness, cv2.LINE_AA)  # Colored line

        # Draw angle
        if settings['show_angles'] and len(pts) >= 3 and all(pts[k] for k in [0, 1, 2]):
            angle = calculate_angle(pts[0], pts[1], pts[2])
            if angle is not None:
                pos = pts[1]
                angle_text = f"{int(angle)}°"
                
                # Enhanced angle text
                (text_width, text_height), baseline = cv2.getTextSize(
                    angle_text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 2
                )
                
                # Background rectangle
                cv2.rectangle(frame, 
                    (pos[0] + 10, pos[1] - text_height - 10), 
                    (pos[0] + text_width + 20, pos[1] + 5), 
                    (0, 0, 0), -1)
                cv2.rectangle(frame, 
                    (pos[0] + 10, pos[1] - text_height - 10), 
                    (pos[0] + text_width + 20, pos[1] + 5), 
                    (255, 255, 255), 2)
                
                # Angle text
                cv2.putText(frame, angle_text, (pos[0] + 15, pos[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw bounding box and label
        if box is not None:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Enhanced bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), line_thickness + 1)  # Black outline
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)  # Colored box
                
                # Label text
                display_text = label_text
                if settings['show_confidence']:
                    display_text += f" ({conf_score:.1%})"
                
                # Label background
                (text_width, text_height), _ = cv2.getTextSize(
                    display_text, cv2.FONT_HERSHEY_DUPLEX, text_scale, 2
                )
                
                cv2.rectangle(frame, (x1, y1 - text_height - 15), 
                    (x1 + text_width + 15, y1 - 5), (0, 0, 0), -1)  # Black background
                cv2.rectangle(frame, (x1, y1 - text_height - 15), 
                    (x1 + text_width + 15, y1 - 5), color, -1)  # Colored background
                
                # Label text
                cv2.putText(frame, display_text, (x1 + 7, y1 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, text_scale, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                logger.warning(f"Error drawing bounding box: {e}")

        return frame
    except Exception as e:
        logger.error(f"Error in draw_pose_annotations: {e}")
        return frame

def process_frame_pose_detection(frame, settings):
    """Process frame for pose detection"""
    try:
        start_time = time.time()
        
        # Get model from session state
        model = st.session_state.pose_model
        if model is None:
            return frame, 0, []
        
        # Determine image size based on quality
        if "Low" in quality_mode:
            img_size = 320
        elif "High" in quality_mode:
            img_size = 640
        else:
            img_size = 480
        
        # Run YOLO inference
        results = model.predict(
            frame, 
            imgsz=img_size,
            conf=settings['confidence_threshold'],
            save=False,
            verbose=False
        )

        detection_count = 0
        detections = []

        # Process results
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is not None and keypoints is not None:
                for box, kpts in zip(boxes, keypoints):
                    try:
                        label = int(box.cls.cpu().item())
                        conf_score = float(box.conf.cpu().item())
                        bbox = box.xyxy[0].cpu().numpy().tolist()
                        
                        # Draw annotations
                        frame = draw_pose_annotations(frame, kpts, label, box, conf_score, settings)
                        
                        detection_count += 1
                        detections.append(Detection(
                            label=CLASS_LABELS.get(label, 'Unknown'),
                            confidence=conf_score,
                            bbox=bbox
                        ))
                    except Exception as e:
                        logger.warning(f"Error processing detection: {e}")
                        continue

        # Draw FPS
        if settings['show_fps']:
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # FPS background
            cv2.rectangle(frame, (10, 5), (120, 35), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        return frame, detection_count, detections
        
    except Exception as e:
        logger.error(f"Error in pose processing: {e}")
        return frame, 0, []

# WebRTC Video Transformer following official patterns
class PoseDetectionTransformer(VideoTransformerBase):
    """Video transformer for real-time pose detection"""
    
    def __init__(self):
        self.frame_count = 0
        self.detection_count = 0
        self.good_posture_count = 0
        self.bad_posture_count = 0
        self.fps_counter = time.time()
        self.lock = threading.Lock()
        
        # Statistics queue for thread-safe updates
        self.result_queue = queue.Queue(maxsize=30)
        
    def transform(self, frame):
        """Transform video frame with pose detection"""
        try:
            # Convert frame
            img = frame.to_ndarray(format="bgr24")
            
            # Get current settings
            settings = st.session_state.detection_settings
            
            # Process frame
            processed_img, detection_count, detections = process_frame_pose_detection(img, settings)
            
            # Update statistics thread-safely
            with self.lock:
                self.frame_count += 1
                self.detection_count = detection_count
                
                # Count posture types
                for detection in detections:
                    if detection.label == 'Postur Baik':
                        self.good_posture_count += 1
                    else:
                        self.bad_posture_count += 1
                
                # Update session stats periodically
                if self.frame_count % 30 == 0:  # Every 30 frames
                    try:
                        # Non-blocking queue put
                        stats_update = {
                            'total_frames': self.frame_count,
                            'good_posture': self.good_posture_count,
                            'bad_posture': self.bad_posture_count,
                            'current_detections': detection_count
                        }
                        self.result_queue.put_nowait(stats_update)
                    except queue.Full:
                        pass  # Skip if queue is full
            
            return processed_img
            
        except Exception as e:
            logger.error(f"Transform error: {e}")
            return frame.to_ndarray(format="bgr24")

# Get optimal media constraints
def get_optimal_constraints():
    """Get optimal media constraints based on device and settings"""
    
    base_constraints = {
        "audio": False,
        "video": {
            "frameRate": {"ideal": 24, "min": 15, "max": 30}
        }
    }
    
    # Adjust based on quality mode
    if "Low" in quality_mode:
        base_constraints["video"].update({
            "width": {"ideal": 320, "min": 240, "max": 480},
            "height": {"ideal": 240, "min": 180, "max": 360},
            "frameRate": {"ideal": 20, "min": 15, "max": 24}
        })
    elif "High" in quality_mode:
        base_constraints["video"].update({
            "width": {"ideal": 640, "min": 480, "max": 1280},
            "height": {"ideal": 480, "min": 360, "max": 720},
            "frameRate": {"ideal": 30, "min": 20, "max": 30}
        })
    else:  # Medium
        base_constraints["video"].update({
            "width": {"ideal": 480, "min": 320, "max": 640},
            "height": {"ideal": 360, "min": 240, "max": 480},
            "frameRate": {"ideal": 24, "min": 15, "max": 30}
        })
    
    # Mobile-specific adjustments
    if device_info["is_mobile"]:
        base_constraints["video"]["facingMode"] = "user"  # Front camera
        # Reduce constraints for mobile
        if "frameRate" in base_constraints["video"]:
            base_constraints["video"]["frameRate"]["ideal"] = min(20, base_constraints["video"]["frameRate"]["ideal"])
    
    return base_constraints

# Main interface
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📹 Live Detection", "📷 Image Analysis", "🎬 Video Analysis"])

# Tab 1: WebRTC Live Detection
with tab1:
    st.markdown("### 📹 Real-time Pose Detection")
    
    # Instructions based on device
    if device_info["is_mobile"]:
        st.markdown("""
        <div class="info-box">
            <h4>📱 Mobile WebRTC Instructions</h4>
            <p><strong>Recommended Browser:</strong> Chrome Mobile or Safari</p>
            <p><strong>Steps:</strong></p>
            <p>1. Click START button below</p>
            <p>2. Allow camera access when prompted</p>
            <p>3. Wait 5-10 seconds for connection</p>
            <p>4. Use landscape orientation for better results</p>
            <p>5. Ensure stable internet connection</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h4>💻 Desktop WebRTC Instructions</h4>
            <p><strong>Recommended Browser:</strong> Chrome or Firefox</p>
            <p><strong>Steps:</strong></p>
            <p>1. Click START button below</p>
            <p>2. Allow camera access in browser popup</p>
            <p>3. Position yourself 1-2 meters from camera</p>
            <p>4. Ensure good lighting conditions</p>
            <p>5. Pose detection will start automatically</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Current settings display
    constraints = get_optimal_constraints()
    video_settings = constraints["video"]
    
    st.markdown(f"""
    <div class="success-box">
        <h4>⚙️ Current Configuration</h4>
        <p><strong>Quality:</strong> {quality_mode}</p>
        <p><strong>Resolution:</strong> {video_settings.get('width', {}).get('ideal', 'Auto')}x{video_settings.get('height', {}).get('ideal', 'Auto')}</p>
        <p><strong>Frame Rate:</strong> {video_settings.get('frameRate', {}).get('ideal', 24)} FPS</p>
        <p><strong>Device:</strong> {device_info['type'].title()} ({device_info['browser']})</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Control buttons
    col1, col2 = st.columns([3, 1])
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
    
    # WebRTC Streamer using official best practices
    try:
        webrtc_ctx = webrtc_streamer(
            key="pose-detection-official",
            mode=WebRtcMode.SENDRECV,
            client_settings=ClientSettings(
                rtc_configuration=RTCConfiguration({
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                    ]
                }),
                media_stream_constraints=constraints,
            ),
            video_transformer_factory=PoseDetectionTransformer,
            async_processing=True,
        )
        
        # Connection status
        if webrtc_ctx.state.playing:
            st.markdown("""
            <div style="text-align: center; margin: 15px 0;">
                <span class="status-indicator status-online"></span>
                <strong style="color: #00b894; font-size: 1.1em;">🟢 Live Detection Active</strong>
            </div>
            """, unsafe_allow_html=True)
        elif webrtc_ctx.state.signalling:
            st.markdown("""
            <div style="text-align: center; margin: 15
