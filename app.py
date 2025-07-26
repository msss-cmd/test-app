import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import cv2
import mediapipe as mp
import numpy as np
import face_recognition # For face authentication
import dlib # Required by face_recognition, also for shape_predictor if used directly
from scipy.spatial import distance as dist # For EAR calculation if needed, but using user's blink logic
import time
import os
from typing import List, Union

# --- Configuration and Constants ---
# Path to dlib's pre-trained facial landmark predictor for face_recognition and dlib's direct use
# Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract it and place it in the same directory as this script.
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Blink detection threshold (from user's provided code)
# This is a very sensitive threshold, might need calibration
BLINK_THRESHOLD = 0.004
# Time to hold eyes closed for a "click" (from previous logic)
GAZE_HOLD_DURATION = 2.0 # seconds

# Face authentication configuration
KNOWN_FACE_IMAGE_PATH = "known_face.jpg"
FACE_AUTH_TOLERANCE = 0.6 # Lower value means stricter match (0.6 is common for face_recognition)

# --- Streamlit Session State Initialization ---
# This is crucial for managing state across Streamlit reruns
if 'current_screen' not in st.session_state:
    st.session_state.current_screen = 'permissions'
if 'camera_permission' not in st.session_state:
    st.session_state.camera_permission = False
if 'microphone_permission' not in st.session_state:
    st.session_state.microphone_permission = False
if 'accessibility_permission' not in st.session_state:
    st.session_state.accessibility_permission = False
if 'face_detected_login' not in st.session_state:
    st.session_state.face_detected_login = False
if 'cursor_sensitivity' not in st.session_state:
    st.session_state.cursor_sensitivity = 50
if 'eye_close_sound_enabled' not in st.session_state:
    st.session_state.eye_close_sound_enabled = True
if 'authenticated_status' not in st.session_state:
    st.session_state.authenticated_status = False
if 'known_face_encoding' not in st.session_state:
    st.session_state.known_face_encoding = None
if 'total_clicks' not in st.session_state:
    st.session_state.total_clicks = 0
if 'is_app_open' not in st.session_state:
    st.session_state.is_app_open = False # For simulated gallery app overlay
if 'last_click_time' not in st.session_state:
    st.session_state.last_click_time = 0.0 # To prevent rapid clicks
if 'simulated_cursor_pos' not in st.session_state:
    st.session_state.simulated_cursor_pos = (0, 0) # (x, y) for on-screen cursor
if 'blink_start_time' not in st.session_state:
    st.session_state.blink_start_time = None

# --- Helper Function for Loading Known Face (for Streamlit) ---
def load_known_face_encoding_streamlit(image_path):
    """Helper to load known face for Streamlit, with user feedback."""
    if not os.path.exists(image_path):
        st.warning(f"Known face image not found at '{image_path}'. "
                   "Please place a clear image of the authorized user there for face authentication. "
                   "Face authentication will be skipped for now.")
        return None
    try:
        known_image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(known_image)
        if len(face_encodings) > 0:
            st.success(f"Successfully loaded known face from '{image_path}'.")
            return face_encodings[0]
        else:
            st.warning(f"No face found in '{image_path}'. Face authentication will be skipped.")
            return None
    except Exception as e:
        st.error(f"Error loading known face image '{image_path}': {e}. Face authentication will be skipped.")
        return None

# --- Computer Vision Processor Class for Streamlit ---
class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        # Initialize dlib's face detector for face_recognition
        self.dlib_detector = dlib.get_frontal_face_detector()
        try:
            # dlib shape predictor is needed by face_recognition for landmarks
            self.dlib_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        except RuntimeError:
            st.error(f"Error: Could not find '{SHAPE_PREDICTOR_PATH}'. "
                     "Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2, "
                     "extract it, and place it in the same directory as this script.")
            st.stop() # Stop Streamlit execution if essential file is missing

        # Load known face encoding once when processor is initialized
        if st.session_state.known_face_encoding is None:
            st.session_state.known_face_encoding = load_known_face_encoding_streamlit(KNOWN_FACE_IMAGE_PATH)
        self.face_auth_enabled = st.session_state.known_face_encoding is not None

        # Internal state for blink detection within the processor
        self._blink_start_time = None
        self._last_click_time = 0.0 # To prevent rapid clicks

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Flip horizontally for mirror view
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output = self.face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks

        frame_h, frame_w, _ = img.shape

        # --- Face Authentication Logic ---
        if self.face_auth_enabled:
            # Use dlib detector to get face locations for face_recognition
            rects = self.dlib_detector(rgb_frame, 0)
            current_face_locations = [(d.top(), d.right(), d.bottom(), d.left()) for d in rects]
            current_face_encodings = face_recognition.face_encodings(rgb_frame, current_face_locations)

            if len(current_face_encodings) > 0:
                matches = face_recognition.compare_faces([st.session_state.known_face_encoding], current_face_encodings[0], FACE_AUTH_TOLERANCE)
                st.session_state.authenticated_status = matches[0]
            else:
                st.session_state.authenticated_status = False

        # Display authentication status
        auth_status_text = "Authenticated: YES" if st.session_state.authenticated_status else "Authenticated: NO"
        auth_color = (0, 255, 0) if st.session_state.authenticated_status else (0, 0, 255)
        cv2.putText(img, auth_status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, auth_color, 2)

        # --- Eye Movement and Blink Detection (only if authenticated or auth is disabled) ---
        if st.session_state.authenticated_status or not self.face_auth_enabled:
            if landmark_points:
                landmarks = landmark_points[0].landmark

                # --- Eye Movement Control (Simulated Cursor) ---
                # Using right eye's iris landmarks (474:478) for cursor control
                # As per the user's provided code snippet, using landmark ID 1 (which is 475)
                if len(landmarks) > 475: # Ensure landmark exists
                    iris_landmark = landmarks[475] # Index 1 in the slice 474:478
                    
                    # Apply sensitivity to the normalized landmark position
                    sensitivity_factor = st.session_state.cursor_sensitivity / 50.0 # 50 is base sensitivity
                    
                    # Map normalized landmark position to frame dimensions, then apply sensitivity
                    # This creates a "zooming" effect for sensitivity
                    cursor_x_raw = iris_landmark.x * frame_w
                    cursor_y_raw = iris_landmark.y * frame_h

                    # Calculate deviation from center and scale by sensitivity
                    center_x, center_y = frame_w / 2, frame_h / 2
                    
                    # Calculate new position based on sensitivity
                    # This moves the cursor further for the same eye movement if sensitivity is high
                    sim_cursor_x = center_x + (cursor_x_raw - center_x) * sensitivity_factor
                    sim_cursor_y = center_y + (cursor_y_raw - center_y) * sensitivity_factor

                    # Clamp to frame boundaries
                    sim_cursor_x = max(0, min(int(sim_cursor_x), frame_w - 1))
                    sim_cursor_y = max(0, min(int(sim_cursor_y), frame_h - 1))

                    # Update session state for Streamlit UI to draw the cursor
                    st.session_state.simulated_cursor_pos = (sim_cursor_x, sim_cursor_y)
                    
                    # Draw a circle on the video feed to show the tracked point
                    cv2.circle(img, (sim_cursor_x, sim_cursor_y), 5, (0, 255, 255), -1) # Yellow circle

                # --- Blink Detection (Simulated Click/App Open) ---
                # Using left eye landmarks 145 (lower eyelid) and 159 (upper eyelid)
                if len(landmarks) > 159: # Ensure landmarks exist
                    left_eye_lower = landmarks[145]
                    left_eye_upper = landmarks[159]

                    # Calculate vertical distance between upper and lower eyelid landmarks
                    # A smaller value indicates the eye is more closed
                    vertical_eye_distance = abs(left_eye_upper.y - left_eye_lower.y)

                    if vertical_eye_distance < BLINK_THRESHOLD:
                        # Eye is considered closed
                        cv2.putText(img, "EYES CLOSED!", (frame_w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if self._blink_start_time is None:
                            self._blink_start_time = time.time() # Start timer

                        # Check if eyes have been closed for the duration
                        if self._blink_start_time is not None and (time.time() - self._blink_start_time) >= GAZE_HOLD_DURATION:
                            # Trigger click/app open if not already open and enough time passed since last click
                            if not st.session_state.is_app_open and (time.time() - self._last_click_time) > (GAZE_HOLD_DURATION + 0.5): # Add small buffer
                                st.session_state.total_clicks += 1
                                st.session_state.is_app_open = True
                                self._last_click_time = time.time() # Update last click time
                                # Play sound (handled by JS injection in UI)
                                st.experimental_rerun() # Rerun to update UI for app open
                            cv2.putText(img, "CLICK/APP OPENED!", (frame_w - 250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    else:
                        # Eye is open
                        self._blink_start_time = None # Reset blink timer
                        if st.session_state.is_app_open: # If app was open, close it on eyes open
                            st.session_state.is_app_open = False
                            st.experimental_rerun() # Rerun to update UI for app close
        
        return img

# --- Streamlit UI Components ---

def play_beep_sound_js():
    """Injects and calls JavaScript to play a simple beep sound."""
    if st.session_state.eye_close_sound_enabled:
        st.components.v1.html("""
            <script>
                function playBeep() {
                    try {
                        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                        const oscillator = audioContext.createOscillator();
                        const gainNode = audioContext.createGain();

                        oscillator.connect(gainNode);
                        gainNode.connect(audioContext.destination);

                        oscillator.type = 'sine'; // Sine wave for a clean beep
                        oscillator.frequency.setValueAtTime(880, audioContext.currentTime); // A5 note, higher pitch
                        gainNode.gain.setValueAtTime(0.5, audioContext.currentTime); // Volume

                        oscillator.start();
                        oscillator.stop(audioContext.currentTime + 0.15); // 0.15 seconds duration
                    } catch (e) {
                        console.error("Error playing beep sound:", e);
                    }
                }
                playBeep(); // Call the function immediately when script is loaded/rerun
            </script>
        """, height=0)

def permission_screen():
    st.header("Permissions Required", divider='rainbow')
    st.markdown("""
    To enable the simulated cursor control, we need conceptual access to your device's camera, microphone, and accessibility features.
    
    _(**Note:** Actual browser accessibility control, direct camera/mic for eye tracking, and real facial recognition are complex and simulated for this demo within the web environment.)_
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Grant Camera", disabled=st.session_state.camera_permission, use_container_width=True):
            st.session_state.camera_permission = True
            st.experimental_rerun()
        if st.session_state.camera_permission:
            st.success("✅ Camera Granted")

    with col2:
        if st.button("Grant Microphone", disabled=st.session_state.microphone_permission, use_container_width=True):
            st.session_state.microphone_permission = True
            st.experimental_rerun()
        if st.session_state.microphone_permission:
            st.success("✅ Microphone Granted")

    with col3:
        if st.button("Grant Accessibility", disabled=st.session_state.accessibility_permission, use_container_width=True):
            st.session_state.accessibility_permission = True
            st.experimental_rerun()
        if st.session_state.accessibility_permission:
            st.success("✅ Accessibility Granted")

    st.markdown("---")
    if st.session_state.camera_permission and st.session_state.microphone_permission and st.session_state.accessibility_permission:
        if st.button("Continue to Face Login", type="primary", use_container_width=True):
            st.session_state.current_screen = 'faceLogin'
            st.experimental_rerun()
    else:
        st.info("Please grant all permissions to continue.")

def face_login_screen():
    st.header("Face Login System", divider='rainbow')
    st.markdown("""
    For security, please ensure your face is visible to the camera.
    
    _(**Note:** This is a simulated face detection for demo purposes. In a real system, this would involve live camera feed and robust face recognition.)_
    """)

    if not st.session_state.face_detected_login:
        if st.button("Simulate Face Detection", type="primary", use_container_width=True):
            st.session_state.face_detected_login = True
            st.success("Face Detected! Redirecting to settings...")
            time.sleep(1.5) # Simulate login delay
            st.session_state.current_screen = 'settings'
            st.experimental_rerun()
    else:
        st.success("✅ Face Detected! You are logged in.")
        st.info("Click 'Continue' to proceed to settings.")
        if st.button("Continue to Settings", type="primary", use_container_width=True):
            st.session_state.current_screen = 'settings'
            st.experimental_rerun()

def settings_screen():
    st.header("Important Settings", divider='rainbow')

    st.subheader("Cursor Sensitivity")
    st.session_state.cursor_sensitivity = st.slider(
        "Adjust Cursor Sensitivity",
        min_value=1, max_value=100, value=st.session_state.cursor_sensitivity, step=1,
        help="Controls how fast the simulated cursor moves with your eye/head movements."
    )
    st.write(f"Current Sensitivity: **{st.session_state.cursor_sensitivity}%**")

    st.subheader("Controls & User Manual")
    st.info("""
        - **Cursor Movement:** Your head/eye movements (as detected by the webcam) will control the custom cursor on the video feed.
        - **Open App (Simulated Click):** Close your eyes for approximately **2 seconds**. You will see "EYES CLOSED!" and then "CLICK/APP OPENED!". A simulated "Gallery App" overlay will appear.
        - **Close App:** Open your eyes to close the simulated app.
        - **Sound Feedback:** A short beep will play when a "click" is registered, if enabled below.
    """)

    st.subheader("Sound for Eye Closing / Click")
    st.session_state.eye_close_sound_enabled = st.checkbox(
        "Enable Sound for Simulated Click",
        value=st.session_state.eye_close_sound_enabled,
        help="Play a short beep sound when a 'click' action is detected (eyes closed for 2 seconds)."
    )
    if st.session_state.eye_close_sound_enabled and st.session_state.is_app_open:
        play_beep_sound_js() # Play sound when app opens due to click

    st.subheader("Profile Settings (Placeholder)")
    st.markdown("User profile management features would go here (e.g., name, preferences, etc.). This section is a placeholder for future development.")

    st.markdown("---")
    if st.button("Start Cursor Control", type="primary", use_container_width=True):
        st.session_state.current_screen = 'cursorControl'
        st.experimental_rerun()

def cursor_control_screen():
    st.header("Live Cursor Control", divider='rainbow')
    st.markdown("Observe the video feed below. Move your head/eyes to control the yellow circle cursor. Close your eyes for 2 seconds to 'click' and open the simulated app.")

    st.info(f"Simulated Clicks/App Opens: **{st.session_state.total_clicks}**")

    # Webcam Stream with VisionProcessor
    webrtc_streamer(
        key="webcam_stream",
        video_processor_factory=VisionProcessor,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}, # Only video, audio not needed for processor
        ),
        async_processing=True, # Process frames asynchronously to keep UI responsive
    )

    # Simulated Gallery App overlay
    if st.session_state.is_app_open:
        st.markdown(
            """
            <style>
            .gallery-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.9); /* Darker overlay */
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1000;
                backdrop-filter: blur(5px); /* Optional: blur background */
            }
            .gallery-content {
                background: linear-gradient(135deg, #f0f9ff, #c9e6ff); /* Light blue gradient */
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4); /* Stronger shadow */
                text-align: center;
                max-width: 90%;
                max-height: 90%;
                overflow-y: auto;
                animation: fadeInScale 0.3s ease-out forwards;
            }
            @keyframes fadeInScale {
                from { opacity: 0; transform: scale(0.9); }
                to { opacity: 1; transform: scale(1); }
            }
            .gallery-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); /* Larger items */
                gap: 20px; /* More space */
                margin-top: 30px;
            }
            .gallery-item {
                background: white;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15); /* Item shadow */
                transition: transform 0.2s ease-in-out;
            }
            .gallery-item:hover {
                transform: translateY(-5px) scale(1.02);
            }
            .gallery-item img {
                width: 100%;
                height: 100px; /* Fixed height for consistency */
                object-fit: cover;
                border-bottom: 1px solid #eee;
            }
            .gallery-item p {
                padding: 10px;
                font-size: 0.9em;
                color: #555;
            }
            </style>
            <div class="gallery-overlay">
                <div class="gallery-content">
                    <h3 style="color: #2c3e50; font-size: 2.5em; margin-bottom: 20px; font-weight: bold;">📸 Simulated Gallery App</h3>
                    <p style="color: #4a698c; margin-bottom: 30px; font-size: 1.1em;">
                        This app opened because you "closed your eyes" (held them closed for 2 seconds).
                    </p>
                    <div class="gallery-grid">
                        <div class="gallery-item"><img src="https://placehold.co/120x100/A8DADC/FFFFFF?text=Nature" alt="Nature"><p>Nature Scene</p></div>
                        <div class="gallery-item"><img src="https://placehold.co/120x100/F7CAC9/FFFFFF?text=City" alt="City"><p>Cityscape</p></div>
                        <div class="gallery-item"><img src="https://placehold.co/120x100/87CEEB/FFFFFF?text=Ocean" alt="Ocean"><p>Ocean View</p></div>
                        <div class="gallery-item"><img src="https://placehold.co/120x100/FFDAB9/FFFFFF?text=Mountain" alt="Mountain"><p>Mountain Peak</p></div>
                        <div class="gallery-item"><img src="https://placehold.co/120x100/CDB7F6/FFFFFF?text=Forest" alt="Forest"><p>Deep Forest</p></div>
                        <div class="gallery-item"><img src="https://placehold.co/120x100/B0E0E6/FFFFFF?text=Desert" alt="Desert"><p>Desert Dunes</p></div>
                    </div>
                    <p style="color: #777; font-size: 1em; margin-top: 30px;">Open your eyes to close this app.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    if st.button("Back to Settings", use_container_width=True):
        st.session_state.current_screen = 'settings'
        st.session_state.is_app_open = False # Ensure app is closed when navigating back
        st.experimental_rerun()

# --- Main App Flow ---
st.sidebar.title("App Navigation")
st.sidebar.markdown("Use the buttons below to navigate between the app screens.")

if st.sidebar.button("🏠 Permissions"):
    st.session_state.current_screen = 'permissions'
    st.experimental_rerun()
if st.sidebar.button("👤 Face Login"):
    st.session_state.current_screen = 'faceLogin'
    st.experimental_rerun()
if st.sidebar.button("⚙️ Settings"):
    st.session_state.current_screen = 'settings'
    st.experimental_rerun()
if st.sidebar.button("👁️ Cursor Control"):
    st.session_state.current_screen = 'cursorControl'
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.info(f"Current Screen: **{st.session_state.current_screen.replace('_', ' ').title()}**")

# Render the current screen based on session state
if st.session_state.current_screen == 'permissions':
    permission_screen()
elif st.session_state.current_screen == 'faceLogin':
    face_login_screen()
elif st.session_state.current_screen == 'settings':
    settings_screen()
elif st.session_state.current_screen == 'cursorControl':
    cursor_control_screen()
