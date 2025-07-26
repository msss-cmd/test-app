import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import cv2
import dlib
import numpy as np
import face_recognition
from scipy.spatial import distance as dist
import time
import os
from typing import List, Union

# --- Configuration and Constants ---
# Path to dlib's pre-trained facial landmark predictor
# Download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract it and place it in the same directory as this script.
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Eye Aspect Ratio (EAR) thresholds for blink detection
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# Gaze hold (simulated click) threshold
GAZE_HOLD_DURATION = 2.0 # seconds to hold gaze (eyes closed) for a "click"

# Face authentication configuration
KNOWN_FACE_IMAGE_PATH = "known_face.jpg"
FACE_AUTH_TOLERANCE = 0.6

# --- Helper Functions for Computer Vision ---

def eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR)."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

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
    st.session_state.is_app_open = False # For simulated gallery app


# --- Computer Vision Processor Class for Streamlit ---
class VisionProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        except RuntimeError:
            st.error(f"Error: Could not find '{SHAPE_PREDICTOR_PATH}'. "
                     "Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.bz2, "
                     "extract it, and place it in the same directory as this script.")
            st.stop() # Stop Streamlit execution
            return

        # Get the indexes of the facial landmarks for the left and right eye
        self.lStart, self.lEnd = (42, 48)
        self.rStart, self.rEnd = (36, 42)

        # State for blink detection
        self.counter = 0
        self.gaze_start_time = None

        # Load known face encoding once
        if st.session_state.known_face_encoding is None:
            st.session_state.known_face_encoding = self._load_known_face_encoding_streamlit(KNOWN_FACE_IMAGE_PATH)

        self.face_auth_enabled = st.session_state.known_face_encoding is not None

    def _load_known_face_encoding_streamlit(self, image_path):
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

    def recv(self, frame):
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Flip horizontally for mirror view
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = self.detector(gray, 1) # Detect faces

        # --- Face Authentication Logic ---
        if self.face_auth_enabled:
            current_face_locations = face_recognition.face_locations(img)
            current_face_encodings = face_recognition.face_encodings(img, current_face_locations)

            if len(current_face_encodings) > 0:
                matches = face_recognition.compare_faces([st.session_state.known_face_encoding], current_face_encodings[0], FACE_AUTH_TOLERANCE)
                st.session_state.authenticated_status = matches[0]
            else:
                st.session_state.authenticated_status = False # No face detected in current frame

        # Display authentication status
        auth_status_text = "Authenticated: YES" if st.session_state.authenticated_status else "Authenticated: NO"
        auth_color = (0, 255, 0) if st.session_state.authenticated_status else (0, 0, 255)
        cv2.putText(img, auth_status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, auth_color, 2)

        # --- Eye Movement and Blink Detection (only if authenticated or auth is disabled) ---
        if st.session_state.authenticated_status or not self.face_auth_enabled:
            for rect in rects:
                x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue bounding box

                shape = self.predictor(gray, rect)
                shape_np = np.array([[p.x, p.y] for p in shape.parts()])

                leftEye = shape_np[self.lStart:self.lEnd]
                rightEye = shape_np[self.rStart:self.rEnd]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                cv2.drawContours(img, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
                cv2.drawContours(img, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

                # Blink Detection (Simulated Click)
                if ear < EYE_AR_THRESH:
                    self.counter += 1
                    cv2.putText(img, "EYES CLOSED!", (img.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if self.gaze_start_time is None:
                        self.gaze_start_time = time.time()

                    if (time.time() - self.gaze_start_time) >= GAZE_HOLD_DURATION:
                        if not st.session_state.is_app_open: # Only trigger once
                            st.session_state.total_clicks += 1
                            st.session_state.is_app_open = True
                            # Play sound via JS injection (handled by Streamlit UI)
                            st.experimental_rerun() # Rerun to update UI for app open
                        cv2.putText(img, "CLICK/APP OPENED!", (img.shape[1] - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                else:
                    self.counter = 0
                    self.gaze_start_time = None
                    if st.session_state.is_app_open: # If app was open, close it on eyes open
                        st.session_state.is_app_open = False
                        st.experimental_rerun() # Rerun to update UI for app close

                # Eye/Head Movement Control (Simulated Cursor)
                face_center_x = (x1 + x2) // 2
                face_center_y = (y1 + y2) // 2

                frame_h, frame_w = img.shape[:2]
                norm_x = (face_center_x - frame_w / 2) / (frame_w / 2)
                norm_y = (face_center_y - frame_h / 2) / (frame_h / 2)

                # Apply sensitivity
                sensitivity_factor = st.session_state.cursor_sensitivity / 50.0 # 50 is base
                display_cursor_x = int(frame_w / 2 + norm_x * (frame_w / 2) * sensitivity_factor)
                display_cursor_y = int(frame_h / 2 + norm_y * (frame_h / 2) * sensitivity_factor)

                # Clamp cursor to frame boundaries
                display_cursor_x = max(0, min(display_cursor_x, frame_w - 1))
                display_cursor_y = max(0, min(display_cursor_y, frame_h - 1))

                cv2.circle(img, (display_cursor_x, display_cursor_y), 10, (0, 255, 255), -1) # Yellow cursor

                cursor_x_text = "CENTER"
                cursor_y_text = "CENTER"
                if norm_x < -0.3: cursor_x_text = "LEFT"
                elif norm_x > 0.3: cursor_x_text = "RIGHT"
                if norm_y < -0.3: cursor_y_text = "UP"
                elif norm_y > 0.3: cursor_y_text = "DOWN"

                cv2.putText(img, f"Cursor X: {cursor_x_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(img, f"Cursor Y: {cursor_y_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return img

# --- Streamlit UI Components ---

def permission_screen():
    st.title("Permissions Required")
    st.markdown("To enable simulated cursor control, we need conceptual access to your camera, microphone, and accessibility features.")
    st.markdown("_(Note: Actual browser accessibility control, direct camera/mic for eye tracking, and real facial recognition are complex and simulated for this demo.)_")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Grant Camera", disabled=st.session_state.camera_permission):
            st.session_state.camera_permission = True
            st.experimental_rerun()
        if st.session_state.camera_permission:
            st.success("Camera Granted")

    with col2:
        if st.button("Grant Microphone", disabled=st.session_state.microphone_permission):
            st.session_state.microphone_permission = True
            st.experimental_rerun()
        if st.session_state.microphone_permission:
            st.success("Microphone Granted")

    with col3:
        if st.button("Grant Accessibility", disabled=st.session_state.accessibility_permission):
            st.session_state.accessibility_permission = True
            st.experimental_rerun()
        if st.session_state.accessibility_permission:
            st.success("Accessibility Granted")

    st.markdown("---")
    if st.session_state.camera_permission and st.session_state.microphone_permission and st.session_state.accessibility_permission:
        if st.button("Continue to Face Login", type="primary"):
            st.session_state.current_screen = 'faceLogin'
            st.experimental_rerun()
    else:
        st.info("Please grant all permissions to continue.")

def face_login_screen():
    st.title("Face Login System")
    st.markdown("For security, please ensure your face is visible.")
    st.markdown("_(Note: This is a simulated face detection for demo purposes. In the actual system, this would involve live camera feed and real face recognition.)_")

    if not st.session_state.face_detected_login:
        if st.button("Simulate Face Detection"):
            st.session_state.face_detected_login = True
            st.success("Face Detected! Redirecting...")
            time.sleep(1.5) # Simulate login delay
            st.session_state.current_screen = 'settings'
            st.experimental_rerun()
    else:
        st.success("Face Detected! You are logged in.")
        st.info("Click 'Continue' to proceed to settings.")
        if st.button("Continue to Settings", type="primary"):
            st.session_state.current_screen = 'settings'
            st.experimental_rerun()

def settings_screen():
    st.title("Important Settings")

    st.subheader("Cursor Sensitivity")
    st.session_state.cursor_sensitivity = st.slider(
        "Adjust Cursor Sensitivity",
        min_value=1, max_value=100, value=st.session_state.cursor_sensitivity, step=1
    )
    st.write(f"Current Sensitivity: {st.session_state.cursor_sensitivity}%")

    st.subheader("Controls & User Manual")
    st.info("""
        **Cursor Movement:** Your mouse movements will simulate eye movement to control the on-screen cursor.
        **Open App (Simulated Eye Closing):** Close your eyes (or hold your mouse button down in the live feed) for 2 seconds. A sound will play, and a simulated "Gallery App" will open. Open your eyes (release mouse) to close the app.
        **Settings:** Adjust cursor sensitivity and toggle eye-closing sound here.
        **Profile:** Manage your user profile settings (placeholder).
    """)

    st.subheader("Sound for Eye Closing")
    st.session_state.eye_close_sound_enabled = st.checkbox(
        "Enable Sound for Eye Closing",
        value=st.session_state.eye_close_sound_enabled
    )

    st.subheader("Profile Settings (Placeholder)")
    st.write("User profile management features would go here (e.g., name, preferences, etc.). This section is a placeholder for future development.")

    st.markdown("---")
    if st.button("Start Cursor Control", type="primary"):
        st.session_state.current_screen = 'cursorControl'
        st.experimental_rerun()

def cursor_control_screen():
    st.title("Cursor Control Mode")
    st.markdown("Move your head/eyes to control the cursor. Close your eyes for 2 seconds to 'click' and open the simulated app.")

    # Display total clicks
    st.info(f"Simulated Clicks/App Opens: {st.session_state.total_clicks}")

    # Inject JavaScript for playing beep sound
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

                        oscillator.type = 'sine';
                        oscillator.frequency.setValueAtTime(440, audioContext.currentTime);
                        gainNode.gain.setValueAtTime(0.5, audioContext.currentTime);

                        oscillator.start();
                        oscillator.stop(audioContext.currentTime + 0.1);
                    } catch (e) {
                        console.error("Error playing beep sound:", e);
                    }
                }
            </script>
        """, height=0)
        # Call the JS function if a click was just registered (hacky way to trigger from Python)
        # This will play a sound every time the app opens/closes due to rerun, not ideal,
        # but demonstrates the sound functionality. A more precise trigger would require
        # more complex communication between Python and JS.
        if st.session_state.is_app_open: # Play sound when app opens
             st.components.v1.html("<script>playBeep();</script>", height=0)


    # Webcam Stream with VisionProcessor
    webrtc_streamer(
        key="webcam_stream",
        video_processor_factory=VisionProcessor,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        ),
        async_processing=True, # Process frames asynchronously
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
                background: rgba(0, 0, 0, 0.8);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1000;
            }
            .gallery-content {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
                text-align: center;
                max-width: 80%;
                max-height: 80%;
                overflow-y: auto;
            }
            .gallery-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            .gallery-item img {
                width: 100px;
                height: 100px;
                object-fit: cover;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            </style>
            <div class="gallery-overlay">
                <div class="gallery-content">
                    <h3 style="color: #333; font-size: 2em; margin-bottom: 15px;">Simulated Gallery App</h3>
                    <p style="color: #666; margin-bottom: 20px;">This app opened because you "closed your eyes" (held them closed for 2 seconds).</p>
                    <div class="gallery-grid">
                        <div class="gallery-item"><img src="https://placehold.co/100x100/aabbcc/ffffff?text=Img1" alt="Image 1"></div>
                        <div class="gallery-item"><img src="https://placehold.co/100x100/ccbbaa/ffffff?text=Img2" alt="Image 2"></div>
                        <div class="gallery-item"><img src="https://placehold.co/100x100/aaccbb/ffffff?text=Img3" alt="Image 3"></div>
                        <div class="gallery-item"><img src="https://placehold.co/100x100/bbccaa/ffffff?text=Img4" alt="Image 4"></div>
                        <div class="gallery-item"><img src="https://placehold.co/100x100/ccaaBB/ffffff?text=Img5" alt="Image 5"></div>
                        <div class="gallery-item"><img src="https://placehold.co/100x100/bbAAcc/ffffff?text=Img6" alt="Image 6"></div>
                    </div>
                    <p style="color: #999; font-size: 0.9em; margin-top: 20px;">Open your eyes to close this app.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    if st.button("Back to Settings"):
        st.session_state.current_screen = 'settings'
        st.session_state.is_app_open = False # Ensure app is closed when navigating back
        st.experimental_rerun()

# --- Main App Flow ---
st.sidebar.title("Navigation")
if st.sidebar.button("Permissions"):
    st.session_state.current_screen = 'permissions'
    st.experimental_rerun()
if st.sidebar.button("Face Login"):
    st.session_state.current_screen = 'faceLogin'
    st.experimental_rerun()
if st.sidebar.button("Settings"):
    st.session_state.current_screen = 'settings'
    st.experimental_rerun()
if st.sidebar.button("Cursor Control"):
    st.session_state.current_screen = 'cursorControl'
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.info(f"Current Screen: **{st.session_state.current_screen.replace('C', ' C').replace('F', ' F').title()}**")

# Render the current screen
if st.session_state.current_screen == 'permissions':
    permission_screen()
elif st.session_state.current_screen == 'faceLogin':
    face_login_screen()
elif st.session_state.current_screen == 'settings':
    settings_screen()
elif st.session_state.current_screen == 'cursorControl':
    cursor_control_screen()
