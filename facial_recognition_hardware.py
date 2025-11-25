import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
from gpiozero import LED
import serial

# ============ LOAD ENCODINGS ============
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# ============ CAMERA SETUP ============
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (1920, 1080)}  # keep full HD
    )
)
picam2.start()

# ============ GPIO ============
output = LED(14)

# ============ UART SETUP ============
# Use /dev/serial0 by default on Raspberry Pi (mapped to main UART)
# Make sure you enabled Serial in raspi-config and wiring matches your ESP32 RX/TX.
try:
    uart = serial.Serial(
        port="/dev/serial0",  # adjust if needed, e.g. "/dev/ttyAMA0" or USB like "/dev/ttyUSB0"
        baudrate=115200,
        timeout=0.1
    )
    print("[INFO] UART initialized on /dev/serial0 at 115200 baud")
except Exception as e:
    uart = None
    print(f"[WARN] Could not open UART: {e}")

# ============ GLOBALS & SETTINGS ============
cv_scaler = 4  # downscale factor for recognition (must be integer)

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# Each item:
#   {
#     "name": str,
#     "bbox": (left, top, right, bottom),
#     "center": (cx, cy),
#     "area": int
#   }
face_coords = []

# List of names that will trigger the GPIO pin
authorized_names = ["john", "alice", "bob"]  # CASE-SENSITIVE


def send_faces_over_uart(face_coords_list):
    """
    Send face coordinates to ESP32 over UART.

    Protocol:
      - One line per face:
        FACE:<name>,<cx>,<cy>,<left>,<top>,<right>,<bottom>,<area>\n
      - If no faces: NOFACE\n
    """
    if uart is None or not uart.is_open:
        return  # UART not available

    if not face_coords_list:
        try:
            uart.write(b"NOFACE\n")
        except Exception:
            pass
        return

    for fc in face_coords_list:
        name = fc["name"]
        (left, top, right, bottom) = fc["bbox"]
        (cx, cy) = fc["center"]
        area = fc["area"]

        line = f"FACE:{name},{cx},{cy},{left},{top},{right},{bottom},{area}\n"
        try:
            uart.write(line.encode("utf-8"))
        except Exception:
            # Ignore UART write errors for now
            pass


def process_frame(frame):
    """
    Detect & recognize faces on a single frame.
    Also fills 'face_coords' with bbox + center + area in FULL-RES coordinates.
    Sends coordinates via UART to ESP32.
    """
    global face_locations, face_encodings, face_names, face_coords

    # Resize the frame using cv_scaler to increase performance
    resized_frame = cv2.resize(
        frame, (0, 0), fx=(1 / cv_scaler), fy=(1 / cv_scaler)
    )

    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    # Use the SMALL model for speed instead of 'large'
    face_encodings = face_recognition.face_encodings(
        rgb_resized_frame, face_locations, model="small"
    )

    face_names = []
    authorized_face_detected = False

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                # Check if the detected face is in our authorized list
                if name in authorized_names:
                    authorized_face_detected = True

        face_names.append(name)

    # Control the GPIO pin based on face detection
    if authorized_face_detected:
        output.on()  # Turn on Pin
    else:
        output.off()  # Turn off Pin

    # ====== BUILD FULL-RES FACE COORDINATES FOR ESP32 / SERVOS ======
    face_coords = []  # clear previous frame data

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up to original frame size
        top_full = top * cv_scaler
        right_full = right * cv_scaler
        bottom_full = bottom * cv_scaler
        left_full = left * cv_scaler

        # Clamp values to frame boundaries (optional safety)
        h, w, _ = frame.shape
        top_full = max(0, min(top_full, h - 1))
        bottom_full = max(0, min(bottom_full, h - 1))
        left_full = max(0, min(left_full, w - 1))
        right_full = max(0, min(right_full, w - 1))

        width = right_full - left_full
        height = bottom_full - top_full
        area = max(0, width * height)

        # Center point of the face in full-resolution coordinates
        cx = left_full + width // 2
        cy = top_full + height // 2

        face_coords.append(
            {
                "name": name,
                "bbox": (left_full, top_full, right_full, bottom_full),
                "center": (cx, cy),
                "area": area,
            }
        )

    # Debug print
    if face_coords:
        print("Detected faces:")
        for fc in face_coords:
            print(
                f"  Name: {fc['name']}, "
                f"Center: {fc['center']}, "
                f"BBox: {fc['bbox']}, "
                f"Area: {fc['area']}"
            )
    else:
        print("No faces detected")

    # ====== SEND OVER UART ======
    send_faces_over_uart(face_coords)

    return frame


def draw_results(frame):
    """
    Draw face boxes and labels.
    Nearest face (largest area) gets a different color.
    """
    if not face_coords:
        return frame

    # Find the "nearest" face by largest area
    max_area = max(fc["area"] for fc in face_coords)

    for fc in face_coords:
        name = fc["name"]
        left, top, right, bottom = fc["bbox"]
        area = fc["area"]

        # Choose colors based on distance (area)
        # Nearest face (largest area) -> one color, others -> another
        if area == max_area:
            box_color = (0, 255, 0)      # GREEN for nearest face
            label_color = (0, 255, 0)
        else:
            box_color = (0, 165, 255)    # ORANGE for other faces
            label_color = (0, 165, 255)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)

        # Draw label background
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 8), font, 1.0, (255, 255, 255), 1)

        # Add an indicator if the person is authorized
        if name in authorized_names:
            cv2.putText(
                frame,
                "Authorized",
                (left + 6, bottom + 23),
                font,
                0.6,
                label_color,
                1,
            )

    return frame


def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps


# ============ MAIN LOOP ============
try:
    while True:
        # Capture a frame from camera
        frame = picam2.capture_array()

        # Process the frame: detect, recognize, fill face_coords & send via UART
        processed_frame = process_frame(frame)

        # Draw result boxes with distance-based colors
        display_frame = draw_results(processed_frame)

        # Calculate and update FPS
        current_fps = calculate_fps()

        # Attach FPS counter to the text and boxes
        cv2.putText(
            display_frame,
            f"FPS: {current_fps:.1f}",
            (display_frame.shape[1] - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Display video feed
        cv2.imshow("Video", display_frame)

        # Break the loop and stop the script if 'q' is pressed
        if cv2.waitKey(1) == ord("q"):
            break

finally:
    # Cleanup
    cv2.destroyAllWindows()
    picam2.stop()
    output.off()
    if uart is not None and uart.is_open:
        uart.close()
