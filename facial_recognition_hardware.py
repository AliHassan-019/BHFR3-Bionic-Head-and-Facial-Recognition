import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
from gpiozero import LED
import serial  # <-- UART

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
UART_PORT = "/dev/ttyAMA0"   # Raspberry Pi 5 UART
UART_BAUD = 115200

print(f"[INFO] Opening UART port {UART_PORT} @ {UART_BAUD} baud")
uart = None
uart_ok = False
try:
    uart = serial.Serial(
        UART_PORT,
        UART_BAUD,
        timeout=0,        # non-blocking read
        write_timeout=0   # non-blocking write
    )
    uart_ok = True
    print("[INFO] UART opened successfully.")
except Exception as e:
    print("[ERROR] Could not open UART:", e)
    uart_ok = False

# Track last TX status for drawing UART logo
last_uart_tx_time = 0.0
last_uart_tx_had_face = False

# ============ GLOBALS & SETTINGS ============
cv_scaler = 4  # downscale factor for recognition (must be integer)

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# This will hold coordinates & info for each detected face in full-resolution coords
# Each item: {
#   "name": str,
#   "bbox": (left, top, right, bottom),
#   "center": (cx, cy),
#   "area": int
# }
face_coords = []

# List of names that will trigger the GPIO pin
authorized_names = ["john", "alice", "bob"]  # CASE-SENSITIVE


def process_frame(frame):
    """
    Detect & recognize faces on a single frame.
    Also fills 'face_coords' with bbox + center + area in FULL-RES coordinates.
    """
    global face_locations, face_encodings, face_names, face_coords

    # Resize the frame using cv_scaler to increase performance
    resized_frame = cv2.resize(
        frame, (0, 0), fx=(1.0 / cv_scaler), fy=(1.0 / cv_scaler)
    )

    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame)

    # Use the SMALL model for speed instead of 'large'
    if face_locations:
        face_encodings = face_recognition.face_encodings(
            rgb_resized_frame, face_locations, model="small"
        )
    else:
        face_encodings = []

    face_names = []
    authorized_face_detected = False

    # Recognize each detected face
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

    if face_locations:
        h, w, _ = frame.shape

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up to original frame size
            top_full = top * cv_scaler
            right_full = right * cv_scaler
            bottom_full = bottom * cv_scaler
            left_full = left * cv_scaler

            # Clamp values to frame boundaries (safety)
            top_full = max(0, min(top_full, h - 1))
            bottom_full = max(0, min(bottom_full, h - 1))
            left_full = max(0, min(left_full, w - 1))
            right_full = max(0, min(right_full, w - 1))

            width = right_full - left_full
            height = bottom_full - top_full
            area = max(0, width * height)

            # Center point of the face in full-resolution coordinates
            cx = left_full + (width // 2)
            cy = top_full + (height // 2)

            face_coords.append(
                {
                    "name": name,
                    "bbox": (left_full, top_full, right_full, bottom_full),
                    "center": (cx, cy),
                    "area": area,
                }
            )

    # If you want verbose debug, uncomment:
    # if face_coords:
    #     print("Detected faces:")
    #     for fc in face_coords:
    #         print(
    #             f"  Name: {fc['name']}, "
    #             f"Center: {fc['center']}, "
    #             f"BBox: {fc['bbox']}, "
    #             f"Area: {fc['area']}"
    #         )

    return frame


def send_first_face_over_uart():
    """
    Send data for the FIRST detected face over UART.

    Format:
      FACE,<name>,<cx>,<cy>,<width>,<height>\\n
    If no face:
      NOFACE\\n
    """
    global last_uart_tx_time, last_uart_tx_had_face

    if not uart_ok or uart is None or not uart.is_open:
        return

    try:
        if face_coords:
            fc = face_coords[0]  # first detected face
            name = fc["name"]
            (cx, cy) = fc["center"]
            (left, top, right, bottom) = fc["bbox"]
            width = right - left
            height = bottom - top

            # Ensure name has no commas to keep CSV simple
            safe_name = str(name).replace(",", "_")

            message = f"FACE,{safe_name},{cx},{cy},{width},{height}\n"
            last_uart_tx_had_face = True
        else:
            message = "NOFACE\n"
            last_uart_tx_had_face = False

        uart.write(message.encode("utf-8"))
        last_uart_tx_time = time.time()
        # Optional debug:
        # print("[UART TX]", message.strip())
    except Exception:
        # Ignore UART exceptions to keep loop running
        pass


def draw_uart_logo(frame):
    """
    Draw a small 'serial communication logo' / indicator on the frame.
    Color encodes UART status + recent TX status:

      - Red   : UART error / not opened
      - Green : UART OK, last TX had face
      - Orange: UART OK, last TX was NOFACE
      - Yellow: UART OK, but no TX recently
    """
    x, y = 40, 40
    radius = 12

    if not uart_ok or uart is None or not uart.is_open:
        color = (0, 0, 255)        # RED -> UART error
        text = "UART ERR"
    else:
        now = time.time()
        if now - last_uart_tx_time < 0.3:  # recent TX in last 300 ms
            if last_uart_tx_had_face:
                color = (0, 255, 0)        # GREEN -> face data sent
            else:
                color = (0, 165, 255)      # ORANGE -> NOFACE sent
        else:
            color = (0, 255, 255)          # YELLOW -> idle
        text = "UART"

    # Circle icon
    cv2.circle(frame, (x, y), radius, color, -1)

    # Text next to icon
    cv2.putText(
        frame,
        text,
        (x + 20, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

    return frame


def draw_results(frame):
    """
    Draw face boxes and labels.
    Nearest face (largest area) gets a different color.
    Also draws UART logo.
    """
    if face_coords:
        # Find the "nearest" face by largest area
        max_area = max(fc["area"] for fc in face_coords)

        for fc in face_coords:
            name = fc["name"]
            left, top, right, bottom = fc["bbox"]
            area = fc["area"]

            # Nearest face (largest area) -> GREEN, others -> ORANGE
            if area == max_area:
                box_color = (0, 255, 0)      # GREEN for nearest face
                label_color = (0, 255, 0)
            else:
                box_color = (0, 165, 255)    # ORANGE for other faces
                label_color = (0, 165, 255)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)

            # Draw label background
            cv2.rectangle(
                frame,
                (left - 3, top - 35),
                (right + 3, top),
                box_color,
                cv2.FILLED,
            )
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

    # Draw UART status/logo on top
    frame = draw_uart_logo(frame)

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
while True:
    # Capture a frame from camera
    frame = picam2.capture_array()

    # Process the frame: detect, recognize, fill face_coords
    processed_frame = process_frame(frame)

    # Send first face data over UART (name + coordinates)
    send_first_face_over_uart()

    # Draw result boxes and UART logo
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

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
output.off()
if uart is not None and uart.is_open:
    uart.close()
