import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
from gpiozero import LED
import serial

# ============ SETTINGS ============
CV_SCALER = 4                     # downscale factor for recognition
UART_PORTS = ["/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"]
UART_BAUD = 115200                # MUST match ESP32
UART_WRITE_TIMEOUT = 0.01         # seconds (non-blocking-ish)
SEND_EVERY_N_FRAMES = 1           # 1 = every frame, 2 = every 2nd frame, ...

AUTHORIZED_NAMES = ["john", "alice", "bob"]  # CASE-SENSITIVE

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
        main={"format": "XRGB8888", "size": (1920, 1080)}
    )
)
picam2.start()

frame_width = 1920
frame_height = 1080

# ============ GPIO ============
output = LED(14)

# ============ UART INIT ============
uart = None
for port in UART_PORTS:
    try:
        uart = serial.Serial(
            port=port,
            baudrate=UART_BAUD,
            timeout=0,
            write_timeout=UART_WRITE_TIMEOUT
        )
        print(f"[INFO] UART opened on {port} at {UART_BAUD} baud")
        break
    except Exception as e:
        print(f"[WARN] Could not open {port}: {e}")

if uart is None:
    print("[ERROR] No UART port could be opened. Running without UART.")

# ============ GLOBALS ============
face_coords = []  # list of dicts as before
frame_count = 0
fps = 0.0
start_time = time.time()


def send_face_over_uart(face_list):
    """
    Send nearest face (largest area) over UART.
    Format:
        FACE:<name>,<cx>,<cy>,<left>,<top>,<right>,<bottom>,<area>\n
    If no face:
        NOFACE\n
    """
    if uart is None or not uart.is_open:
        return

    try:
        if not face_list:
            uart.write(b"NOFACE\n")
            return

        # pick nearest face (largest area)
        nearest = max(face_list, key=lambda fc: fc["area"])
        name = nearest["name"]
        (left, top, right, bottom) = nearest["bbox"]
        (cx, cy) = nearest["center"]
        area = nearest["area"]

        line = f"FACE:{name},{cx},{cy},{left},{top},{right},{bottom},{area}\n"
        uart.write(line.encode("utf-8"))

    except Exception as e:
        # Don't spam; just one line each time something goes wrong
        print(f"[UART ERROR] {e}")


def process_frame(frame, do_send_uart=True):
    """
    Detect & recognize faces on a single frame.
    Updates global 'face_coords' with full-res bbox, center, area.
    Optionally sends nearest face over UART.
    """
    global face_coords

    # Downscale for faster processing
    resized_frame = cv2.resize(
        frame, (0, 0), fx=(1 / CV_SCALER), fy=(1 / CV_SCALER)
    )

    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_resized_frame, face_locations, model="small"
    )

    face_names = []
    authorized_face_detected = False

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name in AUTHORIZED_NAMES:
                    authorized_face_detected = True

        face_names.append(name)

    # GPIO control
    if authorized_face_detected:
        output.on()
    else:
        output.off()

    # Build full-res coordinates
    face_coords = []
    h, w, _ = frame.shape
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top_full = max(0, min(top * CV_SCALER, h - 1))
        bottom_full = max(0, min(bottom * CV_SCALER, h - 1))
        left_full = max(0, min(left * CV_SCALER, w - 1))
        right_full = max(0, min(right * CV_SCALER, w - 1))

        width = right_full - left_full
        height = bottom_full - top_full
        area = max(0, width * height)

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

    # UART TX (only nearest face)
    if do_send_uart:
        send_face_over_uart(face_coords)

    return frame


def draw_results(frame):
    if not face_coords:
        return frame

    max_area = max(fc["area"] for fc in face_coords)

    for fc in face_coords:
        name = fc["name"]
        left, top, right, bottom = fc["bbox"]
        area = fc["area"]

        if area == max_area:
            box_color = (0, 255, 0)   # nearest = green
            label_color = (0, 255, 0)
        else:
            box_color = (0, 165, 255) # others = orange
            label_color = (0, 165, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
        cv2.rectangle(
            frame, (left - 3, top - 35), (right + 3, top), box_color, cv2.FILLED
        )
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 8), font, 1.0, (255, 255, 255), 1)

        if name in AUTHORIZED_NAMES:
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
    elapsed = time.time() - start_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
    return fps


# ============ MAIN LOOP ============
try:
    while True:
        frame = picam2.capture_array()

        # To keep CPU high, avoid unnecessary work:
        # - detection & UART each frame
        # - sending is light (very short line)
        do_send = (frame_count % SEND_EVERY_N_FRAMES == 0)

        processed = process_frame(frame, do_send_uart=do_send)
        display = draw_results(processed)

        current_fps = calculate_fps()
        cv2.putText(
            display,
            f"FPS: {current_fps:.1f}",
            (display.shape[1] - 220, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Video", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    output.off()
    if uart is not None and uart.is_open:
        uart.close()
    print("[INFO] Clean exit")
