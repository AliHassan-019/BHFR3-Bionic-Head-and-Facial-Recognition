#!/usr/bin/env python3
import cv2
import face_recognition
import numpy as np
from picamera2 import Picamera2
import threading
import time
import math
import serial

# ================= CONFIG =================
FRAME_W, FRAME_H = 1280, 720
FRAME_SIZE = (FRAME_W, FRAME_H)

CV_SCALER = 4
DETECT_EVERY_N = 3

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
UART_RATE = 0.05  # 20 Hz

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ================= SERIAL (CDC) =================
ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.2)
time.sleep(2)
print("[PI] CDC connected")

# Guaranteed centering
ser.write(b"CENTER\n")
print("[PI -> ESP32] CENTER")

last_uart = 0
def send_move(nx, ny):
    global last_uart
    now = time.time()
    if now - last_uart < UART_RATE:
        return
    msg = f"MOVE,{nx:.3f},{ny:.3f}\n"
    ser.write(msg.encode())
    print("[PI -> ESP32]", msg.strip())
    last_uart = now

# ================= CAMERAS =================
def init_camera(idx):
    cam = Picamera2(camera_num=idx)
    cam.configure(cam.create_preview_configuration(
        main={"format": "BGR888", "size": FRAME_SIZE}
    ))
    cam.start()
    return cam

cam0 = init_camera(0)
cam1 = init_camera(1)

frame0, frame1 = None, None
lock = threading.Lock()

def grab(cam, idx):
    global frame0, frame1
    while True:
        img = cam.capture_array()
        with lock:
            if idx == 0:
                frame0 = img
            else:
                frame1 = img

threading.Thread(target=grab, args=(cam0,0), daemon=True).start()
threading.Thread(target=grab, args=(cam1,1), daemon=True).start()

# ================= FACE DETECTION =================
def detect_faces(frame):
    small = cv2.resize(frame, None, fx=1/CV_SCALER, fy=1/CV_SCALER)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locs = face_recognition.face_locations(rgb, model="hog")
    faces = []

    for (t, r, b, l) in locs:
        l*=CV_SCALER; t*=CV_SCALER; r*=CV_SCALER; b*=CV_SCALER
        cx = (l + r) / 2
        cy = (t + b) / 2
        dist = math.hypot(cx - FRAME_W/2, cy - FRAME_H/2)
        faces.append((l, t, r, b, cx, cy, dist))

    return faces

# ================= MAIN LOOP =================
faces0, faces1 = [], []
frame_id = 0

print("[PI] Dual-camera face tracking running (q to quit)")

try:
    while True:
        with lock:
            img0 = frame0.copy() if frame0 is not None else None
            img1 = frame1.copy() if frame1 is not None else None

        if img0 is None or img1 is None:
            continue

        if frame_id % DETECT_EVERY_N == 0:
            faces0 = detect_faces(img0)
            faces1 = detect_faces(img1)

        all_faces = [(0,f) for f in faces0] + [(1,f) for f in faces1]

        target = None
        if all_faces:
            cam_idx, f = min(all_faces, key=lambda x: x[1][6])
            l,t,r,b,cx,cy,_ = f

            nx = (cx - FRAME_W/2) / (FRAME_W/2)
            ny = (cy - FRAME_H/2) / (FRAME_H/2)

            nx = max(-1.0, min(1.0, nx))
            ny = max(-1.0, min(1.0, ny))

            send_move(nx, ny)

        # ---------- DRAW ----------
        for img, faces in [(img0, faces0), (img1, faces1)]:
            for f in faces:
                l,t,r,b,_,_,_ = f
                cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2)

        combined = np.hstack((img0, img1))
        cv2.imshow("BHFR3 | Dual Camera Tracking", combined)

        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam0.stop()
    cam1.stop()
    ser.close()
    cv2.destroyAllWindows()
    print("[PI] Shutdown complete")