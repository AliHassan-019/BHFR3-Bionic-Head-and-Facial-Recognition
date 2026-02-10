#!/usr/bin/env python3
import cv2
import face_recognition
import numpy as np
from picamera2 import Picamera2
import threading, time, math, serial, pickle

# ===== CONFIG =====
FRAME_W, FRAME_H = 1280, 720
CV_SCALER = 4
DETECT_EVERY_N = 3
FACE_THRESH = 0.5

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200
UART_RATE = 0.02
NO_FACE_TIMEOUT = 5.0

LOCK_DIST_THRESH = 120   # pixels (robust lock radius)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ===== LOAD ENCODINGS =====
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)
known_enc = data["encodings"]
known_names = data["names"]

# ===== SERIAL =====
ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.2)
time.sleep(2)
ser.write(b"CENTER\n")

last_uart = 0
def send_move(nx, ny):
    global last_uart
    if time.time() - last_uart < UART_RATE:
        return
    ser.write(f"MOVE,{nx:.3f},{ny:.3f}\n".encode())
    last_uart = time.time()

# ===== CAMERAS =====
def init_cam(idx):
    cam = Picamera2(camera_num=idx)
    cam.configure(cam.create_preview_configuration(
        main={"format":"BGR888","size":(FRAME_W,FRAME_H)}
    ))
    cam.start()
    return cam

cam0 = init_cam(0)
cam1 = init_cam(1)

frame0 = frame1 = None
lock = threading.Lock()

def grab(cam, idx):
    global frame0, frame1
    while True:
        img = cam.capture_array()
        with lock:
            if idx == 0: frame0 = img
            else: frame1 = img

threading.Thread(target=grab,args=(cam0,0),daemon=True).start()
threading.Thread(target=grab,args=(cam1,1),daemon=True).start()

# ===== FACE PIPELINE =====
def detect(frame):
    small = cv2.resize(frame,None,fx=1/CV_SCALER,fy=1/CV_SCALER)
    rgb = cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb,model="hog")
    encs = face_recognition.face_encodings(rgb,locs)

    faces=[]
    for (t,r,b,l),enc in zip(locs,encs):
        name="Unknown"
        if known_enc:
            d=face_recognition.face_distance(known_enc,enc)
            i=np.argmin(d)
            if d[i]<FACE_THRESH:
                name=known_names[i]

        l*=CV_SCALER; t*=CV_SCALER; r*=CV_SCALER; b*=CV_SCALER
        cx=(l+r)/2; cy=(t+b)/2
        dist=math.hypot(cx-FRAME_W/2,cy-FRAME_H/2)
        faces.append((l,t,r,b,cx,cy,dist,name))
    return faces

# ===== STATE =====
faces0 = faces1 = []
locked_face = None      # (cx, cy)
locked_idx = None
last_face_time = time.time()
center_sent = False
fid = 0

print("[PI] BHFR3 FINAL RUNNING")

# ===== MAIN LOOP =====
while True:
    with lock:
        i0 = frame0.copy() if frame0 is not None else None
        i1 = frame1.copy() if frame1 is not None else None
    if i0 is None or i1 is None:
        continue

    if fid % DETECT_EVERY_N == 0:
        faces0 = detect(i0)
        faces1 = detect(i1)

    all_faces = faces0 + faces1

    tracked_idx = None

    if all_faces:
        last_face_time = time.time()
        center_sent = False

        # ----- LOCKED TARGET SEARCH -----
        if locked_face is not None:
            lx, ly = locked_face

            dists = [
                math.hypot(f[4]-lx, f[5]-ly)
                for f in all_faces
            ]

            min_d = min(dists)
            if min_d < LOCK_DIST_THRESH:
                tracked_idx = dists.index(min_d)
            else:
                # Locked person disappeared
                locked_face = None
                locked_idx = None

        # ----- ACQUIRE NEW LOCK -----
        if locked_face is None:
            tracked_idx = min(
                range(len(all_faces)),
                key=lambda i: all_faces[i][6]
            )

        # ----- TRACK -----
        f = all_faces[tracked_idx]
        l,t,r,b,cx,cy,_,name = f
        locked_face = (cx, cy)
        locked_idx = tracked_idx

        nx = (cx - FRAME_W/2)/(FRAME_W/2)
        ny = (cy - FRAME_H/2)/(FRAME_H/2)
        nx = max(-1,min(1,nx))
        ny = max(-1,min(1,ny))

        send_move(nx, ny)

    else:
        locked_face = None
        locked_idx = None
        if time.time() - last_face_time > NO_FACE_TIMEOUT and not center_sent:
            ser.write(b"CENTER\n")
            center_sent = True

    # ----- DRAW -----
    for img,faces,offset in [(i0,faces0,0),(i1,faces1,len(faces0))]:
        for i,f in enumerate(faces):
            l,t,r,b,_,_,_,name = f
            gidx = i + offset
            color = (0,0,255) if gidx == locked_idx else (0,255,0)
            cv2.rectangle(img,(l,t),(r,b),color,2)
            cv2.putText(img,name,(l,t-8),FONT,0.6,color,2)

    cv2.imshow("BHFR3 FINAL", np.hstack((i0,i1)))
    fid += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam0.stop()
cam1.stop()
ser.close()
cv2.destroyAllWindows()