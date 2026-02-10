## BHFR3: Bionic Head and Facial Recognition

BHFR3 is a **bionic robotic head** that uses **dual-camera facial detection and recognition** to physically track and follow a person using **four servos** driven by an **ESP32 + PCA9685** servo controller.  
The **Raspberry Pi 5** handles all computer vision, recognition, and target selection, while the **ESP32** performs smooth, calibrated motion control of the neck, eyes, and jaw.

The project demonstrates a full pipeline:

- **Data collection → training facial encodings → real-time recognition**
- **Dual-camera face tracking** with target locking and re-acquisition
- **Smooth servo control** of a robotic head via a simple serial protocol

---

## Features

- **Real-time dual-camera face tracking**
  - Two Raspberry Pi cameras capture 1280×720 video streams.
  - Faces are detected using the `face_recognition` library (HOG model).
  - The system selects and **locks onto the best target**, maintaining identity as the person moves.

- **Face recognition (known vs unknown)**
  - Support for **multiple known people** trained from images stored under `dataset/`.
  - Encodings are stored in `encodings.pickle` and loaded at runtime for recognition.
  - Displays recognized names above faces; unknown faces are labeled as `"Unknown"`.

- **Bionic head motion**
  - **4 × servo channels** (driven from PCA9685 via ESP32):
    - **Neck pan**: head movement left/right.
    - **Eye tilt**: eyes look up/down.
    - **Eye pan**: eyes move left/right.
    - **Jaw**: jaw open/close (mechanically present and calibrated; behavior is extendable in code).
  - **Smooth, rate-limited motion**:
    - Eyes move quickly to track small, fast movements.
    - Head pans more slowly and only when the eyes have turned far enough (to keep tracking natural).

- **Clean ESP32 ⇄ Raspberry Pi serial protocol**
  - Text-based UART at 115200 baud:
    - `CENTER` → recenter head and eyes.
    - `MOVE,nx,ny` → normalized offsets in \([-1, 1]\) for x and y from the image center.
  - Simple to debug and extend.

- **Documented architecture and calibration**
  - Block diagram and flow chart illustrating system design and software flow.
  - Servo calibration figures and reference images of robotic heads.

---

## Hardware Overview

- **Core compute**
  - **Raspberry Pi 5**
    - Runs all Python scripts (vision, recognition, tracking, serial control).
    - Connects to two camera modules and the ESP32 over USB (or UART via adapter).
  - **ESP32**
    - Runs `Final.ino` and controls servos via I²C to the PCA9685.
    - Receives tracking commands from Raspberry Pi over serial.

- **Actuation**
  - **4 × Hobby Servos**
    - **Servo 1 – Neck pan**: rotates the head left/right.
    - **Servo 2 – Eye tilt**: moves the eyes up/down.
    - **Servo 3 – Eye pan**: moves the eyes left/right.
    - **Servo 4 – Jaw**: opens/closes the jaw.
    - Their **calibration values** (pulse-width ranges / mechanical limits) are captured and visualized in the servo calibration figure (`servo_ca;ibration_values.jpeg`).
  - **PCA9685 16‑channel PWM driver**
    - Controlled over I²C by ESP32.
    - Generates stable 50 Hz servo PWM signals for the servos.

- **Sensing**
  - **2 × Raspberry Pi camera modules**
    - Configured via `Picamera2`.
    - Provide a wide field of view for robust tracking and depth perception by perspective.

- **Power & connectivity**
  - Appropriate **regulated 5 V supply** for servos (with sufficient amperage).
  - **Separate power rails** (recommended): one for logic (Pi, ESP32, PCA9685 VCC) and one for servo power, with common ground.
  - USB/UART cable from Raspberry Pi to ESP32.

---

## System Architecture

<p align="center">
  <img src="RaspberryPi%205%20Code/results/Block%20Diagram.jpeg" alt="BHFR3 Block Diagram" width="450" />
</p>

<p align="center">
  <img src="RaspberryPi%205%20Code/results/Flow_chart.jpeg" alt="BHFR3 System Flow Chart" width="450" />
</p>

- **High-level flow**
  - **Step 1 – Camera acquisition**
    - Raspberry Pi captures frames from two cameras in parallel.
  - **Step 2 – Face detection and recognition**
    - Frames are downscaled and passed to `face_recognition` for detection (and encoding when needed).
    - Encodings are compared to `known_encodings` from `encodings.pickle`.
  - **Step 3 – Target selection & locking**
    - The system chooses a **primary target face**:
      - Prefers the previously locked target if still present (within `LOCK_DIST_THRESH`).
      - Otherwise, selects the face closest to the center.
  - **Step 4 – Motion command generation**
    - Computes normalized offsets:
      - \(nx = \frac{cx - W/2}{W/2}\)
      - \(ny = \frac{cy - H/2}{H/2}\)
    - Clamps to \([-1, 1]\) and sends `MOVE,nx,ny` to ESP32 at a limited rate.
  - **Step 5 – Servo actuation**
    - ESP32 updates eye servos first, within a deadzone and step behavior.
    - When eyes are far enough from center, the head servo slowly follows.
    - Jaw servo is calibrated and available for future behaviors (e.g. speech sync).

- **No-face behavior**
  - When no faces are detected for longer than a configurable timeout, the Pi sends `CENTER` once.
  - ESP32 then recenters servos, returning the head to a neutral pose.

---

## Servo Calibration

The servo ranges for neck, eyes, and jaw were empirically tuned for this specific mechanical build.  
The following figure summarizes the calibration values used in the ESP32 firmware:

<p align="center">
  <img src="RaspberryPi%205%20Code/results/servo_ca%3Bibration_values.jpeg" alt="BHFR3 Servo Calibration Values" width="450" />
</p>

## Getting Started

### 1. Raspberry Pi 5 – Software Setup

- **OS and Python**
  - Use a recent **Raspberry Pi OS** with Python 3 (3.9+ recommended).

- **Install system-level dependencies (for `face_recognition` and OpenCV)**
  - On Raspberry Pi OS, you will typically need:

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv cmake build-essential \
                    libopenblas-dev liblapack-dev libatlas-base-dev \
                    libjpeg-dev libpng-dev libtiff-dev \
                    libavcodec-dev libavformat-dev libswscale-dev \
                    libxvidcore-dev libx264-dev \
                    libgtk-3-dev \
                    libboost-all-dev
```

- **Install Python packages**

```bash
pip3 install face_recognition opencv-python numpy imutils pyserial
```

- **Picamera2**
  - Ensure `Picamera2` is installed and configured according to the Raspberry Pi documentation (for newer Bullseye/Bookworm images, `Picamera2` is usually available via apt or preinstalled).

---

### 2. ESP32 – Firmware Setup

- **Required libraries**
  - `Wire` (I²C)
  - `Adafruit PWM Servo Driver` (PCA9685 library)

- **Flashing**
  - Open `ESP32 Code/Final.ino` in the Arduino IDE or PlatformIO.
  - Select the correct ESP32 board and COM port.
  - Install the required libraries via Library Manager.
  - Upload the sketch.

- **Wiring (typical)**
  - ESP32 I²C:
    - `SDA` → PCA9685 `SDA`
    - `SCL` → PCA9685 `SCL`
  - PCA9685 power:
    - `VCC` → 3.3 V or 5 V logic (per board spec).
    - `V+` → 5–6 V servo power (from external supply).
    - **Common GND** between servo supply, PCA9685, ESP32, and Raspberry Pi.
  - Servos:
    - Plug each servo into PCA9685 channels (e.g., 0 = eye tilt, 1 = eye pan, 2 = head pan, 3 = jaw).

---

### 3. Data Collection & Training

- **Step 1 – Capture images**
  - Edit `RaspberryPi 5 Code/image_capture.py`:
    - Set `PERSON_NAME = "<YourName>"`.
  - Run:

```bash
cd "RaspberryPi 5 Code"
python3 image_capture.py
```

  - Controls:
    - **SPACE**: save a frame for the current person.
    - **'q'**: stop capturing.
  - Repeat for each person you want the system to recognize (change `PERSON_NAME` each time).

- **Step 2 – Train encodings**

```bash
cd "RaspberryPi 5 Code"
python3 model_training.py
```

  - This will:
    - Walk the `dataset/` folder.
    - Compute encodings for all detected faces.
    - Save them to `encodings.pickle`.

---

### 4. Running the System

1. **Connect hardware**
   - Power the Raspberry Pi 5, ESP32, PCA9685, and servos.
   - Ensure the ESP32 is connected to the Pi via USB/UART.

2. **Start the ESP32**
   - After boot, the ESP32:
     - Initializes I²C and PCA9685.
     - Recenters the servos (eyes and head).

3. **Start the Raspberry Pi script**

```bash
cd "RaspberryPi 5 Code"
python3 facial_recognition_hardware.py
```

4. **Observe behavior**
   - A window titled **`BHFR3 FINAL`** appears, showing:
     - Left and right camera feeds concatenated.
     - Bounding boxes around detected faces.
     - Names above recognized faces (using `encodings.pickle`).
   - The bionic head will:
     - Track the locked face smoothly.
     - Recenters automatically when no faces are visible for a while.

---

## Serial Protocol Details

- **Connection**
  - Default: `/dev/ttyUSB0` at **115200 baud**, `8N1`.

- **Commands from Raspberry Pi to ESP32**

  - **Center command**

```text
CENTER\n
```

  - **Move command**

```text
MOVE,nx,ny\n
```

  - **Parameters**
    - `nx`: horizontal offset, \([-1.0, 1.0]\)
      - `-1.0` = far left, `0` = center, `1.0` = far right.
    - `ny`: vertical offset, \([-1.0, 1.0]\)
      - `-1.0` = top, `0` = center, `1.0` = bottom.

- **Behavior on ESP32 side**
  - Ignores invalid lines or malformed commands.
  - Uses a **deadzone** around zero to avoid servo jitter.
  - Clamps all servo positions to their calibrated min/max constants.

---

## Visual Inspiration (Reference Head Images)

The following images are **not** photos of BHFR3 itself, but **reference robotic heads from random clips** used for visual inspiration and documentation of the intended look and feel:

<p align="center">
  <img src="RaspberryPi%205%20Code/results/fig1.jpeg" alt="Reference Robotic Head 1" width="450" />
</p>

<p align="center">
  <img src="RaspberryPi%205%20Code/results/fig2.jpeg" alt="Reference Robotic Head 2" width="450" />
</p>

<p align="center">
  <img src="RaspberryPi%205%20Code/results/fig3.jpeg" alt="Reference Robotic Head 3" width="450" />
</p>

<p align="center">
  <img src="RaspberryPi%205%20Code/results/fig4.jpeg" alt="Reference Robotic Head 4" width="450" />
</p>

<p align="center">
  <img src="RaspberryPi%205%20Code/results/fig5.jpeg" alt="Reference Robotic Head 5" width="450" />
</p>

## Extending BHFR3

- **Jaw behavior**
  - Currently calibrated and wired but minimally used in code.
  - Ideas:
    - Animate jaw with a speech synthesizer or audio amplitude.
    - Trigger jaw movement for certain recognized users or emotions.

- **More expressive motion**
  - Add:
    - Idle scanning behavior when no face is detected.
    - Saccadic eye movements and micro-motions.
    - Smooth interpolation profiles (e.g., ease-in/out) instead of constant-steps.

- **Additional sensing**
  - Integrate:
    - Microphones for sound localization.
    - Additional sensors (ultrasonic, IR) for distance estimation.

- **Software improvements**
  - Replace HOG detector with CNN for better accuracy (if performance allows).
  - Add configuration files for thresholds and timings instead of hardcoded constants.

---

## License & Attribution

- **Images in `fig1.jpeg`–`fig5.jpeg`**
  - These are **reference images of robotic heads from random clips**, included only for documentation and design inspiration.
  - They **do not** necessarily depict the physical BHFR3 build itself.

- Please add your preferred license (e.g., MIT, Apache-2.0) here to clarify reuse of the hardware design and code.


