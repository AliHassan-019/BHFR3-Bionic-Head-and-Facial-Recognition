#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define PCA9685_ADDR 0x40
#define SERVO_FREQ   50

Adafruit_PWMServoDriver pwm(PCA9685_ADDR);

// ===== CALIBRATION =====
#define EYE_UD_MIN 120
#define EYE_UD_CTR 170
#define EYE_UD_MAX 220

#define EYE_LR_MIN 210
#define EYE_LR_CTR 265
#define EYE_LR_MAX 340

#define HEAD_MIN   300
#define HEAD_CTR   350
#define HEAD_MAX   430

// ===== CONTROL (🔥 SPEED INCREASED ONLY) =====
#define STEP_EYE        4     // was 2
#define STEP_HEAD       2     // was 1
#define DEADZONE        0.10
#define EYE_UPDATE_MS   20    // was 40
#define HEAD_UPDATE_MS  60    // was 120
#define HEAD_EYE_OFFSET 18

// ===== STATE =====
int eye_ud  = EYE_UD_CTR;
int eye_lr  = EYE_LR_CTR;
int head_lr = HEAD_CTR;

unsigned long last_eye_ms  = 0;
unsigned long last_head_ms = 0;

// ===== HELPERS =====
void apply_servos() {
  pwm.setPWM(0, 0, eye_ud);
  pwm.setPWM(1, 0, eye_lr);
  pwm.setPWM(2, 0, head_lr);
}

void center_servos() {
  eye_ud  = EYE_UD_CTR;
  eye_lr  = EYE_LR_CTR;
  head_lr = HEAD_CTR;
  apply_servos();
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);

  pwm.begin();
  pwm.setPWMFreq(SERVO_FREQ);
  delay(500);

  unsigned long t0 = millis();
  while (millis() - t0 < 2000) {
    center_servos();
    delay(200);
  }

  Serial.println("[ESP32] BHFR3 FINAL DEBUG READY");
}

// ===== LOOP =====
void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();

  if (line == "CENTER") {
    center_servos();
    return;
  }

  if (!line.startsWith("MOVE")) return;

  float nx = 0, ny = 0;
  if (sscanf(line.c_str(), "MOVE,%f,%f", &nx, &ny) != 2) return;

  unsigned long now = millis();

  // ----- EYES -----
  if (now - last_eye_ms >= EYE_UPDATE_MS) {
    last_eye_ms = now;

    if (abs(nx) > DEADZONE)
      eye_lr += (nx > 0) ? STEP_EYE : -STEP_EYE;

    if (abs(ny) > DEADZONE)
      eye_ud += (ny > 0) ? STEP_EYE : -STEP_EYE;

    eye_lr = constrain(eye_lr, EYE_LR_MIN, EYE_LR_MAX);
    eye_ud = constrain(eye_ud, EYE_UD_MIN, EYE_UD_MAX);
  }

  // ----- NECK -----
  int eye_offset = abs(eye_lr - EYE_LR_CTR);
  if (eye_offset > HEAD_EYE_OFFSET &&
      now - last_head_ms >= HEAD_UPDATE_MS) {

    last_head_ms = now;

    if (eye_lr > EYE_LR_CTR)
      head_lr -= STEP_HEAD;
    else
      head_lr += STEP_HEAD;

    head_lr = constrain(head_lr, HEAD_MIN, HEAD_MAX);
  }

  apply_servos();
}