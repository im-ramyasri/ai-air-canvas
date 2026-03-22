"""
AI Air Canvas - Improved Version
Fixes:
  - Shaky/jittery drawing → Exponential smoothing on finger position
  - Gesture switching too sensitive → Gesture debounce (hold N frames)

Controls:
  - Index finger only  → Draw
  - Index + Middle     → Erase
  - Open palm          → Clear canvas
  - Move index to top  → Select color
  - Press 'S'          → Save canvas
  - Press 'Q'          → Quit
"""

import cv2
import numpy as np
import time
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from shape_recognizer import recognize_shape

# ─── Model Setup ─────────────────────────────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model (~10MB)... please wait.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!")
    else:
        print("Model ready.")

# ─── Canvas Config ────────────────────────────────────────────────────────────
CANVAS_H, CANVAS_W = 480, 640
BRUSH_THICKNESS    = 6
ERASER_THICKNESS   = 35

# ─── Smoothing Config ─────────────────────────────────────────────────────────
SMOOTH_ALPHA       = 0.35   # 0.1 = very smooth but laggy | 0.5 = responsive but jittery
                             # 0.35 is a good balance

# ─── Gesture Debounce Config ──────────────────────────────────────────────────
GESTURE_HOLD_FRAMES = 4     # gesture must be held this many frames to activate
                             # increase to reduce false switches (try 5-6 if still jumpy)

# ─── Color Palette ───────────────────────────────────────────────────────────
PALETTE = [
    ("Red",    (50,  50,  220)),
    ("Orange", (30,  140, 255)),
    ("Yellow", (30,  220, 220)),
    ("Green",  (40,  200, 80 )),
    ("Cyan",   (200, 200, 40 )),
    ("Blue",   (220, 80,  40 )),
    ("Purple", (180, 60,  180)),
    ("White",  (240, 240, 240)),
]

ERASER_COLOR  = (0, 0, 0)
PALETTE_H     = 72
COLOR_BLOCK_W = (CANVAS_W - 80) // len(PALETTE)


# ─── Smoothing Filter ─────────────────────────────────────────────────────────
class PointSmoother:
    """
    Exponential Moving Average smoother for (x, y) coordinates.
    Reduces jitter while keeping drawing responsive.
    """
    def __init__(self, alpha=SMOOTH_ALPHA):
        self.alpha = alpha
        self.sx    = None
        self.sy    = None

    def update(self, x, y):
        if self.sx is None:
            self.sx, self.sy = float(x), float(y)
        else:
            self.sx = self.alpha * x + (1 - self.alpha) * self.sx
            self.sy = self.alpha * y + (1 - self.alpha) * self.sy
        return int(self.sx), int(self.sy)

    def reset(self):
        self.sx = None
        self.sy = None


# ─── Gesture Debouncer ────────────────────────────────────────────────────────
class GestureDebouncer:
    """
    Requires a gesture to be detected for N consecutive frames
    before registering it as active. Prevents flickering between gestures.
    """
    def __init__(self, hold_frames=GESTURE_HOLD_FRAMES):
        self.hold_frames    = hold_frames
        self.candidate      = None
        self.candidate_count = 0
        self.active         = None

    def update(self, gesture):
        if gesture == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate       = gesture
            self.candidate_count = 1

        if self.candidate_count >= self.hold_frames:
            self.active = self.candidate

        return self.active


# ─── Finger State Detection ───────────────────────────────────────────────────
def get_finger_states(landmarks, handedness="Right"):
    TIPS = [4, 8, 12, 16, 20]
    PIPS = [3, 6, 10, 14, 18]
    fingers = []

    if handedness == "Right":
        fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)
    else:
        fingers.append(1 if landmarks[4].x > landmarks[3].x else 0)

    for tip, pip in zip(TIPS[1:], PIPS[1:]):
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)

    return fingers


def classify_gesture(fingers, iy):
    """Map finger state to a gesture string."""
    if sum(fingers[1:]) >= 4:
        return "CLEAR"
    if iy < PALETTE_H:
        return "COLOR_SELECT"
    if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        return "ERASE"
    if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
        return "DRAW"
    return "IDLE"


# ─── UI Drawing ───────────────────────────────────────────────────────────────
def draw_palette_bar(frame, selected_idx):
    cv2.rectangle(frame, (0, 0), (CANVAS_W, PALETTE_H), (30, 30, 30), -1)
    for i, (name, color) in enumerate(PALETTE):
        x1, x2 = i * COLOR_BLOCK_W, i * COLOR_BLOCK_W + COLOR_BLOCK_W
        cv2.rectangle(frame, (x1+4, 6), (x2-4, PALETTE_H-6), color, -1)
        if i == selected_idx:
            cv2.rectangle(frame, (x1+2, 4), (x2-2, PALETTE_H-4), (255,255,255), 2)
        cv2.putText(frame, name, (x1+6, PALETTE_H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0,0,0), 1, cv2.LINE_AA)
    cx1 = CANVAS_W - 76
    cv2.rectangle(frame, (cx1, 6), (CANVAS_W-4, PALETTE_H-6), (60,60,60), -1)
    cv2.putText(frame, "CLEAR", (cx1+4, PALETTE_H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA)


def get_palette_selection(x):
    if x >= CANVAS_W - 76:
        return -1, True
    return min(x // COLOR_BLOCK_W, len(PALETTE)-1), False


def draw_hud(frame, gesture, color_name, color_bgr, shape_label, shape_timer):
    h = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-55), (CANVAS_W, h), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"Gesture: {gesture}", (10, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Color: {color_name}", (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 1, cv2.LINE_AA)
    if shape_label and time.time() - shape_timer < 3.0:
        label = f"Detected: {shape_label}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(frame, label, ((CANVAS_W-tw)//2, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2, cv2.LINE_AA)


HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 220, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255,255,255), -1)


# ─── Main Application ─────────────────────────────────────────────────────────
def main():
    download_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CANVAS_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    canvas         = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    selected_idx   = 0
    current_color  = PALETTE[0][1]
    current_name   = PALETTE[0][0]
    prev_point     = None
    current_stroke = []
    shape_label    = ""
    shape_timer    = 0.0

    smoother   = PointSmoother(alpha=SMOOTH_ALPHA)
    debouncer  = GestureDebouncer(hold_frames=GESTURE_HOLD_FRAMES)

    print("=" * 50)
    print("  AI Air Canvas  —  Press Q to quit, S to save")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame        = cv2.flip(frame, 1)
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        result       = landmarker.detect_for_video(mp_image, timestamp_ms)

        gesture_text = "No Hand Detected"

        if result.hand_landmarks and result.handedness:
            for hand_lms, hand_info in zip(result.hand_landmarks, result.handedness):
                draw_hand(frame, hand_lms)

                handedness = hand_info[0].display_name
                fingers    = get_finger_states(hand_lms, handedness)

                # Raw fingertip position
                raw_x = int(hand_lms[8].x * CANVAS_W)
                raw_y = int(hand_lms[8].y * CANVAS_H)
                raw_x = max(0, min(raw_x, CANVAS_W - 1))
                raw_y = max(0, min(raw_y, CANVAS_H - 1))

                # ── Smooth the fingertip position ──────────────────────
                sx, sy    = smoother.update(raw_x, raw_y)
                finger_pt = (sx, sy)

                # ── Debounce the gesture ───────────────────────────────
                raw_gesture = classify_gesture(fingers, sy)
                gesture     = debouncer.update(raw_gesture)

                # Fingertip dot
                cv2.circle(frame, finger_pt, 10, current_color, -1)
                cv2.circle(frame, finger_pt, 10, (255,255,255), 1)

                # ── Act on debounced gesture ───────────────────────────
                if gesture == "CLEAR":
                    gesture_text = "CLEAR CANVAS"
                    canvas[:] = 0
                    current_stroke.clear()
                    shape_label = ""
                    prev_point  = None
                    smoother.reset()

                elif gesture == "COLOR_SELECT":
                    gesture_text = "COLOR SELECT"
                    idx, is_clear = get_palette_selection(sx)
                    if is_clear:
                        canvas[:] = 0
                        current_stroke.clear()
                        shape_label = ""
                    else:
                        selected_idx  = idx
                        current_color = PALETTE[idx][1]
                        current_name  = PALETTE[idx][0]
                    prev_point = None
                    smoother.reset()

                elif gesture == "ERASE":
                    gesture_text = "ERASING"
                    if current_stroke and len(current_stroke) > 8:
                        shape_label = recognize_shape(current_stroke)
                        shape_timer = time.time()
                        current_stroke.clear()
                    if prev_point:
                        cv2.line(canvas, prev_point, finger_pt,
                                 ERASER_COLOR, ERASER_THICKNESS)
                    prev_point = finger_pt

                elif gesture == "DRAW":
                    gesture_text = f"DRAWING ({current_name})"
                    if prev_point:
                        cv2.line(canvas, prev_point, finger_pt,
                                 current_color, BRUSH_THICKNESS)
                        current_stroke.append(finger_pt)
                    prev_point = finger_pt

                else:
                    gesture_text = "IDLE"
                    if current_stroke and len(current_stroke) > 8:
                        shape_label = recognize_shape(current_stroke)
                        shape_timer = time.time()
                        current_stroke.clear()
                    prev_point = None
                    smoother.reset()

        else:
            gesture_text = "No Hand Detected"
            if current_stroke and len(current_stroke) > 8:
                shape_label = recognize_shape(current_stroke)
                shape_timer = time.time()
                current_stroke.clear()
            prev_point = None
            smoother.reset()

        # Blend canvas onto camera feed
        mask     = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask  = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg       = cv2.bitwise_and(frame,  frame,  mask=mask_inv)
        fg       = cv2.bitwise_and(canvas, canvas, mask=mask)
        merged   = cv2.add(bg, fg)

        draw_palette_bar(merged, selected_idx)
        draw_hud(merged, gesture_text, current_name, current_color,
                 shape_label, shape_timer)

        cv2.imshow("AI Air Canvas", merged)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            fname = f"canvas_{int(time.time())}.png"
            cv2.imwrite(fname, canvas)
            print(f"Canvas saved -> {fname}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()