import cv2
import numpy as np
import pyautogui
import os
import time
from camera_selector import select_camera
from eye_detector import process_frame
from blink_detector import BlinkDetector

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# ── NEW smoothing system ──────────────────────────────────────────
# Exponential Moving Average (EMA) — much stronger than simple rolling
# average at killing high-frequency jitter while still tracking real
# movement. ALPHA controls responsiveness:
#   lower ALPHA  = smoother but slower to follow real eye movement
#   higher ALPHA = faster but more jitter gets through
EMA_ALPHA      = 0.08      # very smooth — tuned for jitter-prone tracking
DEADZONE_PX    = 18        # cursor ignores movement smaller than this
SENSITIVITY_X  = 3.5
SENSITIVITY_Y  = 4.0

EAR_FREEZE_THRESHOLD = 0.23

# ── Landmark averaging ────────────────────────────────────────────
# Instead of using ONE iris landmark point (which is noisy), we average
# MULTIPLE landmarks around each iris to get a much more stable center.
# MediaPipe gives us 5 points per iris when refine_landmarks=True:
# the center point plus 4 boundary points around it.
LEFT_IRIS_POINTS  = [468, 469, 470, 471, 472]
RIGHT_IRIS_POINTS = [473, 474, 475, 476, 477]

NOSE_TIP    = 4
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454


def get_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def get_iris_center_averaged(landmarks, point_indices, w, h):
    """Average multiple iris boundary points for a much more stable center
    than using a single landmark point."""
    pts = [get_pt(landmarks, idx, w, h) for idx in point_indices]
    return np.mean(pts, axis=0)


def get_head_center(landmarks, w, h):
    nose   = get_pt(landmarks, NOSE_TIP,    w, h)
    lcheek = get_pt(landmarks, LEFT_CHEEK,  w, h)
    rcheek = get_pt(landmarks, RIGHT_CHEEK, w, h)
    return (nose + lcheek + rcheek) / 3.0


def get_iris_relative_to_head(landmarks, w, h):
    try:
        head_center = get_head_center(landmarks, w, h)
        left_iris   = get_iris_center_averaged(landmarks, LEFT_IRIS_POINTS,  w, h)
        right_iris  = get_iris_center_averaged(landmarks, RIGHT_IRIS_POINTS, w, h)
        iris_avg    = (left_iris + right_iris) / 2.0
        relative    = iris_avg - head_center
        return relative, iris_avg, head_center
    except Exception:
        return None, None, None


class CalibrationState:
    CORNERS = [
        ("TOP-LEFT",     (0.05, 0.05)),
        ("TOP-RIGHT",    (0.95, 0.05)),
        ("BOTTOM-RIGHT", (0.95, 0.95)),
        ("BOTTOM-LEFT",  (0.05, 0.95)),
    ]

    def __init__(self):
        self.step           = 0
        self.done           = False
        self.samples        = []
        self.eye_points     = []
        self.scr_points     = []
        self.M              = None
        self._collecting    = False
        self._collect_start = 0.0
        self.COLLECT_TIME   = 2.0

    def current_corner_name(self):
        if self.done or self.step >= len(self.CORNERS):
            return ""
        return self.CORNERS[self.step][0]

    def current_screen_target(self):
        if self.done or self.step >= len(self.CORNERS):
            return None
        rx, ry = self.CORNERS[self.step][1]
        return int(rx * SCREEN_W), int(ry * SCREEN_H)

    def start_collecting(self):
        self._collecting    = True
        self._collect_start = time.time()
        self.samples        = []

    def update(self, rel_iris):
        if self.done or not self._collecting:
            return False
        self.samples.append(rel_iris.copy())
        if time.time() - self._collect_start >= self.COLLECT_TIME and len(self.samples) >= 15:
            # Use MEDIAN instead of mean — robust against outlier jitter frames
            avg = np.median(np.array(self.samples), axis=0)
            scr = self.current_screen_target()
            self.eye_points.append(avg)
            self.scr_points.append(list(scr))
            self._collecting = False
            self.step += 1
            if self.step >= len(self.CORNERS):
                self._build_transform()
                self.done = True
            return True
        return False

    def _build_transform(self):
        try:
            src = np.array(self.eye_points, dtype=np.float32)
            dst = np.array(self.scr_points, dtype=np.float32)
            self.M, _ = cv2.estimateAffine2D(src, dst)
            if self.M is None:
                print("[CAL] WARNING: Transform failed — press R to recalibrate")
            else:
                print("\n[CAL] Calibration complete!")
        except Exception as e:
            print(f"[CAL] ERROR: {e}")
            self.M = None

    def iris_to_screen(self, rel_iris):
        if self.M is None:
            return None
        try:
            pt     = np.array([[[rel_iris[0], rel_iris[1]]]], dtype=np.float32)
            result = cv2.transform(pt, self.M)
            x = float(np.clip(result[0][0][0], 0, SCREEN_W - 1))
            y = float(np.clip(result[0][0][1], 0, SCREEN_H - 1))
            if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
                return None
            return x, y
        except Exception:
            return None


class EyeMouse:
    def __init__(self):
        self.cal             = CalibrationState()
        cx, cy                = pyautogui.position()
        self._mx              = float(cx)
        self._my              = float(cy)

        # EMA smoothed target (this is what actually drives the cursor)
        self._ema_x           = None
        self._ema_y           = None

        self._click_text      = ""
        self._click_time      = 0.0
        self._cal_phase       = "waiting"
        self._phase_start     = time.time()
        self._countdown       = 3
        self._cursor_frozen   = False

        self.blink_detector = BlinkDetector(
            on_double_blink=self._left_click,
            on_triple_blink=self._right_click
        )

    def _left_click(self):
        try:
            pyautogui.click()
            self._click_text = "LEFT CLICK"
            self._click_time = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] LEFT CLICK  @ {pyautogui.position()}")
        except Exception as e:
            print(f"[CLICK] left click failed: {e}")

    def _right_click(self):
        try:
            pyautogui.rightClick()
            self._click_text = "RIGHT CLICK"
            self._click_time = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] RIGHT CLICK @ {pyautogui.position()}")
        except Exception as e:
            print(f"[CLICK] right click failed: {e}")

    def _smooth_move(self, tx, ty):
        if self._cursor_frozen:
            return

        # ── Exponential Moving Average ────────────────────────────
        # new_smoothed = alpha * new_value + (1 - alpha) * old_smoothed
        # This gives much stronger noise rejection than a rolling average
        # because old jitter contributes less and less over time instead
        # of all frames in the window counting equally.
        if self._ema_x is None:
            self._ema_x = tx
            self._ema_y = ty
        else:
            self._ema_x = EMA_ALPHA * tx + (1 - EMA_ALPHA) * self._ema_x
            self._ema_y = EMA_ALPHA * ty + (1 - EMA_ALPHA) * self._ema_y

        # Deadzone on top of EMA — final safety net against residual jitter
        if abs(self._ema_x - self._mx) > DEADZONE_PX or abs(self._ema_y - self._my) > DEADZONE_PX:
            self._mx = self._ema_x
            self._my = self._ema_y
            try:
                pyautogui.moveTo(int(self._mx), int(self._my))
            except Exception:
                pass

    def update(self, data):
        if data is None:
            self._cursor_frozen = False
            return

        lm  = data["landmarks"]
        fw  = data["frame_w"]
        fh  = data["frame_h"]
        ear = data["avg_ear"]

        if ear < EAR_FREEZE_THRESHOLD:
            if not self._cursor_frozen:
                self._cursor_frozen = True
                # Reset EMA so blink-era frames don't pull cursor when unfrozen
                self._ema_x = None
                self._ema_y = None
        else:
            self._cursor_frozen = False

        rel_iris, iris_abs, head_center = get_iris_relative_to_head(lm, fw, fh)
        if rel_iris is None:
            return

        scaled = np.array([rel_iris[0] * SENSITIVITY_X,
                           rel_iris[1] * SENSITIVITY_Y])

        if not self.cal.done:
            now = time.time()
            if self._cal_phase == "waiting":
                if now - self._phase_start >= self._countdown:
                    self.cal.start_collecting()
                    self._cal_phase = "collecting"
            elif self._cal_phase == "collecting":
                corner_done = self.cal.update(scaled)
                if corner_done:
                    if self.cal.done:
                        self._cal_phase = "done"
                    else:
                        self._cal_phase   = "next_corner"
                        self._phase_start = now
            elif self._cal_phase == "next_corner":
                if now - self._phase_start >= 2.0:
                    self.cal.start_collecting()
                    self._cal_phase = "collecting"
            return

        result = self.cal.iris_to_screen(scaled)
        if result:
            self._smooth_move(result[0], result[1])
        self.blink_detector.update(ear)

    def draw_overlay(self, frame, data):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 140), (10, 10, 10), -1)

        if data is None:
            cv2.putText(frame, "No face — move closer",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
            return frame

        lm = data["landmarks"]
        fw = data["frame_w"]
        fh = data["frame_h"]
        rel_iris, iris_abs, head_center = get_iris_relative_to_head(lm, fw, fh)

        if iris_abs is not None:
            dot_color = (0, 100, 255) if self._cursor_frozen else (0, 255, 80)
            cv2.circle(frame, tuple(iris_abs.astype(int)),    6, dot_color,     -1)
            cv2.circle(frame, tuple(head_center.astype(int)), 4, (255, 100, 0), -1)

        if not self.cal.done:
            now    = time.time()
            corner = self.cal.current_corner_name()

            if self._cal_phase == "waiting":
                remaining = max(0, self._countdown - (now - self._phase_start))
                cv2.putText(frame, f"Look at {corner} corner of screen",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(frame, f"Starting in {remaining:.1f}s ...",
                            (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
            elif self._cal_phase == "collecting":
                pct = int(min((time.time() - self.cal._collect_start) / self.cal.COLLECT_TIME * 100, 100))
                cv2.putText(frame, f"HOLD GAZE — {corner}  ({pct}%)",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
                bar_w = int((w - 20) * pct / 100)
                cv2.rectangle(frame, (10, 50), (w - 10, 68), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, 50), (10 + bar_w, 68), (0, 200, 80), -1)
            elif self._cal_phase == "next_corner":
                nxt = self.cal.CORNERS[self.cal.step][0] if self.cal.step < len(self.cal.CORNERS) else "done"
                cv2.putText(frame, f"Done! Now look at: {nxt}",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Corner {self.cal.step + 1} / {len(self.cal.CORNERS)}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
            return frame

        ear_color = (0, 80, 255) if data["eye_closed"] else (0, 220, 80)
        cv2.putText(frame, f"EAR: {data['avg_ear']:.3f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
        if self._cursor_frozen:
            cv2.putText(frame, "FROZEN", (w - 120, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        mx, my = pyautogui.position()
        cv2.putText(frame, f"Cursor: ({mx}, {my})  |  R=recalibrate",
                    (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
        bstatus = self.blink_detector.get_status_text()
        if bstatus:
            color = (0, 255, 255) if "CLICK" in bstatus else (160, 160, 255)
            cv2.putText(frame, bstatus, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        if time.time() - self._click_time < 1.0:
            cv2.putText(frame, self._click_text, (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        cv2.putText(frame, "Double blink=Left  Triple blink=Right  R=Recal  Q=Quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1)
        return frame


def main():
    print("\n" + "="*55)
    print("   EYE MOUSE v3 — by Areeb (Anti-Jitter Edition)")
    print("="*55)
    print("Look at each RED DOT — hold gaze for 2 seconds.")
    print("4 corners: TOP-LEFT -> TOP-RIGHT -> BOTTOM-RIGHT -> BOTTOM-LEFT")
    print()
    print("Double blink=Left click   Triple blink=Right click")
    print("R=Recalibrate   Q=Quit")
    print("="*55 + "\n")

    result = select_camera()
    if result is None:
        return

    cap, kind = result
    app       = EyeMouse()

    cal_screen    = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    prev_cal_step = -1

    cv2.namedWindow("cal_dot", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("cal_dot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cam_win = "Eye Mouse v3 — Q to quit"
    cv2.namedWindow(cam_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cam_win, 480, 360)
    cv2.moveWindow(cam_win, SCREEN_W - 500, SCREEN_H - 420)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Only flip for laptop camera — phone camera via IP webcam
        # is already correctly oriented after rotation in camera_selector
        if kind == "local":
            frame = cv2.flip(frame, 1)

        try:
            data  = process_frame(frame)
            app.update(data)
            frame = app.draw_overlay(frame, data)
        except Exception as e:
            print(f"[FRAME] Error: {e}")
            continue

        if not app.cal.done:
            target = app.cal.current_screen_target()
            if target and app.cal.step != prev_cal_step:
                cal_screen[:] = 15
                cv2.circle(cal_screen, target, 30, (0, 0, 255), -1)
                cv2.circle(cal_screen, target, 34, (255, 255, 255), 3)
                corner_name = app.cal.current_corner_name()
                lx = 50 if target[0] < SCREEN_W // 2 else SCREEN_W - 350
                ly = 80 if target[1] < SCREEN_H // 2 else SCREEN_H - 50
                cv2.putText(cal_screen, f"Look at this dot: {corner_name}",
                            (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                prev_cal_step = app.cal.step
            cv2.imshow("cal_dot", cal_screen)
            # Bring calibration window to front every frame during calibration
            cv2.setWindowProperty("cal_dot", cv2.WND_PROP_TOPMOST, 1)
        else:
            try:
                cv2.destroyWindow("cal_dot")
                cv2.resizeWindow(cam_win, 480, 360)
                cv2.moveWindow(cam_win, 20, 20)
            except Exception:
                pass

        cv2.imshow(cam_win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            app.cal           = CalibrationState()
            app._cal_phase    = "waiting"
            app._phase_start  = time.time()
            app._ema_x        = None
            app._ema_y        = None
            prev_cal_step     = -1
            cal_screen[:]     = 15
            cv2.namedWindow("cal_dot", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("cal_dot", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow(cam_win, 480, 360)
            cv2.moveWindow(cam_win, SCREEN_W - 500, SCREEN_H - 420)
            print("\n[RECAL] Recalibrating...")

    cap.release()
    cv2.destroyAllWindows()
    print("\nEye Mouse closed.")


if __name__ == "__main__":
    main()