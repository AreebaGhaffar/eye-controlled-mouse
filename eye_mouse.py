import cv2
import numpy as np
import pyautogui
import os
import time
from camera_selector import select_camera
from eye_detector import process_frame
from blink_detector import BlinkDetector

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# ── Tuning knobs ─────────────────────────────────────────────────
SMOOTH_FRAMES  = 5      # lower = more responsive, higher = smoother
SENSITIVITY_X  = 3.5   # increase if cursor doesn't reach edges left/right
SENSITIVITY_Y  = 4.0   # increase if cursor doesn't reach edges top/bottom


# ── Head-pose reference landmarks (stable nose/cheek points) ─────
NOSE_TIP       = 4
LEFT_CHEEK     = 234
RIGHT_CHEEK    = 454

LEFT_IRIS      = 468
RIGHT_IRIS     = 473


def get_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def get_head_center(landmarks, w, h):
    nose   = get_pt(landmarks, NOSE_TIP,    w, h)
    lcheek = get_pt(landmarks, LEFT_CHEEK,  w, h)
    rcheek = get_pt(landmarks, RIGHT_CHEEK, w, h)
    return (nose + lcheek + rcheek) / 3.0


def get_iris_relative_to_head(landmarks, w, h):
    head_center = get_head_center(landmarks, w, h)
    left_iris   = get_pt(landmarks, LEFT_IRIS,  w, h)
    right_iris  = get_pt(landmarks, RIGHT_IRIS, w, h)
    iris_avg    = (left_iris + right_iris) / 2.0
    relative    = iris_avg - head_center
    return relative, iris_avg, head_center


class CalibrationState:
    CORNERS = [
        ("TOP-LEFT",     (0.05, 0.05)),
        ("TOP-RIGHT",    (0.95, 0.05)),
        ("BOTTOM-RIGHT", (0.95, 0.95)),
        ("BOTTOM-LEFT",  (0.05, 0.95)),
    ]

    def __init__(self):
        self.step         = 0
        self.done         = False
        self.samples      = []
        self.eye_points   = []
        self.scr_points   = []
        self.M            = None
        self._collecting  = False
        self._collect_start = 0.0
        self.COLLECT_TIME = 1.5

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
        self._collecting     = True
        self._collect_start  = time.time()
        self.samples         = []

    def update(self, rel_iris):
        if self.done or not self._collecting:
            return False
        self.samples.append(rel_iris.copy())
        if time.time() - self._collect_start >= self.COLLECT_TIME and len(self.samples) >= 10:
            avg = np.mean(self.samples, axis=0)
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
        src = np.array(self.eye_points, dtype=np.float32)
        dst = np.array(self.scr_points, dtype=np.float32)
        self.M, _ = cv2.estimateAffine2D(src, dst)
        print("\n[CAL] Calibration complete!")

    def iris_to_screen(self, rel_iris):
        if self.M is None:
            return None
        pt     = np.array([[[rel_iris[0], rel_iris[1]]]], dtype=np.float32)
        result = cv2.transform(pt, self.M)
        x = float(np.clip(result[0][0][0], 0, SCREEN_W - 1))
        y = float(np.clip(result[0][0][1], 0, SCREEN_H - 1))
        return x, y


class EyeMouse:
    def __init__(self):
        self.cal = CalibrationState()
        self._sx = []
        self._sy = []
        cx, cy   = pyautogui.position()
        self._mx = float(cx)
        self._my = float(cy)
        self._click_text = ""
        self._click_time = 0.0
        self._cal_phase   = "waiting"
        self._phase_start = time.time()
        self._countdown   = 3

        self.blink_detector = BlinkDetector(
            on_double_blink=self._left_click,
            on_triple_blink=self._right_click
        )

    def _left_click(self):
        pyautogui.click()
        self._click_text = "LEFT CLICK"
        self._click_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] LEFT CLICK  @ {pyautogui.position()}")

    def _right_click(self):
        pyautogui.rightClick()
        self._click_text = "RIGHT CLICK"
        self._click_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] RIGHT CLICK @ {pyautogui.position()}")

    def _smooth_move(self, tx, ty):
        self._sx.append(tx)
        self._sy.append(ty)
        if len(self._sx) > SMOOTH_FRAMES:
            self._sx.pop(0)
            self._sy.pop(0)
        sx = float(np.mean(self._sx))
        sy = float(np.mean(self._sy))
        if abs(sx - self._mx) > 2 or abs(sy - self._my) > 2:
            self._mx = sx
            self._my = sy
            pyautogui.moveTo(int(self._mx), int(self._my))

    def update(self, data):
        if data is None:
            return
        lm = data["landmarks"]
        fw = data["frame_w"]
        fh = data["frame_h"]
        rel_iris, iris_abs, head_center = get_iris_relative_to_head(lm, fw, fh)
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
                if now - self._phase_start >= 1.5:
                    self.cal.start_collecting()
                    self._cal_phase = "collecting"
            return

        result = self.cal.iris_to_screen(scaled)
        if result:
            self._smooth_move(result[0], result[1])
        self.blink_detector.update(data["avg_ear"])

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

        cv2.circle(frame, tuple(iris_abs.astype(int)),    6, (0, 255, 80),  -1)
        cv2.circle(frame, tuple(head_center.astype(int)), 4, (255, 100, 0), -1)

        if not self.cal.done:
            now    = time.time()
            corner = self.cal.current_corner_name()
            target = self.cal.current_screen_target()

            if self._cal_phase == "waiting":
                remaining = max(0, self._countdown - (now - self._phase_start))
                cv2.putText(frame, f"CALIBRATION — Look at {corner} corner",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(frame, f"Starting in {remaining:.1f}s  |  Look at RED DOT on screen",
                            (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

            elif self._cal_phase == "collecting":
                pct = int(min((time.time() - self.cal._collect_start) / self.cal.COLLECT_TIME * 100, 100))
                cv2.putText(frame, f"HOLD GAZE at {corner}  ({pct}%)",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
                bar_w = int((w - 20) * pct / 100)
                cv2.rectangle(frame, (10, 50), (w - 10, 68), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, 50), (10 + bar_w, 68), (0, 200, 80), -1)

            elif self._cal_phase == "next_corner":
                nxt = self.cal.CORNERS[self.cal.step][0] if self.cal.step < 4 else "done"
                cv2.putText(frame, f"Good! Now look at: {nxt}",
                            (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if target:
                tx_cam = int(target[0] / SCREEN_W * w)
                ty_cam = int(target[1] / SCREEN_H * h)
                cv2.drawMarker(frame, (tx_cam, ty_cam), (0, 0, 255),
                               cv2.MARKER_CROSS, 30, 2)

            cv2.putText(frame, f"Step {self.cal.step + 1} / {len(self.cal.CORNERS)}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
            return frame

        # Normal operation
        ear_color = (0, 80, 255) if data["eye_closed"] else (0, 220, 80)
        cv2.putText(frame, f"EAR: {data['avg_ear']:.3f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
        mx, my = pyautogui.position()
        cv2.putText(frame, f"Cursor: ({mx}, {my})  |  R = recalibrate",
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
    print("   EYE MOUSE v2 — by Areeb")
    print("="*55)
    print("CALIBRATION STEPS:")
    print("  A full-screen dark window will appear with a RED DOT.")
    print("  Look at each red dot — hold your gaze for 1.5 seconds.")
    print("  It goes through 4 corners: TL → TR → BR → BL")
    print()
    print("After calibration:")
    print("  Eyes move  →  Cursor follows")
    print("  Double blink  →  Left click")
    print("  Triple blink  →  Right click")
    print("  R key  →  Recalibrate anytime")
    print("  Mouse to TOP-LEFT  →  Emergency stop")
    print("  Q  →  Quit")
    print("="*55 + "\n")

    result = select_camera()
    if result is None:
        return

    cap, _ = result
    app    = EyeMouse()

    cal_screen    = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    prev_cal_step = -1

    cv2.namedWindow("cal_dot", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("cal_dot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Lost camera feed.")
            break

        frame = cv2.flip(frame, 1)
        data  = process_frame(frame)

        app.update(data)
        frame = app.draw_overlay(frame, data)

        # Calibration dot window
        if not app.cal.done:
            target = app.cal.current_screen_target()
            if target and app.cal.step != prev_cal_step:
                cal_screen[:] = 20
                cv2.circle(cal_screen, target, 22, (0, 0, 255), -1)
                cv2.circle(cal_screen, target, 25, (255, 255, 255), 2)
                lbl = app.cal.current_corner_name()
                cv2.putText(cal_screen, f"Look here: {lbl}",
                            (target[0] - 90, target[1] + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                prev_cal_step = app.cal.step
            cv2.imshow("cal_dot", cal_screen)
        else:
            try:
                cv2.destroyWindow("cal_dot")
            except:
                pass

        cv2.imshow("Eye Mouse v2 — Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            app.cal           = CalibrationState()
            app._cal_phase    = "waiting"
            app._phase_start  = time.time()
            prev_cal_step     = -1
            cal_screen[:]     = 20
            cv2.namedWindow("cal_dot", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("cal_dot", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            print("\n[RECAL] Recalibrating...")

    cap.release()
    cv2.destroyAllWindows()
    print("\nEye Mouse closed.")


if __name__ == "__main__":
    main()