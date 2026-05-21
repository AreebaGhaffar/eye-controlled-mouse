# Eye-Controlled Mouse for Disabled People
A hands-free mouse controlled entirely by eye movements and blinks. No keyboard or mouse needed. Built for people with motor disabilities.
- Look in any direction → cursor moves there
- Double blink → Left Click
- Triple blink → Right Click

---

## Project Files
| File | Purpose |
|------|---------|
| `camera_selector.py` | Detects available cameras and lets user pick laptop cam or DroidCam |
| `eye_detector.py` | Uses MediaPipe to track iris position and measure eye openness (EAR) |
| `blink_detector.py` | Detects double and triple blink patterns and triggers click actions |
| `eye_mouse.py` | Main file — runs calibration, moves cursor, handles the full app loop |

---

## Requirements
- **Python 3.10.11** (must be 3.10, not 3.11 or higher — MediaPipe requires it)

## Installation
```bash
py -3.10 -m venv eye_mouse_env
eye_mouse_env\Scripts\activate
pip install mediapipe==0.10.9
pip install opencv-python==4.8.1.78
pip install numpy==1.26.4
pip install pyautogui==0.9.54
pip install Pillow==10.2.0
pip install pygetwindow==0.0.9
pip install screeninfo==0.8.1
```

## How to 
```bash
eye_mouse_env\Scripts\activate
python eye_mouse.py
```

## Calibration
When the app starts, a dark screen appears with a red dot. Look at each red dot as it moves through 4 corners of the screen. Hold your gaze for 1.5 seconds at each dot. After all 4 corners, the cursor starts following your eyes.

Press **R** to recalibrate anytime. Press **Q** to quit.