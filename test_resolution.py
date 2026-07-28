import cv2
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Common resolutions to test
resolutions = [
    (640,  480,  "640x480  (current)"),
    (1280, 720,  "1280x720 (HD)"),
    (1920, 1080, "1920x1080 (Full HD)"),
    (960,  540,  "960x540"),
    (800,  600,  "800x600"),
]

print("\nTesting your camera's supported resolutions...\n")

cap = cv2.VideoCapture(0)

for w, h, label in resolutions:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    supported = "✓ SUPPORTED" if actual_w == w and actual_h == h else f"✗ fell back to {actual_w}x{actual_h}"
    print(f"  {label:25s}  →  {supported}")

cap.release()
print("\nDone! Tell Areeb which ones say SUPPORTED.")
input("Press Enter to close.")