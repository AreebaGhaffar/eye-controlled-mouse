import cv2
import time

def scan_cameras(max_check=5):
    """Scan and return list of available camera indices."""
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)          # no DSHOW — avoids Windows C++ exception
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available.append(i)
            cap.release()
        time.sleep(0.1)
    return available

def get_camera_label(index):
    """Try to give a human-friendly label for each camera index."""
    # Index 0 is almost always the built-in laptop webcam
    # DroidCam usually appears as index 1 or 2
    if index == 0:
        return "Laptop built-in camera"
    else:
        return f"External / DroidCam camera (index {index})"

def select_camera():
    """
    Scan available cameras, show a menu, let user pick one.
    Returns an open cv2.VideoCapture object ready to use.
    """
    print("\n" + "="*50)
    print("   EYE MOUSE — CAMERA SETUP")
    print("="*50)
    print("\nScanning for available cameras, please wait...\n")

    cameras = scan_cameras()

    if not cameras:
        print("[ERROR] No cameras found!")
        print("  - Make sure your laptop webcam is not blocked.")
        print("  - If using DroidCam, open the app on your phone first,")
        print("    then connect via WiFi or USB before running this script.")
        input("\nPress Enter to exit.")
        return None

    print(f"Found {len(cameras)} camera(s):\n")
    for idx, cam_index in enumerate(cameras):
        label = get_camera_label(cam_index)
        print(f"  [{idx + 1}] {label}")

    print()

    # If only one camera found, auto-select it
    if len(cameras) == 1:
        print(f"Only one camera found. Auto-selecting: {get_camera_label(cameras[0])}")
        choice = 0
    else:
        while True:
            try:
                raw = input(f"Enter your choice (1 to {len(cameras)}): ").strip()
                choice = int(raw) - 1
                if 0 <= choice < len(cameras):
                    break
                else:
                    print(f"  Please enter a number between 1 and {len(cameras)}.")
            except ValueError:
                print("  Invalid input. Please enter a number.")

    selected_index = cameras[choice]
    label = get_camera_label(selected_index)

    print(f"\nOpening: {label} ...")
    cap = cv2.VideoCapture(selected_index)   # no DSHOW — cleaner on Windows

    # Set a decent resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {selected_index}.")
        print("  Try closing other apps that might be using the camera.")
        input("\nPress Enter to exit.")
        return None

    # Quick test — grab a frame to confirm it works
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] Camera opened but could not read a frame.")
        print("  If using DroidCam, make sure the phone app is running")
        print("  and both devices are on the same WiFi network.")
        cap.release()
        input("\nPress Enter to exit.")
        return None

    print(f"\n[OK] Camera ready! Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("="*50 + "\n")

    return cap, selected_index


# ── Quick test — run this file directly to check your cameras ──
if __name__ == "__main__":
    result = select_camera()
    if result is not None:
        cap, index = result
        print("Camera test: showing live feed for 5 seconds...")
        print("Press Q to quit early.\n")
        start = time.time()
        while time.time() - start < 5:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)   # 1 = horizontal flip, fixes mirror
                cv2.imshow("Camera Test - Press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        print("Camera test done. Everything is working!")
    input("\nPress Enter to close.")