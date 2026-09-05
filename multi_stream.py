import cv2
import time
import threading
import numpy as np
from ultralytics import YOLO

# ----------------- CONFIGURATION -----------------
# Axis F9114 Main Unit IP & RTSP stream paths
AXIS_IP = "10.42.0.178"
RTSP_USER = "root"
RTSP_PASS = "pass"  # Update with your Axis authentication credentials

# Stream URLs for 4 camera channels (Axis F2105-RE sensor heads)
CAMERA_CHANNELS = {
    1: f"rtsp://{RTSP_USER}:{RTSP_PASS}@{AXIS_IP}:554/axis-media/media.amp?camera=1&resolution=1920x1080&fps=30",
    2: f"rtsp://{RTSP_USER}:{RTSP_PASS}@{AXIS_IP}:554/axis-media/media.amp?camera=2&resolution=1920x1080&fps=30",
    3: f"rtsp://{RTSP_USER}:{RTSP_PASS}@{AXIS_IP}:554/axis-media/media.amp?camera=3&resolution=1920x1080&fps=30",
    4: f"rtsp://{RTSP_USER}:{RTSP_PASS}@{AXIS_IP}:554/axis-media/media.amp?camera=4&resolution=1920x1080&fps=30",
}

# COCO Class IDs of interest: 0: person, 14: bird, 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow, 21: bear
TARGET_CLASSES = [0, 14, 15, 16, 17, 18, 19, 21]
GRID_WIDTH, GRID_HEIGHT = 1920, 1080
HALF_W, HALF_H = GRID_WIDTH // 2, GRID_HEIGHT // 2

# ----------------- RTSP THREADED CAPTURE -----------------
class VideoStreamThread:
    """Non-blocking RTSP stream reader to avoid Jetson buffer backlog."""
    def __init__(self, ch_id, rtsp_url):
        self.ch_id = ch_id
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(1)
                self.cap.open(self.rtsp_url, cv2.CAP_FFMPEG)
                continue
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

# ----------------- MAIN PROCESSING PIPELINE -----------------
def main():
    print("[INIT] Loading YOLO model (TensorRT engine or PyTorch weights)...")
    # Use 'yolov8n.engine' if compiled with TensorRT for max Orin NX FPS
    model = YOLO("yolov8n.pt")

    print("[INIT] Starting Axis camera capture threads...")
    streams = {ch: VideoStreamThread(ch, url) for ch, url in CAMERA_CHANNELS.items()}

    cv2.namedWindow("Axis Quad Stream - Security HUD", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Axis Quad Stream - Security HUD", GRID_WIDTH, GRID_HEIGHT)

    zoomed_channel = None

    try:
        while True:
            frames = {}
            alerts = {ch: False for ch in CAMERA_CHANNELS}

            for ch, stream in streams.items():
                ret, frame = stream.read()
                if not ret or frame is None:
                    # Render placeholder if stream drops
                    blank = np.zeros((HALF_H, HALF_W, 3), dtype=np.uint8)
                    cv2.putText(blank, f"CAM {ch}: CONNECTING...", (40, HALF_H // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    frames[ch] = blank
                    continue

                # Run inference on native 1080p frame
                results = model(frame, classes=TARGET_CLASSES, conf=0.45, verbose=False)

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        label = f"{model.names[cls_id]} {conf:.2f}"
                        # Bounding box & label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        alerts[ch] = True

                # Overlay Channel Identifier
                cv2.putText(frame, f"CAM {ch} [1080p]", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                frames[ch] = frame

            # Auto-zoom on detection or return to 2x2 grid
            active_alerts = [ch for ch, detected in alerts.items() if detected]
            if active_alerts:
                zoomed_channel = active_alerts[0]  # Focus on first active alert channel
            else:
                zoomed_channel = None

            if zoomed_channel and frames[zoomed_channel].shape[0] == 1080:
                display_canvas = cv2.resize(frames[zoomed_channel], (GRID_WIDTH, GRID_HEIGHT))
                cv2.putText(display_canvas, f"ALERT: CAM {zoomed_channel} ACTIVE", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                # Build 2x2 Matrix
                top_row = np.hstack([
                    cv2.resize(frames[1], (HALF_W, HALF_H)),
                    cv2.resize(frames[2], (HALF_W, HALF_H))
                ])
                bottom_row = np.hstack([
                    cv2.resize(frames[3], (HALF_W, HALF_H)),
                    cv2.resize(frames[4], (HALF_W, HALF_H))
                ])
                display_canvas = np.vstack([top_row, bottom_row])

            cv2.imshow("Axis Quad Stream - Security HUD", display_canvas)

            # Keyboard controls: 'q' quit, '1'-'4' manual zoom toggle, '0' reset grid
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                zoomed_channel = int(chr(key))
            elif key == ord('0'):
                zoomed_channel = None

    finally:
        print("[CLEANUP] Stopping capture threads and closing windows...")
        for stream in streams.values():
            stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
