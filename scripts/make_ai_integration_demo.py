from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
from ultralytics import YOLO

INPUT_VIDEO = Path("assets_raw/ai-source.mp4")
TEMP_VIDEO = Path("public/ai-integration-demo-temp.mp4")
OUTPUT_VIDEO = Path("public/ai-integration-demo.mp4")

MAX_SECONDS = 10
OUTPUT_WIDTH = 960
FPS_LIMIT = 24
CONF_THRES = 0.18

MODEL_NAME = "yolov8n.pt"

# COCO classes commonly useful for this demo:
# 0 person, 39 bottle, 41 cup, 62 chair, 63 couch, 64 potted plant,
# 66 dining table, 67 toilet, 72 tv, 73 laptop, 74 mouse, 75 remote,
# 76 keyboard, 77 cell phone, 84 book, etc.
TARGET_CLASSES = None  # Use None to allow all YOLO classes


def resize_keep_aspect(frame, width):
    h, w = frame.shape[:2]
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def draw_text_box(frame, text, x, y, color=(0, 255, 255), scale=0.45):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 12, y + 6), (5, 10, 25), -1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 12, y + 6), color, 1)
    cv2.putText(frame, text, (x + 6, y - 4), font, scale, color, thickness, cv2.LINE_AA)


def draw_header(frame, frame_idx, detections, latency_ms):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 82), (5, 10, 28), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    cv2.putText(
        frame,
        "REAL-TIME AI INTEGRATION // EDGE VISION PIPELINE",
        (22, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"YOLO inference   detections: {detections:02d}   latency: {latency_ms:05.1f} ms   mode: embedded_ai",
        (22, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (225, 245, 255),
        1,
        cv2.LINE_AA,
    )

    scan_y = int((frame_idx * 7) % h)
    cv2.line(frame, (0, scan_y), (w, scan_y), (0, 255, 255), 2)
    cv2.line(frame, (0, min(scan_y + 8, h - 1)), (w, min(scan_y + 8, h - 1)), (168, 85, 247), 1)


def draw_side_panel(frame, class_counts, avg_conf):
    h, w = frame.shape[:2]
    panel_w = 270
    x0 = w - panel_w - 18
    y0 = 98
    y1 = min(h - 25, y0 + 245)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (w - 18, y1), (2, 6, 23), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.rectangle(frame, (x0, y0), (w - 18, y1), (0, 255, 255), 1)

    cv2.putText(
        frame,
        "SYSTEM STATUS",
        (x0 + 14, y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    lines = [
        ("OpenCV stream", "ACTIVE"),
        ("YOLO model", "READY"),
        ("Edge runtime", "SIM"),
        ("HW sync", "OK"),
    ]

    y = y0 + 60
    for label, value in lines:
        cv2.putText(frame, label, (x0 + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 165, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (w - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 140), 1, cv2.LINE_AA)
        y += 25

    y += 10
    cv2.putText(frame, "CONFIDENCE", (x0 + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 165, 190), 1, cv2.LINE_AA)
    bar_x = x0 + 14
    bar_y = y + 13
    bar_w = panel_w - 50
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (30, 41, 59), -1)

    conf_w = int(bar_w * np.clip(avg_conf, 0, 1))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_w, bar_y + 10), (0, 255, 255), -1)

    y = bar_y + 38
    cv2.putText(frame, "TOP CLASSES", (x0 + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 165, 190), 1, cv2.LINE_AA)

    y += 24
    for cls_name, count in list(class_counts.items())[:4]:
        cv2.putText(frame, f"{cls_name[:18]}", (x0 + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (225, 245, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str(count), (w - 50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 140), 1, cv2.LINE_AA)
        y += 22


def draw_neural_overlay(frame, frame_idx):
    h, w = frame.shape[:2]

    # Small animated graph on lower-left corner
    cx = 110
    cy = h - 95

    nodes = [
        (cx, cy),
        (cx + 55, cy - 35),
        (cx + 55, cy + 35),
        (cx + 115, cy),
        (cx + 170, cy - 28),
        (cx + 170, cy + 28),
    ]

    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if j <= i:
                continue
            if abs(i - j) <= 2:
                alpha = 0.12 + 0.10 * np.sin(frame_idx * 0.08 + i + j)
                color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
                cv2.line(frame, a, b, color, 1)

    for i, (x, y) in enumerate(nodes):
        pulse = 1 + 0.35 * np.sin(frame_idx * 0.12 + i)
        radius = int(4 * pulse)
        cv2.circle(frame, (x, y), radius, (0, 255, 255), -1)

    draw_text_box(frame, "neural.runtime(feed='camera', target='objects')", 22, h - 28, color=(168, 85, 247), scale=0.42)


def encode_h264_with_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")

    # fallback for Windows winget installation
    if not ffmpeg:
        possible = list(Path.home().glob("AppData/Local/Microsoft/WinGet/Packages/**/ffmpeg.exe"))
        if possible:
            ffmpeg = str(possible[0])

    if not ffmpeg:
        print("FFmpeg not found. Keeping OpenCV mp4 output.")
        TEMP_VIDEO.replace(OUTPUT_VIDEO)
        return

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(TEMP_VIDEO),
        "-an",
        "-vf", "format=yuv420p",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.1",
        "-crf", "28",
        "-preset", "medium",
        "-movflags", "+faststart",
        str(OUTPUT_VIDEO),
    ]

    subprocess.run(cmd, check=True)
    TEMP_VIDEO.unlink(missing_ok=True)


def main():
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Missing input video: {INPUT_VIDEO}")

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {INPUT_VIDEO}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(min(src_fps if src_fps > 0 else 24, FPS_LIMIT))
    max_frames = int(MAX_SECONDS * fps)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    frame = resize_keep_aspect(frame, OUTPUT_WIDTH)
    h, w = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(TEMP_VIDEO), fourcc, fps, (w, h))

    frame_idx = 0

    while ret and frame_idx < max_frames:
        frame = resize_keep_aspect(frame, OUTPUT_WIDTH)
        frame = cv2.convertScaleAbs(frame, alpha=0.96, beta=-4)

        start = time.perf_counter()
        result = model.predict(frame, conf=CONF_THRES, verbose=False)[0]
        latency_ms = (time.perf_counter() - start) * 1000

        detections = 0
        conf_values = []
        class_counts = {}

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if TARGET_CLASSES is not None and cls_id not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                name = model.names.get(cls_id, str(cls_id))

                detections += 1
                conf_values.append(conf)
                class_counts[name] = class_counts.get(name, 0) + 1

                color = (0, 255, 170) if conf >= 0.45 else (0, 190, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    fill = roi.copy()
                    fill[:] = color
                    cv2.addWeighted(fill, 0.10, roi, 0.90, 0, roi)

                draw_text_box(frame, f"{name.upper()} {conf:.2f}", x1, max(y1, 22), color=color, scale=0.42)

        avg_conf = float(np.mean(conf_values)) if conf_values else 0.0

        draw_header(frame, frame_idx, detections, latency_ms)
        draw_side_panel(frame, class_counts, avg_conf)
        draw_neural_overlay(frame, frame_idx)

        writer.write(frame)

        ret, frame = cap.read()
        frame_idx += 1

    cap.release()
    writer.release()

    encode_h264_with_ffmpeg()

    size_mb = OUTPUT_VIDEO.stat().st_size / (1024 * 1024)
    print(f"Created: {OUTPUT_VIDEO}")
    print(f"Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()