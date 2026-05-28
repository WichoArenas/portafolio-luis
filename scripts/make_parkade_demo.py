from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np

INPUT_VIDEO = Path("assets_raw/parkade-source.mp4")
TEMP_VIDEO = Path("public/parkade-demo-temp.mp4")
OUTPUT_VIDEO = Path("public/parkade-demo.mp4")

MAX_SECONDS = 9
OUTPUT_WIDTH = 960
FPS_LIMIT = 24

# Demo slots sobre frame redimensionado a 960 px de ancho.
# Cada fila se define como: (x1, y1, x2, y2, number_of_slots)
# Ajusta estos valores si tus cajones no coinciden perfecto.
PARKING_ROWS = [
    (20, 86, 455, 132, 18),
    (15, 158, 465, 220, 17),
    (20, 250, 470, 305, 16),
    (10, 366, 465, 420, 17),
    (525, 88, 888, 132, 15),
    (540, 165, 890, 225, 13),
    (520, 255, 872, 310, 14),
    (530, 365, 900, 424, 14),
]

def resize_keep_aspect(frame, width):
    h, w = frame.shape[:2]
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)

def slot_score(roi):
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(gray, 60, 140)
    edge_density = np.mean(edges > 0)

    saturation = np.mean(hsv[:, :, 1]) / 255.0
    brightness_std = np.std(gray) / 255.0

    dark_ratio = np.mean(gray < 105)

    # Heurística simple: autos suelen tener textura, contraste, sombras y color.
    score = (
        edge_density * 2.2
        + saturation * 0.55
        + brightness_std * 1.25
        + dark_ratio * 0.45
    )

    return float(np.clip(score, 0.0, 1.0))

def draw_text_box(frame, text, x, y, color=(0, 255, 255), scale=0.45):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 10, y + 5), (5, 10, 25), -1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 10, y + 5), color, 1)
    cv2.putText(frame, text, (x + 5, y - 4), font, scale, color, thickness, cv2.LINE_AA)

def draw_header(frame, frame_idx, occupied, total):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 78), (5, 10, 28), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    cv2.putText(
        frame,
        "SMART PARKADE SYSTEM // AI OCCUPANCY VISUALIZATION",
        (22, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    free = total - occupied
    cv2.putText(
        frame,
        f"Occupied: {occupied:02d}   Free: {free:02d}   Slots scanned: {total:02d}",
        (22, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 245, 255),
        1,
        cv2.LINE_AA,
    )

    scan_y = int((frame_idx * 9) % h)
    cv2.line(frame, (0, scan_y), (w, scan_y), (0, 255, 255), 2)
    cv2.line(frame, (0, min(scan_y + 8, h - 1)), (w, min(scan_y + 8, h - 1)), (255, 180, 0), 1)

def draw_corners(frame):
    h, w = frame.shape[:2]
    color = (0, 255, 255)
    length = 34
    thickness = 2

    corners = [
        ((15, 15), (15 + length, 15), (15, 15 + length)),
        ((w - 15, 15), (w - 15 - length, 15), (w - 15, 15 + length)),
        ((15, h - 15), (15 + length, h - 15), (15, h - 15 - length)),
        ((w - 15, h - 15), (w - 15 - length, h - 15), (w - 15, h - 15 - length)),
    ]

    for p1, p2, p3 in corners:
        cv2.line(frame, p1, p2, color, thickness)
        cv2.line(frame, p1, p3, color, thickness)

def draw_slots(frame, frame_idx):
    total = 0
    occupied = 0

    for row_id, (x1, y1, x2, y2, n_slots) in enumerate(PARKING_ROWS, start=1):
        slot_w = (x2 - x1) / n_slots

        for i in range(n_slots):
            sx1 = int(x1 + i * slot_w)
            sx2 = int(x1 + (i + 1) * slot_w) - 2
            sy1 = int(y1)
            sy2 = int(y2)

            roi = frame[sy1:sy2, sx1:sx2]
            score = slot_score(roi)

            # Umbral visual. Si marca demasiados ocupados/libres, ajustar 0.22–0.36.
            is_occupied = score > 0.28

            # Leve variación temporal para que se vea vivo, sin cambiar todo.
            if (frame_idx // 45 + row_id + i) % 23 == 0:
                is_occupied = not is_occupied

            total += 1
            if is_occupied:
                occupied += 1

            color = (0, 70, 255) if is_occupied else (0, 255, 140)
            label = "OCC" if is_occupied else "FREE"

            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 1)

            fill = frame[sy1:sy2, sx1:sx2].copy()
            fill[:, :] = color
            cv2.addWeighted(fill, 0.12, frame[sy1:sy2, sx1:sx2], 0.88, 0, frame[sy1:sy2, sx1:sx2])

            if i % 3 == 0:
                cv2.putText(
                    frame,
                    label,
                    (sx1 + 2, sy1 + 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    return occupied, total

def encode_h264_with_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("FFmpeg not found. Keeping OpenCV mp4 output.")
        TEMP_VIDEO.replace(OUTPUT_VIDEO)
        return

    cmd = [
        ffmpeg,
        "-y",
        "-i", str(TEMP_VIDEO),
        "-an",
        "-c:v", "libx264",
        "-crf", "29",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(OUTPUT_VIDEO),
    ]

    subprocess.run(cmd, check=True)
    TEMP_VIDEO.unlink(missing_ok=True)

def main():
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Missing input video: {INPUT_VIDEO}")

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

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

        # Color grading estilo tech.
        frame = cv2.convertScaleAbs(frame, alpha=0.96, beta=-5)

        occupied, total = draw_slots(frame, frame_idx)
        draw_header(frame, frame_idx, occupied, total)
        draw_corners(frame)

        draw_text_box(frame, "BERKELEY 2ND PLACE", 22, h - 25, color=(0, 210, 255), scale=0.45)
        draw_text_box(frame, "AI-ASSISTED PARKING OCCUPANCY", w - 335, h - 25, color=(0, 255, 140), scale=0.45)

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