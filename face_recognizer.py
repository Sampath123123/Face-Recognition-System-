import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition
import numpy as np

try:
    import config
except ImportError:
    print("ERROR: config.py not found.")
    raise SystemExit(1)
except Exception as exc:
    print(f"ERROR: Failed to import config.py.\nDetails: {exc}")
    raise SystemExit(1)


BASE_DIR = Path(__file__).resolve().parent


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


OFFICER_DATABASE_CSV = resolve_path(
    getattr(config, "OFFICER_DATABASE_CSV", "database/officers.csv")
)
OFFICER_IMAGES_DIR = resolve_path(
    getattr(config, "OFFICER_IMAGES_DIR", "database/officers")
)
ALERTS_DIR = resolve_path(getattr(config, "ALERTS_DIR", "alerts"))
CAMERA_INDEX = getattr(config, "CAMERA_INDEX", 0)
CAMERA_BACKEND = getattr(config, "CAMERA_BACKEND", "DEFAULT").upper()
CAMERA_FOURCC = getattr(config, "CAMERA_FOURCC", "MJPG")
CAMERA_FPS = int(getattr(config, "CAMERA_FPS", 30))
CAMERA_FRAME_WIDTH = int(getattr(config, "CAMERA_FRAME_WIDTH", 1280))
CAMERA_FRAME_HEIGHT = int(getattr(config, "CAMERA_FRAME_HEIGHT", 720))
RECOGNITION_TOLERANCE = getattr(config, "RECOGNITION_TOLERANCE", 0.48)
RESIZE_FACTOR = getattr(config, "RESIZE_FACTOR", 0.25)
FACE_DETECTION_UPSAMPLE = int(getattr(config, "FACE_DETECTION_UPSAMPLE", 1))
FACE_DETECTION_MODEL = getattr(config, "FACE_DETECTION_MODEL", "hog")
PROCESS_EVERY_N_FRAMES = int(getattr(config, "PROCESS_EVERY_N_FRAMES", 1))
OFFICER_LABEL = getattr(config, "OFFICER_LABEL", "Officer")
UNKNOWN_LABEL = getattr(config, "UNKNOWN_LABEL", "Unknown - Alert")
ALERT_BANNER_TEXT = getattr(config, "ALERT_BANNER_TEXT", "ALERT: UNKNOWN PERSON DETECTED")
UNKNOWN_ALERT_COOLDOWN_SECONDS = getattr(config, "UNKNOWN_ALERT_COOLDOWN_SECONDS", 8)
SAVE_UNKNOWN_SNAPSHOTS = getattr(config, "SAVE_UNKNOWN_SNAPSHOTS", True)
ENABLE_TERMINAL_BELL = getattr(config, "ENABLE_TERMINAL_BELL", True)
AUTO_CLEAN_INVALID_SAMPLES = getattr(config, "AUTO_CLEAN_INVALID_SAMPLES", True)


CONFIG_ERRORS = []
if RECOGNITION_TOLERANCE <= 0:
    CONFIG_ERRORS.append("RECOGNITION_TOLERANCE must be greater than 0.")
if RESIZE_FACTOR <= 0 or RESIZE_FACTOR > 1:
    CONFIG_ERRORS.append("RESIZE_FACTOR must be between 0 and 1.")
if FACE_DETECTION_UPSAMPLE < 0:
    CONFIG_ERRORS.append("FACE_DETECTION_UPSAMPLE must be 0 or greater.")
if FACE_DETECTION_MODEL not in {"hog", "cnn"}:
    CONFIG_ERRORS.append("FACE_DETECTION_MODEL must be 'hog' or 'cnn'.")
if PROCESS_EVERY_N_FRAMES < 1:
    CONFIG_ERRORS.append("PROCESS_EVERY_N_FRAMES must be 1 or greater.")

if CONFIG_ERRORS:
    print("ERROR: Configuration issues found:")
    for error in CONFIG_ERRORS:
        print(f"- {error}")
    raise SystemExit(1)


known_face_data = []


def open_camera():
    backend = cv2.CAP_V4L2 if CAMERA_BACKEND == "V4L2" and hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
    video_capture = cv2.VideoCapture(CAMERA_INDEX, backend)
    if not video_capture.isOpened():
        return video_capture

    if CAMERA_FOURCC and len(CAMERA_FOURCC) == 4:
        video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
    if CAMERA_FPS > 0:
        video_capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    return video_capture


def ensure_runtime_directories():
    OFFICER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)


def rewrite_database_csv(rows):
    with OFFICER_DATABASE_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["name", "image_file", "badge_id"])
        writer.writeheader()
        writer.writerows(rows)


def load_officer_database():
    global known_face_data

    ensure_runtime_directories()

    if not OFFICER_DATABASE_CSV.exists():
        print(f"ERROR: Officer database CSV not found: {OFFICER_DATABASE_CSV}")
        return False

    with OFFICER_DATABASE_CSV.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            print("ERROR: Officer database CSV is empty.")
            return False

        required_columns = {"name", "image_file"}
        missing_columns = required_columns.difference({name.strip() for name in reader.fieldnames})
        if missing_columns:
            print(
                "ERROR: Officer database CSV must contain these headers: "
                "name,image_file[,badge_id]"
            )
            print(f"Missing headers: {sorted(missing_columns)}")
            return False

        loaded_profiles = []
        cleaned_rows = []
        invalid_entries_found = False
        for line_number, row in enumerate(reader, start=2):
            name = str(row.get("name", "")).strip()
            image_file = str(row.get("image_file", "")).strip()
            badge_id = str(row.get("badge_id", "")).strip()

            if not name or not image_file:
                print(f"Skipping line {line_number}: missing name or image_file.")
                invalid_entries_found = True
                continue

            image_path = Path(image_file)
            if not image_path.is_absolute():
                image_path = OFFICER_IMAGES_DIR / image_file

            if not image_path.exists():
                print(f"Skipping {name}: image not found at {image_path}")
                invalid_entries_found = True
                continue

            try:
                image = face_recognition.load_image_file(str(image_path))
                encodings = face_recognition.face_encodings(image)
            except Exception as exc:
                print(f"Skipping {name}: failed to read image. Details: {exc}")
                invalid_entries_found = True
                if AUTO_CLEAN_INVALID_SAMPLES:
                    try:
                        image_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                continue

            if not encodings:
                print(f"Skipping {name}: no face found in {image_path.name}")
                invalid_entries_found = True
                if AUTO_CLEAN_INVALID_SAMPLES:
                    try:
                        image_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                continue

            cleaned_rows.append(
                {
                    "name": name,
                    "image_file": image_file,
                    "badge_id": badge_id,
                }
            )
            loaded_profiles.append(
                {
                    "name": name,
                    "badge_id": badge_id,
                    "encoding": encodings[0],
                }
            )

    if AUTO_CLEAN_INVALID_SAMPLES and invalid_entries_found:
        rewrite_database_csv(cleaned_rows)
        print("Cleaned invalid sample entries from the officer database CSV.")

    known_face_data = loaded_profiles
    if not known_face_data:
        print("ERROR: No valid officer face encodings were loaded.")
        return False

    print(f"Loaded {len(known_face_data)} officer face profiles.")
    return True


def identify_face(face_encoding):
    known_encodings = [profile["encoding"] for profile in known_face_data]
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = int(np.argmin(distances))
    best_distance = float(distances[best_match_index])

    if best_distance <= RECOGNITION_TOLERANCE:
        matched = known_face_data[best_match_index]
        return {
            "matched": True,
            "name": matched["name"],
            "badge_id": matched["badge_id"],
            "label": OFFICER_LABEL,
        }

    return {
        "matched": False,
        "name": "Unknown",
        "badge_id": "",
        "label": UNKNOWN_LABEL,
    }


def save_unknown_snapshot(frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = ALERTS_DIR / f"unknown_{timestamp}.jpg"
    cv2.imwrite(str(image_path), frame)
    return image_path


def trigger_unknown_alert(frame, unknown_count):
    if ENABLE_TERMINAL_BELL:
        print("\a", end="", flush=True)

    snapshot_message = ""
    if SAVE_UNKNOWN_SNAPSHOTS:
        image_path = save_unknown_snapshot(frame)
        snapshot_message = f" Snapshot saved: {image_path}"

    print(
        f"[ALERT] {unknown_count} unknown face(s) detected at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.{snapshot_message}"
    )


def draw_face_label(frame, face_box, profile):
    top, right, bottom, left = face_box
    is_match = profile["matched"]
    box_color = (0, 180, 0) if is_match else (0, 0, 255)

    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

    lines = [profile["name"], profile["label"]]
    if profile["badge_id"]:
        lines.append(f"ID: {profile['badge_id']}")

    panel_height = 24 * len(lines) + 10
    panel_top = max(top - panel_height, 0)
    cv2.rectangle(frame, (left, panel_top), (right, top), box_color, cv2.FILLED)

    for index, text in enumerate(lines):
        y = panel_top + 20 + (index * 22)
        cv2.putText(
            frame,
            text,
            (left + 6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_alert_banner(frame, unknown_count):
    banner_height = 52
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_height), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.putText(
        frame,
        ALERT_BANNER_TEXT,
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Unknown faces: {unknown_count}",
        (12, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run_recognition():
    video_capture = open_camera()
    if not video_capture.isOpened():
        print("ERROR: Could not open the webcam.")
        return

    last_unknown_alert_at = 0.0
    frame_counter = 0
    last_results = []
    print("Press 'q' to quit.")

    while True:
        ok, frame = video_capture.read()
        if not ok:
            continue

        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(
                rgb_small_frame,
                number_of_times_to_upsample=FACE_DETECTION_UPSAMPLE,
                model=FACE_DETECTION_MODEL,
            )
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            refreshed_results = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                profile = identify_face(face_encoding)
                scaled_box = (
                    int(top / RESIZE_FACTOR),
                    int(right / RESIZE_FACTOR),
                    int(bottom / RESIZE_FACTOR),
                    int(left / RESIZE_FACTOR),
                )
                refreshed_results.append((scaled_box, profile))
            last_results = refreshed_results

        unknown_count = 0
        for scaled_box, profile in last_results:
            draw_face_label(frame, scaled_box, profile)
            if not profile["matched"]:
                unknown_count += 1

        if unknown_count > 0:
            draw_alert_banner(frame, unknown_count)
            now = time.monotonic()
            if now - last_unknown_alert_at >= UNKNOWN_ALERT_COOLDOWN_SECONDS:
                trigger_unknown_alert(frame, unknown_count)
                last_unknown_alert_at = now

        cv2.imshow("Officer Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_counter += 1

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if load_officer_database():
        run_recognition()
    else:
        print("Exiting: failed to load officer face database.")
