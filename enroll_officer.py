import csv
import time
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
CAMERA_INDEX = getattr(config, "CAMERA_INDEX", 0)
CAMERA_BACKEND = getattr(config, "CAMERA_BACKEND", "DEFAULT").upper()
CAMERA_FOURCC = getattr(config, "CAMERA_FOURCC", "MJPG")
CAMERA_FPS = int(getattr(config, "CAMERA_FPS", 30))
CAMERA_FRAME_WIDTH = int(getattr(config, "CAMERA_FRAME_WIDTH", 1280))
CAMERA_FRAME_HEIGHT = int(getattr(config, "CAMERA_FRAME_HEIGHT", 720))
FACE_DETECTION_UPSAMPLE = int(getattr(config, "FACE_DETECTION_UPSAMPLE", 1))
FACE_DETECTION_MODEL = getattr(config, "FACE_DETECTION_MODEL", "hog")
ENROLLMENT_POSES = list(getattr(config, "ENROLLMENT_POSES", ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]))
ENROLLMENT_IMAGES_PER_POSE = int(getattr(config, "ENROLLMENT_IMAGES_PER_POSE", 4))
ENROLLMENT_FRAME_SKIP = int(getattr(config, "ENROLLMENT_FRAME_SKIP", 4))
ENROLLMENT_COUNTDOWN_SECONDS = int(getattr(config, "ENROLLMENT_COUNTDOWN_SECONDS", 2))
ENROLLMENT_POSE_HOLD_FRAMES = int(getattr(config, "ENROLLMENT_POSE_HOLD_FRAMES", 6))
ENROLLMENT_CAPTURE_GAP_FRAMES = int(getattr(config, "ENROLLMENT_CAPTURE_GAP_FRAMES", 3))
ENROLLMENT_CENTER_YAW_THRESHOLD = float(getattr(config, "ENROLLMENT_CENTER_YAW_THRESHOLD", 0.10))
ENROLLMENT_SIDE_YAW_THRESHOLD = float(getattr(config, "ENROLLMENT_SIDE_YAW_THRESHOLD", 0.14))
ENROLLMENT_UP_PITCH_THRESHOLD = float(getattr(config, "ENROLLMENT_UP_PITCH_THRESHOLD", 0.35))
ENROLLMENT_DOWN_PITCH_THRESHOLD = float(getattr(config, "ENROLLMENT_DOWN_PITCH_THRESHOLD", 0.49))


def ensure_paths():
    OFFICER_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not OFFICER_DATABASE_CSV.exists():
        with OFFICER_DATABASE_CSV.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["name", "image_file", "badge_id"])
            writer.writeheader()


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


def prompt_value(label):
    value = input(f"{label}: ").strip()
    if not value:
        print(f"ERROR: {label} is required.")
        raise SystemExit(1)
    return value


def draw_overlay(frame, badge_id, name, pose, captured_count, total_target, status_lines):
    lines = [
        f"Officer: {name}",
        f"Badge ID: {badge_id}",
        f"Target pose: {pose}",
        f"Captured: {captured_count}/{total_target}",
    ]
    lines.extend(status_lines)

    panel_height = 30 * len(lines) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (frame.shape[1] - 10, min(panel_height, frame.shape[0] - 10)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (24, 40 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2 if index < 3 else 1,
            cv2.LINE_AA,
        )


def average_point(points):
    return np.mean(np.asarray(points, dtype=np.float32), axis=0)


def classify_pose(landmarks):
    left_eye_center = average_point(landmarks["left_eye"])
    right_eye_center = average_point(landmarks["right_eye"])
    nose_tip_center = average_point(landmarks["nose_tip"])
    chin_center = np.asarray(landmarks["chin"][8], dtype=np.float32)

    eye_midpoint = (left_eye_center + right_eye_center) / 2.0
    face_width = max(abs(right_eye_center[0] - left_eye_center[0]), 1.0)
    face_height = max(chin_center[1] - eye_midpoint[1], 1.0)

    yaw = float((nose_tip_center[0] - eye_midpoint[0]) / face_width)
    pitch = float((nose_tip_center[1] - eye_midpoint[1]) / face_height)

    if abs(yaw) <= ENROLLMENT_CENTER_YAW_THRESHOLD:
        if pitch <= ENROLLMENT_UP_PITCH_THRESHOLD:
            return "UP", yaw, pitch
        if pitch >= ENROLLMENT_DOWN_PITCH_THRESHOLD:
            return "DOWN", yaw, pitch
        return "CENTER", yaw, pitch

    if yaw <= -ENROLLMENT_SIDE_YAW_THRESHOLD:
        return "LEFT", yaw, pitch
    if yaw >= ENROLLMENT_SIDE_YAW_THRESHOLD:
        return "RIGHT", yaw, pitch
    return "CENTER", yaw, pitch


def face_analysis_from_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(
        rgb_frame,
        number_of_times_to_upsample=FACE_DETECTION_UPSAMPLE,
        model=FACE_DETECTION_MODEL,
    )
    if len(face_locations) == 0:
        return {"status": "NO_FACE"}
    if len(face_locations) > 1:
        return {"status": "MULTIPLE_FACES"}

    top, right, bottom, left = face_locations[0]
    landmarks_list = face_recognition.face_landmarks(rgb_frame, [face_locations[0]])
    if not landmarks_list:
        return {"status": "LANDMARKS_MISSING"}

    pose, yaw, pitch = classify_pose(landmarks_list[0])

    padding = 30
    top = max(top - padding, 0)
    left = max(left - padding, 0)
    bottom = min(bottom + padding, frame.shape[0])
    right = min(right + padding, frame.shape[1])
    return {
        "status": "OK",
        "pose": pose,
        "yaw": yaw,
        "pitch": pitch,
        "crop": frame[top:bottom, left:right],
    }


def crop_has_encodable_face(face_crop):
    rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(
        rgb_crop,
        number_of_times_to_upsample=FACE_DETECTION_UPSAMPLE,
        model=FACE_DETECTION_MODEL,
    )
    if len(face_locations) != 1:
        return False
    encodings = face_recognition.face_encodings(rgb_crop, face_locations)
    return bool(encodings)


def capture_pose_samples(video_capture, badge_id, name, pose):
    target_count = ENROLLMENT_IMAGES_PER_POSE
    captured_paths = []
    frame_index = 0
    matched_hold_frames = 0
    matched_gap_frames = 0

    pose_dir = OFFICER_IMAGES_DIR / badge_id
    pose_dir.mkdir(parents=True, exist_ok=True)

    countdown_start = time.monotonic()
    while True:
        ok, frame = video_capture.read()
        if not ok:
            continue

        elapsed = time.monotonic() - countdown_start
        if elapsed < ENROLLMENT_COUNTDOWN_SECONDS:
            status_lines = [
                "Keep one face in frame",
                f"Starting in {ENROLLMENT_COUNTDOWN_SECONDS - int(elapsed)}",
            ]
            draw_overlay(frame, badge_id, name, pose, len(captured_paths), target_count, status_lines)
            cv2.imshow("Officer Enrollment", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
            continue

        frame_index += 1
        analysis = face_analysis_from_frame(frame)

        if analysis["status"] == "NO_FACE":
            matched_hold_frames = 0
            status_lines = ["No face detected", "Move your face into the frame"]
        elif analysis["status"] == "MULTIPLE_FACES":
            matched_hold_frames = 0
            status_lines = ["Multiple faces detected", "Keep only one person in view"]
        elif analysis["status"] == "LANDMARKS_MISSING":
            matched_hold_frames = 0
            status_lines = ["Face detected but landmarks failed", "Try better lighting"]
        else:
            current_pose = analysis["pose"]
            if current_pose == pose:
                matched_hold_frames += 1
                hold_needed = max(ENROLLMENT_POSE_HOLD_FRAMES - matched_hold_frames, 0)
                if hold_needed > 0:
                    status_lines = [
                        f"Current pose: {current_pose}",
                        f"Hold steady for {hold_needed} more frame(s)",
                    ]
                else:
                    status_lines = [
                        f"Current pose: {current_pose}",
                        "Pose accepted. Capturing samples",
                    ]
            else:
                matched_hold_frames = 0
                status_lines = [
                    f"Current pose: {current_pose}",
                    f"Waiting for target pose: {pose}",
                ]

        draw_overlay(frame, badge_id, name, pose, len(captured_paths), target_count, status_lines)
        cv2.imshow("Officer Enrollment", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

        if frame_index % ENROLLMENT_FRAME_SKIP != 0:
            continue

        if analysis["status"] != "OK":
            continue

        if analysis["pose"] != pose or matched_hold_frames < ENROLLMENT_POSE_HOLD_FRAMES:
            matched_gap_frames = 0
            continue

        matched_gap_frames += 1
        if matched_gap_frames < ENROLLMENT_CAPTURE_GAP_FRAMES:
            continue

        matched_gap_frames = 0
        if not crop_has_encodable_face(analysis["crop"]):
            continue

        file_name = f"{pose.lower()}_{len(captured_paths) + 1:02d}.jpg"
        image_path = pose_dir / file_name
        cv2.imwrite(str(image_path), analysis["crop"])
        captured_paths.append(image_path.relative_to(OFFICER_IMAGES_DIR).as_posix())

        if len(captured_paths) >= target_count:
            return captured_paths


def remove_existing_rows(rows, badge_id):
    filtered = []
    for row in rows:
        existing_badge = str(row.get("badge_id", "")).strip()
        if existing_badge != badge_id:
            filtered.append(row)
    return filtered


def lookup_existing_enrollment(badge_id):
    if not OFFICER_DATABASE_CSV.exists():
        return None

    with OFFICER_DATABASE_CSV.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        matches = []
        for row in reader:
            existing_badge = str(row.get("badge_id", "")).strip()
            if existing_badge == badge_id:
                matches.append(
                    {
                        "name": str(row.get("name", "")).strip(),
                        "image_file": str(row.get("image_file", "")).strip(),
                        "badge_id": existing_badge,
                    }
                )

    if not matches:
        return None

    enrolled_name = matches[0]["name"] or "Unknown"
    return {
        "name": enrolled_name,
        "badge_id": badge_id,
        "samples": len(matches),
    }


def update_database_csv(name, badge_id, image_files):
    rows = []
    with OFFICER_DATABASE_CSV.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "name": str(row.get("name", "")).strip(),
                    "image_file": str(row.get("image_file", "")).strip(),
                    "badge_id": str(row.get("badge_id", "")).strip(),
                }
            )

    rows = remove_existing_rows(rows, badge_id)
    for image_file in image_files:
        rows.append(
            {
                "name": name,
                "image_file": image_file,
                "badge_id": badge_id,
            }
        )

    with OFFICER_DATABASE_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["name", "image_file", "badge_id"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    ensure_paths()
    name = prompt_value("Officer name")
    badge_id = prompt_value("Badge ID")

    existing_enrollment = lookup_existing_enrollment(badge_id)
    if existing_enrollment:
        print(
            f"Officer already enrolled: {existing_enrollment['name']} "
            f"({existing_enrollment['badge_id']}) with "
            f"{existing_enrollment['samples']} sample(s)."
        )
        raise SystemExit(0)

    print("Enrollment will capture guided face poses.")
    print("Required sequence:", ", ".join(ENROLLMENT_POSES))
    print("Press q in the camera window to cancel.")

    video_capture = open_camera()
    if not video_capture.isOpened():
        print("ERROR: Could not open the webcam.")
        raise SystemExit(1)

    captured_files = []
    try:
        for pose in ENROLLMENT_POSES:
            pose_files = capture_pose_samples(video_capture, badge_id, name, pose)
            captured_files.extend(pose_files)
            print(f"Captured {len(pose_files)} sample(s) for pose {pose}.")
    except KeyboardInterrupt:
        print("\nEnrollment cancelled.")
        video_capture.release()
        cv2.destroyAllWindows()
        raise SystemExit(1)

    video_capture.release()
    cv2.destroyAllWindows()

    update_database_csv(name, badge_id, captured_files)
    print(f"Enrollment complete for {name} ({badge_id}).")
    print(f"Saved {len(captured_files)} face samples and updated {OFFICER_DATABASE_CSV}.")


if __name__ == "__main__":
    main()
