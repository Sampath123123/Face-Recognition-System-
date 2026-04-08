# Officer Face Recognition Alert System

This version keeps face detection and face recognition as the primary feature.

The system:

- loads a local database of officer face images
- matches webcam faces against that officer database
- labels matched people as `Officer`
- labels non-matched people as `Possible Trespassing Civilian`
- raises an alert banner and stores a snapshot when an unknown face is seen

## Folder structure

```text
Face-Recognition-System--main/
  face_recognizer.py
  enroll_officer.py
  config.py
  database/
    officers.csv
    officers/
      officer_1.jpg
      officer_2.jpg
  alerts/
```

## Better enrollment flow

Instead of manually collecting many selfies, use the guided enrollment script:

```bash
python enroll_officer.py
```

It will ask for:

- officer name
- badge ID

Then it opens the webcam and guides the officer through these poses:

- center
- left
- right
- up
- down

For each pose it now waits until the detected face orientation matches the requested pose, then captures several face crops automatically, saves them under a badge-specific folder inside [database/officers](/home/aiml/Desktop/Face-Recognition-System--main/database/officers), and updates [database/officers.csv](/home/aiml/Desktop/Face-Recognition-System--main/database/officers.csv).

If the entered badge ID is already present in the database, the script shows that the officer is already enrolled and stops instead of recording duplicate samples.

The enrollment step now also validates each captured crop before saving it, so obviously bad samples are rejected earlier.

This is the recommended way to build the face database.

## Officer database CSV

The file [database/officers.csv](/home/aiml/Desktop/Face-Recognition-System--main/database/officers.csv) must contain:

```csv
name,image_file,badge_id
Ravi Kumar,RG-101/center_01.jpg,RG-101
Ravi Kumar,RG-101/left_01.jpg,RG-101
Ravi Kumar,RG-101/right_01.jpg,RG-101
```

Put the matching image files inside [database/officers](/home/aiml/Desktop/Face-Recognition-System--main/database/officers).

## Setup

Install the required packages:

```bash
pip install face_recognition opencv-python numpy
```

Then either:

- use `python enroll_officer.py` to generate entries automatically
- or add rows manually if needed

If distant faces are not detected well enough, tune these values in [config.py](/home/aiml/Desktop/Face-Recognition-System--main/config.py):

- `CAMERA_FRAME_WIDTH`
- `CAMERA_FRAME_HEIGHT`
- `RESIZE_FACTOR`
- `FACE_DETECTION_UPSAMPLE`
- `PROCESS_EVERY_N_FRAMES`

Higher resolution and higher upsample improve distant-face detection, but they also make processing slower.
Higher `PROCESS_EVERY_N_FRAMES` improves FPS by running full detection less often.

## Run

```bash
python face_recognizer.py
```

Press `q` to quit.

## What happens on unknown faces

If a face is not found in the officer database:

- the face box turns red
- the screen shows an alert banner
- a snapshot is saved in [alerts](/home/aiml/Desktop/Face-Recognition-System--main/alerts)

If invalid sample images are still present in the database, the recognizer removes those broken entries from the CSV automatically when `AUTO_CLEAN_INVALID_SAMPLES = True` in [config.py](/home/aiml/Desktop/Face-Recognition-System--main/config.py).

Weapon detection is intentionally not part of this step. Get face recognition stable first, then add weapon detection as a second stage.
