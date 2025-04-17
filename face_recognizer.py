# Main script using Google Drive API for downloads. Reads config from config.py
# Improved version with better error handling and modularization.

import face_recognition
import cv2
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from io import BytesIO
import os
import time
import json
import google.generativeai as genai
import PIL
import re

# --- Import Configuration ---
try:
    import config
except ImportError:
    print("ERROR: config.py not found. Please create a config.py file with the required settings.")
    exit()
except Exception as config_err:
    print(f"ERROR: Failed to import config.py.\nDetails: {config_err}")
    exit()

# --- Load Configuration ---
try:
    SCOPES = getattr(config, 'GOOGLE_API_SCOPES', None)
    CREDS_FILE = getattr(config, 'GOOGLE_CREDS_FILE', 'credentials.json')
    SHEET_ID = getattr(config, 'GOOGLE_SHEET_ID', None)
    SHEET_NAME = getattr(config, 'GOOGLE_SHEET_NAME', None)
    WORKSHEET_NAME = getattr(config, 'GOOGLE_WORKSHEET_NAME', 'Sheet1')
    SHEET_COLUMNS = getattr(config, 'SHEET_COLUMNS', None)
    RECOGNITION_TOLERANCE = getattr(config, 'RECOGNITION_TOLERANCE', 0.55)
    RESIZE_FACTOR = getattr(config, 'RESIZE_FACTOR', 0.25)
    GEMINI_API_KEY = getattr(config, 'GEMINI_API_KEY', None)
    GEMINI_PLACEHOLDER = "YOUR_GEMINI_API_KEY_HERE"
    GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'
except Exception as load_config_err:
    print(f"ERROR: Failed to load variables from config.py.\nDetails: {load_config_err}")
    exit()

# --- Validate Essential Configuration ---
CONFIG_ERRORS = []
if not SCOPES:
    CONFIG_ERRORS.append("GOOGLE_API_SCOPES missing in config.py.")
if not os.path.exists(CREDS_FILE):
    CONFIG_ERRORS.append(f"Credentials file '{CREDS_FILE}' not found.")
if not SHEET_ID and not SHEET_NAME:
    CONFIG_ERRORS.append("Set GOOGLE_SHEET_ID or GOOGLE_SHEET_NAME in config.py.")
if not SHEET_COLUMNS or not isinstance(SHEET_COLUMNS, dict) or "name" not in SHEET_COLUMNS or "image_url" not in SHEET_COLUMNS:
    CONFIG_ERRORS.append("SHEET_COLUMNS invalid/missing in config.py.")
if not GEMINI_API_KEY or GEMINI_API_KEY == GEMINI_PLACEHOLDER:
    CONFIG_ERRORS.append("GEMINI_API_KEY missing or set to placeholder in config.py.")

if CONFIG_ERRORS:
    print("ERROR: Configuration issues found:")
    for err in CONFIG_ERRORS:
        print(f"- {err}")
    exit()

# --- Gemini API Setup ---
gemini_model = None
if GEMINI_API_KEY and GEMINI_API_KEY != GEMINI_PLACEHOLDER:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"Gemini API configured successfully (Model: {GEMINI_MODEL_NAME}).")
    except Exception as gemini_err:
        print(f"ERROR: Failed to configure Gemini API. Q&A disabled.\nDetails: {gemini_err}")

# --- Global Variables ---
known_face_data = {}
drive_id_regex = re.compile(r"(?:/d/|id=)([-\w]{25,})")

# --- Functions ---
def load_known_faces_from_sheet():
    """
    Reads the target Google Sheet, downloads images via Drive API, and encodes faces.
    """
    global known_face_data, drive_id_regex
    print("\nConnecting to Google APIs & loading known faces...")
    service_account_email, spreadsheet, creds, drive_service = None, None, None, None

    # --- Authorize and Build Services ---
    try:
        creds = Credentials.from_service_account_file(CREDS_FILE, scopes=SCOPES)
        client = gspread.authorize(creds)
        drive_service = build('drive', 'v3', credentials=creds)

        # --- Open Target Sheet ---
        if SHEET_ID:
            spreadsheet = client.open_by_key(SHEET_ID)
        elif SHEET_NAME:
            spreadsheet = client.open(SHEET_NAME)
        else:
            raise ValueError("Sheet ID or Name not configured")

        sheet = spreadsheet.worksheet(WORKSHEET_NAME)
    except Exception as e_setup:
        print(f"ERROR during API setup/Sheet connection.\nDetails: {e_setup}")
        return False

    # --- Fetching Sheet Data ---
    try:
        records = sheet.get_all_records()
        if not records:
            print("INFO: No data records to process.")
            return True

        header = list(records[0].keys())
        missing_cols = [col for col in SHEET_COLUMNS.values() if col not in header]
        if missing_cols:
            print(f"ERROR: Missing columns in header: {missing_cols}. Check config.py & sheet.")
            return False
    except Exception as e_fetch:
        print(f"ERROR reading sheet data.\nDetails: {e_fetch}")
        return False

    # --- Process Records ---
    local_known_face_data = {}
    for index, row in enumerate(records):
        name = row.get(SHEET_COLUMNS["name"], "").strip()
        original_url = row.get(SHEET_COLUMNS["image_url"], "").strip()

        if not name or not original_url:
            continue

        match = drive_id_regex.search(original_url)
        if not match:
            continue
        file_id = match.group(1)

        try:
            request = drive_service.files().get_media(fileId=file_id)
            fh = BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            fh.seek(0)
            image = face_recognition.load_image_file(fh)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                encoding = face_encodings[0]
                local_known_face_data[name] = {'encoding': encoding}
        except Exception:
            continue

    known_face_data = local_known_face_data
    return True

def run_recognition():
    """
    Runs the face recognition loop using the webcam.
    """
    global known_face_data
    known_encodings_list = [d['encoding'] for d in known_face_data.values()]
    known_names_list = list(known_face_data.keys())

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("ERROR: No video capture device found.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings_list, face_encoding, tolerance=RECOGNITION_TOLERANCE)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names_list[first_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# --- Main Execution ---
if __name__ == "__main__":
    if load_known_faces_from_sheet():
        run_recognition()
    else:
        print("Exiting: Failed to load known faces.")