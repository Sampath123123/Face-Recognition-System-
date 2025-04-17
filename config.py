# config.py
# Configuration file for the Face Recognition Project

# --- Google API Scopes ---
# Define the permissions your script needs. Add more if required by other APIs.
GOOGLE_API_SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly', # To read sheet data
    'https://www.googleapis.com/auth/drive.readonly'    # To download files from Drive via API
]

# --- Google Service Account Credentials ---
# Path to your downloaded Google Service Account JSON key file.
# Place this file in the same directory as your scripts or provide the full path.
GOOGLE_CREDS_FILE = 'credentials.json'

# --- Google Sheet Configuration (TARGET SHEET: "Known Faces") ---

# OPTION 1: Use Sheet ID (Recommended for robustness)
# Find this ID in the URL of your "Known Faces" Google Sheet:
# e.g., https://docs.google.com/spreadsheets/d/THIS_IS_THE_ID/edit
# Paste the ID inside the quotes below. If filled, SHEET_NAME is ignored.
GOOGLE_SHEET_ID = "1M49zBvB13u1mKXSY0FphLEBOB4yU2A_zqS9OjT2BDrk"  # <<< PASTE THE ID of "Known Faces" sheet HERE

# OPTION 2: Use Sheet Name (Fallback if SHEET_ID is None or fails)
# The exact name of your target Google Sheet document.
GOOGLE_SHEET_NAME = 'Known Faces' # <<< UPDATE if you prefer using name

# The exact name of the worksheet (tab) within your target Google Sheet.
GOOGLE_WORKSHEET_NAME = 'Sheet1' # <<< Usually 'Sheet1', change if needed


# --- Gemini API Configuration ---

# Your Google Generative AI (Gemini) API Key. Get from Google AI Studio.
# IMPORTANT: Keep this key secret! Do not share it publicly.
# !!! Replace "YOUR_GEMINI_API_KEY_HERE" with your actual key !!!
GEMINI_API_KEY = "AIzaSyAq2RnjENjhJ43kboM809JaiP5No36kziw"


# --- Face Recognition Settings ---

# How much distance between faces to consider it a match. Lower is stricter.
# 0.6 is typical default, 0.5 - 0.55 often works well. Experiment needed.
RECOGNITION_TOLERANCE = 0.55

# Factor to resize video frames BEFORE processing for faces. Saves computation.
# 0.25 means processing an image 1/4 the width and height (1/16 the pixels).
# Smaller values are faster but might miss small/distant faces.
RESIZE_FACTOR = 0.25

# --- Target Sheet Column Header Names ---
# Define the exact header names expected in Row 1 of your "Known Faces" sheet.
# Keys (left side) are used internally in the Python script.
# Values (right side, in quotes) MUST exactly match your sheet headers (case-sensitive).
SHEET_COLUMNS = {
    "name": "Name",             # Header for the person's name column
    "image_url": "Image_URL",   # Header for the Google Drive link column (standard link)
    "gender": "Gender",         # Header for Gender column
    "college": "College",       # Header for College column
    "studying": "Studying",      # Header for Studying column (Matches your sheet's 'Studing')
    "branch": "Branch"          # Header for Branch column
    # Add/remove lines here if your target sheet has different/more/fewer columns
}


