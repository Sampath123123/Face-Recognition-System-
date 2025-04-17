# Face Recognition System with Google Sheets Integration

This project implements a face recognition system using the `face_recognition` library and integrates with Google Sheets to manage known faces. It leverages the Google Drive API to download images of known individuals stored in a Google Sheet and uses these images to train the face recognition model.  The system then uses a webcam to perform real-time face recognition, identifying known faces based on the data in the Google Sheet.

## Key Features

*   **Face Recognition:** Uses the `face_recognition` library for accurate and efficient face detection and recognition.
*   **Google Sheets Integration:** Reads names and image URLs from a Google Sheet to populate the known faces database.  Supports both Sheet ID and Sheet Name for configuration.
*   **Google Drive API:** Downloads images from Google Drive using the provided URLs.  Handles authentication and authorization through a service account.
*   **Real-time Recognition:** Processes webcam footage to identify faces in real-time.
*   **Configurable Tolerance:** Allows adjusting the recognition tolerance for fine-tuning accuracy.
*   **Modular Design:**  Separates configuration loading, data fetching, and recognition logic for improved maintainability.
*   **Error Handling:** Includes robust error handling to manage potential issues with API connections, file downloads, and data processing.

## Prerequisites

*   Python 3.6 or higher
*   A Google Cloud project with the Google Drive API and Google Sheets API enabled.
*   A Google service account with access to the Google Sheet and Google Drive folder containing the images.
*   A `credentials.json` file containing the service account credentials.
*   A Google Sheet containing a list of names and corresponding Google Drive image URLs.
*   Libraries listed in `requirements.txt`.

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd face-recognition-google-sheets
    ```

2.  Create a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4.  Configure the project:

    *   Create a `config.py` file (or modify the existing `config_example.py`) and set the following variables:

        *   `GOOGLE_API_SCOPES`: The required Google API scopes.
        *   `GOOGLE_CREDS_FILE`: The path to your `credentials.json` file.
        *   `GOOGLE_SHEET_ID` or `GOOGLE_SHEET_NAME`:  The ID or name of your Google Sheet.
        *   `WORKSHEET_NAME`:  The name of the worksheet within the Google Sheet.
        *   `SHEET_COLUMNS`: A dictionary mapping column names in the sheet to the expected data (e.g., `{"name": "Name", "image_url": "Image URL"}`).
        *   `RECOGNITION_TOLERANCE`: The face recognition tolerance.

5.  Run the script:

    ```bash
    python main.py
    ```

## Configuration

The `config.py` file is used to configure the project.  Refer to the `config_example.py` file for an example configuration.  Ensure that the service account has the necessary permissions to access the Google Sheet and Google Drive files.

## Usage

Once the script is running, it will access your webcam and attempt to identify faces based on the data in your Google Sheet.  Press `q` to quit the application.

## Contributing

Contributions are welcome! Please submit pull requests with bug fixes, new features, or improvements to the documentation.

## License

This project is licensed under the [MIT License](LICENSE).
