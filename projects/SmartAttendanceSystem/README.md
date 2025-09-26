# Smart Attendance System

Face recognition-based attendance logger using OpenCV and `face_recognition`.

For a detailed architecture, data flow, and call graph, see `docs/ARCHITECTURE.md`.

Open the browser-friendly documentation: `docs/index.html`.

For a practical What/Why/How explanation, see:
- Markdown: `docs/IMPLEMENTATION_GUIDE.md`
- HTML (in the SPA): Implementation section in `docs/index.html`

## Project layout

```
SmartAttendanceSystem/
├── smart_attendance_system.py   # core functions and main entry
├── run_webcam.py                # convenience launcher for webcam mode
├── run_image.py                 # recognize faces in a single image
├── requirements.txt             # Python dependencies
├── known_faces/                 # put subfolders per person with their photos
│   ├── Alice/
│   │   └── alice1.png
│   └── Bob/
│       └── bob1.png
├── test_images/                 # optional folder to test on images
└── attendance.csv               # created at runtime
```

## Setup (Windows)

1. Install Python 3.10 or 3.11 from the Microsoft Store or python.org.
2. Create and activate a virtual environment (recommended):

```
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```
pip install --upgrade pip wheel
pip install -r requirements.txt
```

If `face_recognition` fails to build due to `dlib`, use a prebuilt wheel for your Python version (search "dlib‑<version>‑cp311‑win_amd64.whl") and install it first, then install `face_recognition`.

## Prepare known faces

- Create a folder per person inside `known_faces/` with 1–3 clear, frontal photos.
- File types supported: .jpg, .jpeg, .png
- Folder name is used as the person's display name.

Example:

```
known_faces/
  Alice/
    alice1.png
  Bob/
    bob1.png
```

## Run

- Webcam attendance (press `q` to quit):

```
python run_webcam.py
```

- Recognize a single image:

```
python run_image.py test_images\\alice1.png
```

- Run the original script directly and edit the `__main__` section if you want to experiment:

```
python smart_attendance_system.py
```

Attendance will be saved to `attendance.csv` in the project root. Duplicates for the same person per day are skipped by default.

## Notes

- Threshold is set to 0.6; lower it for stricter matching.
- For many identities, consider precomputing encodings and saving to disk.
- You can open `attendance.csv` with Excel to review logs.
