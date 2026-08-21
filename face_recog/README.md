# Python `face_recognition` Practice Course for Beginners 🚀

Welcome! This directory contains a complete, step-by-step beginner practice guide for Python's `face_recognition` library.

---

## 🛠️ Environment Setup

Always activate the dedicated Conda environment before running any script:

```bash
conda activate face
```

---

## 📁 Practice Folder Overview

```text
face_recog/
├── images/                        # Practice images
│   ├── person_a.jpg               # Known Person A (Alice photo 1)
│   ├── person_a_2.jpg             # Known Person A (Alice photo 2 - outdoor)
│   ├── person_b.jpg               # Known Person B (Bob photo)
│   └── group.jpg                  # Group image with multiple faces
│
├── 01_load_image.py               # Lesson 1: load_image_file()
├── 02_face_detection.py           # Lesson 2: face_locations() & Bounding Boxes
├── 03_batch_face_locations.py     # Lesson 3: batch_face_locations()
├── 04_face_landmarks.py           # Lesson 4: face_landmarks() (Eyes, Nose, Lips, Chin)
├── 05_face_encodings.py           # Lesson 5: face_encodings() (128D Embeddings)
├── 06_face_encodings_from_locations.py # Lesson 6: Encoding with known locations
├── 07_face_comparison_and_distance.py  # Lesson 7: compare_faces() & face_distance()
├── 08_complete_workflow.py        # Lesson 8: Full End-to-End Recognition System
└── README.md                      # This documentation
```

---

## 🧠 Core `face_recognition` API Reference

| # | Purpose | Method | Beginner Description |
|---|---|---|---|
| 1 | **Image Loading** | `load_image_file(path)` | Loads image file into a 3D NumPy array of RGB pixel values |
| 2 | **Face Detection** | `face_locations(img, model="hog")` | Finds bounding box coordinates `(top, right, bottom, left)` for all faces |
| 3 | **Batch Detection** | `batch_face_locations(images)` | Detects face locations across multiple images at once (useful for videos/large batches) |
| 4 | **Facial Landmarks** | `face_landmarks(img)` | Extracts 68 facial points (`chin`, `eyes`, `eyebrows`, `nose`, `lips`) |
| 5 | **Face Encoding** | `face_encodings(img)` | Converts detected face into a unique 128-dimensional biometric vector |
| 6 | **Speed Optimization**| `face_encodings(img, locations)` | Reuses pre-detected face locations to generate encodings faster |
| 7 | **Face Distance** | `face_distance(known_list, test)` | Calculates numerical Euclidean distance between face embeddings |
| 8 | **Face Comparison** | `compare_faces(known_list, test, tolerance=0.6)` | Returns `True`/`False` matches based on distance tolerance threshold |

---

## 🔄 Complete Recognition Pipeline Workflow

```text
┌──────────────────────────────┐
│  1. Load Image               │  ← face_recognition.load_image_file()
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  2. Detect Face Locations    │  ← face_recognition.face_locations()
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  3. Extract Landmarks        │  ← face_recognition.face_landmarks() [Optional]
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  4. Compute 128D Encodings   │  ← face_recognition.face_encodings()
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  5. Compare / Match Distance │  ← face_recognition.compare_faces()
│                              │  ← face_recognition.face_distance()
└──────────────────────────────┘
```

---

## 🏃 How to Run the Lessons

Run each python script sequentially:

```bash
# Activate conda environment first
conda activate face

# Lesson 1: Load image & inspect array dimensions
python 01_load_image.py

# Lesson 2: Detect face locations & draw bounding box
python 02_face_detection.py

# Lesson 3: Detect faces in batch across multiple images
python 03_batch_face_locations.py

# Lesson 4: Extract 9 facial landmark features & draw landmark lines
python 04_face_landmarks.py

# Lesson 5: Generate 128-dimensional face embedding vector
python 05_face_encodings.py

# Lesson 6: Generate encodings using pre-computed face locations
python 06_face_encodings_from_locations.py

# Lesson 7: Calculate numerical face distance & test tolerance thresholds
python 07_face_comparison_and_distance.py

# Lesson 8: Full face identification pipeline (Gallery vs Unknown)
python 08_complete_workflow.py
```

Output visualization files saved automatically:
- `output_faces_detected.jpg` (Lesson 2 output)
- `output_landmarks.jpg` (Lesson 4 output)
- `output_final_recognition.jpg` (Lesson 8 output)
