# attendance_scrfd.py
"""
SCRFD + ArcFace attendance script (InsightFace, GPU-ready).

- Detects faces with SCRFD (great on small/distant faces)
- Extracts embeddings with ArcFace via FaceAnalysis
- Matches by cosine similarity against dataset/ (one image per student; filename = student name)
- Writes attendance.csv and classroom_out.jpg
- Draws boxes/labels ONLY for recognized faces (sim >= THRESHOLD)

Dependencies:
  pip install insightface opencv-python numpy pandas
  # GPU:
  pip install onnxruntime-gpu
"""

import os
import cv2
import numpy as np
import pandas as pd
import traceback
import insightface
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"
CLASSROOM_IMAGE = "classroom.jpeg"
OUTPUT_CSV = "attendance.csv"
OUTPUT_IMAGE = "classroom_out.jpg"

# cosine similarity threshold (tune on your data)
THRESHOLD = 0.50

# GPU: 0 for first GPU, -1 for CPU
CTX_ID = 1
DET_SIZE = (640, 640)  # SCRFD input size (trade-off between speed & small-face recall)
SCRFD_CANDIDATES = [
    "scrfd_10g_bnkps",  # good balance (small-face friendly)
    "scrfd_2.5g_bnkps", # lighter
    "scrfd_34g_bnkps",  # heavier / strongest
]

# ----------------------------------------

def load_scrfd_detector(ctx_id=1, det_size=(640, 640)):
    """Try SCRFD variants from model zoo; return the first that works."""
    for name in SCRFD_CANDIDATES:
        try:
            det = insightface.model_zoo.get_model(name)
            # SCRFD uses .prepare(ctx_id=?, input_size=?)
            det.prepare(ctx_id=ctx_id, input_size=det_size)
            print(f"[INFO] Loaded SCRFD detector '{name}' (ctx_id={ctx_id}, input_size={det_size})")
            return det
        except Exception as e:
            print(f"[WARN] Could not load '{name}': {e}")
    print("[WARN] No SCRFD model loaded; will fall back to FaceAnalysis detector if available.")
    return None

def create_face_analysis(ctx_id=1, providers=None):
    """
    FaceAnalysis bundles high-quality ArcFace embeddings (e.g., 'buffalo_l').
    We’ll use it for recognition, and as a fallback detector if SCRFD fails.
    """
    if providers is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # GPU→CPU fallback
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    print(f"[INFO] FaceAnalysis prepared (ctx_id={ctx_id}, providers={providers})")
    return app

def detect_scrfd(detector, img_bgr, thresh=0.3):
    """
    SCRFD detect: returns (bboxes[N,4], kps[N,5,2]) in image coords.
    """
    try:
        bboxes, kps = detector.detect(img_bgr, thresh, input_size=DET_SIZE)
        return bboxes, kps
    except Exception as e:
        print("[WARN] SCRFD detect() failed:", e)
        return None, None

def l2norm(x, eps=1e-12):
    return x / (np.linalg.norm(x) + eps)

def build_db(app, scrfd=None):
    """
    For each student image:
      1) detect face (prefer SCRFD; fallback to app.get)
      2) compute ArcFace embedding
    """
    db = {}
    files = sorted(os.listdir(DATASET_DIR))
    for fn in files:
        if not fn.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        name = os.path.splitext(fn)[0]
        path = os.path.join(DATASET_DIR, fn)
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Cannot read {path}, skipping.")
            continue

        emb = None

        # prefer SCRFD detection, then embed via app.get on the crop
        if scrfd is not None:
            bboxes, _ = detect_scrfd(scrfd, img, thresh=0.3)
            if bboxes is not None and len(bboxes) > 0:
                # pick largest box
                areas = (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
                x1, y1, x2, y2 = bboxes[np.argmax(areas)][:4].astype(int)
                crop = img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                faces = app.get(crop)
                if faces:
                    emb = faces[0].normed_embedding

        # fallback: let FaceAnalysis detect & embed directly
        if emb is None:
            faces = app.get(img)
            if faces:
                # choose largest face
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                emb = face.normed_embedding

        if emb is None:
            print(f"[WARN] No face/embedding for {fn}, skipping.")
            continue

        emb = l2norm(emb)
        db[name] = emb
        print(f"[INFO] Registered {name}")
    return db

def recognize_and_draw(app, db, scrfd=None, classroom_path=CLASSROOM_IMAGE, threshold=THRESHOLD):
    """
    Detect faces (SCRFD preferred). For each detection:
      - compute embedding
      - match against DB
      - if sim >= threshold → mark PRESENT and draw box/label
    """
    attendance = {name: "ABSENT" for name in db.keys()}

    img = cv2.imread(classroom_path)
    if img is None:
        raise FileNotFoundError(f"Could not read classroom image: {classroom_path}")

    names = list(db.keys())
    db_embs = np.vstack([db[n] for n in names]) if names else np.zeros((0,512))
    annotated = img.copy()
    draws = []

    used_faceanalysis_detection = False

    if scrfd is not None:
        bboxes, _ = detect_scrfd(scrfd, img, thresh=0.3)
        if bboxes is None or len(bboxes) == 0:
            print("[INFO] SCRFD found no faces; falling back to FaceAnalysis detection.")
            used_faceanalysis_detection = True
        else:
            for i, box in enumerate(bboxes):
                x1, y1, x2, y2 = box[:4].astype(int)
                crop = img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                if crop.size == 0:
                    continue
                faces_in_crop = app.get(crop)
                if not faces_in_crop:
                    continue
                emb = l2norm(faces_in_crop[0].normed_embedding)
                if db_embs.shape[0] == 0:
                    continue
                sims = db_embs.dot(emb)
                j = int(np.argmax(sims))
                sim = float(sims[j])
                if sim >= threshold:
                    name = names[j]
                    attendance[name] = "PRESENT"
                    draws.append((x1, y1, x2, y2, name, sim))
    else:
        used_faceanalysis_detection = True

    if used_faceanalysis_detection:
        faces = app.get(img)
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            emb = l2norm(f.normed_embedding)
            if db_embs.shape[0] == 0:
                continue
            sims = db_embs.dot(emb)
            j = int(np.argmax(sims))
            sim = float(sims[j])
            if sim >= threshold:
                name = names[j]
                attendance[name] = "PRESENT"
                draws.append((x1, y1, x2, y2, name, sim))

    # draw only recognized faces
    annotated = draw_boxes(annotated, draws)
    cv2.imwrite(OUTPUT_IMAGE, annotated)
    print(f"[INFO] Annotated image saved to {OUTPUT_IMAGE}")

    # optional: display window (ignored on headless)
    try:
        cv2.imshow("Attendance (recognized only)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass

    return attendance

def draw_boxes(img, items):
    out = img.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(w, h) / 800.0)
    thick = max(1, int(round(scale * 2)))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x1, y1, x2, y2, name, sim) in items:
        x1 = max(0, min(x1, w-1)); y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1)); y2 = max(0, min(y2, h-1))
        color = (36, 255, 12)
        label = f"{name} {sim:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(label, font, fontScale=scale, thickness=thick)
        ty = y1 - 8
        if ty - th - 4 < 0: ty = y1 + th + 10
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty), color, cv2.FILLED)
        cv2.putText(out, label, (x1 + 2, ty - 4), font, fontScale=scale, color=(255,255,255), thickness=thick, lineType=cv2.LINE_AA)

    return out

def save_csv(attendance, path=OUTPUT_CSV):
    df = pd.DataFrame(list(attendance.items()), columns=["Name", "Status"])
    df.to_csv(path, index=False)
    print(f"[INFO] Saved attendance -> {path}")

def main():
    try:
        # Prepare embedding/analysis app (ArcFace), with GPU→CPU provider list
        try:
            app = create_face_analysis(ctx_id=CTX_ID, providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        except Exception as e:
            print(f"[WARN] FaceAnalysis GPU init failed: {e} -> using CPU only.")
            app = create_face_analysis(ctx_id=-1, providers=['CPUExecutionProvider'])

        # Load SCRFD detector on GPU if possible
        scrfd = None
        try:
            scrfd = load_scrfd_detector(ctx_id=CTX_ID, det_size=DET_SIZE)
        except Exception as e:
            print(f"[WARN] SCRFD init failed: {e}")

        # Build gallery
        db = build_db(app, scrfd=scrfd)
        if not db:
            print("[ERROR] No registered students found in dataset/. Aborting.")
            return

        # Recognize classroom and write outputs
        attendance = recognize_and_draw(app, db, scrfd=scrfd, classroom_path=CLASSROOM_IMAGE, threshold=THRESHOLD)
        save_csv(attendance)

    except Exception as e:
        print("[FATAL] Unexpected error:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
