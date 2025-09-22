# attendance_retinaface_draw_recognized_only.py
"""
RetinaFace + InsightFace attendance script (DRAW BOXES ONLY FOR RECOGNIZED FACES).

- Builds a DB from dataset/ (one image per student: name.jpg)
- Detects & recognizes faces in classroom.jpg
- Writes attendance.csv (Name,Status)
- Draws boxes + names only for recognized faces (sim >= THRESHOLD)
- Saves annotated image as classroom_out.jpg and attempts to display it.
"""

import os
import cv2
import numpy as np
import pandas as pd
import traceback
from insightface.app import FaceAnalysis
import insightface

DATASET_DIR = "dataset"
CLASSROOM_IMAGE = "classroom.jpeg"
OUTPUT_CSV = "attendance.csv"
OUTPUT_IMAGE = "classroom_out.jpg"

THRESHOLD = 0.50           # cosine threshold for matching (tweak to your data)
PREFERRED_CTX_ID = 0        # 0 => GPU, -1 => CPU
FORCE_CPU_IF_FAIL = True

def try_load_retina_detector(ctx_id=0):
    try:
        candidates = [
            "retinaface_r50_v1",
            "retinaface_mnet025_v1",
            "retinaface_mnet025_v2",
            "retinaface_r50_v2",
        ]
        for name in candidates:
            try:
                detector = insightface.model_zoo.get_model(name)
                detector.prepare(ctx_id=ctx_id, det_size=(640, 640))
                print(f"[INFO] Loaded RetinaFace detector '{name}' (ctx_id={ctx_id})")
                return detector
            except Exception:
                continue
        print("[WARN] No common retinaface model found in model_zoo. Falling back to FaceAnalysis detector.")
        return None
    except Exception as ex:
        print("[WARN] Exception while loading retinaface detector:", ex)
        return None

def create_face_analysis(ctx_id=0, det_size=(640,640), providers=None):
    try:
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=det_size)
        print(f"[INFO] FaceAnalysis prepared (ctx_id={ctx_id})")
        return app
    except Exception as e:
        print("[WARN] Failed to prepare FaceAnalysis:", e)
        raise

def detect_with_detector(detector, img_bgr, threshold=0.35, scale=1.0):
    try:
        bboxes, kps = detector.detect(img_bgr, threshold=threshold, scale=scale)
        return bboxes, kps
    except Exception as e:
        print("[WARN] detector.detect() failed:", e)
        return None, None

def build_db(app, detector=None):
    db = {}
    files = sorted(os.listdir(DATASET_DIR))
    for fn in files:
        if not fn.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        name = os.path.splitext(fn)[0]
        path = os.path.join(DATASET_DIR, fn)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"[WARN] Unable to read {path}, skipping")
            continue

        try:
            if detector is not None:
                bboxes, kps = detect_with_detector(detector, img_bgr, threshold=0.35, scale=1.0)
                if bboxes is None or len(bboxes) == 0:
                    faces = app.get(img_bgr)
                    if not faces:
                        print(f"[WARN] No face found in {fn}. skipping.")
                        continue
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    emb = face.normed_embedding
                else:
                    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes]
                    idx = int(np.argmax(areas))
                    box = bboxes[idx].astype(int)
                    x1,y1,x2,y2 = box[0], box[1], box[2], box[3]
                    crop = img_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                    if crop.size == 0:
                        faces = app.get(img_bgr)
                        if not faces:
                            print(f"[WARN] No face embedding available for {fn}. skipping.")
                            continue
                        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                        emb = face.normed_embedding
                    else:
                        faces_in_crop = app.get(crop)
                        if not faces_in_crop:
                            faces = app.get(img_bgr)
                            if not faces:
                                print(f"[WARN] No face embedding available for {fn}. skipping.")
                                continue
                            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                            emb = face.normed_embedding
                        else:
                            emb = faces_in_crop[0].normed_embedding
            else:
                faces = app.get(img_bgr)
                if not faces:
                    print(f"[WARN] No face found in {fn}. skipping.")
                    continue
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                emb = face.normed_embedding

            emb = emb / (np.linalg.norm(emb) + 1e-12)
            db[name] = emb
            print(f"[INFO] Registered: {name}")
        except Exception as e:
            print(f"[ERROR] Error processing {fn}: {e}")
            traceback.print_exc()
            continue

    return db

def recognize_and_annotate(app, db, detector=None, classroom_path=CLASSROOM_IMAGE, threshold=THRESHOLD):
    attendance = {name: "ABSENT" for name in db.keys()}
    img_bgr = cv2.imread(classroom_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read classroom image: {classroom_path}")

    annotated = img_bgr.copy()
    detections_for_draw = []  # list of dicts: {bbox, name, sim}

    names = list(db.keys())
    db_embs = np.vstack([db[n] for n in names]) if names else np.zeros((0,512))

    # Use RetinaFace detector if available to get accurate boxes
    use_faceanalysis = False
    if detector is not None:
        bboxes, kps = detect_with_detector(detector, img_bgr, threshold=0.35, scale=1.0)
        if bboxes is None or len(bboxes) == 0:
            print("[INFO] RetinaFace found no faces; falling back to FaceAnalysis detection.")
            use_faceanalysis = True
        else:
            for i, box in enumerate(bboxes):
                box = box.astype(int)
                x1,y1,x2,y2 = box[0], box[1], box[2], box[3]
                crop = img_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                if crop.size == 0:
                    continue
                faces_in_crop = app.get(crop)
                if not faces_in_crop:
                    continue
                emb = faces_in_crop[0].normed_embedding
                emb = emb / (np.linalg.norm(emb) + 1e-12)
                if db_embs.shape[0] == 0:
                    continue
                sims = db_embs.dot(emb)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_name = names[best_idx]
                print(f"[DETECT] bbox#{i} matched {best_name} (sim={best_sim:.3f})")
                # Only record/draw if similarity passes threshold (recognized)
                if best_sim >= threshold:
                    attendance[best_name] = "PRESENT"
                    detections_for_draw.append({
                        "bbox": (x1, y1, x2, y2),
                        "name": best_name,
                        "sim": best_sim
                    })

    else:
        use_faceanalysis = True

    # FaceAnalysis detection/embedding path (fallback or if detector found nothing)
    if use_faceanalysis:
        faces = app.get(img_bgr)
        if not faces:
            print("[INFO] No faces found by FaceAnalysis either.")
        else:
            for f in faces:
                x1, y1, x2, y2 = [int(x) for x in f.bbox]
                emb = f.normed_embedding
                emb = emb / (np.linalg.norm(emb) + 1e-12)
                if db_embs.shape[0] == 0:
                    continue
                sims = db_embs.dot(emb)
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_name = names[best_idx]
                print(f"[FaceAnalysis] matched {best_name} (sim={best_sim:.3f})")
                # Only record/draw if similarity passes threshold (recognized)
                if best_sim >= threshold:
                    attendance[best_name] = "PRESENT"
                    detections_for_draw.append({
                        "bbox": (x1, y1, x2, y2),
                        "name": best_name,
                        "sim": best_sim
                    })

    # Draw rectangles and labels ONLY for recognized faces (detections_for_draw)
    annotated = draw_detections(annotated, detections_for_draw)

    # Save annotated image
    cv2.imwrite(OUTPUT_IMAGE, annotated)
    print(f"[INFO] Annotated image saved to {OUTPUT_IMAGE}")

    # Try to display (will fail on headless servers)
    try:
        window_name = "Attendance - Press any key to close"
        cv2.imshow(window_name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass

    return attendance

def draw_detections(img, detections):
    """
    img: BGR image
    detections: list of {bbox: (x1,y1,x2,y2), name: str, sim: float}
    Returns annotated image (BGR) with boxes/labels only for recognized faces.
    """
    out = img.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(w, h) / 800.0)
    thickness = max(1, int(round(scale * 2)))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))

        color = (36, 255, 12)   # green-ish for recognized
        label = f"{det['name']} {det['sim']:.2f}"

        # draw rectangle
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        # put filled rectangle behind text for readability
        ((text_w, text_h), _) = cv2.getTextSize(label, font, fontScale=scale, thickness=thickness)
        text_offset_x = x1
        text_offset_y = y1 - 8
        if text_offset_y - text_h - 4 < 0:
            text_offset_y = y1 + text_h + 10

        box_coords = ((text_offset_x, text_offset_y - text_h - 4), (text_offset_x + text_w + 6, text_offset_y))
        cv2.rectangle(out, box_coords[0], box_coords[1], color, cv2.FILLED)
        cv2.putText(out, label, (text_offset_x + 2, text_offset_y - 4), font, fontScale=scale,
                    color=(255,255,255), thickness=thickness, lineType=cv2.LINE_AA)

    return out

def save_csv(attendance, out_file=OUTPUT_CSV):
    df = pd.DataFrame(list(attendance.items()), columns=["Name","Status"])
    df.to_csv(out_file, index=False)
    print(f"[INFO] Saved attendance -> {out_file}")

def main():
    detector = None
    app = None
    try:
        try:
            app = create_face_analysis(ctx_id=PREFERRED_CTX_ID)
        except Exception:
            if FORCE_CPU_IF_FAIL:
                print("[WARN] Falling back to CPU for FaceAnalysis")
                app = create_face_analysis(ctx_id=-1, providers=['CPUExecutionProvider'])
            else:
                raise

        try:
            detector = try_load_retina_detector(ctx_id=PREFERRED_CTX_ID)
            if detector is None:
                print("[INFO] Will use FaceAnalysis's detector.")
        except Exception as ex:
            print("[WARN] Could not create retinaface detector:", ex)
            detector = None

        db = build_db(app, detector=detector)
        if not db:
            print("[ERROR] No registered students found in dataset/ — aborting.")
            return

        attendance = recognize_and_annotate(app, db, detector=detector)
        save_csv(attendance)

    except Exception as e:
        print("[FATAL] Unexpected error:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
