#!/usr/bin/env python3
"""
SCRFD + ArcFace attendance app (photo OR multi-streams)

This variant hardcodes configuration at the top (no CLI args),
prefers CUDA (will fall back to CPU if unavailable), and uses
timestamp format: 30-12-2020_12-24-25 for all saved filenames.

Run:
  python attendance_scrfd_multi_cuda_hardcoded.py

Dependencies:
  pip install insightface opencv-python numpy pandas onnxruntime-gpu
"""

import os
import cv2
import time
import math
import queue
import threading
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from insightface.app import FaceAnalysis
import insightface

# ---------------------- HARDCODED CONFIG ----------------------
# Put your dataset folder here (one image per student: name.jpg)
DATASET_DIR = "dataset"
# Mode: "photo" or "streams"
MODE = "streams"
# Photo path (used when MODE == "photo")
PHOTO_PATH = "classroom.jpeg"
# Default streams. 0 selects the laptop/default camera. Add more sources as needed (including RTSP/HTTP URLs).
STREAM_SOURCES = [2]

# Use GPU if available. CTX_ID = 0 -> GPU, -1 -> CPU
# Keep 0 for CUDA preference; the code will catch exceptions and retry on CPU if truly unavailable.
CTX_ID = 0

# Matching/config
THRESHOLD = 0.50           # cosine similarity threshold (tune)
DET_SIZE = (640, 640)      # SCRFD input size (increase for tiny faces)
SCRFD_CANDIDATES = ["scrfd_10g_bnkps", "scrfd_2.5g_bnkps", "scrfd_34g_bnkps"]

CAPTURE_QUEUE_SIZE = 2     # per-stream queue size (keep small)
PROCESS_INTERVAL = 0.01     # seconds between processing frames
RECOGNIZE_MIN_SEEN = 1     # frames required to mark present

# ---------------------------------------------------------------

# Helper: consistent timestamp format dd-mm-YYYY_HH-MM-SS
def ts_now_str():
    return datetime.now().strftime("%d-%m-%Y_%H-%M-%S")


def l2norm(x, eps=1e-12):
    return x / (np.linalg.norm(x) + eps)


def load_scrfd(ctx_id=0, det_size=DET_SIZE):
    for name in SCRFD_CANDIDATES:
        try:
            det = insightface.model_zoo.get_model(name)
            # try to set input size if available
            try:
                det.prepare(ctx_id=ctx_id, input_size=det_size)
            except Exception:
                pass
            print(f"[INFO] Loaded SCRFD detector {name}")
            return det
        except Exception as e:
            print(f"[WARN] SCRFD '{name}' not available: {e}")
    print("[WARN] No SCRFD model loaded; will fallback to FaceAnalysis detection.")
    return None


def create_face_app(ctx_id=CTX_ID):
    # prefer CUDA execution provider when available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
    print("[INFO] FaceAnalysis prepared (ArcFace embeddings ready).")
    return app


# ---------- gallery building ----------
def build_gallery(app, scrfd, dataset_dir):
    db = {}
    for entry in sorted(os.listdir(dataset_dir)):
        entry_path = os.path.join(dataset_dir, entry)

        # case 1: folder with multiple images
        if os.path.isdir(entry_path):
            name = entry.replace("-", " ")  # folder name is the student's name
            embs = []
            for fn in sorted(os.listdir(entry_path)):
                if not fn.lower().endswith(('.jpg', 'jpeg', 'png')):
                    continue
                path = os.path.join(entry_path, fn)
                img = cv2.imread(path)
                if img is None:
                    print(f"[WARN] cannot read {path}")
                    continue

                emb = None
                if scrfd is not None:
                    try:
                        bboxes, _ = scrfd.detect(img, 0.3, input_size=DET_SIZE)
                    except Exception:
                        bboxes = None
                    if bboxes is not None and len(bboxes) > 0:
                        # pick largest box
                        areas = (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
                        x1,y1,x2,y2 = bboxes[np.argmax(areas)][:4].astype(int)
                        crop = img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                        faces = app.get(crop)
                        if faces:
                            emb = faces[0].normed_embedding

                if emb is None:
                    faces = app.get(img)
                    if faces:
                        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                        emb = face.normed_embedding

                if emb is not None:
                    embs.append(l2norm(emb))
                    print(f"[INFO] Registered {name} from {fn}")
                else:
                    print(f"[WARN] No face found in {path}")

            if embs:
                # average embeddings from multiple images
                db[name] = l2norm(np.mean(embs, axis=0))

        # case 2: single image file
        elif entry.lower().endswith(('.jpg','jpeg','png')):
            name = os.path.splitext(entry)[0]
            img = cv2.imread(entry_path)
            if img is None:
                print(f"[WARN] cannot read {entry_path}")
                continue

            emb = None
            if scrfd is not None:
                try:
                    bboxes, _ = scrfd.detect(img, 0.3, input_size=DET_SIZE)
                except Exception:
                    bboxes = None
                if bboxes is not None and len(bboxes) > 0:
                    areas = (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
                    x1,y1,x2,y2 = bboxes[np.argmax(areas)][:4].astype(int)
                    crop = img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                    faces = app.get(crop)
                    if faces:
                        emb = faces[0].normed_embedding

            if emb is None:
                faces = app.get(img)
                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    emb = face.normed_embedding

            if emb is not None:
                db[name] = l2norm(emb)
                print(f"[INFO] Registered {name}")
            else:
                print(f"[WARN] No face found for {entry}")

    if len(db) == 0:
        raise SystemExit("No registered students found in dataset.")

    names = list(db.keys())
    embs = np.vstack([db[n] for n in names])
    return names, embs, db


# ---------- capture thread ----------
class CaptureThread(threading.Thread):
    def __init__(self, src, out_q, name=None):
        super().__init__(daemon=True)
        self.src = src
        self.q = out_q
        self.name = name or str(src)
        self.cap = None
        self.running = True

    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                print(f"[ERROR] Capture failed for {self.src}")
                self.running = False
                return
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    # stream ended or dropped - sleep briefly then continue
                    time.sleep(0.1)
                    continue
                try:
                    if self.q.full():
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                    self.q.put_nowait(frame)
                except Exception:
                    pass
            self.cap.release()
        except Exception as e:
            print(f"[ERROR] CaptureThread exception for {self.src}: {e}")
            traceback.print_exc()
            self.running = False

    def stop(self):
        self.running = False


# ---------- processing ----------
def process_frame_and_match(app, scrfd, frame, names, db_embs):
    draws = []
    try:
        if scrfd is not None:
            try:
                bboxes, _ = scrfd.detect(frame, 0.3, input_size=DET_SIZE)
            except Exception:
                bboxes = None
            if bboxes is None or len(bboxes)==0:
                faces = app.get(frame)
                if faces:
                    for f in faces:
                        x1,y1,x2,y2 = [int(x) for x in f.bbox]
                        emb = l2norm(f.normed_embedding)
                        sims = db_embs.dot(emb)
                        best = int(np.argmax(sims)); sim = float(sims[best])
                        if sim >= THRESHOLD:
                            draws.append((x1,y1,x2,y2,names[best],sim))
                return draws
            else:
                for box in bboxes:
                    x1,y1,x2,y2 = [int(v) for v in box[:4]]
                    crop = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                    if crop.size==0: continue
                    faces = app.get(crop)
                    if not faces: continue
                    emb = l2norm(faces[0].normed_embedding)
                    sims = db_embs.dot(emb)
                    best = int(np.argmax(sims)); sim = float(sims[best])
                    if sim >= THRESHOLD:
                        draws.append((x1,y1,x2,y2,names[best],sim))
                return draws
        else:
            faces = app.get(frame)
            for f in faces:
                x1,y1,x2,y2 = [int(x) for x in f.bbox]
                emb = l2norm(f.normed_embedding)
                sims = db_embs.dot(emb)
                best = int(np.argmax(sims)); sim = float(sims[best])
                if sim >= THRESHOLD:
                    draws.append((x1,y1,x2,y2,names[best],sim))
            return draws
    except Exception as e:
        print("[WARN] processing error:", e)
        return draws


def draw_boxes(img, draws):
    out = img.copy()
    h,w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(w,h)/900.0)
    thickness = max(1,int(round(scale*2)))
    for (x1,y1,x2,y2,name,sim) in draws:
        color = (36,255,12)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,thickness)
        label = f"{name} {sim:.2f}"
        (tw,th),_ = cv2.getTextSize(label,font,scale,thickness)
        ty = y1-8
        if ty-th-4 < 0: ty = y1 + th + 10
        cv2.rectangle(out,(x1,ty-th-4),(x1+tw+6,ty),color,cv2.FILLED)
        cv2.putText(out,label,(x1+2,ty-4),font,scale,(255,255,255),thickness,cv2.LINE_AA)
    return out


# ---------- photo mode ----------
def run_photo_mode(app, scrfd, names, db_embs, photo_path):
    img = cv2.imread(photo_path)
    if img is None:
        raise FileNotFoundError(photo_path)
    draws = process_frame_and_match(app, scrfd, img, names, db_embs)
    annotated = draw_boxes(img, draws)
    out_img = "classroom_out.jpg"
    cv2.imwrite(out_img, annotated)
    ts = ts_now_str()
    csv_name = f"attendance_photo_{ts}.csv"
    attendance = {n:"ABSENT" for n in names}
    for (_,_,_,_,name,_) in draws:
        attendance[name] = "PRESENT"
    pd.DataFrame(list(attendance.items()), columns=["Name","Status"]).to_csv(csv_name,index=False)
    print(f"[INFO] Photo mode done. Saved {out_img} and {csv_name}")


# ---------- streams mode ----------
def run_streams_mode(app, scrfd, names, db_embs, stream_sources):
    streams = []
    for i, src in enumerate(stream_sources):
        q = queue.Queue(maxsize=CAPTURE_QUEUE_SIZE)
        t = CaptureThread(src, q, name=f"stream{i}")
        t.start()
        streams.append({
            "src": src,
            "thread": t,
            "queue": q,
            "seen_counts": {n:0 for n in names},
            "present": {n:False for n in names},
            "last_draw": None,
        })

    print("[INFO] Streams started. Processing frames. Press 'q' to quit, 's' to save snapshots for all streams.")
    try:
        while True:
            start = time.time()
            for i, s in enumerate(streams):
                try:
                    frame = s["queue"].get_nowait()
                except queue.Empty:
                    frame = None
                if frame is None:
                    continue
                draws = process_frame_and_match(app, scrfd, frame, names, db_embs)
                seen_this_frame = set([d[4] for d in draws])
                for nm in names:
                    if nm in seen_this_frame:
                        s["seen_counts"][nm] += 1
                    else:
                        s["seen_counts"][nm] = max(0, s["seen_counts"][nm]-1)
                    if s["seen_counts"][nm] >= RECOGNIZE_MIN_SEEN:
                        s["present"][nm] = True
                annotated = draw_boxes(frame, draws)
                s["last_draw"] = annotated
                win = f"stream-{i}"
                try:
                    cv2.imshow(win, annotated)
                except Exception:
                    pass
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    raise KeyboardInterrupt()
                if key == ord('s'):
                    fn = f"stream{i}_snap_{ts_now_str()}.jpg"
                    cv2.imwrite(fn, annotated)
                    print(f"[INFO] Saved snapshot {fn}")
            elapsed = time.time() - start
            sleep_for = max(0.0, PROCESS_INTERVAL - elapsed)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("[INFO] Stopping stream processing...")
    finally:
        for s in streams:
            s["thread"].stop()
        cv2.destroyAllWindows()
        for i, s in enumerate(streams):
            csv_name = f"attendance_stream{i}_{ts_now_str()}.csv"
            pd.DataFrame(list(s["present"].items()), columns=["Name","Status"]).replace({True:"PRESENT", False:"ABSENT"}).to_csv(csv_name, index=False)
            print(f"[INFO] Saved {csv_name}")
        print("[INFO] All done.")


# ---------- main ----------
def main():
    # ensure CUDA device visibility is forced (if you want to lock to a GPU index, change below)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    # prepare models (try GPU first; fall back to CPU automatically)
    try:
        app = create_face_app(ctx_id=CTX_ID)
    except Exception as e:
        print("[WARN] FaceAnalysis failed with GPU, retrying CPU:", e)
        app = create_face_app(ctx_id=-1)

    scrfd = load_scrfd(ctx_id=CTX_ID, det_size=DET_SIZE)

    # build gallery
    names, db_embs, db_map = build_gallery(app, scrfd, DATASET_DIR)

    if MODE == "photo":
        run_photo_mode(app, scrfd, names, db_embs, PHOTO_PATH)
    else:
        if not STREAM_SOURCES:
            raise SystemExit("No streams provided. Set STREAM_SOURCES at top of file.")
        run_streams_mode(app, scrfd, names, db_embs, STREAM_SOURCES)


if __name__ == "__main__":
    main()
