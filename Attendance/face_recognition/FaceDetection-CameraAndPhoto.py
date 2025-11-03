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
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import logging
import argparse
import mysql.connector
from datetime import date
import sys
import os
import importlib.util

# Load config module dynamically from the parent directory
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.py'))
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
sys.modules['config'] = config
spec.loader.exec_module(config)


# ---------------------- HARDCODED CONFIG ----------------------
# Put your dataset folder here (one image per student: name.jpg)
DATASET_DIR = "dataset"

# Use GPU if available. CTX_ID = 0 -> GPU, -1 -> CPU
# Keep 0 for CUDA preference; the code will catch exceptions and retry on CPU if truly unavailable.
CTX_ID = 0

# Matching/config
THRESHOLD = 0.50           # cosine similarity threshold (tune)
DET_SIZE = (640, 640)      # SCRFD input size (increase for tiny faces)
SCRFD_CANDIDATES = ["scrfd_10g_bnkps", "scrfd_2.5g_bnkps", "scrfd_34g_bnkps"]

CAPTURE_QUEUE_SIZE = 2     # per-stream queue size (keep small)
PROCESS_INTERVAL = 0.1     # seconds between processing frames
RECOGNIZE_MIN_SEEN = 1     # frames required to mark present

HTTP_PORT = 8081

# ---------------------------------------------------------------

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = queue.Queue(maxsize=1)
        self.condition = threading.Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it is available
            with self.condition:
                self.frame = buf
                self.condition.notify_all()

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/stream.mjpg')
            self.end_headers()
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(b'Content-Length: ' + str(len(frame)).encode() + b'\r\n')
                    self.wfile.write(b'\r\n')
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

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
                if not fn.lower().endswith((".jpg", "jpeg", "png")):
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
        elif entry.lower().endswith((".jpg","jpeg","png")):
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
    
    try:
        with open("attendance.json", "w") as f:
            json.dump(attendance, f)
        print("[INFO] attendance.json updated.")
    except IOError as e:
        print(f"[ERROR] Failed to write to attendance.json: {e}")

    try:
        cnx = mysql.connector.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME
        )
        cursor = cnx.cursor()
        today = date.today().strftime('%Y-%m-%d')

        for student_name, status in attendance.items():
            # Get student_id from name
            cursor.execute("SELECT id FROM students WHERE name = %s", (student_name,))
            student = cursor.fetchall() # Explicitly fetch all results
            if student:
                student_id = student[0][0] # Access the ID from the fetched tuple
                # Check if attendance for this student on this day already exists
                cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND date = %s", (student_id, today))
                existing_attendance = cursor.fetchall() # Explicitly fetch all results
                if existing_attendance:
                    cursor.execute("UPDATE attendance SET status = %s WHERE student_id = %s AND date = %s", (status, student_id, today))
                else:
                    cursor.execute("INSERT INTO attendance (student_id, status, date) VALUES (%s, %s, %s)", (student_id, status, today))

        cnx.commit()
        cursor.close()
        cnx.close()
        print("[INFO] Database updated.")
    except mysql.connector.Error as err:
        print(f"[ERROR] Database update failed: {err}")

    print(f"[INFO] Photo mode done. Saved {out_img} and {csv_name}")


# ---------- streams mode ----------
def run_streams_mode(app, scrfd, names, db_embs, stream_sources):
    global output
    output = StreamingOutput()
    server = StreamingServer(('', HTTP_PORT), StreamingHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print(f"[INFO] Streaming server started on port {HTTP_PORT}")

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
                
                ret, jpeg = cv2.imencode('.jpg', annotated)
                if ret:
                    output.write(jpeg.tobytes())
                    # print(f"[INFO] Wrote JPEG frame to output for stream {s['src']}")

                # Update attendance json
                attendance_data = {name: ("PRESENT" if present else "ABSENT") for name, present in s["present"].items()}
                print(f"[DEBUG] Attendance data to write for stream {s['src']}: {attendance_data}")
                try:
                    with open("attendance.json", "w") as f:
                        json.dump(attendance_data, f)
                    print(f"[INFO] attendance.json updated for stream {s['src']}")
                except IOError as e:
                    print(f"[ERROR] Failed to write to attendance.json for stream {s['src']}: {e}")

            elapsed = time.time() - start
            sleep_for = max(0.0, PROCESS_INTERVAL - elapsed)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("[INFO] Stopping stream processing...")
    finally:
        for s in streams:
            s["thread"].stop()
        server.shutdown()
        server.server_close()
        for i, s in enumerate(streams):
            csv_name = f"attendance_stream{i}_{ts_now_str()}.csv"
            pd.DataFrame(list(s["present"].items()), columns=["Name","Status"]).replace({True:"PRESENT", False:"ABSENT"}).to_csv(csv_name, index=False)
            print(f"[INFO] Saved {csv_name}")
        print("[INFO] All done.")


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="streams", help="Operating mode: streams or photo")
    parser.add_argument("--stream_sources", type=str, default="0", help="Comma-separated list of stream sources")
    parser.add_argument("--photo_path", type=str, help="Path to the photo for photo mode")
    args = parser.parse_args()

    try:
        stream_sources = [int(x) for x in args.stream_sources.split(',')]
    except ValueError:
        stream_sources = [args.stream_sources]


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

    if args.mode == "photo":
        if not args.photo_path:
            raise SystemExit("Photo mode requires --photo_path argument.")
        run_photo_mode(app, scrfd, names, db_embs, args.photo_path)
    else:
        if not stream_sources:
            raise SystemExit("No streams provided. Set --stream_sources at top of file.")
        run_streams_mode(app, scrfd, names, db_embs, stream_sources)


if __name__ == "__main__":
    main()
