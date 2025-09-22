from flask import Flask, render_template, request, jsonify, Response, send_file
import subprocess
import os
import json
import atexit
import requests
import time
import csv
from io import StringIO
import mysql.connector
from datetime import date

app = Flask(__name__)

# --- Database Configuration ---
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "nipe-attendance"

# Global variable to hold the face recognition process
face_recognition_process = None

def get_db_connection():
    """Establishes a connection to the database."""
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

def kill_face_recognition_process():
    global face_recognition_process
    if face_recognition_process:
        if os.name == 'nt':
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(face_recognition_process.pid)])
        else:
            os.killpg(os.getpgid(face_recognition_process.pid), subprocess.SIGTERM)
        face_recognition_process.wait()
        face_recognition_process = None

atexit.register(kill_face_recognition_process)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/manual")
def manual_attendance_page():
    """Displays the manual attendance page with a list of students."""
    try:
        cnx = get_db_connection()
        cursor = cnx.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students")
        students = cursor.fetchall()
        cursor.close()
        cnx.close()
        return render_template("manual_attendance.html", students=students)
    except mysql.connector.Error as err:
        return f"Error: {err}", 500

@app.route("/manual", methods=["POST"])
def submit_manual_attendance():
    """Submits manual attendance data to the database."""
    try:
        attendance_data = request.form
        today = date.today().strftime('%Y-%m-%d')
        cnx = get_db_connection()
        cursor = cnx.cursor()

        for student_id, status in attendance_data.items():
            if student_id.startswith("status_"):
                s_id = student_id.split("_")[1]
                # Check if attendance for this student on this day already exists
                cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND date = %s", (s_id, today))
                if cursor.fetchone():
                    cursor.execute("UPDATE attendance SET status = %s WHERE student_id = %s AND date = %s", (status, s_id, today))
                else:
                    cursor.execute("INSERT INTO attendance (student_id, status, date) VALUES (%s, %s, %s)", (s_id, status, today))

        cnx.commit()
        cursor.close()
        cnx.close()
        return render_template("success.html", message="Attendance submitted successfully!")
    except mysql.connector.Error as err:
        return f"Error: {err}", 500

@app.route("/qr")
def qr_code_page():
    """Displays the QR code attendance page."""
    return render_template("qr_attendance.html")

@app.route("/qr", methods=["POST"])
def submit_qr_attendance():
    """Submits QR code attendance data to the database."""
    try:
        data = request.json
        student_id = data.get("student_id")
        today = date.today().strftime('%Y-%m-%d')

        cnx = get_db_connection()
        cursor = cnx.cursor()

        # Check if attendance for this student on this day already exists
        cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND date = %s", (student_id, today))
        if cursor.fetchone():
            cursor.execute("UPDATE attendance SET status = 'Present' WHERE student_id = %s AND date = %s", (student_id, today))
        else:
            cursor.execute("INSERT INTO attendance (student_id, status, date) VALUES (%s, 'Present', %s)", (student_id, today))

        cnx.commit()
        cursor.close()
        cnx.close()
        return jsonify({"success": True, "message": "Attendance marked successfully!"})
    except mysql.connector.Error as err:
        return jsonify({"success": False, "message": f"Error: {err}"}), 500

@app.route("/face")
def face_page():
    return render_template("face_recognition.html")

@app.route("/start_face_recognition", methods=['POST'])
def start_face_recognition():
    global face_recognition_process
    if face_recognition_process is None:
        data = request.json
        mode = data.get("mode", "streams")
        stream_sources = data.get("stream_sources", "0")

        cmd = [
            "python",
            "face_recognition/FaceDetection-CameraAndPhoto.py",
            "--mode", mode,
            "--stream_sources", stream_sources
        ]

        if os.name == 'nt':
            face_recognition_process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            face_recognition_process = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid
            )
        time.sleep(2) # Give the script time to start
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})

@app.route('/recognize_photo', methods=['POST'])
def recognize_photo():
    if 'photo' not in request.files:
        return 'No photo part', 400
    file = request.files['photo']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)
        
        cmd = [
            "python", 
            "face_recognition/FaceDetection-CameraAndPhoto.py",
            "--mode", "photo",
            "--photo_path", filepath
        ]
        
        if os.name == 'nt':
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            subprocess.Popen(
                cmd,
                preexec_fn=os.setsid
            )
        return jsonify({'status': 'processing'})

@app.route("/stop_face_recognition")
def stop_face_recognition():
    kill_face_recognition_process()
    return Response(status=204)

@app.route("/video_feed")
def video_feed():
    def generate():
        try:
            req = requests.get("http://127.0.0.1:8081/stream.mjpg", stream=True, timeout=5)
            for chunk in req.iter_content(chunk_size=1024):
                yield chunk
        except requests.exceptions.ConnectionError:
            print("Connection to video stream failed.")
            pass
        except requests.exceptions.Timeout:
            print("Connection to video stream timed out.")
            pass
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=FRAME')

@app.route("/api/face_attendance")
def face_attendance_api():
    """Stores face recognition attendance data in the database."""
    try:
        with open("attendance.json", "r") as f:
            attendance_data = json.load(f)
        
        today = date.today().strftime('%Y-%m-%d')
        cnx = get_db_connection()
        cursor = cnx.cursor()

        for student_name, status in attendance_data.items():
            # Get student_id from name
            cursor.execute("SELECT id FROM students WHERE name = %s", (student_name,))
            student = cursor.fetchone()
            if student:
                student_id = student[0]
                # Check if attendance for this student on this day already exists
                cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND date = %s", (student_id, today))
                if cursor.fetchone():
                    cursor.execute("UPDATE attendance SET status = %s WHERE student_id = %s AND date = %s", (status, student_id, today))
                else:
                    cursor.execute("INSERT INTO attendance (student_id, status, date) VALUES (%s, %s, %s)", (student_id, status, today))

        cnx.commit()
        cursor.close()
        cnx.close()
        return jsonify(attendance_data)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({})
    except mysql.connector.Error as err:
        return jsonify({"error": str(err)}), 500

@app.route("/attendance")
def attendance_report():
    """Displays the attendance report for the current day."""
    try:
        today = date.today().strftime('%Y-%m-%d')
        cnx = get_db_connection()
        cursor = cnx.cursor(dictionary=True)
        cursor.execute("""
            SELECT s.name, a.status
            FROM students s
            LEFT JOIN attendance a ON s.id = a.student_id AND a.date = %s
        """, (today,))
        attendance_records = cursor.fetchall()
        cursor.close()
        cnx.close()
        return render_template("attendance_report.html", attendance=attendance_records, today=date.today().strftime('%B %d, %Y'))
    except mysql.connector.Error as err:
        return f"Error: {err}", 500

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, threaded=True)