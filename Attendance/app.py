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

import config

app = Flask(__name__)

# --- Database Configuration ---v
DB_HOST = config.DB_HOST
DB_USER = config.DB_USER    
DB_PASSWORD = config.DB_PASSWORD
DB_NAME = config.DB_NAME

# Global variable to hold the face recognition process
face_recognition_process = None
qr_tokens = {}

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
        print(f"[INFO] Attempting to terminate face recognition process with PID: {face_recognition_process.pid}")
        if os.name == 'nt':
            subprocess.call(['taskkill', '/F', '/T', '/PID', str(face_recognition_process.pid)])
        else:
            os.killpg(os.getpgid(face_recognition_process.pid), subprocess.SIGTERM)
        face_recognition_process.wait()
        face_recognition_process = None
        print("[INFO] Face recognition process terminated.")

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
        # Replace hyphens with spaces in student names
        for student in students:
            if student.get('name'):
                student['name'] = student['name'].replace('-', ' ')
        cursor.close()
        cnx.close()
        return render_template("manual_attendance.html", students=students)
    except mysql.connector.Error as err:
        return f"Error: {err}", 500

@app.route("/manual", methods=["POST"])
def submit_manual_attendance():
    """Submits manual attendance data to the database."""
    try:
        today = date.today().strftime('%Y-%m-%d')
        cnx = get_db_connection()
        cursor = cnx.cursor()

        # Iterate through all form items to find attendance statuses
        for key, value in request.form.items():
            if key.startswith("status_"):
                s_id = key.split("_")[1]
                status = value # Value will be 'Present' or 'Absent'

                # Check if attendance for this student on this day already exists
                cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND date = %s", (s_id, today))
                existing_attendance = cursor.fetchall() # Explicitly fetch all results
                if existing_attendance:
                    cursor.execute("UPDATE attendance SET status = %s WHERE student_id = %s AND date = %s", (status, s_id, today))
                else:
                    cursor.execute("INSERT INTO attendance (student_id, status, date) VALUES (%s, %s, %s)", (s_id, status, today))

        cnx.commit()
        cursor.close()
        cnx.close()

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": True, "message": "Attendance saved successfully!"})
        else:
            return render_template("success.html", message="Attendance submitted successfully!")
    except mysql.connector.Error as err:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"success": False, "message": f"Database error: {err}"}), 500
        else:
            return f"Error: {err}", 500

@app.route("/qr")
def qr_code_page():
    """Displays the QR code attendance page."""
    return render_template("qr_attendance.html")

@app.route("/api/qr_token")
def qr_token():
    """Generates and returns a short-lived token for QR code."""
    import secrets
    token = secrets.token_hex(16)
    # Store the token with a timestamp (e.g., valid for 5 seconds)
    qr_tokens[token] = time.time() + 5
    return jsonify({"token": token})


@app.route("/qr", methods=["POST"])
def submit_qr_attendance():
    """Submits QR code attendance data to the database."""
    try:
        data = request.json
        token = data.get("token")
        student_id = data.get("student_id") # Assuming student_id is sent from the client

        # Validate the token
        if token not in qr_tokens or time.time() > qr_tokens[token]:
            return jsonify({"success": False, "message": "Invalid or expired QR code."}), 400
        
        del qr_tokens[token] # Invalidate the token after use

        today = date.today().strftime('%Y-%m-%d')

        cnx = get_db_connection()
        cursor = cnx.cursor()

        # Check if attendance for this student on this day already exists
        cursor.execute("SELECT id FROM attendance WHERE student_id = %s AND date = %s", (student_id, today))
        existing_attendance = cursor.fetchall() # Explicitly fetch all results
        if existing_attendance:
            cursor.execute("UPDATE attendance SET status = 'Present' WHERE student_id = %s AND date = %s", (student_id, today))
        else:
            cursor.execute("INSERT INTO attendance (student_id, status, date) VALUES (%s, 'Present', %s)", (student_id, today))

        cnx.commit()
        cursor.close()
        cnx.close()
        return jsonify({"success": True, "message": "Attendance marked successfully!"})
    except mysql.connector.Error as err:
        return jsonify({"success": False, "message": f"Error: {err}"}), 500

@app.route("/submit_student_qr_attendance", methods=["POST"])
def submit_student_qr_attendance():
    """Receives student QR attendance submission and marks attendance."""
    try:
        data = request.json
        student_id = data.get("student_id")
        session_id = data.get("session_id")
        print(f"[INFO] Received QR attendance submission for student_id: {student_id}, session_id: {session_id}")

        if not student_id or not session_id:
            return jsonify({"success": False, "message": "Missing student_id or session_id."}), 400

        today = date.today().strftime('%Y-%m-%d')

        cnx = get_db_connection()

        # First, verify if the student_id exists in the students table
        cursor_student_check = cnx.cursor()
        cursor_student_check.execute("SELECT id FROM students WHERE id = %s", (student_id,))
        student_exists = cursor_student_check.fetchall() # Explicitly fetch all results
        cursor_student_check.close()

        if not student_exists:
            cnx.close()
            return jsonify({"success": False, "message": f"Student with ID {student_id} not found."}), 200

        # Now, handle attendance operations with a new cursor
        cursor_attendance = cnx.cursor()

        # Check if attendance for this student on this day already exists
        cursor_attendance.execute("SELECT id FROM attendance WHERE student_id = %s AND date = %s", (student_id, today))
        attendance_record = cursor_attendance.fetchall() # Explicitly fetch all results
        if attendance_record:
            cursor_attendance.execute("UPDATE attendance SET status = 'Present' WHERE student_id = %s AND date = %s", (student_id, today))
        else:
            cursor_attendance.execute("INSERT INTO attendance (student_id, status, date) VALUES (%s, 'Present', %s)", (student_id, today))

        cnx.commit()
        cursor_attendance.close()
        cnx.close()
        return jsonify({"success": True, "message": f"Attendance marked for student {student_id} in session {session_id}!"})
    except mysql.connector.Error as err:
        print(f"Database error in submit_student_qr_attendance: {err}") # Log detailed error
        return jsonify({"success": False, "message": "A database error occurred while marking attendance."}), 500
    except Exception as e:
        print(f"Unexpected error in submit_student_qr_attendance: {e}") # Log detailed error
        return jsonify({"success": False, "message": "An unexpected error occurred while marking attendance."}), 500

@app.route("/face")
def face_page():
    return render_template("face_recognition.html")

@app.route("/student")
def student_page():
    return render_template("student.html")

@app.route("/start_face_recognition", methods=['POST'])
def start_face_recognition():
    global face_recognition_process
    if face_recognition_process is None:
        data = request.json
        mode = data.get("mode", "streams")
        stream_sources = data.get("stream_sources", "0") # This will now be the deviceId

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
        print("Attempting to connect to video stream at http://127.0.0.1:8081/stream.mjpg")
        try:
            req = requests.get("http://127.0.0.1:8081/stream.mjpg", stream=True, timeout=5)
            print("Successfully connected to video stream.")
            for chunk in req.iter_content(chunk_size=1024):
                yield chunk
        except requests.exceptions.ConnectionError as e:
            print(f"Connection to video stream failed: {e}")
            pass
        except requests.exceptions.Timeout as e:
            print(f"Connection to video stream timed out: {e}")
            pass
        except Exception as e:
            print(f"An unexpected error occurred in video stream generation: {e}")
            pass
        except Exception as e:
            print(f"An unexpected error occurred in video stream generation: {e}")
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
        
        # Replace hyphens with spaces in the response keys
        sanitized_data = {name.replace('-', ' '): status for name, status in attendance_data.items()}
        return jsonify(sanitized_data)
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
        # Replace hyphens with spaces in student names
        for record in attendance_records:
            if record.get('name'):
                record['name'] = record['name'].replace('-', ' ')
        cursor.close()
        cnx.close()
        return render_template("attendance_report.html", attendance=attendance_records, today=date.today().strftime('%B %d, %Y'))
    except mysql.connector.Error as err:
        return f"Error: {err}", 500

@app.route("/download_attendance")
def download_attendance():
    """Downloads the attendance data from face recognition as a CSV file."""
    try:
        with open("attendance.json", "r") as f:
            attendance_data = json.load(f)
        
        si = StringIO()
        cw = csv.writer(si)
        cw.writerow(["Student Name", "Status"])
        for name, status in attendance_data.items():
            cw.writerow([name.replace('-', ' '), status])
        
        output = si.getvalue()
        today = date.today().strftime('%Y-%m-%d')
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename=face_attendance_{today}.csv"}
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return "Attendance file not found or is empty.", 404

@app.route("/download_report")
def download_report():
    """Downloads the attendance report as a CSV file."""
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

        # Create CSV in memory
        si = StringIO()
        cw = csv.writer(si)
        cw.writerow(["Student Name", "Status"])
        for record in attendance_records:
            name = record['name'].replace('-', ' ') if record.get('name') else ''
            cw.writerow([name, record['status'] or 'Absent'])
        
        output = si.getvalue()
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename=attendance_report_{today}.csv"}
        )
    except mysql.connector.Error as err:
        return f"Error: {err}", 500

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, threaded=True, host='0.0.0.0')