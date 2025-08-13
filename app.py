import time, secrets, uuid, hmac, hashlib, io
from collections import defaultdict
from flask import Flask, render_template, jsonify, request, send_file, abort, redirect, url_for
from flask_socketio import SocketIO, emit, join_room
import qrcode

app = Flask(__name__)
app.config['SECRET_KEY'] = 'change_this_secret_in_prod'   # for sockets
socketio = SocketIO(app, cors_allowed_origins="*")

# ===== learning-only in-memory store =====
SESSIONS = {}  # session_id -> session dict

# defaults
TOKEN_LIFETIME = 2.0       # seconds each QR is valid (teacher can override)
DEFAULT_DURATION = 30.0    # session length
HMAC_SECRET = b'put_a_long_random_secret_here'  # change in real use

def make_sig(session_id, idx, code):
    msg = f"{session_id}|{idx}|{code}".encode()
    return hmac.new(HMAC_SECRET, msg, hashlib.sha256).hexdigest()

def random_6digit():
    return f"{secrets.randbelow(1_000_000):06d}"

@app.route('/')
def home():
    return redirect(url_for('teacher'))

# -------- TEACHER ----------
@app.route('/teacher')
def teacher():
    return render_template('teacher.html')

@app.route('/teacher/start', methods=['POST'])
def teacher_start():
    data = request.get_json() or {}
    duration = float(data.get('duration', DEFAULT_DURATION))
    lifetime = float(data.get('lifetime', TOKEN_LIFETIME))

    now = time.time()
    token_count = max(1, int(duration / lifetime))
    session_id = uuid.uuid4().hex

    tokens = []
    for i in range(token_count):
        code = random_6digit()
        start = now + i * lifetime
        sig = make_sig(session_id, i, code)
        tokens.append({'idx': i, 'code': code, 'start': start, 'sig': sig, 'used_by': set()})

    SESSIONS[session_id] = {
        'session_id': session_id,
        'created_at': now,
        'duration': duration,
        'lifetime': lifetime,
        'tokens': tokens,
        'scans': defaultdict(set),   # student_id -> set(indices)
        'present': set()
    }
    return jsonify({'session_id': session_id, 'token_count': token_count})

@app.route('/teacher/session/<session_id>')
def teacher_session(session_id):
    s = SESSIONS.get(session_id)
    if not s: return "Session not found", 404
    return render_template('session.html', s=s)

@app.route('/teacher/add', methods=['POST'])
def teacher_add():
    data = request.get_json() or {}
    session_id = data.get('session_id'); student_id = data.get('student_id')
    s = SESSIONS.get(session_id)
    if not s or not student_id: return jsonify({'ok': False}), 400
    s['present'].add(student_id)
    socketio.emit('present_update', {'student_id': student_id, 'session_id': session_id}, room=session_id)
    return jsonify({'ok': True})

# -------- QR IMAGE (for current token) ----------
@app.route('/qr/<session_id>/<int:idx>.png')
def qr_image(session_id, idx):
    s = SESSIONS.get(session_id)
    if not s or idx < 0 or idx >= len(s['tokens']): return abort(404)
    t = s['tokens'][idx]
    payload = f"{session_id}|{idx}|{t['code']}|{t['sig']}"
    img = qrcode.make(payload)
    buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
    return send_file(buf, mimetype='image/png')

# -------- STUDENT ----------
@app.route('/student')
def student_page():
    return render_template('student.html')

@app.route('/student/scan', methods=['POST'])
def student_scan():
    data = request.get_json() or {}
    session_id = data.get('session_id')
    student_id = data.get('student_id')
    idx = data.get('idx')
    code = data.get('code')
    sig = data.get('sig')

    if not all([session_id, student_id]) or idx is None:
        return jsonify({'ok': False, 'error': 'missing params'}), 400

    s = SESSIONS.get(session_id)
    if not s: return jsonify({'ok': False, 'error': 'invalid session'}), 400

    try:
        idx = int(idx)
    except:
        return jsonify({'ok': False, 'error': 'bad idx'}), 400

    if idx < 0 or idx >= len(s['tokens']):
        return jsonify({'ok': False, 'error': 'invalid token idx'}), 400

    token = s['tokens'][idx]

    # verify signature & time window (+ small grace)
    if sig != make_sig(session_id, idx, token['code']):
        return jsonify({'ok': False, 'error': 'bad signature'}), 400

    now = time.time()
    if not (token['start'] <= now < token['start'] + s['lifetime'] + 0.8):
        return jsonify({'ok': False, 'error': 'token expired or inactive'}), 400

    if student_id in token['used_by']:
        return jsonify({'ok': False, 'error': 'token already used by this student'}), 400

    token['used_by'].add(student_id)
    s['scans'][student_id].add(idx)

    if len(s['scans'][student_id]) >= 2:
        s['present'].add(student_id)
        socketio.emit('present_update', {'student_id': student_id, 'session_id': session_id}, room=session_id)
        return jsonify({'ok': True, 'attendance': 'present', 'scans_count': len(s['scans'][student_id])})
    else:
        socketio.emit('scan_pending', {'student_id': student_id, 'session_id': session_id, 'scans_count': len(s['scans'][student_id])}, room=session_id)
        return jsonify({'ok': True, 'attendance': 'not_yet', 'scans_count': len(s['scans'][student_id])})

# -------- SOCKET.IO ----------
@socketio.on('join')
def on_join(data):
    session_id = data.get('session_id')
    join_room(session_id)

if __name__ == '__main__':
    # eventlet gives reliable websockets on Windows for dev
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
