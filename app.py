from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
import cv2
import torch
import joblib
import os
import csv
from datetime import datetime, timedelta
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils import recognize_face
from EmailAlert2 import send_alert_email
from functools import wraps

app = Flask(__name__, template_folder='templates')
app.secret_key = 'classroom_intrusion_detection_secret_key_2024'

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load classifier and encoder
BASE_DIR = os.path.dirname(__file__)
clf = joblib.load(os.path.join(BASE_DIR, 'models', 'classifier.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))

cap = None
camera_on = False
THRESHOLD = 0.7
attendance_today = set()

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def prepare_attendance_csv():
    global attendance_file
    student_dir = os.path.join(BASE_DIR, 'data', 'student_db')
    attendance_dir = os.path.join(BASE_DIR, 'attendance')
    os.makedirs(attendance_dir, exist_ok=True)

    today_str = datetime.now().strftime('%Y%m%d')
    attendance_file = os.path.join(attendance_dir, f"attendance_{today_str}.csv")

    if not os.path.exists(student_dir):
        os.makedirs(student_dir, exist_ok=True)
        return

    current_students = sorted(os.listdir(student_dir), key=lambda s: s.lower())

    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Status', 'Time'])
            for student in current_students:
                writer.writerow([student, 'Absent', ''])
        return

    with open(attendance_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        existing_data = {row[0]: row for row in reader if len(row) >= 3}

    updated_rows = []
    for student in current_students:
        if student in existing_data:
            updated_rows.append(existing_data[student])
        else:
            updated_rows.append([student, 'Absent', ''])

    updated_rows = sorted(updated_rows, key=lambda row: row[0].lower())

    with open(attendance_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(updated_rows)

def get_attendance_data():
    """Get current attendance data for dashboard"""
    if not os.path.exists(attendance_file):
        return {'present': 0, 'absent': 0, 'total': 0, 'students': []}

    with open(attendance_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        students = list(reader)

    present = sum(1 for s in students if len(s) > 1 and s[1] == 'Present')
    total = len(students)
    absent = total - present

    return {
        'present': present,
        'absent': absent,
        'total': total,
        'students': students
    }

@app.route('/')
@login_required
def index():
    prepare_attendance_csv()
    attendance_data = get_attendance_data()
    return render_template('dashboard.html', 
                         camera_on=camera_on,
                         attendance=attendance_data)

@app.route('/start', methods=['POST'])
@login_required
def start_camera():
    global cap, camera_on
    if not camera_on:
        cap = cv2.VideoCapture(0)
        camera_on = True
        flash('Camera started successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/stop', methods=['POST'])
@login_required
def stop_camera():
    global cap, camera_on
    if cap:
        cap.release()
        cap = None

        # Update attendance file with final status
        updated_rows = []
        with open(attendance_file, 'r', newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                name = row[0]
                if name in attendance_today:
                    updated_rows.append([name, 'Present', row[2] if len(row) > 2 else ''])
                else:
                    updated_rows.append([name, 'Absent', ''])

        with open(attendance_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(updated_rows)

        camera_on = False
        flash('Camera stopped successfully!', 'info')
    return redirect(url_for('index'))

@app.route('/add-student', methods=['GET', 'POST'])
@login_required
def add_student():
    global cap
    message = None

    if request.method == 'POST':
        name = request.form.get('student_name')
        if not name:
            flash('Name is required.', 'error')
        else:
            # Create directory for student
            save_path = os.path.join('data', 'student_db', name)
            os.makedirs(save_path, exist_ok=True)

            # Initialize camera if not already done
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)

            count = len(os.listdir(save_path)) + 1
            saved = 0
            duration = 5  # seconds to capture images
            end_time = datetime.now() + timedelta(seconds=duration)

            while datetime.now() < end_time and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.imshow("Capturing Images", frame)
                filename = os.path.join(save_path, f"{name}_{count + saved}.jpg")
                cv2.imwrite(filename, frame)
                saved += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
            flash(f'✅ {saved} images captured for {name}.', 'success')

    return render_template('add_student.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Login credentials
        if username == 'teacher' and password == 'classroom2024':
            session['logged_in'] = True
            session['login_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            flash('Login successful!', 'success')

            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/video')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global cap, attendance_today

    while camera_on and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)
        faces = mtcnn(frame_rgb)

        if boxes is not None and faces is not None:
            h, w, _ = frame.shape
            for box, face in zip(boxes, faces):
                x1, y1, x2, y2 = [int(b) for b in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if face is not None:
                    with torch.no_grad():
                        embedding = resnet(face.unsqueeze(0).to(device))
                        embedding_np = embedding[0].cpu().numpy()

                    label, prob = recognize_face(embedding_np, clf, label_encoder, threshold=THRESHOLD)

                    # Mark attendance
                    if label != "Intruder" and label not in attendance_today:
                        attendance_today.add(label)
                        now = datetime.now().strftime('%H:%M:%S')

                        # Update attendance file
                        with open(attendance_file, 'r') as f:
                            rows = list(csv.reader(f))

                        for i in range(1, len(rows)):
                            if rows[i][0] == label:
                                rows[i][1] = 'Present'
                                rows[i][2] = now
                                break

                        with open(attendance_file, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerows(rows)

                        print(f"✅ Marked Present: {label} at {now}")

                    # Send alert for intruders
                    if label == "Intruder":
                        send_alert_email()

                    # Draw rectangle and label
                    color = (0, 255, 0) if label != "Intruder" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)
