from flask import Flask, render_template, Response
import cv2
import os
import face_recognition
from datetime import datetime, date
import requests
import numpy as np
import cloudinary
import cloudinary.api
import cloudinary.uploader

app = Flask(__name__)

cloudinary.config(
    cloud_name="dimnbv1qj",
    api_key="685589194883471",
    api_secret="Zn7snqhOiBaL5u52XL0dUtQOzhs",
    secure=True,
)

def load_student_images():
    images = []
    class_names = []
    folder = 'student_images'
    result = cloudinary.api.resources(type="upload", prefix=folder, max_results=500)
    for resource in result["resources"]:
        if resource["format"].lower() in ["jpg", "jpeg", "png"]:
            image_url = resource["secure_url"]
            response = requests.get(image_url)
            image_data = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
                class_name = os.path.splitext(os.path.basename(resource["public_id"]))[0]
                class_names.append(class_name)
            else:
                print(f"Error loading image: {image_url}")
    return images, class_names

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encoded_face = face_recognition.face_encodings(img)[0]
            encode_list.append(encoded_face)
        except IndexError:
            print("Face not found in the image.")
    return encode_list

def mark_attendance(name, attendance_file_path, marked_names, class_names):
    global recognition_status
    if name in class_names and name not in marked_names:
        now = datetime.now()
        # Adding date in the format YYYY-MM-DD and time in 12-hour format with AM/PM
        dtString = now.strftime('%Y-%m-%d %I:%M:%S %p')
        with open(attendance_file_path, 'a') as f:
            # Writing the date and time in the specified format
            f.writelines(f'\n{name},{dtString}')
        recognition_status = "recognized"
        marked_names.add(name)
        print(f"Marked attendance for {name}")
        return True
    elif name not in class_names:
        recognition_status = "unrecognized"
    return False

def create_or_open_attendance_file(attendance_file_path):
    if not os.path.isfile(attendance_file_path):
        with open(attendance_file_path, 'w') as f:
            f.write("Name, Date\n")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognition_complete')
def recognition_complete():
    # to access the next_page.html
    return render_template('next_page.html')  

recognition_status = "processing"  # could be 'processing', 'recognized', or 'unrecognized'

@app.route('/recognition_status')
def get_recognition_status():
    global recognition_status
    return {"status": recognition_status}



@app.route('/submit_page')
def submit_page():
    # to access the submit_page.html
    return render_template('submit_page.html')


def gen_frames():
    images, class_names = load_student_images()
    encoded_face_train = find_encodings(images)
    currentDate = date.today().strftime('%m%d%Y')
    attendance_file_path = 'Attendance_' + currentDate + '.csv'
    create_or_open_attendance_file(attendance_file_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    marked_names = set()
    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                break
            
            # Convert the frame to RGB for face recognition
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Find face locations and encodings in the frame
            faces_in_frame = face_recognition.face_locations(img_rgb)
            encodings_in_frame = face_recognition.face_encodings(img_rgb, faces_in_frame)

            for face_loc, encoding in zip(faces_in_frame, encodings_in_frame):
                matches = face_recognition.compare_faces(encoded_face_train, encoding, tolerance=0.5)
                name = "Scanning..."

                if True in matches:
                    matched_index = matches.index(True)
                    name = class_names[matched_index]
                else:
                    name = "Unknown"
                if mark_attendance(name, attendance_file_path, marked_names, class_names):
                    yield "REDIRECT"
                    return  # Stops the generator after successful recognition

                y1, x2, y2, x1 = face_loc
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the frame and yield for displaying
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=False)