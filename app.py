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
import time
import io


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


def mark_attendance(name, class_names):
    global recognition_status
    dtString = datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')
    attendance_record = f"{name},{dtString}\n"
    
    if name in class_names or name == "Unknown User":
        recognition_status = "recognized" if name in class_names else "unrecognized"
        # Convert the attendance record to a bytes-like object
        attendance_data = io.StringIO(attendance_record)
        file_name = f"{name}_attendance_data_and_time_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
        
        # Upload the record to Cloudinary
        response = cloudinary.uploader.upload(attendance_data, resource_type='raw',
                                              public_id=f"attendance_files/{file_name}")
        
        print(f"Uploaded attendance for {name} to Cloudinary")
        return True


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
    marked_names = set()

    try:
        while True:
            recognized_names = []  # Reset recognized names list for each scanning period
            start_scan_time = time.time()  # Reset the scan timer for each user

            while time.time() - start_scan_time < 5:  # Continuous scanning for 5 seconds
                success, img = cap.read()
                if not success:
                    break

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces_in_frame = face_recognition.face_locations(img_rgb)
                encodings_in_frame = face_recognition.face_encodings(img_rgb, faces_in_frame)

                for face_loc, encoding in zip(faces_in_frame, encodings_in_frame):
                    matches = face_recognition.compare_faces(encoded_face_train, encoding, tolerance=0.5)
                    if True in matches:
                        matched_index = matches.index(True)
                        recognized_names.append(class_names[matched_index])
                    else:
                        recognized_names.append("Unknown")  # Append "Unknown" for unmatched faces

                # Frame processing and encoding for display
                ret, buffer = cv2.imencode('.jpg', img)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Decision making after 5 seconds of scanning
            if recognized_names:
                most_common_name = max(set(recognized_names), key=recognized_names.count)
                if most_common_name != "Unknown":
                    mark_attendance(most_common_name, class_names)
                else:
                    mark_attendance("Unknown User", class_names)

                # Signal that recognition is complete
                yield (b'--RECOGNITION_COMPLETE--')
                cap.release()
                return  # Stop the generator after making a decision
    finally:
        cap.release()


        

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=False)