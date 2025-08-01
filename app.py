


# copy of basic 21
# html ui improvements
# face recognition try 3 - less laggy - with frame skip - display matched face to frontend




import os
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from ultralytics import YOLO
import face_recognition

app = Flask(__name__)

MODEL_DIR = "models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
default_model = 'yolov8s.pt'

is_paused = False
last_frame = None

# Directory to store detected face images
MATCHED_FACES_DIR = "matched_faces"
if not os.path.exists(MATCHED_FACES_DIR):
    os.makedirs(MATCHED_FACES_DIR)

matched_faces = []  # Store {"name": "Person", "img_url": "/matched_faces/person.jpg"}

@app.route('/')
def index():
    video_files = [f for f in os.listdir('videos') if f.endswith(('.mp4', '.avi', '.mkv'))]
    return render_template('index.html', video_files=video_files, model_files=model_files, default_model=default_model)

# Load known faces
known_faces = []
known_names = []

def load_known_faces():
    """Loads all face images from 'faces/' directory."""
    global known_faces, known_names
    face_dir = "faces"

    if not os.path.exists(face_dir):
        os.makedirs(face_dir)
        return

    for filename in os.listdir(face_dir):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(face_dir, filename)

            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_faces.append(encodings[0])
                known_names.append(name)
                print(f"Loaded face for {name}")
            else:
                print(f"Warning: No face found in {filename}, skipping.")

load_known_faces()

def generate_frames(video_source, model_path):
    global is_paused, last_frame

    model = YOLO(model_path, task='detect')
    labels = model.names

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    prev_frame_time = 0
    frame_skip = 10  
    frame_count = 0

    while True:
        if is_paused and last_frame is not None:
            ret, buffer = cv2.imencode('.jpg', last_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            print('Reached the end of the video file.')
            break

        frame_count += 1

        results = model(frame, verbose=False)
        detections = results[0].boxes
        object_count = 0

        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()

            if conf >= 0.5:
                color = (0, 255, 0)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'{classname}: {int(conf * 100)}%'
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

                if classname == 'person':
                    if frame_count % frame_skip != 0:
                        continue  

                    object_count += 1
                    person_crop = frame[ymin:ymax, xmin:xmax]
                    rgb_face = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

                    face_locations = face_recognition.face_locations(rgb_face)
                    face_encodings = face_recognition.face_encodings(rgb_face, face_locations)

                    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
                        name = "Unknown"

                        if True in matches:
                            match_index = matches.index(True)
                            name = known_names[match_index]

                        cv2.rectangle(frame, (xmin + left, ymin + top), (xmin + right, ymin + bottom), (255, 0, 0), 2)
                        cv2.putText(frame, name, (xmin + left, ymin + top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        face_filename = f"{name}.jpg"
                        face_path = os.path.join(MATCHED_FACES_DIR, face_filename)

                        if name != "Unknown" and not any(f["name"] == name for f in matched_faces):
                            face_crop = person_crop[top:bottom, left:right]
                            cv2.imwrite(face_path, face_crop)
                            matched_faces.append({"name": name, "img_url": f"/matched_faces/{face_filename}"})

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f'People detected: {object_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        last_frame = frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    video_source = request.args.get('source', 'Webcam')
    selected_model = request.args.get('model', default_model)

    if video_source == 'Webcam':
        video_source = 1  
    else:
        video_source = os.path.join('videos', video_source)

    model_path = os.path.join(MODEL_DIR, selected_model)
    
    return Response(generate_frames(video_source, model_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    return jsonify({"paused": is_paused})

@app.route('/matched_faces')
def get_matched_faces():
    return jsonify(matched_faces)

@app.route('/matched_faces/<filename>')
def serve_matched_face(filename):
    return send_from_directory(MATCHED_FACES_DIR, filename)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)





































