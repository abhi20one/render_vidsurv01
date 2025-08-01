import os
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

MODEL_DIR = "models"  # Directory where YOLO models are stored
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
default_model = 'yolov8s.pt'

# Global pause flag
is_paused = False
last_frame = None  # Store last frame for pausing

@app.route('/')
def index():
    """Render index page with dropdown options for video files and models."""
    video_files = [f for f in os.listdir('videos') if f.endswith(('.mp4', '.avi', '.mkv'))]
    return render_template('index.html', video_files=video_files, model_files=model_files, default_model=default_model)

def generate_frames(video_source, model_path):
    """Process video stream and yield frames."""
    global is_paused, last_frame

    model = YOLO(model_path, task='detect')
    labels = model.names

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    frame_rate_buffer = []
    fps_avg_len = 10

    while True:
        if is_paused and last_frame is not None:
            # Send the last frame while paused
            ret, buffer = cv2.imencode('.jpg', last_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
            continue  # Skip further processing while paused

        ret, frame = cap.read()
        if not ret:
            print('Reached the end of the video file. Exiting program.')
            break

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
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                              (xmin + label_size[0], label_ymin + base_line - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

                object_count += 1

        # Store the last frame before sending
        last_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    video_source = request.args.get('source', 'Webcam')
    selected_model = request.args.get('model', default_model)

    if video_source == 'Webcam':
        video_source = 0  # Use the default webcam
    else:
        video_source = os.path.join('videos', video_source)

    model_path = os.path.join(MODEL_DIR, selected_model)
    
    return Response(generate_frames(video_source, model_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    """Toggle pause/resume for video streaming."""
    global is_paused
    is_paused = not is_paused
    return jsonify({"paused": is_paused})


if __name__ == '__main__':
    app.run()