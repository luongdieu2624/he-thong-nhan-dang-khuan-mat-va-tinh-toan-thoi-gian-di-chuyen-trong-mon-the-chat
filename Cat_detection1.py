from flask import Flask, render_template, Response, request
import cv2
from threading import Thread, Lock
import time
import numpy as np
import os

app = Flask(__name__)

# Địa chỉ stream video
stream_url = 'http://172.16.34.134:4747/video'
cap = cv2.VideoCapture(stream_url)
time.sleep(2)  # Chờ để đảm bảo kết nối ổn định

if not cap.isOpened():
    print("❌ Không thể mở camera. Kiểm tra lại URL hoặc kết nối camera.")
    exit()

width, height = 640, 480
video_frame = None
lock = Lock()
recording = False
out = None

# Load MobileNet SSD
model_prototxt = 'deploy.prototxt'
model_weights = 'mobilenet_iter_73000.caffemodel'

if not os.path.exists(model_prototxt) or not os.path.exists(model_weights):
    print("❌ Lỗi: Không tìm thấy file model. Kiểm tra lại đường dẫn.")
    exit()

net = cv2.dnn.readNetFromCaffe(model_prototxt, model_weights)
print("✅ Model MobileNet SSD đã tải thành công!")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def camera_stream():
    global cap, video_frame, recording, out
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        object_detected = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                if (endX - startX) > 50 and (endY - startY) > 50:
                    label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    object_detected = True

        if recording and object_detected and out is not None:
            out.write(frame)

        with lock:
            video_frame = frame.copy()

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, out
    if not recording:
        filename = f"static/video_{int(time.time())}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 10, (width, height))
        recording = True
        return "Bắt đầu quay video khi phát hiện vật thể."
    return "Đã quay video rồi."

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording, out
    if recording:
        recording = False
        if out is not None:
            out.release()
            out = None
        return "Dừng quay video."
    return "Không có video nào đang quay."

def gen_frames():
    global video_frame
    while True:
        with lock:
            if video_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', video_frame)
            if not ret:
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    camera_thread = Thread(target=camera_stream, daemon=True)
    camera_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    cap.release()