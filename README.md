# 🎯 Face Recognition and Movement Tracking System
<div align="center" dir="auto">
<p align="center" dir="auto">
  <a target="_blank" rel="noopener noreferrer" href=""><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Logo_DAI_NAM.png/1200px-Logo_DAI_NAM.png" alt="DaiNam University Logo" width="200" style="max-width: 100%;"></a>
  <a target="_blank" rel="noopener noreferrer" href=""><img src="https://raw.githubusercontent.com/drkhanusa/DNU_PlagiarismChecker/main/docs/images/AIoTLab_logo.png" alt="AIoTLab Logo" width="170" style="max-width: 100%;"></a>
</p>
<p dir="auto"><a href="https://fit.dainam.edu.vn" rel="nofollow"><img src="https://camo.githubusercontent.com/14375b31490acab17dd414aef749f3c109a82abaeae50592667c9955b79ce09a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4d616465253230627925323041496f544c61622d626c75653f7374796c653d666f722d7468652d6261646765" alt="Made by AIoTLab" data-canonical-src="https://img.shields.io/badge/Made%20by%20AIoTLab-blue?style=for-the-badge" style="max-width: 100%;"></a>
<a href="https://fit.dainam.edu.vn" rel="nofollow"><img src="https://camo.githubusercontent.com/f33b9e36f6d7e3878c31898033ff8514d824d4f51d8cab187bf3eddc84e2a99e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f466163756c74792532306f66253230496e666f726d6174696f6e253230546563686e6f6c6f67792d677265656e3f7374796c653d666f722d7468652d6261646765" alt="Faculty of IT" data-canonical-src="https://img.shields.io/badge/Faculty%20of%20Information%20Technology-green?style=for-the-badge" style="max-width: 100%;"></a>
<a href="https://dainam.edu.vn" rel="nofollow"><img src="https://camo.githubusercontent.com/b503f479f429296dbff6eb7e1e583a962657044af1feb98e6dfc4a68a106a49e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4461694e616d253230556e69766572736974792d7265643f7374796c653d666f722d7468652d6261646765" alt="DaiNam University" data-canonical-src="https://img.shields.io/badge/DaiNam%20University-red?style=for-the-badge" style="max-width: 100%;"></a></p>
</div>
<div align="center">
  
[![Python](https://img.shields.io/badge/Python-3.6+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

🚀 An intelligent surveillance system using face recognition and motion detection to track user movement across 3 cameras, featuring real-time web interface and sound alerts.

</div>

## 📝 System Diagram
"
<div align="center">
  <img src="https://github.com/luongdieu2624/he-thong-nhan-dang-khuan-mat-va-tinh-toan-thoi-gian-di-chuyen-trong-mon-the-chat/blob/main/System%20Diagram.jpg" alt="System Diagram" width="800"/>
</div>

## ✨ Highlighted Features

### 🎭 Face Recognition
- 🧠 Uses ResNet model for face feature extraction
- 👥 Supports multiple faces in single frame
- 🎯 High accuracy with 0.5 matching threshold
- 📚 Easy training with custom dataset

### 🎬 Motion Detection
- 📹 Uses MOG2 Background Subtraction algorithm
- 🎛️ Customizable motion detection threshold
- 🔍 Efficient noise and shadow handling

### 📸 Multi-Camera Surveillance
- 🎥 Camera A: Face recognition
- 📹 Cameras B & C: Motion detection
- ⏱️ Time synchronization between cameras

### 🖥️ Web Interface
<div align="center">
  <img src="https://github.com/luongdieu2624/he-thong-nhan-dang-khuan-mat-va-tinh-toan-thoi-gian-di-chuyen-trong-mon-the-chat/blob/main/Web%20Interface.jpg" alt="System Interface" width="800"/>
</div>

Features:
- 📺 Live video streams from 3 cameras
- ⚡ Real-time status updates
- ⏲️ Travel time display between points
- 🏁 Cycle completion notifications

### 🔔 Smart Alerts
- 🚨 Sound alerts for exceeded travel time
- 🔈 "Ting" notification for too fast movement
- 💾 Movement data storage to MockAPI

## 🛠️ System Requirements

### 💻 Hardware
- 3️⃣ IP cameras (RTSP/HTTP stream support)
- 📊 Recommended resolution: 160x120 pixels
- 🌐 Stable network connection

### 📦 Software Dependencies
```bash
python>=3.6         # 🐍 Core runtime
face_recognition    # 👤 Face detection & recognition
opencv-python>=4.0  # 📸 Image processing
flask>=2.0         # 🌐 Web framework
pygame             # 🔊 Audio playback
numpy              # 🔢 Numerical operations
pillow            # 🖼️ Image handling
requests          # 🌍 HTTP client
```

## 🚀 Quick Start

### 1️⃣ Environment Setup
```bash
# 🏗️ Create virtual environment
python -m venv venv

# 🌟 Activate virtual environment
source venv/bin/activate  # 🐧 Linux/Mac
venv\Scripts\activate     # 🪟 Windows

# 📦 Install dependencies
pip install -r requirements_dev.txt
```

### 2️⃣ Camera Configuration
```python
# 🎥 In demo.py
CAM_A_IP = 'http://192.168.1.101:4747/video'  # 👤 Face recognition
CAM_B_IP = 'http://192.168.1.102:4747/video'  # 📹 Motion detection 1
CAM_C_IP = 'http://192.168.1.110:4747/video'  # 🎥 Motion detection 2
```

### 3️⃣ Dataset Preparation
```plaintext
📁 dataset/
 ├── 👤 person1.jpg        # Filename is label
 ├── 👥 person2-name.jpg   # Part after hyphen is label
 └── ...
```

### 4️⃣ Model Training
```bash
# 🧠 Train face recognition model
python face_recognition_train.py
```

## ⚙️ System Configuration

### 🎛️ Key Parameters
```python
# ⏱️ Travel times (seconds)
max_travel_time = 5    # ⏰ Maximum between points
min_travel_time = 2    # ⌛ Minimum between points
min_valid_time = 0.5   # ✅ Minimum confirmation

# 👤 Face recognition
face_recognition_interval = 0.3  # 🔄 Scan frequency
face_recognition_tolerance = 0.5 # 🎯 Match threshold

# 📹 Motion detection
motion_threshold = 1.5  # 📊 Pixel change percentage
```

### 🌐 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | 🏠 Main web page |
| `GET /video_feed_[a\|b\|c]` | 📹 Camera streams |
| `GET /face_status` | 👤 Recognition status |
| `GET /current_time` | ⏱️ Cycle time |
| `GET /travel_time` | 🕒 Travel times |
| `GET /completion_status` | ✅ Cycle status |

## 🔄 Operation Process

### 1️⃣ System Startup
- 🎥 Initialize cameras
- 🧠 Load face model
- 🌐 Start web server

### 2️⃣ Surveillance Cycle
- 👤 Face verification
- ⏱️ Time counting
- 📹 Motion detection
- ✅ Return confirmation

### 3️⃣ Data Recording
```json
{
    "start_time": "2025-03-18T18:56:44",
    "end_time": "2025-03-18T18:57:11",
    "travel_time_a_b": 3.5,
    "travel_time_b_c": 4.2,
    "travel_time_c_a": 3.8
}
```

## 📜 License

See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. 🍴 Fork repository
2. 🌟 Create feature branch
3. ✍️ Commit changes
4. 📤 Push to branch
5. 📫 Create Pull Request

## 🔒 Safety Notes

- 🏢 Place cameras securely
- 🌐 Check network regularly
- 💾 Backup data periodically
- 🔑 Update security settings
<p dir="auto">© 2025 AIoTLab, Faculty of Information Technology, DaiNam University. All rights reserved.</p>
---
<div align="center">
  
Made with ❤️ for security and efficiency

[⬆ Back to top](#-face-recognition-and-movement-tracking-system)

</div>
