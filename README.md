# 🚦 Traffic Detection, Tracking & Count Analysis

An AI-powered traffic analytics application built using **YOLO11n, ByteTrack, OpenCV, PyTorch and Streamlit**.

The system detects and tracks vehicles in traffic videos, counts vehicle crossings, estimates approximate vehicle speeds, analyzes traffic congestion and presents the results through an interactive dark-themed dashboard.

---

## 📌 Project Overview

Traffic monitoring is important for understanding road usage, vehicle movement and congestion.

This project uses computer vision and deep learning to automate traffic video analysis.

Users can upload traffic videos and analyze vehicle movement through a Streamlit application. The system detects vehicles, assigns tracking IDs, records line-crossing events and generates traffic analytics.

## ✨ Key Features

### 1. Vehicle Detection
- YOLO11n-based vehicle detection.
- Detection of cars, trucks, buses and motorbikes.
- Bounding-box visualization.
- Vehicle confidence information.

### 2. Vehicle Tracking
- ByteTrack multi-object tracking.
- Unique tracking IDs for vehicles.
- Continuous tracking across video frames.
- Reduced duplicate vehicle counting.

### 3. Vehicle Counting
- Virtual line-crossing detection.
- Direction-based vehicle counting.
- UP and DOWN traffic counts.
- Vehicle-class-wise statistics.
- Crossing-event history.

### 4. Vehicle Speed Analysis
- Perspective-based approximate speed estimation.
- Road calibration support.
- Vehicle-wise speed information.
- Average speed analytics.
- Speed visualization in the dashboard.

**Note:** Speed values are approximate estimates and depend on camera perspective and calibration accuracy. They are not certified speed measurements.

### 5. Traffic Congestion Analysis
- Zone-based traffic monitoring.
- Vehicle occupancy analysis.
- Congestion score calculation.
- Congestion level classification.
- Zone-wise traffic statistics.
- Congestion history visualization.

Congestion scores are heuristic indicators based on the configured zones and occupancy assumptions.

### 6. Interactive Streamlit Dashboard
- Compact dark-themed dashboard.
- Traffic KPI cards.
- Vehicle count statistics.
- Direction analysis.
- Vehicle classification charts.
- Vehicle speed analysis.
- Vehicle crossing-event history.
- CSV export functionality.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLO11n | Vehicle detection |
| PyTorch | Deep learning framework |
| ByteTrack | Object tracking |
| OpenCV | Image and video processing |
| Streamlit | Interactive dashboard |
| NumPy | Numerical operations |
| Pandas | Data processing |
| Plotly | Interactive visualizations |

## 🏗️ System Workflow

1. Upload a traffic video.
2. Extract video frames.
3. Detect vehicles using YOLO11n.
4. Track vehicles using ByteTrack.
5. Identify vehicle movement and line crossings.
6. Record vehicle crossing events.
7. Estimate approximate vehicle speeds.
8. Analyze zone-based congestion.
9. Generate traffic statistics.
10. Display results in the Streamlit dashboard.

## 📂 Project Structure

```text
traffic_analytics/
│
├── analytics/
│   ├── speed_estimator.py
│   └── line_counter.py
│
├── tracking/
│   ├── tracker.py
│   └── video_processor.py
│
├── models/
│
├── utils/
│
├── outputs/
│
├── app.py
├── config.py
├── requirements.txt
└── README.md
```

*The structure above is representative. Keep the folder names consistent with the actual files in your repository.*

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/ASWINJ1710/traffic_detection_-_tracking_count_analysis.git
```

### Step 2: Open the Project

```bash
cd traffic_detection_-_tracking_count_analysis
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

PyTorch installation may require a compatible CPU or CUDA build depending on your hardware.

### Step 5: Configure the YOLO Model

Place your trained YOLO model weights in the configured model directory.

Ensure the model path in the project configuration points to the correct `.pt` file.

## ▶️ Run the Application

Execute:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## 📊 Dashboard Analytics

The dashboard provides:

| Analytics | Description |
|---|---|
| Total Vehicles | Total recorded vehicle crossings |
| UP Count | Vehicles moving in the UP direction |
| DOWN Count | Vehicles moving in the DOWN direction |
| Vehicle Classification | Counts by vehicle category |
| Speed Analysis | Approximate vehicle speed estimates |
| Congestion Analysis | Zone-based congestion indicators |
| Crossing History | Recorded vehicle crossing events |

## 📁 Generated Outputs

The application can generate:

- Processed traffic videos.
- Vehicle crossing-event records.
- CSV files for traffic events.
- Vehicle counting summaries.
- Speed analysis results.
- Congestion statistics.

## ⚠️ Limitations

- Detection accuracy depends on the trained model and video quality.
- Vehicle tracking may be affected by occlusion and crowded scenes.
- Counting accuracy depends on tracking and virtual line placement.
- Speed estimation requires camera-specific calibration.
- Congestion scores are heuristic and depend on configured road zones.

## 🚀 Future Improvements

- Improve vehicle detection accuracy.
- Improve tracking in crowded traffic scenes.
- Validate speed estimates using real-world measurements.
- Improve congestion analysis.
- Add advanced traffic reports.
- Add additional analytics and visualizations.

## 👨‍💻 Author

**ASWIN J**

GitHub: https://github.com/ASWINJ1710

Project Repository:

https://github.com/ASWINJ1710/traffic_detection_-_tracking_count_analysis

---

⭐ If you find this project interesting, consider giving the repository a star!
