
# Traffic Detection, Tracking & Count Analysis

An AI-powered traffic video analytics system using YOLO11n, ByteTrack, OpenCV, and Streamlit.

## Project Overview

This project detects and tracks vehicles from traffic videos and counts vehicles crossing a predefined virtual counting line.

It focuses on four vehicle categories:

- Car
- Truck
- Bus
- Motorbike

## Key Features

- Vehicle detection using custom-trained YOLO11n
- Multi-object tracking using ByteTrack
- Unique vehicle tracking IDs
- Virtual line-crossing detection
- UP and DOWN direction counting
- Confidence-weighted vehicle classification across tracks
- Vehicle-wise traffic statistics
- CSV export of crossing events
- Annotated video generation
- Streamlit dashboard

## Tech Stack

- Python
- PyTorch
- YOLO11 (Ultralytics)
- ByteTrack
- OpenCV
- Streamlit
- NumPy
- Pandas
- Plotly

## Project Structure

```text
traffic_analytics/
├── analytics/
│   └── line_counter.py
├── models/
│   └── detector.py
├── tracking/
│   ├── tracker.py
│   └── video_processor.py
├── data/
│   ├── models/
│   └── videos/
├── outputs/
├── app.py
├── config.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/ASWINJ1710/traffic_detection_-_tracking_count_analysis.git
cd traffic_detection_-_tracking_count_analysis
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Install a compatible PyTorch build for your hardware if needed. For NVIDIA CUDA 12.8:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Model Setup

Place the trained YOLO11 model at:

```text
data/models/best.pt
```

The trained model weights are not included in this repository.

## Run the Application

```bash
streamlit run app.py
```

Open the local Streamlit URL displayed in your terminal.

## Output

The system generates:

- Annotated traffic videos
- Vehicle crossing event CSV files
- Vehicle class-wise counts
- Direction-wise traffic counts

## Current Status

Core vehicle detection, tracking, line-crossing counting, and track-level classification have been tested on a sample traffic video.

The Streamlit dashboard and further real-world testing are ongoing.

## Author

ASWIN J

GitHub: https://github.com/ASWINJ1710
```
