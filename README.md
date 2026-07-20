# AI-Based Real-Time Object Detection and Multi-Object Tracking System

## Overview

This project implements a professional real-time object detection and multi-object tracking application using Python, OpenCV, YOLOv8, and Deep SORT. The application supports live webcam input and uploaded videos, logs detections to SQLite, and provides a CustomTkinter dashboard with statistics, search, screenshot, and recording features.

## Folder Structure

- `app.py` - main application launcher
- `requirements.txt` - Python dependencies
- `models/` - contains `yolov8n.pt`
- `detector/yolo_detector.py` - YOLO detection module
- `tracker/deep_sort_tracker.py` - Deep SORT tracker module
- `database/database.py` - SQLite manager
- `gui/dashboard.py` - CustomTkinter UI
- `utils/video_utils.py` - video processing helpers
- `utils/counter.py` - object counting utilities
- `utils/logger.py` - logging utilities
- `output/screenshots/` - saved frames
- `output/recordings/` - saved processed videos

## Installation

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download `yolov8n.pt` and place it in the `models/` folder.

## Running the Application

From the project root:

```bash
python app.py
```

## Features

- Live webcam object detection and tracking
- Video upload processing (.mp4, .avi, .mov)
- YOLOv8-based object detection
- Deep SORT multi-object tracking
- Real-time FPS and statistics
- Detection history stored in SQLite
- Screenshot capture and output video recording
- Search history by object name, tracking ID, and date

## Troubleshooting

- If the GUI does not open, confirm `customtkinter` is installed.
- If the model does not load, ensure `models/yolov8n.pt` exists.
- If camera access fails, verify the webcam is available and not used by another app.
- If video processing is slow, resize the frame or run with CUDA-enabled GPU.

## Notes

- The app automatically selects CUDA if available.
- The tracker maintains stable IDs and counts objects that enter/leave.
- The `detections` table stores each detection event for search and auditing.
