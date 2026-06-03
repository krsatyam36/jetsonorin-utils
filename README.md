# Jetson Orin Utilities 🚀

![Jetson](https://img.shields.io/badge/NVIDIA-Jetson-76B900?style=for-the-badge&logo=nvidia)
![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

A growing collection of tools, scripts, and utilities for working with NVIDIA Jetson devices. This repository is designed to be the central hub for development utilities needed for Jetson-based drone and computer vision projects.

## ✨ Current Features

- **Network Video Streaming (`src/stream.py`)**: A lightweight Flask-based web server that streams high-performance, hardware-accelerated MJPG video from an attached Arducam (or any V4L2 camera) to any device on the local network.

## 🗺️ Roadmap & Planned Utilities

As the project grows, this repository will include:
- **Computer Vision**: AI inference integration for real-time object detection.
- **Hardware Control**: GPIO interfacing scripts for peripheral control.
- **Flight Telemetry**: MAVLink communication scripts for interacting with flight controllers (e.g., Pixhawk).
- **System Monitors**: Utilities to check Jetson temperature, CPU/GPU usage, and power modes.

## 🚀 Getting Started

### Prerequisites
Make sure your Jetson is up to date and has the following dependencies installed:

```bash
sudo apt update
sudo apt install python3-flask python3-opencv v4l-utils -y
```

### Usage

1. Clone the repository:
```bash
git clone https://github.com/krsatyam36/jetsonorin-utils.git
cd jetsonorin-utils
```

2. Run the Web Streamer:
```bash
python3 src/stream.py
```

Then, open a web browser on any device connected to the same local network and navigate to:
```
http://<JETSON_IP_ADDRESS>:5000
```

## 📁 Project Structure

```
jetsonorin-utils/
├── .gitignore
├── README.md
└── src/
    └── stream.py       # Flask + OpenCV camera streaming server
```
