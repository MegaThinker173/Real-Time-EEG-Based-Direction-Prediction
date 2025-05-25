# Real-Time EEG-Based Direction Prediction

This project implements a real-time system that predicts directional intent (left, right, up, down) using EEG brainwave signals. It utilizes wavelet/spectrogram-based preprocessing, deep learning (Keras/TensorFlow), and Pygame for live visual feedback — enabling brain-controlled directional navigation in a game environment.

---

## Project Overview

- **Input**: Raw EEG data stream from a connected Emotiv headset
- **Processing**:
  - Sliding window of EEG samples (800 samples per window)
  - Baseline normalization
  - Wavelet or spectrogram transformation
  - Classification using a trained CNN model
- **Output**: Real-time direction prediction (up/down/left/right)
- **Visualization**: A dot on-screen moves according to the predicted thought direction

---

## Core Components

### 1. `testing_eeg_model.py`
- Main logic for baseline recording, EEG streaming, preprocessing, and prediction
- Handles real-time interaction with the user and visual dot movement
- Connects to Cortex WebSocket API to receive EEG data from the headset
- Predicts direction using the trained `.h5` model
- Logs user interaction data (start/end times + actions) into a CSV file

### 2. `my_multiprocessing.py`
- A more robust, multiprocessing-enabled version of the system
- Launches EEG acquisition in a separate process to avoid blocking the game loop
- Uses Python `multiprocessing.Queue` to pass EEG windows to the main process
- Maintains synchronized real-time prediction and movement logic

### 3. `preprocess.py`
- Contains the preprocessing logic for EEG data
- Converts raw EEG signals into image-like tensors using spectrogram generation
- Ensures input compatibility with CNN-based models
- Supports both PyTorch and NumPy-based operations

---

## Requirements

Install dependencies using pip:

```bash
pip install pygame numpy tensorflow torch scipy pywt websockets