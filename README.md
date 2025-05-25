# Real-Time EEG-Based Direction Prediction

This repository contains a full implementation of a real-time EEG-based direction prediction system. It enables directional control (up, down, left, right) using brainwave signals collected from an Emotiv headset. The predicted directions are used to control a visual dot in a Pygame interface.

This documentation is intended to help future developers and students understand the project and begin working with it quickly.

> **Note:** If you are viewing this file in Google Drive and want to see the properly formatted version, please visit the GitHub repository:  
> [https://github.com/MegaThinker173/Real-Time-EEG-Based-Direction-Prediction](https://github.com/MegaThinker173/Real-Time-EEG-Based-Direction-Prediction)

---

## Overview

The system uses real-time EEG data to predict the direction a user is thinking of. A convolutional neural network (CNN) model is trained on EEG spectrogram representations, and predictions are visualized in an interactive Pygame interface.

Components of the system include:

- Real-time EEG signal acquisition from Emotiv EpocX via the Cortex API
- Preprocessing EEG time-series into spectrogram tensors
- A trained CNN model that outputs directional intent
- Visualization and interaction with a user through a dot movement interface

---

## Repository Structure

| File | Description |
|------|-------------|
| `testing_eeg_model.py` | Single-threaded implementation of the real-time EEG prediction game |
| `my_multiprocessing.py` | Optimized version using multiprocessing for smoother performance |
| `preprocess.py` | Spectrogram-based EEG preprocessing pipeline |
| `train_eeg_model_to_predict_direction.ipynb` | Jupyter notebook for training and exporting the model and label encoder |
| `may25_eeg_prediction_model.h5` | Pre-trained Keras model used for prediction |
| `may25_label_encoder.pkl` | Label encoder for mapping model output indices to direction labels |

---

## System Flow

```
[ EEG Headset ]
      ↓
[ Cortex API (WebSocket) ]
      ↓
[ EEG Buffer (deque of 800x14) ]
      ↓
[ Baseline Subtraction (first 30s) ]
      ↓
[ Preprocessing (Spectrogram) ]
      ↓
[ CNN Model (Keras .h5) ]
      ↓
[ Predicted Direction (softmax) ]
      ↓
[ Game Interface (pygame) ]
```

---

## Getting Started

### Requirements

Install required packages using pip:

```bash
pip install pygame numpy tensorflow torch scipy pywt websockets
```

You also need:
- Python 3.8+
- Access to Emotiv’s Cortex API with valid client credentials
- Trusted SSL setup for `wss://localhost:6868`

---

### Running the Game

**Option 1: Single-threaded version**

```bash
python testing_eeg_model.py
```

**Option 2: Multiprocessing version (recommended)**

```bash
python my_multiprocessing.py
```

This version improves UI responsiveness by running EEG acquisition in a separate process.

---

## File-Level Documentation

### testing_eeg_model.py

Handles the full prediction loop including:

- Connecting to Emotiv Cortex API via WebSocket
- Collecting EEG data in real-time
- Applying a 30-second baseline calculation
- Subtracting baseline and preprocessing EEG data
- Making predictions with a trained CNN model
- Updating the Pygame interface based on prediction accuracy

**Key Functions:**

- `process_baseline(segment)`: Computes the average EEG signal during baseline.
- `predict_and_move(segment)`: Applies preprocessing and makes a prediction.
- `move_ball(predicted, target)`: Moves the dot if prediction matches the correct direction.
- `get_eeg_input()`: Async coroutine that manages EEG data streaming.

---

### my_multiprocessing.py

This script mirrors the functionality of `testing_eeg_model.py` but improves performance using multiprocessing.

- Spawns a background process to stream EEG data and send it through a queue
- Main process handles game logic and model inference
- Prevents interface freezing during model prediction

**Key Functions:**

- `eeg_worker(queue)`: Background process that streams and queues EEG segments.
- `predict_and_move(segment)`: Same as in single-threaded version.
- `process_baseline(segment)`: Calculates baseline mean from EEG.
- `move_ball(pred, target)`: Executes correct movement action if prediction is accurate.

---

### preprocess.py

Responsible for converting raw EEG signals into CNN-compatible image tensors using spectrograms.

**Key Functions:**

- `eeg_to_spectrogram_tensor(eeg_raw)`: Converts raw EEG into a stack of normalized spectrogram images.
- `pre_process_item(series)`: Calls the spectrogram transformer and reshapes the result into `(1, 14, 64, 64)` format for CNN input.

---

### train_eeg_model_to_predict_direction.ipynb

A Jupyter notebook for training the CNN model. It performs the following:

- Loads and preprocesses EEG data
- Encodes direction labels using `LabelEncoder`
- Splits data into training and validation sets
- Trains a CNN classifier
- Saves:
  - The trained Keras model as `.h5`
  - The label encoder as `.pkl`

Use this notebook when collecting new training data or retraining the model.

---

## Output Files

- `game_data_YYYYMMDD_HHMMSS.csv`: Automatically saved logs of prediction sessions, including action timing and direction.
- `may25_eeg_prediction_model.h5`: Trained CNN model used in real-time prediction.
- `may25_label_encoder.pkl`: Label encoder used to decode the predicted class index.

---

## Notes for Future Developers

- Always match your model output order with the saved label encoder.
- The first 30 seconds of any session are used to compute a personalized baseline — do not skip this step.
- Multiprocessing is more robust in real-time usage; prefer `my_multiprocessing.py`.
- Training new models requires labeled EEG data in the same format as used previously (`800 x 14` window per sample).
- All predictions are softmax-based. You may optionally inspect prediction confidence or adjust thresholds.

---

## Potential Improvements

- Add real-time EEG graphing for signal visualization
- Implement confidence thresholding for predictions
- Add calibration options per user
- Extend to support more commands or controls (e.g., object selection)
- Add GUI options to change model or device in runtime

---

## License

This project is for educational and research purposes only. Not intended for commercial or clinical use.
