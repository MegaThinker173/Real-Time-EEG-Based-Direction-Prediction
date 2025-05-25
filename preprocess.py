import numpy as np
import pywt

'''
# Get 2D scalogram
def get_scalogram(signal, wavelet='morl', scales=np.arange(1,64)):
    coef, freqs = pywt.cwt(signal, scales, wavelet)
    coef = coef.astype(np.float32)
    return np.abs(coef)

# series shape: (800, 14)
# return model input shape: ( 63, 400, 1 )
def pre_process_item(series):
    series = np.mean(series, axis=1)  # shape: (800,)
    series = series[::2]  # optional: downsample to 400

    scalogram = get_scalogram(series)
    return scalogram.reshape(1, 63, 400, 1)
'''

import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F

def eeg_to_spectrogram_tensor(
    eeg_raw,
    fs=128,
    nperseg=64,
    noverlap=32,
    resize_to=(64, 64),
    normalize=True,
    log_scale=True,
):
    """
    Chuyển tín hiệu EEG [T, 14] thành tensor ảnh [14, H, W] (dùng spectrogram)
    """
    n_channels = eeg_raw.shape[1]
    spectrograms = []

    for ch in range(n_channels):
        f, t, Sxx = scipy.signal.spectrogram(eeg_raw[:, ch], fs=fs, nperseg=nperseg, noverlap=noverlap)

        if log_scale:
            Sxx = np.log1p(Sxx)

        if normalize:
            mean = np.mean(Sxx)
            std = np.std(Sxx)
            Sxx = (Sxx - mean) / (std + 1e-8)

        # Resize từng ảnh về cùng kích thước nếu cần
        Sxx_resized = F.interpolate(
            input=torch.tensor(Sxx).unsqueeze(0).unsqueeze(0),  # [1, 1, F, T]
            size=resize_to,
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()  # [H, W]

        spectrograms.append(Sxx_resized)

    # Stack lại thành tensor [14, H, W]
    spec_tensor = np.stack(spectrograms, axis=0)
    return spec_tensor


def pre_process_item(series):
    return eeg_to_spectrogram_tensor(series).reshape(1,  14, 64, 64)