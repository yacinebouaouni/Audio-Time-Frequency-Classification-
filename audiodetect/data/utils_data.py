"""This module provides functions for audio processing:

* load_audio_file(file_path): Loads audio data from a file.
* compute_log_mel_spectrogram(audio, config): Computes log Mel spectrogram.

Requires librosa library for audio processing tasks.
"""
import librosa
import numpy as np

def load_audio_file(file_path):
    """load audio from file_path"""
    audio, _ = librosa.load(file_path, sr=None)
    return audio

def compute_log_mel_spectrogram(audio, config):
    """
    Compute the log Mel spectrogram of an audio signal.

    Parameters:
        audio (numpy.ndarray): The input audio signal.
        config: An object containing hyperparameters for the spectrogram computation.
            Required attributes:
                - sampling_rate (int): Sampling rate of the audio signal.
                - n_fft (int): Number of samples in each short-time Fourier transform (STFT).
                - hop_length (int): Number of samples between consecutive STFT frames.
                - n_mels (int): Number of Mel bands to generate.
                - fmin (float): Minimum frequency for Mel filter banks.
                - fmax (float): Maximum frequency for Mel filter banks.

    Returns:
        numpy.ndarray: The computed log Mel spectrogram.

    """
    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=config.FS,
                                                     n_fft=config.N_FFT,
                                                     hop_length=config.HOP,
                                                     n_mels=config.N_MELS,
                                                     fmin=config.MIN_FREQ,
                                                     fmax=config.MAX_FREQ)
    # Compute log Mel spectrogram
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram
