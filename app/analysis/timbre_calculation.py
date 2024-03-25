import librosa
import numpy as np
import scipy
from scipy import signal

from .timbral_roughness import timbral_roughness


# returns a 28 dimensional vector
def calculate_timbre(y: np.ndarray, sr: int, beat_timings: np.ndarray):
    # convert beat timings to beat_frames
    hop_length = 512
    beat_frames = librosa.time_to_frames(times=beat_timings, sr=sr, hop_length=hop_length)

    # Now we need to consider 1-beat duration half-overlapping frames
    # First, let's make beat_frames half-overlapping
    half_overlapping_beat_frames = []
    for i in range(len(beat_frames) - 1):
        half_beat = beat_frames[i] + (beat_frames[i + 1] - beat_frames[i]) // 2
        half_overlapping_beat_frames.extend([beat_frames[i], half_beat])
    half_overlapping_beat_frames.append(beat_frames[-1])  # include the last beat frame

    # Feature bands from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=969559
    # Robust matching of audio signals using spectral flatness features (Herre, Allamanche, and Hellmuth (2001))
    # Define the frequency bands in Hz with uniform subdivisions
    bands = [(300 + i * (6000 - 300) / 4, 300 + (i + 1) * (6000 - 300) / 4) for i in range(4)]

    mfcc = calculate_mfcc(y, sr, half_overlapping_beat_frames)
    spectral_flatness = calculate_spectral_flatness(y, sr, bands, half_overlapping_beat_frames, hop_length=hop_length)

    roughness = timbral_roughness(audio_samples=y, fs=sr, bands=bands)

    timbre_vector = np.hstack([mfcc, spectral_flatness, roughness])
    return timbre_vector


def calculate_mfcc(y: np.ndarray, sr: int, half_overlapping_beat_frames):
    # Compute MFCCs per frame
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # Now, let's get MFCCs for each half-overlapping beat frames
    beat_mfcc = librosa.util.sync(mfcc, half_overlapping_beat_frames, aggregate=np.mean)

    # Finally, let's average MFCCs over all frames
    average_mfcc = np.mean(beat_mfcc, axis=1)
    return average_mfcc


def calculate_spectral_flatness(y: np.ndarray, sr: int, bands: list[tuple[float, float]], half_overlapping_beat_frames,
                                hop_length: int = 512):
    # Small constant to prevent division by zero or near-zero values
    eps = 1e-10

    # Calculate the spectral flatness for each half overlapping beat frame
    sf_bands = []
    for beat_frame in half_overlapping_beat_frames:
        frame = y[beat_frame * hop_length: (beat_frame + 1) * hop_length]  # extract frame

        f, Pxx_den = signal.periodogram(frame, sr)  # compute power spectral density (PSD)

        # For each frequency band, calculate the spectral flatness
        sf_band = []
        for band in bands:
            band_indices = np.where((f >= band[0]) & (f <= band[1]))  # indices corresponding to the frequency band
            band_Pxx_den = Pxx_den[band_indices]  # PSD values corresponding to the frequency band

            if len(band_Pxx_den) > 0:  # avoid dividing by zero
                sf_band.append(scipy.stats.gmean(band_Pxx_den) / (np.mean(band_Pxx_den) + eps))
            else:
                sf_band.append(0)
        sf_bands.append(sf_band)

    # Compute the average spectral flatness for each band
    average_sf_bands = np.mean(sf_bands, axis=0)  # average spectral flatness for each band
    return average_sf_bands
