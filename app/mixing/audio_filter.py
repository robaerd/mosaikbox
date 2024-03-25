import logging

import pydub
import scipy.signal as signal
import numpy as np
from pydub import AudioSegment

from ..services import utilities, audio_loader

logger = logging.getLogger(__name__)


def normalize(y: np.ndarray) -> (np.ndarray, float):
    audio = np.array(y)
    max_value = np.iinfo(audio.dtype).max
    return audio / np.iinfo(audio.dtype).max, max_value


def denormalize(y_normalized: np.ndarray, max_value: float) -> np.ndarray:
    return y_normalized * max_value


def _linear_filter(audio: np.ndarray, fs: int, filter_type: str, cutoff_frequency: int, target_factor: float,
                   starting_factor: float = 1.0) -> np.ndarray:
    assert target_factor >= 0.0
    assert target_factor <= 1.0
    assert starting_factor >= 0.0
    assert starting_factor <= 1.0

    starting_factor_inverse = 1 - starting_factor
    target_factor_inverse = 1 - target_factor

    y, audio_original_dtype_max = normalize(audio)

    order = 5
    normal_cutoff = cutoff_frequency / (0.5 * fs)
    b, a = signal.butter(order, normal_cutoff, btype=filter_type)

    y_bass_removed = signal.filtfilt(b, a, y)  # apply the filter to the signal

    y_out = np.zeros_like(y)  # init output signal

    # Compute the blend factor for each sample and use it to blend the original and filtered signals
    for i in range(len(y)):
        blend_factor = starting_factor_inverse - (starting_factor_inverse - target_factor_inverse) * (i / len(y))
        y_out[i] = (1 - blend_factor) * y[i] + blend_factor * y_bass_removed[i]

    # normalize the output signal to prevent clipping
    if np.max(y_out) > 1.0:
        logger.debug("Normalizing again after filter application to prevent clipping")
        y_out /= np.abs(y_out).max()
    return denormalize(y_out, max_value=audio_original_dtype_max)


def _linear_low_filter(audio: np.ndarray, fs: int, target_factor: float, starting_factor: float = 1.0) -> np.ndarray:
    LOW_CUTOFF_FREQUENCY = 140
    return _linear_filter(audio, fs, 'high', LOW_CUTOFF_FREQUENCY, target_factor, starting_factor)


def _linear_high_filter(audio: np.ndarray, fs: int, target_factor: float, starting_factor: float = 1.0) -> np.ndarray:
    LOW_CUTOFF_FREQUENCY = 2800
    return _linear_filter(audio, fs, 'low', LOW_CUTOFF_FREQUENCY, target_factor, starting_factor)


def _linear_bandstop_filter(audio: np.ndarray, fs: int, lowcut: int, highcut: int, target_factor: float,
                            starting_factor: float = 1.0) -> np.ndarray:
    """
    Apply a bandstop filter to the audio.
    :param audio: Input audio data.
    :param fs: Sampling rate of the audio data.
    :param lowcut: Lower bound frequency to start attenuation.
    :param highcut: Higher bound frequency to end attenuation.
    :param target_factor: The amount of attenuation at mid frequencies.
    :param starting_factor: Filter starting factor.
    :return: Filtered audio data.
    """
    assert target_factor >= 0.0
    assert target_factor <= 1.0
    assert starting_factor >= 0.0
    assert starting_factor <= 1.0

    starting_factor_inverse = 1 - starting_factor
    target_factor_inverse = 1 - target_factor

    y, audio_original_dtype_max = normalize(audio)

    order = 5
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='bandstop')

    y_filtered = signal.filtfilt(b, a, y) # apply the filter to the signal

    # Compute the blend factor for each sample and use it to blend the original and filtered signals
    y_out = np.zeros_like(y)
    for i in range(len(y)):
        blend_factor = starting_factor_inverse - (starting_factor_inverse - target_factor_inverse) * (i / len(y))
        y_out[i] = (1 - blend_factor) * y[i] + blend_factor * y_filtered[i]

    # normalize the output signal to prevent clipping
    if np.max(y_out) > 1.0:
        logger.debug("Normalizing again after filter application to prevent clipping")
        y_out /= np.abs(y_out).max()
    return denormalize(y_out, max_value=audio_original_dtype_max)


def linear_mid_filter(audio: pydub.AudioSegment, target_factor: float,
                      starting_factor: float = 1.0) -> pydub.AudioSegment:
    """
    Apply mid filter on the audio segment.
    :param audio: Input audio segment.
    :param target_factor: The amount of attenuation at mid frequencies.
    :param starting_factor: Filter starting factor.
    :return: Filtered audio segment.
    """
    logger.debug(f"Applying mid filter. Start: {starting_factor}, Target: {target_factor}")

    LOW_CUTOFF_FREQUENCY = 140
    HIGH_CUTOFF_FREQUENCY = 2800

    y = audio.get_array_of_samples()
    y_mid = _linear_bandstop_filter(y, audio.frame_rate, LOW_CUTOFF_FREQUENCY, HIGH_CUTOFF_FREQUENCY, target_factor,
                                    starting_factor)
    return utilities.convert_audio_array_of_samples_to_audio_segment(y_mid, audio.sample_width, audio.frame_rate,
                                                                     audio.channels)


def linear_low_filter(audio: pydub.AudioSegment,
                      target_factor: float,
                      starting_factor: float = 1.0) -> pydub.AudioSegment:
    logger.debug(f"Applying low filter. Start: {starting_factor}, Target: {target_factor}")
    y = audio.get_array_of_samples()
    y_filtered = _linear_low_filter(y, audio.frame_rate, target_factor=target_factor, starting_factor=starting_factor)
    return utilities.convert_audio_array_of_samples_to_audio_segment(y_filtered, audio.sample_width, audio.frame_rate,
                                                                     audio.channels)


def linear_high_filter(audio: pydub.AudioSegment,
                       target_factor: float,
                       starting_factor: float = 1.0) -> pydub.AudioSegment:
    logger.debug(f"Applying high filter. Start: {starting_factor}, Target: {target_factor}")
    y = audio.get_array_of_samples()
    y_filtered = _linear_high_filter(y, audio.frame_rate, target_factor=target_factor, starting_factor=starting_factor)
    return utilities.convert_audio_array_of_samples_to_audio_segment(y_filtered, audio.sample_width, audio.frame_rate,
                                                                     audio.channels)


def _linear_volume_transformation(audio: np.ndarray, target_factor: float, starting_factor: float = 1.0) -> np.ndarray:
    audio, audio_original_dtype_max = normalize(audio)
    # Create a linear ramp from starting_factor to target_factor
    ramp = np.linspace(starting_factor, target_factor, num=len(audio))
    # Apply the ramp to the audio
    audio_ramp = audio * ramp
    return denormalize(audio_ramp, max_value=audio_original_dtype_max)


def linear_volume_transformation(audio: pydub.AudioSegment, target_factor: float,
                                 starting_factor: float = 1.0) -> AudioSegment:
    logger.debug(f"Applying linear volume transformation. Start: {starting_factor}, Target: {target_factor}")
    y = audio.get_array_of_samples()
    y_filtered = _linear_volume_transformation(y, target_factor=target_factor, starting_factor=starting_factor)
    return utilities.convert_audio_array_of_samples_to_audio_segment(y_filtered, audio.sample_width, audio.frame_rate,
                                                                     audio.channels)


def normalize_loudness(audio: pydub.AudioSegment, manual_gain=None) -> (AudioSegment, float):
    """
    Normalize the loudness of the audio segment to -24 LUFS if manual_gain is not set
    If manu_gain is set then only the given gain is applied to the audio segment
    :param audio:
    :param manual_gain:
    :return:
    """
    logger.debug(f"Normalizing loudness. Manual gain: {manual_gain}")
    audio_np = audio.get_array_of_samples()

    y_normalized, audio_original_dtype_max = normalize(audio_np)
    if manual_gain is None:
        y_out, gain = audio_loader.normalize_loudness(y_normalized, audio.frame_rate, target_loudness_lufs=-24)
    else:
        y_out = audio_loader.apply_gain_lufs(y_normalized, manual_gain)
        gain = manual_gain
    y_denormalized = denormalize(y_out, max_value=audio_original_dtype_max)

    return (utilities.convert_audio_array_of_samples_to_audio_segment(y_denormalized, audio.sample_width,
                                                                      audio.frame_rate, audio.channels), gain)
