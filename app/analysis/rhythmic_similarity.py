import logging
from typing import Dict
import numpy as np

from app.config.config import RHYTHM_SIMILARITY_QUANTIZATION_OVER_BEATS
from sklearn.metrics.pairwise import cosine_similarity
from ADTLib import ADT

logger = logging.getLogger(__name__)


def transcribe_percussion(audio: np.ndarray, sr: int, beats: np.ndarray):
    logger.debug("Transcribing percussion")
    assert np.ndim(beats) == 1  # we do not want beat type here

    percussion_timing = ADT(audio, sample_rate=sr)
    quantized_beats = _quantize(beats, RHYTHM_SIMILARITY_QUANTIZATION_OVER_BEATS, percussion_timing)

    return quantized_beats


def calculate_percussive_similarity(quantized_beats_1, quantized_beats_2, features: [str] = ['kick', 'snare', 'hihat']):
    similarities = []
    if 'kick' in features:
        # kick_similarity = cosine_similarity(quantized_beats_1[:, 1].reshape(1, -1), quantized_beats_2[:, 1].reshape(1, -1))
        kick_similarity = matching_positions_similarity(quantized_beats_1[:, 1], quantized_beats_2[:, 1])
        similarities.append(kick_similarity)
    if 'snare' in features:
        # snare_similarity = cosine_similarity(quantized_beats_1[:, 2].reshape(1, -1), quantized_beats_2[:, 2].reshape(1, -1))
        snare_similarity = matching_positions_similarity(quantized_beats_1[:, 2], quantized_beats_2[:, 2])
        similarities.append(snare_similarity)
    if 'hihat' in features:
        # hihat_similarity = cosine_similarity(quantized_beats_1[:, 3].reshape(1, -1), quantized_beats_2[:, 3].reshape(1, -1))
        hihat_similarity = matching_positions_similarity(quantized_beats_1[:, 3], quantized_beats_2[:, 3])
        similarities.append(hihat_similarity)

    similarity = np.mean(similarities)
    return similarity


def matching_positions_similarity(a, b):
    # Calculate matches for both 1 and 0
    matches = np.sum(a == b)
    # Total number of elements
    total_elements = len(a)
    # Compute similarity: twice the number of matches divided by the total elements
    similarity = 2. * matches / (total_elements + total_elements)
    return similarity


def _quantize(beats: np.ndarray, num_positions: int, percusstion_timing: Dict) -> np.ndarray:
    # if 4 is changed to amount of beats * num_positions, then it will quantize to the amount of beats2o
    beat_amount = len(beats)
    max_beat_timing = np.max(beats)
    sub_beats = np.linspace(beats[0], beats[-1], num_positions * (beat_amount - 1) + 1, endpoint=True)

    # extract the beat positions from the percussion timing where the instrument is a kick drum
    beat_positions_kd = percusstion_timing['Kick']
    beat_positions_sd = percusstion_timing['Snare']
    beat_positions_hh = percusstion_timing['Hihat']

    # filter all beat positions that are smaller than the max beat timing
    if len(beat_positions_kd) > 0:
        beat_positions_kd = beat_positions_kd[beat_positions_kd <= max_beat_timing]
    if len(beat_positions_sd) > 0:
        beat_positions_sd = beat_positions_sd[beat_positions_sd <= max_beat_timing]
    if len(beat_positions_hh) > 0:
        beat_positions_hh = beat_positions_hh[beat_positions_hh <= max_beat_timing]

    quantized_beats_kd_pos = np.digitize(beat_positions_kd, sub_beats)
    quantized_beats_sd_pos = np.digitize(beat_positions_sd, sub_beats)
    quantized_beats_hh_pos = np.digitize(beat_positions_hh, sub_beats)
    # convert quaniized beats to 1d array of 0s and 1s
    quantized_beats_kd = np.zeros(len(sub_beats))
    quantized_beats_sd = np.zeros(len(sub_beats))
    quantized_beats_hh = np.zeros(len(sub_beats))
    quantized_beats_kd[quantized_beats_kd_pos] = 1
    quantized_beats_sd[quantized_beats_sd_pos] = 1
    quantized_beats_hh[quantized_beats_hh_pos] = 1

    stacked_timing_quantized_beats = np.stack((sub_beats, quantized_beats_kd, quantized_beats_sd, quantized_beats_hh),
                                              axis=1)

    return stacked_timing_quantized_beats


def _calculate_similarity(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    similarity = cosine_similarity(arr1, arr2)
    return similarity


# Uses isclose since np.linspace has floating point precision issues at around 1e-15
def get_subset_of_percussion_timing(quantized_percussion: np.ndarray, start_time, end_time) -> np.ndarray:
    # necessary because np.linspace has floating point precision issues at around 1e-15
    atol = 1e-10  # to have a bigger tolerance than the error
    rtol = 0.0  # we don't want relative tolerance as we know that the error is in the area of 1e-15
    is_close_start_time = np.isclose(quantized_percussion[:, 0], start_time, rtol=rtol, atol=atol)
    is_close_end_time = np.isclose(quantized_percussion[:, 0], end_time, rtol=rtol, atol=atol)
    close_or_ge_start_time = np.logical_or(quantized_percussion[:, 0] >= start_time, is_close_start_time)
    close_or_ge_end_time = np.logical_or(quantized_percussion[:, 0] <= end_time, is_close_end_time)
    mask = np.logical_and(close_or_ge_start_time, close_or_ge_end_time)

    return quantized_percussion[mask]

def adjust_beat_timing_to_start_at_0(beat_timing: np.ndarray) -> np.ndarray:
    # subtract the first timing from all timings
    percussion_timing_adjusted = np.copy(beat_timing)
    first = percussion_timing_adjusted[0, 0]

    if first == 0.0:
        logger.debug("Percussion/Beat timing already starts at 0")
        return percussion_timing_adjusted

    percussion_timing_adjusted[:, 0] = percussion_timing_adjusted[:, 0] - first
    return percussion_timing_adjusted
