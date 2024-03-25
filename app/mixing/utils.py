import logging

import numpy as np
from pydub import AudioSegment

from app.analysis import vocal
from app.mixing import audio_filter
from app.services import utilities

logger = logging.getLogger(__name__)

def extract_audio_segment(
        audio_segment: AudioSegment,
        start_time_seconds: float,
        end_time_seconds: float,
        stretch_playback_rate: float = 1.0
) -> np.ndarray:
    reversed_stretch_playback_rate = 1 / stretch_playback_rate
    unstretched_start_time_seconds = start_time_seconds * reversed_stretch_playback_rate
    unstretched_end_time_seconds = end_time_seconds * reversed_stretch_playback_rate

    start_time_ms = unstretched_start_time_seconds * 1000  # pydub works with ms
    end_time_ms = unstretched_end_time_seconds * 1000

    return audio_segment[start_time_ms:end_time_ms]


def load_and_normalize_audiosegment(audio_path: str, manual_gain: float = None):
    """
    If manual_gain is given, then we don't normalize to a target lufs, instead we just apply the gain
    Used for loading and normalizing main tracks and loading and setting gain of stems of the main track
    :param audio_path:
    :return:
    """
    logger.debug("Loading and normalizing audio {}".format(audio_path))
    audio_segment = AudioSegment.from_file(audio_path)
    audio_segment_normalized, gain = audio_filter.normalize_loudness(audio_segment, manual_gain=manual_gain)
    return audio_segment_normalized, gain


def extract_and_prepare_audio_segment(audio_segment: AudioSegment, start_time: float, end_time: float,
                                      stretch_playback_rate: float, key_shift: int):
    
    audio_segment = extract_audio_segment(audio_segment,
                                          start_time,
                                          end_time,
                                          stretch_playback_rate=stretch_playback_rate)
    return utilities.adjust_tempo_and_key_audio_segment(audio_segment, playback_rate=stretch_playback_rate,
                                                        key_shift=key_shift)


def linear_drum_volume_transformation(audio_segment_without_drums: AudioSegment,
                                      audio_segment_only_drums,
                                      starting_factor: float,
                                      target_factor: float):
    audio_segment_only_drums_linear_vol_decrease = audio_filter.linear_volume_transformation(audio_segment_only_drums,
                                                                                             target_factor=target_factor,
                                                                                             starting_factor=starting_factor)
    return audio_segment_without_drums.overlay(audio_segment_only_drums_linear_vol_decrease)


def should_remove_end_vocals(current_vocal_intervals: list[tuple[int, int]],
                             current_song_end_time_stretched: int,
                             overlay_duration_before_transition_ms: int,
                             overlay_duration_after_transition_ms: int,
                             next_song_vocal_intervals: list[tuple[int, int]],
                             next_song_start_time_stretched: int) -> bool:
    current_song_end_vocals = vocal.extract_segment_shift_to_zero(current_vocal_intervals,
                                                                  current_song_end_time_stretched - overlay_duration_before_transition_ms,
                                                                  current_song_end_time_stretched + overlay_duration_after_transition_ms)

    next_song_start_vocals = vocal.extract_segment_shift_to_zero(next_song_vocal_intervals,
                                                                 next_song_start_time_stretched - overlay_duration_before_transition_ms,
                                                                 next_song_start_time_stretched + overlay_duration_after_transition_ms)

    intersection_ms = vocal.calculate_intersection_duration(current_song_end_vocals, next_song_start_vocals)
    return intersection_ms > 2000
