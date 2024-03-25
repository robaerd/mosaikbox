import logging

from app.services.music_source_separation import get_song_stem_filepath, StemType

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

logger = logging.getLogger(__name__)


def find_vocal_segments(session_id, audio_path: str) -> list[tuple[int, int]]:
    logger.debug(f"Finding vocal segments for audio {audio_path}")
    vocal_stem_path = get_song_stem_filepath(session_id, audio_path, StemType.VOCALS)

    vocal_audio = AudioSegment.from_wav(vocal_stem_path)
    min_non_silent_len = 400
    min_silent_len = 1000
    chunks = detect_nonsilent(vocal_audio,
                              # must be silent for at least 2 seconds
                              min_silence_len=min_silent_len,
                              # consider it silent if quieter than -40 dBFS
                              silence_thresh=-40,
                              seek_step=10)  # better performance than 1ms, and enough accuracy to detect vocal segments
    # min non silent len filtering
    filtered_chunks = []
    for chunk in chunks:
        if chunk[1] - chunk[0] > min_non_silent_len:
            filtered_chunks.append(chunk)
    logger.debug(f"Found {len(filtered_chunks)} vocal segments for audio {audio_path}")
    return chunks


def extract_segment_shift_to_zero(interval_list: list[tuple[int, int]], start_time: int, end_time: int):
    interval_list_end_filterd = filter(lambda x: x[0] < end_time, interval_list)
    interval_list_start_filterd = filter(lambda x: x[1] >= start_time, interval_list_end_filterd)
    interval_list_start_adjusted = [(start - start_time, end - start_time) for start, end in
                                    interval_list_start_filterd]
    # set every negative start to zero
    interval_list_negative_start_adjusted = [(max(0, start), end) for start, end in interval_list_start_adjusted]
    return interval_list_negative_start_adjusted


def calculate_intersection_duration(list1, list2):
    # Flatten both lists with start/end tags
    events = []
    for start, end in list1:
        events.append((start, 'start', 1))
        events.append((end, 'end', 1))
    for start, end in list2:
        events.append((start, 'start', 2))
        events.append((end, 'end', 2))

    # Sort the events by time, breaking ties by preferring 'end' over 'start'
    events.sort(key=lambda x: (x[0], x[1]))

    # Traverse the sorted events to find intersections
    active_intervals = [0, 0]  # Count of active intervals from each list
    intersection_start = None
    total_intersection_duration = 0

    for time, event_type, list_number in events:
        if event_type == 'start':
            active_intervals[list_number - 1] += 1
            if all(x > 0 for x in active_intervals):
                # We are entering an intersection
                intersection_start = time
        else:
            if all(x > 0 for x in active_intervals):
                # We are leaving an intersection
                total_intersection_duration += time - intersection_start
            active_intervals[list_number - 1] -= 1
            intersection_start = None

    return total_intersection_duration
