import copy
import logging
import os.path
from collections import Counter
from operator import itemgetter, attrgetter
from typing import List, Optional

import msaf
import numpy as np
import jams
from pathlib import Path

from app.config.config import SEGMENT_MIN_TRANSITION_OFFSET
from app.services.utilities import get_file_name_without_extension, get_file_extension, calculate_file_hash, copy_file
from app.config.path_config import get_segmentation_data_path
from . import timbre_calculation
from .Segment import Segment
from ..services.song_schedule_service import SongScheduleService
from ..analysis import beats as bts
from ..analysis import rhythmic_similarity as rs

logger = logging.getLogger(__name__)


def calculate_song_boundaries(session_id: str, audio: np.ndarray, sr: int, audio_path: str, beats: np.ndarray,
                              plot: bool = False) -> (
        np.ndarray, List):
    # first store the audio file in the directory with the reference files (annotations)
    # include the hash of the file in the name
    file_hash = calculate_file_hash(audio_path)
    segmentation_data_path = get_segmentation_data_path(session_id)

    assert_segmentation_paths_exist(segmentation_data_path)

    destination_audio_name = get_file_name_without_extension(
        audio_path) + '_' + file_hash + get_file_extension(audio_path)
    destination_audio_path = os.path.join(segmentation_data_path, 'audio', destination_audio_name)
    # copy the audio file to the destination
    copy_file(audio_path, destination_audio_path)

    # then convert beats to annotations
    destination_jam_name = get_file_name_without_extension(destination_audio_name) + '.jams'
    convert_beats_to_jams_and_save(audio, sr, destination_jam_name, beats, segmentation_data_path)
    # configure parameters
    feature = "mfcc"
    bid = "sf"
    lid = "fmc2d"
    annot_beats = True
    # set the features_tmp_file to prevent race conditions
    msaf.config.features_tmp_file = os.path.join(segmentation_data_path, 'features_tmp',
                                                 f".features_tmp_file_{destination_audio_name}.json")
    boundaries, labels = msaf.process(destination_audio_path, boundaries_id=bid, feature=feature,
                                      annot_beats=annot_beats, labels_id=lid, plot=plot)
    logger.debug(f"Boundaries for {audio_path}: {boundaries}")
    logger.debug(f"Labels for {audio_path}: {labels}")

    return boundaries, labels


def _calculate_segment_length(start: float, end: float, tempo: float) -> int:
    segment_length = end - start
    segment_length_beats = round(segment_length * (tempo / 60))
    return segment_length_beats


def process_song_boundaries_to_segment_timing(boundaries: np.ndarray, labels: list[float], tempo: float,
                                              beats: np.ndarray) -> list[Segment]:
    def get_beat_type(beat_time: float, beats: np.ndarray) -> int:
        def find_nearest_idx(array: np.ndarray, value):
            return (np.abs(array - value)).argmin()

        # find the nearest beat
        nearest_beat_idx = find_nearest_idx(beats[:, 0], beat_time)
        # get the index of the nearest beat
        nearest_beat = beats[nearest_beat_idx]
        return nearest_beat[1]  # return the beat type

    # preprocess boundaries and merge segments
    segment_tuple_list = []  # list of tuples (start, end)
    last_beat_time = beats[-1, 0]

    for idx, boundary_start in enumerate(boundaries):
        if idx == len(boundaries) - 1:
            break

        boundary_end = boundaries[idx + 1]

        # TODO: first conjunct is unnecessary and can be removed
        if idx == len(boundaries) - 2 and (boundary_end > last_beat_time or boundary_end != last_beat_time):
            # boundary_end is longer than last beat
            # we just set the boundary_end to the last beat
            boundary_end = last_beat_time

        current_segment_length = _calculate_segment_length(boundary_start, boundary_end, tempo)
        current_segment = Segment(boundary_start, boundary_end, current_segment_length, labels[idx], 1.0)
        segment_tuple_list.append(current_segment)

    for idx, segment in enumerate(segment_tuple_list):
        segment_tuple_list[idx].start_beat_type = get_beat_type(segment.start_time, beats)
        segment_tuple_list[idx].end_beat_type = get_beat_type(segment.end_time, beats)
    return segment_tuple_list


def find_nearest_idx(array: np.ndarray, value):
    return (np.abs(array - value)).argmin()


def calculate_segment_compatibility_based_on_labels(_segments: list[Segment], audio_length: float):
    segments = copy.deepcopy(_segments)

    intro_outro_compatibility = 0.5
    if len(segments) == 1:
        # we penalize songs with only one segment
        segments[0].compatibility = intro_outro_compatibility
        return segments

    intro_segment_label = segments[0].label
    outro_segment_label = segments[-1].label

    segments_without_intro_outro = list(
        filter(lambda x: (x.label not in [intro_segment_label, outro_segment_label]), segments))


    # Calculate the frequency of each label
    label_freq = Counter(attrgetter("label")(seg) for seg in segments_without_intro_outro)
    # Sort labels by count in descending order
    sorted_label_counts = sorted(label_freq.items(), key=itemgetter(1), reverse=True)

    weight = 1.0
    for key, occurrences in sorted_label_counts:
        for segment in segments:
            if segment.label == key:
                segment.compatibility = weight
        # weight -= 0.10

    # set intro and outro weights and all sections that match their label
    if weight < intro_outro_compatibility:
        intro_outro_compatibility = weight
    for segment in segments:
        if segment.label in [intro_segment_label, outro_segment_label]:
            segment.compatibility = intro_outro_compatibility

    # we want to prevent too early and too late segments to be able to do transitions
    for segment in segments:
        if segment.start_time - SEGMENT_MIN_TRANSITION_OFFSET < 0.0:
            logger.debug(f"Penalizing segment, because it starts too early: start_time: {segment.start_time} < segment_min_transition_offset: {SEGMENT_MIN_TRANSITION_OFFSET}")
            segment.compatibility = 0.0
        if segment.end_time + SEGMENT_MIN_TRANSITION_OFFSET > audio_length:
            logger.debug(f"Penalizing segment, because it ends too late: end_time + segment_min_transition_offset: {segment.end_time + SEGMENT_MIN_TRANSITION_OFFSET} > audio_length: {audio_length}")
            segment.compatibility = 0.0

    return segments


def calculate_timbre_for_segments(audio: np.ndarray, sr: int, beats: np.ndarray, segments: list[Segment]) -> list[
    Segment]:
    for idx, segment in enumerate(segments):
        # get the audio for the segment
        logger.debug(f"Calculating timbre for segment {idx} out of {len(segments)}")
        segment_length = segment.end_time - segment.start_time
        if segment_length < 5:  # we don't want to calculate timbre for segments shorter than 5 seconds
            logger.debug(f"Skipping timbre calculation for segment {idx} because it is shorter than 5 seconds")
            continue
        segment_audio, segment_beats = extract_and_shift_audio_and_beats(audio, sr, beats, segment.start_time,
                                                                         segment.end_time)
        # calculate the timbre for the segment
        segments[idx].timbre_vectors = timbre_calculation.calculate_timbre(segment_audio, sr, segment_beats[:, 0])

    return segments


def map_segments_timbre_to_beat_timbre(segments: list[Segment], beats: np.ndarray, time_until_first_beat: float) -> \
list[np.ndarray]:
    beat_timings = beats[:, 0]
    beat_sync_timbre = []

    atol = 1e-10  # to have a bigger tolerance than the error
    rtol = 0.0  # we don't want relative tolerance as we know that the error is in the area of 1e-15

    for idx, beat_timing in enumerate(beat_timings):
        beat_timing_with_downbeat_offset = beat_timing + time_until_first_beat
        # get the timbre vector for the beat
        segment_found = False
        for segment in segments:
            if np.isclose(segment.start_time, beat_timing_with_downbeat_offset, rtol=rtol, atol=atol) or (
                    segment.start_time <= beat_timing_with_downbeat_offset < segment.end_time):
                beat_sync_timbre.append(segment.timbre_vectors)
                segment_found = True
                break
        if not segment_found:
            beat_sync_timbre.append(0.0)

    return beat_sync_timbre


def extract_and_shift_audio_and_beats(
        audio: np.ndarray,
        sr: int,
        beats: np.ndarray,
        start_beat_time: float,
        end_beat_time: float,
):
    segment_audio = SongScheduleService.extract_audio_vector_segment(audio_vector=audio,
                                                                     start_time_seconds=start_beat_time,
                                                                     end_time_seconds=end_beat_time,
                                                                     sr=sr)

    segment_beats = bts.get_subset_of_beats_by_timing(beats=beats,
                                                      start_time=start_beat_time,
                                                      end_time=end_beat_time)
    segment_beats = rs.adjust_beat_timing_to_start_at_0(segment_beats)

    return segment_audio, segment_beats


def merge_similar_segment_boundaries(segments: list[Segment]):
    merged_segments = []

    segment_to_merge: Optional[Segment] = None
    for idx, _ in enumerate(segments):
        segment = copy.deepcopy(segments[idx])
        if segment_to_merge is None:
            assert segment.label is not None
            segment_to_merge = segment
            continue
        elif segment_to_merge.label == segments[
            idx].label and segment.compatibility > 0:  # compatibility is 0 if segment.start_time ist too early or segment.end_time is too late for transition
            assert segment.label is not None
            logger.debug(f"Merging segments with same label: {segment.label}")
            segment_to_merge.end_time = segment.end_time
            segment_to_merge.beat_amount += segment.beat_amount
            segment_to_merge.end_beat_type = segment.end_beat_type
            continue
        else:
            merged_segments.append(segment_to_merge)
            segment_to_merge = segment
            continue
    merged_segments.append(segment_to_merge)
    return merged_segments


def merge_segments_until_min_length(start_time: float, segments: list[Segment], min_length: float,
                                    is_first_song: bool = False) -> Segment:
    segment_to_merge: Optional[Segment] = None
    for idx, _segment in enumerate(segments):
        if _segment.start_time < start_time:
            continue

        segment = copy.deepcopy(segments[idx])
        if segment_to_merge is None:
            segment_to_merge = segment
            continue

        length = segment_to_merge.end_time - segment_to_merge.start_time
        if length < min_length and (segment.compatibility > 0 or is_first_song):
            assert segment.label is not None
            logger.debug(f"Merging segments with same label: {segment.label}")
            segment_to_merge.end_time = segment.end_time
            segment_to_merge.beat_amount += segment.beat_amount
            segment_to_merge.end_beat_type = segment.end_beat_type
            segment_to_merge.label = segment.label
            continue
        elif length < min_length and segment.compatibility == 0:
            # we don't want to merge segments with compatibility 0
            logger.debug("Not merging segments with compatibility 0. Segment will be below minimum length.")
            break
        else:
            break
    return segment_to_merge


def filter_and_move_segment_boundaries_to_downbeats(segments: list[Segment], beats: np.ndarray, tempo: float) -> list[
    Segment]:
    last_beat_time = beats[-1, 0]
    last_beat_type = beats[-1, 1]
    second_last_beat_time = beats[-2, 0]
    processed_segments = []

    for idx, _ in enumerate(segments):
        segment = copy.deepcopy(segments[idx])

        # since multiple segments can have the same end-time, we have to execute the following check for every segment
        if segment.end_time > last_beat_time:
            logger.debug("Segment end time exceeds last beat time, shifting segment end time to the last beat...")
            # the segment end time is longer than the last beat
            # we just set the segment end time to the last beat
            logger.debug("Segment end time is now {} and beat type changed from {} to {}".format(last_beat_type,
                                                                                          segment.end_beat_type,
                                                                                          last_beat_type))
            segment.end_time = last_beat_time
            segment.end_beat_type = last_beat_type
        elif second_last_beat_time < segment.end_time < last_beat_time:
            # the segment end time is in between the last beat and the second last beat and not aligned
            # we just set it to the last beat
            logger.debug("Segment end time is now {} and beat type changed from {} to {}".format(last_beat_type,
                                                                                          segment.end_beat_type,
                                                                                          last_beat_type))
            segment.end_time = last_beat_time
            segment.end_beat_type = last_beat_type

        # todo: this can be moved out of the loop
        # move the first segment start time to the first beat
        if idx == 0 and segment.start_time < beats[0, 0]:
            logger.debug("Moving segment start time to the first beat...")
            segment.start_time = beats[0, 0]
            segment.start_beat_type = beats[0, 1]

        # we beat align each segment start time and end time
        # if a segment time is not aligned, we shift it to the closest beat
        # start_time
        (beat_index_start,) = np.where(beats[:, 0] == segment.start_time)
        assert len(beat_index_start) <= 1
        if len(beat_index_start) == 0:
            # segment start time is not aligned
            # we shift it to the closest beat
            logger.debug("Segment start time is not aligned, shifting to closest beat...")
            nearest_beat_idx = find_nearest_idx(beats[:, 0], segment.start_time)
            segment.start_time = beats[nearest_beat_idx, 0]
            segment.start_beat_type = beats[nearest_beat_idx, 1]
        # end_time
        (beat_index_end,) = np.where(beats[:, 0] == segment.end_time)
        assert len(beat_index_end) <= 1
        if len(beat_index_end) == 0:
            # segment end time is not aligned
            # we shift it to the closest beat
            logger.debug("Segment end time is not aligned, shifting to closest beat...")
            nearest_beat_idx = find_nearest_idx(beats[:, 0], segment.end_time)
            segment.end_time = beats[nearest_beat_idx, 0]
            segment.end_beat_type = beats[nearest_beat_idx, 1]

        # main alignment procedure
        if segment.start_beat_type == 2:
            logger.debug("Shifting segment start time by one beat to the left")
            # shift the segment to the left by one beat
            (beat_index,) = np.where(beats[:, 0] == segment.start_time)
            assert len(beat_index) == 1
            new_start_time = beats[beat_index[0] - 1, 0]
            segment.start_time = new_start_time
            segment.start_beat_type = 1.0
        elif segment.start_beat_type == 4:
            logger.debug("Shifting segment start time by one beat to the right")
            # shift the segment to the right by one beat
            (beat_index,) = np.where(beats[:, 0] == segment.start_time)
            assert len(beat_index) == 1
            # check if we can shift it to the right
            if beat_index[0] + 1 >= len(beats):
                logger.debug("Segment start time exceeds last beat time, not shifting segment...")
            else:
                new_start_time = beats[beat_index[0] + 1, 0]
                segment.start_time = new_start_time
                segment.start_beat_type = 1.0
        elif segment.start_time == 3:
            logger.debug("Segment starts on the third beat, not shifting segment.")
            # continue

        if segment.end_beat_type == 2:
            logger.debug("Shifting segment end time by one beat to the left")
            # shift the segment to the left by one beat
            (beat_index,) = np.where(beats[:, 0] == segment.end_time)
            assert len(beat_index) == 1
            new_end_time = beats[beat_index[0] - 1, 0]
            segment.end_time = new_end_time
            segment.end_beat_type = 1.0
        elif segment.end_beat_type == 4 and segment.end_time != last_beat_time:
            logger.debug("Shifting segment end time by one beat to the right")
            # shift the segment to the right by one beat
            (beat_index,) = np.where(beats[:, 0] == segment.end_time)
            assert len(beat_index) == 1
            if beat_index[0] + 1 >= len(beats):
                logger.debug("Segment end time exceeds last beat time, not shifting segment...")
            else:
                new_end_time = beats[beat_index[0] + 1, 0]
                segment.end_time = new_end_time
                segment.end_beat_type = 1.0
        elif segment.end_beat_type == 3 and segment.end_time != last_beat_time:
            logger.debug("Segment ends on the third beat, not shifting segment.")
            # continue

        segment.beat_amount = _calculate_segment_length(segment.start_time, segment.end_time, tempo)
        processed_segments.append(segment)
    return processed_segments


def assert_segmentation_paths_exist(segmentation_data_path: str):
    path_segmentation_audio = Path(os.path.join(segmentation_data_path, 'audio'))
    path_segmentation_references = Path(os.path.join(segmentation_data_path, 'references'))
    path_segmentation_features_tmp = Path(os.path.join(segmentation_data_path, 'features_tmp'))

    path_segmentation_audio.mkdir(parents=True, exist_ok=True)
    path_segmentation_references.mkdir(parents=True, exist_ok=True)
    path_segmentation_features_tmp.mkdir(parents=True, exist_ok=True)


def convert_beats_to_jams_and_save(audio: np.ndarray, sr: int, destination_jam_name: str, beat_times: np.ndarray,
                                   segmentation_data_path: str):
    #  We will now save or beats to a JAMS file.
    jam = jams.JAMS()
    jam.file_metadata.duration = len(audio) / sr
    beat_a = jams.Annotation(namespace='beat')
    beat_a.annotation_metadata = jams.AnnotationMetadata(data_source='BeatNet beat tracker')

    #  Add beat timings to the annotation record.
    #  The beat namespace does not require value or confidence fields,
    #  so we can leave those blank.
    for idx, (beat_timing, beat_type) in enumerate(beat_times):
        if idx == 0 and beat_timing != 0.0:
            # append a beat at the beginning of the song
            # necessary for the segmentation algorithm
            beat_a.append(time=0.0, duration=0.0)
        elif idx == len(beat_times) - 1 and beat_timing != jam.file_metadata.duration:
            # append a beat at the end of the song
            # necessary for the segmentation algorithm
            beat_a.append(time=jam.file_metadata.duration, duration=0.0)
        else:
            beat_a.append(time=beat_timing, duration=0.0)

    #  Store the new annotation in the jam file. This need to be located on the references folder
    #  and be named like the audio file except for the jams extension.
    jam.annotations.append(beat_a)
    jam.save(os.path.join(segmentation_data_path, 'references', destination_jam_name))
