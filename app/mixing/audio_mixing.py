import logging

from pydub import AudioSegment

from . import audio_filter
from .utils import load_and_normalize_audiosegment, extract_and_prepare_audio_segment, \
    linear_drum_volume_transformation, should_remove_end_vocals
from ..models.mix import MixModel
from ..models.song_schedule import SongScheduleItem
from ..services import utilities
from ..services import music_source_separation as mss

logger = logging.getLogger(__name__)


def main_transition(base_audio: AudioSegment,
                    cand_audio: AudioSegment,
                    cand_audio_without_drum_stem: AudioSegment,
                    cand_audio_only_drum_stem: AudioSegment,
                    cand_audio_without_vocal_stem: AudioSegment,
                    cand_audio_only_vocal_stem: AudioSegment,
                    cand_audio_without_vocal_and_drum_stem: AudioSegment,
                    remove_drum_cand_beginning: bool,
                    remove_drum_cand_ending: bool,
                    remove_drum_cand_ending_beats: int,
                    remove_end_vocals: bool,
                    downbeat_duration_ms: float,
                    high_rhythmic_similarity: bool = False,
                    high_timbral_similarity: bool = False,
                    high_harmonic_similarity: bool = False,
                    transition_beat_extension: int = 0,  # amount of beats to add to transition
                    transition_scale_factor: float = 1.0):
    base_audio_seg1_start = int(downbeat_duration_ms) * (32 + transition_beat_extension) * transition_scale_factor
    base_audio_seg2_start = int(downbeat_duration_ms) * (16.25 + transition_beat_extension) * transition_scale_factor
    base_audio_seg3_start = int(downbeat_duration_ms) * (16 + transition_beat_extension) * transition_scale_factor
    base_audio_seg4_start = int(downbeat_duration_ms) * 8 * transition_scale_factor

    song1_seg0 = base_audio[:-base_audio_seg1_start]
    song1_seg1 = base_audio[-base_audio_seg1_start:-base_audio_seg2_start]
    song1_seg2 = base_audio[-base_audio_seg2_start:-base_audio_seg3_start]
    song1_seg3 = base_audio[-base_audio_seg3_start:-base_audio_seg4_start]
    song1_seg4 = base_audio[-base_audio_seg4_start:]

    if high_rhythmic_similarity:
        song1_seg2 = audio_filter.linear_low_filter(song1_seg2, target_factor=0.5, starting_factor=1.0)
        song1_seg3 = audio_filter.linear_low_filter(song1_seg3, target_factor=0.4, starting_factor=0.5)
        song1_seg4 = audio_filter.linear_low_filter(song1_seg4, target_factor=0.4, starting_factor=0.4)
    else:
        song1_seg2 = audio_filter.linear_low_filter(song1_seg2, target_factor=0.3, starting_factor=1.0)
        song1_seg3 = audio_filter.linear_low_filter(song1_seg3, target_factor=0.2, starting_factor=0.3)
        song1_seg4 = audio_filter.linear_low_filter(song1_seg4, target_factor=0.2, starting_factor=0.2)

    song1_seg2 = audio_filter.linear_high_filter(song1_seg2, target_factor=0.8, starting_factor=1.0)
    song1_seg3 = audio_filter.linear_high_filter(song1_seg3, target_factor=0.7, starting_factor=0.8)
    song1_seg4 = audio_filter.linear_high_filter(song1_seg4, target_factor=0.7, starting_factor=0.7)

    if not high_timbral_similarity:
        song1_seg1 = audio_filter.linear_mid_filter(song1_seg1, target_factor=0.8, starting_factor=1.0)
        song1_seg2 = audio_filter.linear_mid_filter(song1_seg2, target_factor=0.6, starting_factor=0.8)
        song1_seg3 = audio_filter.linear_mid_filter(song1_seg3, target_factor=0.5, starting_factor=0.6)
        song1_seg4 = audio_filter.linear_mid_filter(song1_seg4, target_factor=0.5, starting_factor=0.5)

    song1_seg4 = audio_filter.linear_volume_transformation(song1_seg4, target_factor=0.0, starting_factor=1.0)

    cand_audio_seg2_start = int(downbeat_duration_ms) * 8 * transition_scale_factor
    cand_audio_seg3_start = int(downbeat_duration_ms) * 15.75 * transition_scale_factor
    cand_audio_seg4_start = int(downbeat_duration_ms) * 16 * transition_scale_factor
    cand_audio_seg5_start = int(downbeat_duration_ms) * (32 + transition_beat_extension) * transition_scale_factor

    cand_audio_seg6_1_start_ending = int(downbeat_duration_ms) * 32 * transition_scale_factor
    cand_audio_seg6_start_ending = int(downbeat_duration_ms) * remove_drum_cand_ending_beats * transition_scale_factor
    cand_audio_seg7_start_transition_ending = int(downbeat_duration_ms) * 16 * transition_scale_factor

    song2_seg1 = cand_audio[:cand_audio_seg2_start] if not remove_drum_cand_beginning else cand_audio_without_drum_stem[
                                                                                           :cand_audio_seg2_start]
    song2_seg2 = cand_audio[
                 cand_audio_seg2_start:cand_audio_seg3_start] if not remove_drum_cand_beginning else linear_drum_volume_transformation(
        cand_audio_without_drum_stem[cand_audio_seg2_start:cand_audio_seg3_start],
        cand_audio_only_drum_stem[cand_audio_seg2_start:cand_audio_seg3_start], starting_factor=0.0, target_factor=0.8)
    song2_seg3 = cand_audio[
                 cand_audio_seg3_start:cand_audio_seg4_start] if not remove_drum_cand_beginning else linear_drum_volume_transformation(
        cand_audio_without_drum_stem[cand_audio_seg3_start:cand_audio_seg4_start],
        cand_audio_only_drum_stem[cand_audio_seg3_start:cand_audio_seg4_start], starting_factor=0.8, target_factor=1.0)
    song2_seg4 = cand_audio[cand_audio_seg4_start:cand_audio_seg5_start]
    song2_seg5 = cand_audio[cand_audio_seg5_start:-cand_audio_seg6_1_start_ending]

    cand_audio_or_vocal_less_audio = cand_audio_without_vocal_stem if remove_end_vocals else cand_audio

    song2_seg6_1 = cand_audio_or_vocal_less_audio[
                   -cand_audio_seg6_1_start_ending:-cand_audio_seg6_start_ending] if not remove_end_vocals else linear_drum_volume_transformation(
        cand_audio_without_vocal_stem[-cand_audio_seg6_1_start_ending:-cand_audio_seg6_start_ending],
        cand_audio_only_vocal_stem[-cand_audio_seg6_1_start_ending:-cand_audio_seg6_start_ending], starting_factor=1.0,
        target_factor=0.0)

    drum_less_or_drum_and_vocal_less_audio = cand_audio_without_drum_stem if not remove_end_vocals else cand_audio_without_vocal_and_drum_stem

    song2_seg6 = cand_audio_or_vocal_less_audio[
                 -cand_audio_seg6_start_ending:-cand_audio_seg7_start_transition_ending] if not remove_drum_cand_ending else linear_drum_volume_transformation(
        drum_less_or_drum_and_vocal_less_audio[-cand_audio_seg6_start_ending:-cand_audio_seg7_start_transition_ending],
        cand_audio_only_drum_stem[-cand_audio_seg6_start_ending:-cand_audio_seg7_start_transition_ending],
        starting_factor=1.0, target_factor=0.4)

    song2_seg7 = cand_audio_or_vocal_less_audio[
                 -cand_audio_seg7_start_transition_ending:] if not remove_drum_cand_ending else linear_drum_volume_transformation(
        drum_less_or_drum_and_vocal_less_audio[-cand_audio_seg7_start_transition_ending:],
        cand_audio_only_drum_stem[-cand_audio_seg7_start_transition_ending:], starting_factor=0.4, target_factor=0.1)

    if high_rhythmic_similarity:
        song2_seg1 = audio_filter.linear_low_filter(song2_seg1, target_factor=0.3, starting_factor=0.3)
        song2_seg2 = audio_filter.linear_low_filter(song2_seg2, target_factor=0.4, starting_factor=0.3)
        song2_seg3 = audio_filter.linear_low_filter(song2_seg3, target_factor=1.0, starting_factor=0.4)
    else:
        song2_seg1 = audio_filter.linear_low_filter(song2_seg1, target_factor=0.2, starting_factor=0.2)
        song2_seg2 = audio_filter.linear_low_filter(song2_seg2, target_factor=0.3, starting_factor=0.2)
        song2_seg3 = audio_filter.linear_low_filter(song2_seg3, target_factor=1.0, starting_factor=0.3)

    if remove_drum_cand_beginning:
        song2_seg1 = audio_filter.linear_high_filter(song2_seg1, target_factor=0.8, starting_factor=0.5)
        song2_seg2 = audio_filter.linear_high_filter(song2_seg2, target_factor=0.9, starting_factor=0.8)
        song2_seg3 = audio_filter.linear_high_filter(song2_seg3, target_factor=1.0, starting_factor=0.9)
    else:
        song2_seg1 = audio_filter.linear_high_filter(song2_seg1, target_factor=0.6, starting_factor=0.3)
        song2_seg2 = audio_filter.linear_high_filter(song2_seg2, target_factor=0.8, starting_factor=0.6)
        song2_seg3 = audio_filter.linear_high_filter(song2_seg3, target_factor=1.0, starting_factor=0.8)

    song2_seg1 = audio_filter.linear_volume_transformation(song2_seg1, target_factor=1.0, starting_factor=0.0)

    # now mix the segments
    song1_transition = song1_seg1 + song1_seg2 + song1_seg3 + song1_seg4
    song2_transition = song2_seg1 + song2_seg2 + song2_seg3 + song2_seg4
    song_overlay = song1_transition.overlay(song2_transition)

    mix = song1_seg0 + song_overlay + song2_seg5 + song2_seg6_1 + song2_seg6 + song2_seg7
    return mix


# assumes that transition is at 40 downbeats
# if transition_scale_factor = 1.0, then transition is 32 downbeats
# for testing it might make sense to set transition_scale_factor = 0.25 (beats instead of downbeats)
def rolling_transition(base_audio: AudioSegment, cand_audio: AudioSegment,
                       cand_audio_without_drum_stem: AudioSegment, remove_drum_cand_beginning: bool,
                       remove_drum_cand_ending: bool, remove_drum_cand_ending_beats: int,
                       downbeat_duration_ms: float, transition_scale_factor: float = 1.0):
    base_audio_seg1_start = int(downbeat_duration_ms) * 40 * transition_scale_factor
    base_audio_seg2_start = int(downbeat_duration_ms) * 25 * transition_scale_factor
    base_audio_seg3_start = int(downbeat_duration_ms) * 24 * transition_scale_factor
    base_audio_seg4_start = int(downbeat_duration_ms) * 8 * transition_scale_factor

    song1_seg0 = base_audio[:-base_audio_seg1_start]
    song1_seg1 = base_audio[-base_audio_seg1_start:-base_audio_seg2_start]
    song1_seg2 = base_audio[-base_audio_seg2_start:-base_audio_seg3_start]
    song1_seg3 = base_audio[-base_audio_seg3_start:-base_audio_seg4_start]
    song1_seg4 = base_audio[-base_audio_seg4_start:]

    # apply filters to song1_seg2
    # song1_seg2 = audio_filter.linear_low_filter(song1_seg2, target_factor=0.0, starting_factor=0.3)
    # song1_seg2 = audio_filter.linear_high_filter(song1_seg2, target_factor=0.0, starting_factor=0.3)
    # song1_seg1 = audio_filter.linear_volume_transformation(song1_seg1, target_factor=1.0, starting_factor=1.0)
    #
    song1_seg2 = audio_filter.linear_low_filter(song1_seg2, target_factor=0.3, starting_factor=1.0)
    song1_seg2 = audio_filter.linear_high_filter(song1_seg2, target_factor=0.4, starting_factor=1.0)
    song1_seg2 = audio_filter.linear_volume_transformation(song1_seg2, target_factor=0.8, starting_factor=1.0)

    song1_seg3 = audio_filter.linear_low_filter(song1_seg3, target_factor=0.2, starting_factor=0.3)
    song1_seg3 = audio_filter.linear_high_filter(song1_seg3, target_factor=0.2, starting_factor=0.4)
    song1_seg3 = audio_filter.linear_volume_transformation(song1_seg3, target_factor=0.8, starting_factor=0.8)

    song1_seg4 = audio_filter.linear_low_filter(song1_seg4, target_factor=0.2, starting_factor=0.2)
    song1_seg4 = audio_filter.linear_high_filter(song1_seg4, target_factor=0.2, starting_factor=0.2)
    song1_seg4 = audio_filter.linear_volume_transformation(song1_seg4, target_factor=0.0, starting_factor=0.8)

    cand_audio_seg2_start = int(downbeat_duration_ms) * 8 * transition_scale_factor
    cand_audio_seg3_start = int(downbeat_duration_ms) * 15 * transition_scale_factor
    cand_audio_seg4_start = int(downbeat_duration_ms) * 16 * transition_scale_factor
    cand_audio_seg5_start = int(downbeat_duration_ms) * 40 * transition_scale_factor

    cand_audio_seg6_start_ending = int(downbeat_duration_ms) * remove_drum_cand_ending_beats * transition_scale_factor

    song2_seg1 = cand_audio[:cand_audio_seg2_start] if not remove_drum_cand_beginning else cand_audio_without_drum_stem[
                                                                                           :cand_audio_seg2_start]
    song2_seg2 = cand_audio[
                 cand_audio_seg2_start:cand_audio_seg3_start] if not remove_drum_cand_beginning else cand_audio_without_drum_stem[
                                                                                                     cand_audio_seg2_start:cand_audio_seg3_start]
    song2_seg3 = cand_audio[
                 cand_audio_seg3_start:cand_audio_seg4_start] if not remove_drum_cand_beginning else cand_audio_without_drum_stem[
                                                                                                     cand_audio_seg3_start:cand_audio_seg4_start]
    song2_seg4 = cand_audio[cand_audio_seg4_start:cand_audio_seg5_start]
    song2_seg5 = cand_audio[cand_audio_seg5_start:-cand_audio_seg6_start_ending]
    song2_seg6 = cand_audio[
                 -cand_audio_seg6_start_ending:] if not remove_drum_cand_ending else cand_audio_without_drum_stem[
                                                                                     -cand_audio_seg6_start_ending:]

    # apply filters to song2_seg1
    song2_seg1 = audio_filter.linear_low_filter(song2_seg1, target_factor=0.2, starting_factor=0.2)
    song2_seg1 = audio_filter.linear_high_filter(song2_seg1, target_factor=0.2, starting_factor=0.2)
    song2_seg1 = audio_filter.linear_volume_transformation(song2_seg1, target_factor=0.8, starting_factor=0.0)
    #
    song2_seg2 = audio_filter.linear_low_filter(song2_seg2, target_factor=0.3, starting_factor=0.2)
    song2_seg2 = audio_filter.linear_high_filter(song2_seg2, target_factor=0.75, starting_factor=0.2)
    song2_seg2 = audio_filter.linear_volume_transformation(song2_seg2, target_factor=0.8, starting_factor=0.8)
    #
    song2_seg3 = audio_filter.linear_low_filter(song2_seg3, target_factor=1.0, starting_factor=0.3)
    song2_seg3 = audio_filter.linear_high_filter(song2_seg3, target_factor=1.0, starting_factor=0.75)
    song2_seg3 = audio_filter.linear_volume_transformation(song2_seg3, target_factor=1.0, starting_factor=0.8)
    #
    # song2_seg4 = audio_filter.linear_volume_transformation(song2_seg4, target_factor=1.0, starting_factor=0.8)

    # now mix the segments
    song1_transition = song1_seg1 + song1_seg2 + song1_seg3 + song1_seg4
    song2_transition = song2_seg1 + song2_seg2 + song2_seg3 + song2_seg4
    song_overlay = song1_transition.overlay(song2_transition)

    mix = song1_seg0 + song_overlay + song2_seg5 + song2_seg6
    return mix


def mix_songs(session_id: str, song_schedule: list[SongScheduleItem], is_drum_removal_enabled: bool,
              is_vocal_removal_enabled) -> AudioSegment:
    logger.info("Starting song mixing for session {}".format(session_id))
    mixed_song = None
    previous_song_end_vocal_interval = []
    for idx, schedule in enumerate(song_schedule):
        logger.debug("Mixing song {} of {}".format(idx + 1, len(song_schedule)))

        audio_path = schedule.audio_path
        start_time = schedule.start_time
        end_time = schedule.end_time
        original_tempo = schedule.original_tempo
        target_tempo = schedule.target_tempo
        key_shift = schedule.key_shift
        timbral_contr = schedule.timbral_contr
        rhytmic_contr = schedule.r_contr
        harmonic_contr = schedule.h_contr
        current_vocal_intervals = schedule.vocal_intervals_time_stretched

        high_rhythmic_similarity = rhytmic_contr > 0.95
        high_timbral_similarity = timbral_contr > 0.95
        high_harmonic_similarity = harmonic_contr > 0.9

        should_remove_drum_stem = False
        should_remove_drum_next_song = False
        audio_segment_without_drum_stem = None
        audio_segment_only_drum_stem = None

        remove_end_vocals = False
        audio_segment_without_vocal_stem = None
        audio_segment_only_vocal_stem = None

        audio_segment_without_drum_and_vocal_stem = None

        rhythmic_contr_threshold = 0.95
        if mixed_song is not None and rhytmic_contr < rhythmic_contr_threshold:
            should_remove_drum_stem = is_drum_removal_enabled

        stretch_playback_rate = utilities.calculate_playback_rate(target_tempo=target_tempo,
                                                                  current_tempo=original_tempo)
        stretch_factor = 1 / stretch_playback_rate

        beat_duration = utilities.get_duration_by_beats(1, tempo=target_tempo)
        downbeat_duration = beat_duration * 4
        transition_scale_factor = 0.5  # use beats instead of downbeats for transition timing

        beat_crossover_before_transition_point_seconds = downbeat_duration * 16 * transition_scale_factor
        beat_crossover_after_transition_point_seconds = downbeat_duration * 16 * transition_scale_factor

        # assume transition after 24 (down-)beats (rolling transition)
        rolling_transition_base_song_crossover = downbeat_duration * 24 * transition_scale_factor

        rolling_transition_timbre_threshold = 0.98
        rolling_transition_next_song = False
        # transition = rolling_transition # we disabled the rolling transition for now
        if len(song_schedule) > idx + 1:
            timbral_contr_next_song = song_schedule[idx + 1].timbral_contr
            rhythmic_contr_next_song = song_schedule[idx + 1].r_contr
            start_time_next_song = song_schedule[idx + 1].start_time
            vocal_intervals_next_song = song_schedule[idx + 1].vocal_intervals_time_stretched
            original_tempo_next_song = song_schedule[idx + 1].original_tempo
            target_tempo_next_song = song_schedule[idx + 1].target_tempo
            stretch_playback_rate_next_song = utilities.calculate_playback_rate(target_tempo=target_tempo_next_song,
                                                                                current_tempo=original_tempo_next_song)
            stretch_factor_next_song = 1 / stretch_playback_rate_next_song
            if timbral_contr_next_song >= rolling_transition_timbre_threshold:
                logger.debug(
                    f"Timbre contribution of next song is high: {timbral_contr_next_song}. Adjusting song boundaries for rolling transition.")
                # rolling_transition_next_song = True  # we disabled the rolling transition for now
                rolling_transition_next_song = False
            if rhythmic_contr_next_song < rhythmic_contr_threshold:
                should_remove_drum_next_song = is_drum_removal_enabled
                logger.debug(
                    f"Rhythmic contribution of next song is low: {rhythmic_contr_next_song}. Removing drum stem of next song to prepare for transition without drum stem.")

            current_song_end_time_stretched_ms = int(end_time * stretch_factor * 1000)
            next_song_start_time_stretched_ms = int(start_time_next_song * stretch_factor_next_song * 1000)
            remove_end_vocals = should_remove_end_vocals(current_vocal_intervals=current_vocal_intervals,
                                                         current_song_end_time_stretched=current_song_end_time_stretched_ms,
                                                         overlay_duration_before_transition_ms=int(
                                                             beat_crossover_before_transition_point_seconds * 1000),
                                                         overlay_duration_after_transition_ms=int(
                                                             beat_crossover_after_transition_point_seconds * 1000),
                                                         next_song_vocal_intervals=vocal_intervals_next_song,
                                                         next_song_start_time_stretched=next_song_start_time_stretched_ms) and is_vocal_removal_enabled
            logger.debug(f"Should remove vocals: {remove_end_vocals}")

        # rolling_transition_current_song = False # we disabled the rolling transition for now
        transition = main_transition
        if timbral_contr >= rolling_transition_timbre_threshold:
            logger.debug(f"Timbre contribution is high: {timbral_contr}. Using rolling transition.")
            # transition = rolling_transition # we disabled the rolling transition for now
            transition = main_transition

        audio_segment, gain = load_and_normalize_audiosegment(audio_path)
        if should_remove_drum_stem or should_remove_drum_next_song:
            # remove drum stem
            mss.mix_song_without_stem(session_id, audio_path, [mss.StemType.DRUMS])
            audio_segment_without_drum_stem, _ = load_and_normalize_audiosegment(
                mss.get_song_without_stem_filepath(session_id, audio_path, [mss.StemType.DRUMS]), manual_gain=gain)
            audio_segment_only_drum_stem, _ = load_and_normalize_audiosegment(
                mss.get_song_stem_filepath(session_id, audio_path, mss.StemType.DRUMS), manual_gain=gain)
        if remove_end_vocals:
            mss.mix_song_without_stem(session_id, audio_path, [mss.StemType.VOCALS])
            audio_segment_without_vocal_stem, _ = load_and_normalize_audiosegment(
                mss.get_song_without_stem_filepath(session_id, audio_path, [mss.StemType.VOCALS]), manual_gain=gain)
            audio_segment_only_vocal_stem, _ = load_and_normalize_audiosegment(
                mss.get_song_stem_filepath(session_id, audio_path, mss.StemType.VOCALS), manual_gain=gain)
        if should_remove_drum_next_song and remove_end_vocals:
            mss.mix_song_without_stem(session_id, audio_path, [mss.StemType.DRUMS, mss.StemType.VOCALS])
            audio_segment_without_drum_and_vocal_stem, _ = load_and_normalize_audiosegment(
                mss.get_song_without_stem_filepath(session_id, audio_path, [mss.StemType.DRUMS, mss.StemType.VOCALS]),
                manual_gain=gain)

        if mixed_song is not None and start_time - beat_crossover_before_transition_point_seconds >= 0:
            start_time -= beat_crossover_before_transition_point_seconds
        elif mixed_song is not None:
            logger.warning("Can not adjust start time to fit transition. Start time is too early.")
            raise ValueError("Can not adjust end time to fit transition. Start time is too early.")

        song_length_seconds = stretch_factor * (len(audio_segment) / 1000.0)
        if idx < len(song_schedule) - 1:
            if rolling_transition_next_song and (
                    end_time + rolling_transition_base_song_crossover < song_length_seconds):
                end_time += rolling_transition_base_song_crossover
            elif (not rolling_transition_next_song) and (
                    end_time + beat_crossover_after_transition_point_seconds < song_length_seconds):
                end_time += beat_crossover_after_transition_point_seconds
            else:
                logger.warning(
                    f"Can not adjust end time to fit transition. End time is too late: end_time={end_time}, song_length_seconds={song_length_seconds}")
                raise ValueError("Can not adjust end time to fit transition. End time is too late.")

        audio_segment = extract_and_prepare_audio_segment(audio_segment, start_time, end_time, stretch_playback_rate,
                                                          key_shift)
        audio_segment_without_drum_stem = extract_and_prepare_audio_segment(audio_segment_without_drum_stem, start_time,
                                                                            end_time, stretch_playback_rate,
                                                                            key_shift) if should_remove_drum_stem or should_remove_drum_next_song else audio_segment_without_drum_stem
        audio_segment_only_drum_stem = extract_and_prepare_audio_segment(audio_segment_only_drum_stem, start_time,
                                                                         end_time, stretch_playback_rate,
                                                                         key_shift) if should_remove_drum_stem or should_remove_drum_next_song else audio_segment_only_drum_stem
        audio_segment_without_vocal_stem = extract_and_prepare_audio_segment(audio_segment_without_vocal_stem,
                                                                             start_time, end_time,
                                                                             stretch_playback_rate,
                                                                             key_shift) if remove_end_vocals else audio_segment_without_vocal_stem
        audio_segment_only_vocal_stem = extract_and_prepare_audio_segment(audio_segment_only_vocal_stem, start_time,
                                                                          end_time, stretch_playback_rate,
                                                                          key_shift) if remove_end_vocals else audio_segment_only_vocal_stem
        audio_segment_without_drum_and_vocal_stem = extract_and_prepare_audio_segment(
            audio_segment_without_drum_and_vocal_stem, start_time, end_time, stretch_playback_rate,
            key_shift) if should_remove_drum_next_song and remove_end_vocals else audio_segment_without_drum_and_vocal_stem

        if idx == 0 or mixed_song is None:
            mixed_song = audio_segment
        else:
            if rolling_transition_next_song:
                remove_drum_cand_ending_beats = 24  # todo: some beats should be added, similar to normal transition, but since rolling transition is disbaled, not relevant
            else:
                remove_drum_cand_ending_beats = 24  # 16.25
            mixed_song = transition(base_audio=mixed_song,
                                    cand_audio=audio_segment,
                                    cand_audio_without_drum_stem=audio_segment_without_drum_stem,
                                    cand_audio_only_drum_stem=audio_segment_only_drum_stem,
                                    cand_audio_without_vocal_stem=audio_segment_without_vocal_stem,
                                    cand_audio_only_vocal_stem=audio_segment_only_vocal_stem,
                                    cand_audio_without_vocal_and_drum_stem=audio_segment_without_drum_and_vocal_stem,
                                    remove_drum_cand_beginning=should_remove_drum_stem,
                                    remove_drum_cand_ending=should_remove_drum_next_song,
                                    remove_drum_cand_ending_beats=remove_drum_cand_ending_beats,
                                    remove_end_vocals=remove_end_vocals,
                                    downbeat_duration_ms=int(downbeat_duration * 1000),
                                    high_rhythmic_similarity=high_rhythmic_similarity,
                                    high_timbral_similarity=high_timbral_similarity,
                                    high_harmonic_similarity=high_harmonic_similarity,
                                    transition_scale_factor=transition_scale_factor)

    return mixed_song
