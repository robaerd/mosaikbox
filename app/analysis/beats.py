import logging

from BeatNet.BeatNet import BeatNet
import numpy as np
from scipy.optimize import dual_annealing
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def determine_beats_and_tempo(audio_path: str, audio_length: float) -> (np.ndarray, int):
    beats_detected = detect_beats(audio_path)
    downbeats_detected, tempo_estimated = detect_downbeats_and_tempo(audio_path=audio_path, beats=beats_detected)

    beat_grid, tempo = create_beat_grid(audio_length, tempo_estimated, beats_detected)
    return beat_grid, tempo


def detect_beats(audio_path: str) -> np.ndarray:
    def estimate_beats(audio_path: str, model: int = 1) -> np.ndarray:
        # TODO: change mps to automatic device inference
        estimator = BeatNet(model, mode='offline', inference_model='DBN', plot=[], thread=False, device='mps')
        return estimator.process(audio_path)

    beat_times = estimate_beats(audio_path, 1)

    return beat_times


def detect_downbeats_and_tempo(audio_path: str = None, beats: np.ndarray = None) -> (
        np.ndarray, int):
    if audio_path is None and beats is None:
        raise RuntimeError('Either audio or beats must be provided!')
    elif audio_path is not None and beats is None:
        beats = detect_beats(audio_path)

    downbeats = beats[beats[:, 1] == 1.0]
    beat_types = np.unique(beats[:, 1])

    tempo = calculate_tempo_most_occurring_downbeat_interval(downbeats, beat_types.size)
    return downbeats, tempo


def calculate_tempo_most_occurring_downbeat_interval(downbeat_times: np.ndarray, amount_beat_types: int) -> float:
    # first we try to determine the tempo by calculating the most occuring interval between downbeats
    beat_intervals = np.diff(downbeat_times[:, 0])
    # we round the intervals to 2 decimals to make sure that we can find the most occuring interval
    beat_intervals_rounded = np.round(beat_intervals, 2)
    # multiply by 100 and convert to int, so we can use np.bincount
    beat_intervals_rounded = (beat_intervals_rounded * 100).astype(int)
    # find the most occuring interval and normalize again by dividing by 100
    # if there are multiple most occuring intervals, we will use the last one. Meaning, we prefer slower tempos to higher tempos
    most_occuring_interval_bins = np.bincount(beat_intervals_rounded)
    most_occuring_interval_indices = np.argwhere(most_occuring_interval_bins == np.amax(most_occuring_interval_bins))
    if most_occuring_interval_indices.size > 1:
        logger.debug("More than one most occuring interval found, using the last one")
        most_occuring_interval = most_occuring_interval_indices[-1][0] / 100
    elif most_occuring_interval_indices.size == 1:
        most_occuring_interval = most_occuring_interval_indices[0][0] / 100
    else:
        most_occuring_interval = 0

    if most_occuring_interval == 0:
        # if the most occuring interval is 0, we will use the median interval instead
        logger.debug("Most occuring interval is 0, using median interval instead")

        most_occuring_interval = np.median(beat_intervals_rounded)

    bpm = (60 * amount_beat_types) / most_occuring_interval

    return bpm


def get_subset_of_beats_by_timing(beats: np.ndarray, start_time, end_time) -> np.ndarray:
    # necessary because np.linspace has floating point precision issues at around 1e-15
    atol = 1e-10  # to have a bigger tolerance than the error
    rtol = 0.0  # we don't want relative tolerance as we know that the error is in the area of 1e-15
    is_close_start_time = np.isclose(beats[:, 0], start_time, rtol=rtol, atol=atol)
    is_close_end_time = np.isclose(beats[:, 0], end_time, rtol=rtol, atol=atol)
    close_or_ge_start_time = np.logical_or(beats[:, 0] >= start_time, is_close_start_time)
    close_or_ge_end_time = np.logical_or(beats[:, 0] <= end_time, is_close_end_time)
    mask = np.logical_and(close_or_ge_start_time, close_or_ge_end_time)

    return beats[mask]


def closest_mean_diff(X, Y):
    # Calculate the mean of the minimum absolute differences between each estimated beat grid position $g_i$ and all
    # detected beat positions $t_j$.
    differences = np.array([np.abs(X - y).min() for y in Y])
    return np.mean(differences)


def estimate_beat_amount(audio_length: float, tempo: float):
    return int(np.floor(audio_length * (tempo / 60)))


def calculate_evenly_distributed_beats(audio_length: float, tempo: float, first_downbeat_offset: float):
    sec_beat = (60 / tempo)
    # since we want to include the last beat we add sec_beat to the end
    beats = np.arange(first_downbeat_offset, audio_length + sec_beat, sec_beat)

    if beats[-1] > audio_length:
        # if the last beat is larger than the audio length, we remove it
        beats = beats[:-1]  # happens probably due to rounding errors

    beat_types = np.array([1, 2, 3, 4])  # beat_types to assign
    beat_types_repeated = np.tile(beat_types, len(beats) // len(beat_types) + 1)  # Assign the beat_types to the beats
    beat_types_trimmed = beat_types_repeated[:len(beats)]  # Trim the repeated beat_types to the length of the beats

    beats_with_beat_types = np.vstack((beats, beat_types_trimmed)).T # Stack the beats with beat_types_trimmed to create a 2D beat-position array
    return beats_with_beat_types


def custom_round_tempo(tempo: float) -> float:
    # Find the closest multiple of 0.1
    round_tenth = round(tempo * 10) / 10
    # Find the closest multiple of 0.25
    round_quarter = round(tempo * 4) / 4

    # Compare the absolute differences
    if abs(tempo - round_tenth) <= abs(tempo - round_quarter):
        return round_tenth
    else:
        return round_quarter


def create_beat_grid(audio_length: float, tempo_estimated: float, downbeats: np.ndarray) -> tuple[np.ndarray, float]:
    def objective(x):
        offset, tempo = x
        estimated_array = calculate_evenly_distributed_beats(audio_length, tempo, offset)[:, 0]
        return closest_mean_diff(estimated_array, original_array)

    def objective_offset_only(offset, tempo):
        estimated_array = calculate_evenly_distributed_beats(audio_length, tempo, offset)[:, 0]
        return closest_mean_diff(estimated_array, original_array)

    def run_optimizer(objective, x0, bounds):
        minimizer_kwargs_restricted = {"method": "L-BFGS-B", "bounds": bounds}
        res = dual_annealing(objective, bounds, minimizer_kwargs=minimizer_kwargs_restricted, maxiter=1400,
                             initial_temp=8200, x0=x0)
        return res

    # bootstrap
    original_array = downbeats[:, 0]
    tempo_bounds = (60, 190)
    # adapt tempo if half of the tempo was estimated
    if tempo_estimated <= (tempo_bounds[1] / 2):
        logger.debug("Estimated tempo is less than the half of the max tempo bound, so we double the tempo")
        tempo_estimated *= 2
        logger.debug("New estimated tempo: {}".format(tempo_estimated))

    tempo_bounds = (60, tempo_estimated * 1.15)

    # estimate the max offset by taking tempo_estimate into account
    beat_length_estimate = 60 / tempo_estimated
    downbeat_length_estimate = beat_length_estimate  # * 4
    max_offset_estimate = downbeat_length_estimate * 1.4  # assume a 40% error in tempo estimation

    # define the bounds
    bounds = [(0, max_offset_estimate), tempo_bounds]
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

    # initial guess
    x0 = np.array([original_array[0], tempo_estimated])
    logger.debug(f"Initial guess: {x0}")
    logger.debug(f"Initial guess objective: {objective(x0)}")

    res = dual_annealing(objective, bounds, minimizer_kwargs=minimizer_kwargs, maxiter=1200, initial_temp=10000, x0=x0)
    res_objective = objective(res.x)
    logger.debug(f"Annealing objective: {res_objective} with x: {res.x}")

    logger.debug("Running optimizer with more restricted bounds")
    # assume a variance of 5 % in the initial tempo estimation
    bounds_restricted = [(0, downbeat_length_estimate * 1.05), (tempo_estimated / 1.05, tempo_estimated * 1.05)]
    res_restricted = run_optimizer(objective, x0, bounds_restricted)
    res_objective_restricted = objective(res_restricted.x)
    logger.debug(f"Restricted optimizer objective: {res_objective_restricted} with x: {res_restricted.x}")
    if res_objective_restricted <= res_objective:
        logger.debug("Restricted bounds yieled better results. Using restricted result")
        res = res_restricted
        res_objective = res_objective_restricted

    # Print the result
    optimized_offset = res.x[0]
    optimized_tempo = res.x[1]

    # check if we could move the offset by downbeat_duration to the left
    downbeat_duration = 60 / optimized_tempo
    offset_bounds = [(0, max_offset_estimate)]
    new_offset_candidate = optimized_offset - downbeat_duration
    if round(new_offset_candidate, 2) >= -0.05:
        logger.debug(f"Setting new offset candidate to 0 as it is too close to 0 ({new_offset_candidate})")
        new_offset_candidate = 0
    while new_offset_candidate >= 0:
        new_res = minimize(objective_offset_only, new_offset_candidate, args=optimized_tempo, method='L-BFGS-B',
                           bounds=offset_bounds)
        new_res_objective = objective_offset_only(new_res.x, optimized_tempo)
        logger.debug(f"New offset candidate {new_res.x} with objective {new_res_objective}")
        if round(new_res_objective, 2) <= round(res_objective, 2):
            logger.debug(f"Found better offset {new_offset_candidate} with objective {new_res_objective}")
            optimized_offset = new_offset_candidate
            res_objective = new_res_objective
        new_offset_candidate -= downbeat_duration
        if round(new_offset_candidate, 2) >= -0.05:
            logger.debug(f"Setting new offset candidate to 0 as it is too close to 0 ({new_offset_candidate})")
            new_offset_candidate = 0

    # now round the tempo and optimize the offset again
    optimized_tempo_rounded = custom_round_tempo(optimized_tempo)
    round_tempo_offset_res = minimize(objective_offset_only, optimized_offset, args=optimized_tempo_rounded,
                                      method='L-BFGS-B', bounds=offset_bounds)
    round_tempo_offset_res_objective = objective_offset_only(round_tempo_offset_res.x, optimized_tempo_rounded)
    current_objective = objective_offset_only(optimized_offset, optimized_tempo_rounded)
    if round_tempo_offset_res_objective < current_objective:
        logger.debug(
            f"Found better offset for rounded tempo {round_tempo_offset_res.x} with objective {round_tempo_offset_res_objective}, old objective was: {current_objective}")
        optimized_offset = round_tempo_offset_res.x

    logger.debug(f"Offset: {optimized_offset}, BPM: {optimized_tempo_rounded}")

    logger.debug(f"BPM rounded: {optimized_tempo_rounded}")
    estimated_beats = calculate_evenly_distributed_beats(audio_length, optimized_tempo_rounded, optimized_offset)
    return estimated_beats, optimized_tempo_rounded
