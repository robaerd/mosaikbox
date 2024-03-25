from ..services import utilities
from ..config import config


def filter_songs_too_short(song_list: list[str]) -> list[str]:
    filtered_songs = []
    for song in song_list:
        if is_song_long_enough(song):
            filtered_songs.append(song)
    return filtered_songs


def is_song_long_enough(song_path: str) -> bool:
    song_duration = utilities.get_audio_length_from_file(song_path)
    return song_duration >= config.SEGMENT_MIN_LENGTH + config.SEGMENT_MIN_TRANSITION_OFFSET * 2
