import logging

import numpy as np

logger = logging.getLogger(__name__)


class SongCache:
    song_limit: int
    song_cache: dict[str, np.ndarray]

    def __init__(self, song_limit: int):
        self.song_limit = song_limit
        self.song_cache = dict()

    def remove_song(self, song_id: str):
        logger.debug(f"Removing song from cache: {song_id}")
        self.song_cache.pop(song_id)

    def add_song(self, song_id: str, song: np.ndarray):
        if len(self.song_cache) < self.song_limit:
            logger.debug(f"Adding song to cache: {song_id}")
            # add to dict
            self.song_cache[song_id] = song
        else:
            logger.debug(f"Cache full, not adding song: {song_id}")

    def get_song(self, song_id: str) -> np.ndarray:
        logger.debug(f"Getting song from cache: {song_id}")
        return self.song_cache[song_id]

    def contains(self, song_id: str) -> bool:
        logger.debug(f"Checking if song is in cache: {song_id}")
        return song_id in self.song_cache
