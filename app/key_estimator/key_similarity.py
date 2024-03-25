import logging

from .key_distance import key_distance_harmonic

logger = logging.getLogger(__name__)


def calculate_key_similarity(key1, key2) -> float:
    logger.debug(f"Calculating similarity between {key1} and {key2}")
    distance, is_harmonic = key_distance_harmonic(key1, key2)
    distance = abs(distance)
    similarity = 1 if distance == 0 else 1.0 / distance
    if is_harmonic:
        logger.debug("Penalizing as closest key is a harmonic variant of the key")
        similarity *= 0.8
    return similarity
