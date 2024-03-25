import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_timbre_similarity(timbre_vector_1: np.ndarray, timbre_vector_2: np.ndarray):
    return cosine_similarity(timbre_vector_1.reshape(1, -1), timbre_vector_2.reshape(1, -1))[0][0]
