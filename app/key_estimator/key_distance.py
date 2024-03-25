from .key_type import KeyType, get_key_type

minor_camelot_scale = ['5A', '12A', '7A', '2A', '9A', '4A', '11A', '6A', '1A', '8A', '3A', '10A']
major_camelot_scale = ['8B', '3B', '10B', '5B', '12B', '7B', '2B', '9B', '4B', '11B', '6B', '1B']


def key_distance(key1, key2):
    if get_key_type(key1) == KeyType.MINOR:
        camelot_scale = minor_camelot_scale
    elif get_key_type(key1) == KeyType.MAJOR:
        camelot_scale = major_camelot_scale
    else:
        raise ValueError('Both keys must be either a major or minor key')

    key1_idx = camelot_scale.index(key1)
    key2_idx = camelot_scale.index(key2)
    if key2_idx < key1_idx:
        diff = key1_idx - key2_idx
        if diff > 6:
            return diff - 12
        else:
            return diff
    else:
        diff = key2_idx - key1_idx
        if diff > 6:
            return 12 - diff
        else:
            return -diff


def key_distance_harmonic(key1, key2) -> (int, bool):
    def get_scale_with_custom_scale_num(key, scale_num) -> str:
        key_scale_char = key[-1]
        return str(scale_num) + key_scale_char

    if get_key_type(key1) != get_key_type(key2):
        return key_distance_different_scale(key1, key2), True

    key2_candidates = [key2]
    key2_num = get_key_without_scale(key2)
    if key2_num == 12:
        key2_candidates.append(get_scale_with_custom_scale_num(key2, 1))
        key2_candidates.append(get_scale_with_custom_scale_num(key2, 11))
    elif key2_num == 1:
        key2_candidates.append(get_scale_with_custom_scale_num(key2, 2))
        key2_candidates.append(get_scale_with_custom_scale_num(key2, 12))
    else:
        key2_candidates.append(get_scale_with_custom_scale_num(key2, key2_num + 1))
        key2_candidates.append(get_scale_with_custom_scale_num(key2, key2_num - 1))

    best_distance = None
    best_key = None
    for key2_candidate in key2_candidates:
        distance = key_distance(key1, key2_candidate)
        if best_distance is None or abs(distance) < abs(best_distance):
            best_distance = distance
            best_key = key2_candidate
    return best_distance, best_key != key2


# based on the harmonic mixing wheel
def is_key_harmonically_compatible(key1, key2) -> bool:
    key1_num = get_key_without_scale(key1)
    key2_num = get_key_without_scale(key2)
    if get_key_without_scale(key1) != get_key_without_scale(key2):
        return key1_num == key2_num
    else:
        diff = abs(key1_num - key2_num)
        return diff <= 1


def get_key_without_scale(key) -> int:
    key_without_scale = key[:-1]
    key_without_scale_int = int(key_without_scale)
    return key_without_scale_int


def key_distance_different_scale(key1, key2):
    assert get_key_type(key1) != get_key_type(key2)
    key1_key2_scale = key1[:-1] + key2[-1]
    return key_distance(key1_key2_scale, key2)


def get_key_by_relative_semitone(key, semitones):
    if get_key_type(key) == KeyType.MINOR:
        camelot_scale = minor_camelot_scale
    elif get_key_type(key) == KeyType.MAJOR:
        camelot_scale = major_camelot_scale
    else:
        raise ValueError('Key must be either a major or minor key')

    key_idx = camelot_scale.index(key)
    shift = key_idx + semitones
    if shift < 0:
        key_shift_idx = 12 + shift
    elif shift >= 12:
        key_shift_idx = shift - 12
    else:
        key_shift_idx = shift
    return camelot_scale[key_shift_idx]
