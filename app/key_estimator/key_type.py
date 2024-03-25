from enum import Enum


class KeyType(Enum):
    MAJOR = 1
    MINOR = 2


def get_key_type(key) -> KeyType:
    if key[-1] == 'A':
        return KeyType.MINOR
    elif key[-1] == 'B':
        return KeyType.MAJOR
    else:
        raise ValueError('Key is not a valid key')
