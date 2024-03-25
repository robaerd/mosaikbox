class NotFoundException(Exception):
    def __init__(self, object_type: str, object_id: str):
        """
        Exception for when an object is not found.
        :param object_type: The type of object that was not found. E.g. "song".
        :param object_id: The ID of the object that was not found. E.g. "123".
        """
        self.object_type = object_type
        self.object_id = object_id
        self.message = f"{self.object_type} with ID {self.object_id} not found"
        super().__init__(self.message)


class ValidationException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class NoCandidateError(Exception):
    """Exception raised when there is no candidate.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="No candidate found"):
        self.message = message
        super().__init__(self.message)


class NoLyricsFoundError(Exception):
    """"Exception raised when no lyrics are found.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="No lyrics found"):
        self.message = message
        super().__init__(self.message)


class TaskCancelledException(Exception):
    def __init__(self, message="Task was cancelled"):
        self.message = message
        super().__init__(self.message)
