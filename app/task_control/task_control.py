class TaskControl:
    def __init__(self):
        self.cancelled = False


class TaskControlManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskControlManager, cls).__new__(cls)
            cls._instance.task_controls = {}
        return cls._instance

    def register_task(self, session_id):
        task_control = TaskControl()
        self.task_controls[session_id] = task_control

    def cancel_task(self, session_id):
        if session_id in self.task_controls:
            self.task_controls[session_id].cancelled = True

    def remove_task(self, session_id):
        if session_id in self.task_controls:
            del self.task_controls[session_id]

    def is_task_cancelled(self, session_id):
        if session_id in self.task_controls:
            return self.task_controls[session_id].cancelled
        return True
