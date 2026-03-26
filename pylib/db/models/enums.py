from enum import Enum


class EnumStatus(str, Enum):
    WAITING = "Waiting"
    RUNNING = "Running"
    DONE = "Done"
    ERROR = "Error"
