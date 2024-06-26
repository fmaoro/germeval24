from enum import Enum


class Task(str, Enum):
    GERMEVAL24 = "GERMEVAL24"
    GERMEVAL24_MUTLICLASS = "GERMEVAL24_MULTICLASS"

    def __str__(self) -> str:
        return self.value
