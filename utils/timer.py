from timeit import default_timer as timer
from datetime import timedelta, time


class Timer:

    def __init__(self, total_steps : int) -> None:
        self._total_steps : int = total_steps
        self._elapsed_time : time = None
        self._runtime : timedelta = None
        self._remaining_time : time = None
        self._start : float = timer()
        self._prev : float = self._start


    def update(self, step : int):
        now = timer()
        self._elapsed_time = timedelta(seconds=now-self._start)
        self._runtime = timedelta(seconds=now-self._prev)
        self._remaining_time = self._runtime*(self._total_steps-step)
        self._prev = now

    @property
    def elapsed_time(self) -> str:
        return strftimedelta(self._elapsed_time)

    @property
    def runtime(self) -> float:
        return self._runtime.total_seconds()

    @property
    def remaining_time(self) -> str:
        return strftimedelta(self._remaining_time)


def strftimedelta(delta):
    days, seconds = delta.days, delta.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    s = ""
    if days > 0:
        s += f"{days}d "
    s += f"{hours}:{minutes}:{seconds}"
    return s