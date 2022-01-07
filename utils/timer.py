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
        self._elapsed_time = convert_timedelta(now-self._start)
        self._runtime = timedelta(seconds=now-self._prev)
        self._remaining_time = convert_timedelta((self._runtime*(self._total_steps-step)).total_seconds())
        self._prev = now

    @property
    def elapsed_time(self) -> str:
        return self._elapsed_time.strftime('%H:%M:%S')

    @property
    def runtime(self) -> float:
        return self._runtime.total_seconds()

    @property
    def remaining_time(self) -> str:
        return self._remaining_time.strftime('%H:%M:%S')


def convert_timedelta(duration : float) -> time:
    delta = timedelta(seconds=duration)
    days, seconds = delta.days, delta.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    t = time(hour=hours, minute=minutes, second=seconds)
    return t