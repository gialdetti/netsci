import time


class Timer:
    def __init__(self, logger=None, desc="Execution"):
        self.logger = logger
        self.desc = desc

    def __enter__(self):
        self.start = time.time_ns()
        return self

    def __exit__(self, *args):
        self.end = time.time_ns()
        self.interval = self.end - self.start
        self.interval_ms = self.interval // 1_000_000

        if self.logger is not None:
            self.logger(f"{self.desc} took {self.interval_ms/1000:.03f} sec.")


if __name__ == "__main__":
    with Timer(logger=print) as t:
        time.sleep(3.14)

    t.interval_ms
