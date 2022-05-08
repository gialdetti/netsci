import time

class Timer:
    def __init__(self, logger=None, desc='Execution'):
        self.logger = logger
        self.desc = desc

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = (self.end - self.start)

        if self.logger is not None:
            self.logger(f'{self.desc} took {self.interval:.03f} sec.')


if __name__ == "__main__":
    with Timer(logger=print):
        time.sleep(3.14)