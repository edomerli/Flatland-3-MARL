from timeit import default_timer

class Timer:
    def __init__(self):
        self.start_time = None
        self.accumulator = 0

    def start(self):
        self.start_time = default_timer()

    def stop(self):
        self.end_time = default_timer()
        self.accumulator += self.end_time - self.start_time
    
    def cumulative_elapsed(self):
        return self.accumulator
