from timeit import default_timer

class Timer:
    def __init__(self):
        self.start_time = None
        self.accumulator = 0
        self.count = 0

    def start(self):
        self.start_time = default_timer()

    def stop(self):
        self.end_time = default_timer()
        self.accumulator += self.end_time - self.start_time
        self.count += 1
    
    def avg_elapsed(self):
        return self.accumulator / self.count
