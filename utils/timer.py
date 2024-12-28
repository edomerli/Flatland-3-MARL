from timeit import default_timer

class Timer:
    def __init__(self):
        """Simple timer class.
        """
        self.start_time = None
        self.accumulator = 0

    def start(self):
        """Start the timer.
        """
        self.start_time = default_timer()

    def stop(self):
        """Stop the timer and accumulate the elapsed time.
        """
        self.end_time = default_timer()
        self.accumulator += self.end_time - self.start_time
    
    def cumulative_elapsed(self):
        """Get the accumulated time.
        """
        return self.accumulator
