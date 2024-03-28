# encoding=utf-8
import time

import torch
    
    
class CostTimer:
    
    t: float = 0
    
    def __enter__(self):
        # torch.cuda.synchronize()
        # CostTimer.t = time.time()
        # self.start = torch.cuda.Event(enable_timing=True)
        # self.end = torch.cuda.Event(enable_timing=True)
        # self.start.record(stream=torch.cuda.current_stream())
        self.timer = Timer("timer")
        self.timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # torch.cuda.synchronize()
        # CostTimer.t = time.time() - CostTimer.t
        # self.end.record(stream=torch.cuda.current_stream())
        # CostTimer.t = self.start.elapsed_time(self.end)
        # self.end.synchronize()
        CostTimer.t = self.timer.elapsed()
        return

    @classmethod
    def get_time(self):
        return CostTimer.t


class Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, "timer has already been started"
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, "timer is not started"
        torch.cuda.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True) -> float:
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self, logger):
        self.logger = logger
        self.timers = {}

    def __call__(self, name) -> Timer:
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def reset(self):
        for _, t in self.timers:
            t.reset()

    def write(self, names, iteration, normalizer=1.0, reset=False, metrics_group='timers'):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # pollutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            self.logger.log_metrics(
                {name: value}, step=iteration, metrics_group=metrics_group)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += " | {}: {:.2f}".format(name, elapsed_time)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(string, flush=True)
        else:
            print(string, flush=True)

def local_rank():
    """Local rank of process"""
    local_rank = os.environ.get("LOCAL_RANK")

    if local_rank is None:
        local_rank = os.environ.get("SLURM_LOCALID")

    if local_rank is None:
        print(
            "utils.local_rank() environment variable LOCAL_RANK not set, defaulting to 0",
            flush=True,
        )
        local_rank = 0
    return int(local_rank)

