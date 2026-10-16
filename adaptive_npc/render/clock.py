"""Fixed-timestep accumulator: game ticks at `tick_rate` regardless of the render fps."""


class TickClock:
    def __init__(self, tick_rate, max_catchup=4):
        if tick_rate <= 0:
            raise ValueError("tick_rate must be positive")
        self.dt = 1.0 / float(tick_rate)
        self.max_catchup = int(max_catchup)
        self.acc = 0.0

    def advance(self, frame_dt):
        """Add elapsed wall time; return how many game steps are due this frame."""
        self.acc += frame_dt
        n = int(self.acc // self.dt)
        if n > self.max_catchup:  # rendering stalled: drop the backlog instead of fast-forwarding
            n = self.max_catchup
            self.acc = 0.0
        else:
            self.acc -= n * self.dt
        return n

    def reset(self):
        self.acc = 0.0
