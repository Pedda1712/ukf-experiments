class Ball():
    def __init__(self, pos: list[float, float, float], velocity: list[float, float, float]):
        self.pos = pos
        self.velocity = velocity

    def propagate(self, delta: float):
        return Ball([p + v * delta for (p, v) in zip(self.pos, self.velocity)], self.velocity)
