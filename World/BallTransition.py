from .State import Ball
from .Info import BallWorldInfo

class BallTransition():
    def __init__(self, world: BallWorldInfo):
        self.world = world

    def _transition_one(self, s: Ball, delta: float) -> Ball:
        result = s.propagate(delta)
        for (i, (position, limit)) in enumerate(zip(result.pos, self.world.dims)):
            if position + self.world.ball_radius > limit/2:
                result.pos[i] = limit/2 - self.world.ball_radius
                result.velocity[i] = -result.velocity[i] * self.world.bounce_discount
            elif position - self.world.ball_radius < -limit/2:
                result.pos[i] = -limit/2 + self.world.ball_radius
                result.velocity[i] = -result.velocity[i] * self.world.bounce_discount
                
            if position + self.world.ball_radius - limit/2 >= 1e-4:
                result.velocity[i] = result.velocity[i] * self.world.ground_discount
            if position - self.world.ball_radius + limit/2 <= 1e-4:
                result.velocity[i] = result.velocity[i] * self.world.ground_discount
                
            result.velocity[i] = result.velocity[i] * self.world.air_discount
        result.velocity[1] -= self.world.gravity * delta
        
        return result
        
    def transition(self, states: list[Ball], delta: float) -> list[Ball]:
        return [self._transition_one(s, delta) for s in states]

