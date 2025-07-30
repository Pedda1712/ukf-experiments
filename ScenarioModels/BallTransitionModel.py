import numpy as np

from Filter import TransitionModel
from World import BallWorldInfo, Ball, BallTransition

class BallTransitionModel(TransitionModel):
    Q: np.ndarray
    world: BallWorldInfo
    _transition: BallTransition

    def __init__(self, world: BallWorldInfo, velocity_variance: np.ndarray):
        self.Q = np.diag(velocity_variance)
        self.world = world
        self._transition = BallTransition(world)

    def get_noise(self) -> np.ndarray:
        return self.Q

    def transition(self, states: np.ndarray, noise: np.ndarray, delta: float) -> np.ndarray:
        balls = [Ball(s[0:3], s[3:6] + q) for (s, q) in zip(states, noise)]
        balls = self._transition.transition(balls, delta)
        return np.array([np.append(np.array(b.pos), np.array(b.velocity)) for b in balls])
        
