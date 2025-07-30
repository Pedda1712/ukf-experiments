import numpy as np

class TransitionModel:
    def __init__(self):
        pass
    def get_noise(self) -> np.ndarray:
        pass
    def transition(self, states: np.ndarray, noise: np.ndarray, delta: float) -> np.ndarray:
        pass
