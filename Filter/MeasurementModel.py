import numpy as np

class MeasurementModel:
    def __init__(self):
        pass
    def get_noise(self) -> np.ndarray:
        pass
    def measure(self, states: np.ndarray, noises: np.ndarray) -> np.ndarray:
        pass
