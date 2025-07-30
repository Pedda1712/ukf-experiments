import numpy as np

from Filter import MeasurementModel

"""
Produces observations of the
3D ball position with added noise.
"""

class Ball3DMeasurementModel(MeasurementModel):
    R: np.ndarray
    def __init__(self, observation_variance: np.ndarray):
        self.R = np.diag(observation_variance)
    def get_noise(self) -> np.ndarray:
        return self.R
    def measure(self, states: np.ndarray, noises: np.ndarray) -> np.ndarray:
        return np.array([s[0:3] + r for (s, r) in zip(states, noises)])
