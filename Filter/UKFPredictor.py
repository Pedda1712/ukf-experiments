import numpy as np

from .TransitionModel import TransitionModel
from .SigmaPointGeneration import SigmaPointGenerationConfig, extend_state_by_normal_noise, sigma_points, transformed_sigma_points_to_gaussian

class UKFPredictor:
    f: TransitionModel
    config: SigmaPointGenerationConfig
    
    def __init__(self, f: TransitionModel, config: SigmaPointGenerationConfig = SigmaPointGenerationConfig()):
        self.f = f
        self.config = config
        
    def predict(self, mk: np.ndarray, Pk: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Uses the UT to approximate the distribution of the transformed
        gaussian variable by another gaussian distribution.
        """
        mk_ex, Pk_ex = extend_state_by_normal_noise(mk, Pk, self.f.get_noise())
        X, Wm, Wc = sigma_points(mk_ex, Pk_ex, self.config)
        
        X = np.array(X)
        state_dim = np.shape(mk)[0]
        states = X[:, :(state_dim)]
        noises = X[:, (state_dim):]
        
        Y = self.f.transition(states, noises, delta)

        m, P, D = transformed_sigma_points_to_gaussian(Y, Wm, Wc, states)
        return m, P, D
