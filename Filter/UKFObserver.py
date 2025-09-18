import numpy as np

from .MeasurementModel import MeasurementModel
from .SigmaPointGeneration import SigmaPointGenerationConfig, extend_state_by_normal_noise, sigma_points, transformed_sigma_points_to_gaussian
class UKFObserver:
    h: MeasurementModel
    config: SigmaPointGenerationConfig
    
    def __init__(self, h: MeasurementModel, parameters: SigmaPointGenerationConfig = SigmaPointGenerationConfig()):
        self.h = h
        self.config = parameters

    def observe(self, m_predict: np.ndarray, P_predict: np.ndarray, y_observe: np.ndarray, observation_uncertainty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m_predict_ex, P_predict_ex = extend_state_by_normal_noise(m_predict, P_predict, observation_uncertainty)
        X, Wm, Wc = sigma_points(m_predict_ex, P_predict_ex, self.config)

        X = np.array(X)
        state_dim = m_predict.shape[0]
        states = X[:, :state_dim]
        noises = X[:, state_dim:]
        
        Y = self.h.measure(states, noises)

        mu, S, C = transformed_sigma_points_to_gaussian(Y, Wm, Wc, states)

        # now the rules for gaussian conditioning apply
        K = C @ np.linalg.pinv(S) # filter gain
        m = m_predict + K @ (y_observe - mu)
        P = P_predict - K @ S @ K.T
        return m, P
