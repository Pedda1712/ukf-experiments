import pygame
from pygame.locals import *

from World import BallWorldInfo, Ball, BallTransition
from Display import GLRenderer, Observer, DisplayConfig, Display

import numpy as np

def sigma_points(mean: np.ndarray, cov: np.ndarray, alpha: float = 1, beta: float = 0, kappa: float = -1) -> tuple[list[np.ndarray], list[float], list[float]]:
    """
    Calculate Sigma Points of the Unscented Transform according to [Särkkä2013]
    """
    n = mean.shape[0]
    
    lam = (alpha**2)*(n + kappa) - n
    fac = np.sqrt(n + lam)
    
    n_points = 2 * n + 1
    X = [np.zeros(n) for _ in range(n_points)]
    Wm = [1/(2*n + 2*lam)] * n_points
    Wc = [1/(2*n + 2*lam)] * n_points
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    P = np.linalg.cholesky(cov)
    X[0] = mean.copy()

    for i in range(n):
        idx = i + 1
        X[idx] = mean + fac * P[:, i]
        X[idx + n] = mean - fac * P[:, i]
    return X, Wm, Wc

def transformed_sigma_points_to_gaussian(X: list[np.ndarray], Wm: list[float], Wc: list[float], X_old: list[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = (np.array(X) * np.array(Wm).reshape(-1, 1)).sum(axis=0)
    centered_X = (np.array(X) - m)
    Pk = centered_X.T @ (centered_X * np.array(Wc).reshape(-1, 1))
    
    if X_old is None:
        return m, Pk, None
    else:
        # for the observations: also calculate covariance of predicted state and observations
        X_old = np.array(X_old)[:, 0:6]
        m_old = (X_old * np.array(Wm).reshape(-1, 1)).sum(axis=0)
        X_old_centered = (X_old - m_old)
        Ck = X_old_centered.T @ (centered_X * np.array(Wc).reshape(-1, 1))
        return m, Pk, Ck

def extend_state_by_normal_noise(mk, Pk, Q):
    stateshape = Pk.shape[0]
    noisedim = Q.shape[0]
    Pk_ex = np.block([
        [Pk, np.zeros((stateshape, noisedim))],
        [np.zeros((noisedim, stateshape)), Q]
    ])
    mk_ex = np.append(mk, np.zeros(noisedim))
    return mk_ex, Pk_ex
    
class BallTransitionModel:
    Q: np.ndarray
    world: BallWorldInfo
    transition: BallTransition
    
    def __init__(self, world: BallWorldInfo, velocity_variance: np.ndarray):
        self.Q = np.diag(velocity_variance)
        self.world = world
        self.transition = BallTransition(world)

    def predict(self, mk: np.ndarray, Pk: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Uses the UT to approximate the distribution of the transformed
        gaussian variable by another gaussian distribution.
        """
        mk_ex, Pk_ex = extend_state_by_normal_noise(mk, Pk, self.Q)
        X, Wm, Wc = sigma_points(mk_ex, Pk_ex)
        
        balls = [Ball(x[0:3], x[3:6] + x[6:9]) for x in X] # noisy balls
        balls = self.transition.transition(balls, delta)

        Y = [np.append(np.array(b.pos), np.array(b.velocity)) for b in balls]

        m, P, _ = transformed_sigma_points_to_gaussian(Y, Wm, Wc)
        return m, P

class Ball3DObservationModel:
    R: np.ndarray

    def __init__(self, observation_variance: np.ndarray):
        self.R = np.diag(observation_variance)

    def observe(self, m_predict, P_predict, y_observe) -> tuple[np.ndarray, np.ndarray]:
        m_predict_ex, P_predict_ex = extend_state_by_normal_noise(m_predict, P_predict, self.R)

        X, Wm, Wc = sigma_points(m_predict_ex, P_predict_ex)
        Y = [x[0:3] + x[6:9] for x in X]

        mu, S, C = transformed_sigma_points_to_gaussian(Y, Wm, Wc, X)

        # now the rules for gaussian conditioning apply
        K = C @ np.linalg.inv(S) # filter gain
        m = m_predict + K @ (y_observe - mu)
        P = P_predict - K @ S @ K.T
        return m, P

if __name__ == "__main__":
    world = BallWorldInfo()
    ball = Ball([0, 0, 0], [20, 9, 13])
    trans = BallTransition(world)

    transition_noise = np.array([0.1, 0.1, 0.1])
    real_observation_noise = np.array([10, 10, 10])
    assumed_observation_noise = np.array([10, 10, 10])

    ukf_predict = BallTransitionModel(world, transition_noise)
    ukf_observe = Ball3DObservationModel(assumed_observation_noise)

    mean = np.array([0, 0, 0, 0, 0, 0])
    uncert = np.diag(np.array([10, 10, 10, 10, 10, 10]))
    
    unfiltered = np.array([0, 0, 0])
    
    display = Display(DisplayConfig())

    while display.display(world, [ball, Ball(mean[0:3].tolist(), mean[3:6].tolist()), Ball(unfiltered.tolist(), np.zeros(3))]):
        ball = trans.transition([ball], display.get_last_delta())[0]
        # noisy observation
        unfiltered = np.random.multivariate_normal(ball.pos, np.diag(real_observation_noise))
        # filter
        mean_dash, uncert_dash = ukf_predict.predict(mean, uncert, display.get_last_delta())
        mean, uncert = ukf_observe.observe(mean_dash, uncert_dash, unfiltered)
