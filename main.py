import pygame
from pygame.locals import *

from World import BallWorldInfo, Ball, BallTransition
from Display import GLRenderer, Observer, DisplayConfig, Display

import numpy as np

from Filter import UKFPredictor, UKFObserver, TransitionModel, MeasurementModel
from ScenarioModels import BallTransitionModel, Ball3DMeasurementModel

if __name__ == "__main__":
    world = BallWorldInfo()
    ball = Ball([0, 0, 0], [20, 9, 13])
    trans = BallTransition(world)

    transition_noise = np.array([0.1, 0.1, 0.1])
    real_observation_noise = np.array([3, 3, 3])
    assumed_observation_noise = np.array([3, 3, 3])

    transition_model = BallTransitionModel(world, transition_noise)
    measurement_model = Ball3DMeasurementModel(assumed_observation_noise)
    ukf_predict = UKFPredictor(transition_model)
    ukf_observe = UKFObserver(measurement_model)

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
