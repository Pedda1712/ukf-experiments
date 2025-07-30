from World import BallWorldInfo, Ball, BallTransition
from Display import Observer, DisplayConfig, Display

import numpy as np

from Filter import UKFPredictor, UKFObserver, TransitionModel, MeasurementModel
from ScenarioModels import BallTransitionModel, Ball3DMeasurementModel, BallTriangulationModel

from dataclasses import dataclass
from typing import Callable

@dataclass
class BallExperimentConfig:
    ball_world: BallWorldInfo
    transition_model: TransitionModel
    assumed_measurement_model: MeasurementModel
    measurement: MeasurementModel
    initial_true_state: Ball
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    observer_list: list[Observer]
    mean_extractor: Callable[np.ndarray, list[float]]

def run_experiment(config: BallExperimentConfig):
    world = config.ball_world
    ball = config.initial_true_state
    trans = BallTransition(world)

    ukf_predict = UKFPredictor(config.transition_model)
    ukf_observe = UKFObserver(config.assumed_measurement_model)

    mean = config.prior_mean
    cov = config.prior_covariance
    
    unfiltered = None
    display = Display(DisplayConfig())

    while display.display(world, [ball, Ball(config.mean_extractor(mean), np.zeros(3))], config.observer_list):
        ball = trans.transition([ball], display.get_last_delta())[0]
        measurement_noise = config.measurement.get_noise()
        unfiltered = config.measurement.measure(np.concat((np.array(ball.pos), np.array(ball.velocity))).reshape(1,6), np.random.multivariate_normal(np.zeros(measurement_noise.shape[0]), measurement_noise))
        if len(unfiltered.shape) == 2:
            unfiltered = unfiltered[0]
        mean_dash, cov_dash = ukf_predict.predict(mean, cov, display.get_last_delta())
        mean, cov = ukf_observe.observe(mean_dash, cov_dash, unfiltered)

if __name__ == "__main__":
    # in this case the observations are the concatenated 2D positions
    # in three cameras
    world = BallWorldInfo()
    observer_list = [
        Observer(25, 0, 0),
        Observer(25, 90, 0),
        Observer(25, 0, -90)
    ]
    config = BallExperimentConfig(
        world,
        BallTransitionModel(world, np.ones(3)*0.1),
        BallTriangulationModel(np.ones(6)*0.2, observer_list),
        BallTriangulationModel(np.ones(6)*0.2, observer_list),
        Ball([0, 0, 0], [20, 9, 13]),
        np.zeros(6),
        np.eye(6)*10,
        observer_list,
        lambda m: m[0:3].tolist()
    )
    run_experiment(config)
    """
    # direct 3D observations
    config = BallExperimentConfig(
        world,
        BallTransitionModel(world, np.ones(3)*0.1),
        Ball3DMeasurementModel(np.ones(3)*5),
        Ball3DMeasurementModel(np.ones(3)*5),
        Ball([0, 0, 0], [20, 9, 13]),
        np.zeros(6),
        np.eye(6)*10,
        [],
        lambda m: m[0:3].tolist()
    )
    run_experiment(config)
    """
