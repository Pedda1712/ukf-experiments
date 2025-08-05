from World import BallWorldInfo, Ball, BallTransition
from Display import Observer, DisplayConfig, Display

import numpy as np

from Filter import UKFPredictor, UKFObserver, TransitionModel, MeasurementModel
from ScenarioModels import BallTransitionModel, Ball3DMeasurementModel, BallTriangulationModel, BallUnknownCameraPositionsTriangulationModel

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
    assumed_observers_extractor: Callable[np.ndarray, list[Observer]]

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

    while display.display(world, [ball, Ball(config.mean_extractor(mean), np.zeros(3))], config.observer_list, config.assumed_observers_extractor(mean)):
        ball = trans.transition([ball], display.get_last_delta())[0]
        measurement_noise = config.measurement.get_noise()
        unfiltered = config.measurement.measure(np.concat((np.array(ball.pos), np.array(ball.velocity))).reshape(1,6), np.random.multivariate_normal(np.zeros(measurement_noise.shape[0]), measurement_noise))
        if len(unfiltered.shape) == 2:
            unfiltered = unfiltered[0]
        mean_dash, cov_dash = ukf_predict.predict(mean, cov, display.get_last_delta())
        mean, cov = ukf_observe.observe(mean_dash, cov_dash, unfiltered)

def get_pitch_yaws(mean: np.ndarray, known: Observer):
    result = [(known.camera_pitch, known.camera_yaw)]
    num_cams = int((mean.shape[0] - 6) / 2)
    for n in range(num_cams):
        s = 6 + n*2
        result.append((mean[s], mean[s+1]))
    return result

if __name__ == "__main__":
    # In this experiment, two camera angles are to be estimated from observations
    world = BallWorldInfo()
    actual_observer_list = [
        Observer(25, 0, 0),
        Observer(25, 90, 0),
        Observer(25, 0, -90),
        Observer(25, 0, -45),
    ]
    ball_transition_variance = 0.1
    camera_transition_variance = 0.01
    assumed_measurement_variance = 0.5
    actual_measurement_variance = 0.5
    ball_prior_variance = 10
    camera_prior_variance = 45
    config = BallExperimentConfig(
        world,
        BallTransitionModel(world, np.hstack((np.ones(3)*ball_transition_variance, np.ones(6) * camera_transition_variance))),
        BallUnknownCameraPositionsTriangulationModel(np.ones(8)*assumed_measurement_variance, actual_observer_list[0]),
        BallTriangulationModel(np.ones(8)*actual_measurement_variance, actual_observer_list),
        Ball([0, 0, 0], [20, 9, 13]),
        np.hstack((np.zeros(6), np.array([45, -45, 45, -45, 45, -45]))),
        np.diag(np.hstack((np.ones(6)*ball_prior_variance,np.ones(6)*camera_prior_variance))),
        actual_observer_list,
        lambda m: m[0:3].tolist(),
        lambda m: [Observer(actual_observer_list[0].camera_dist, pitch, yaw) for (pitch, yaw) in get_pitch_yaws(m, actual_observer_list[0])]
    )
    run_experiment(config)
    """
    # in this case the observations are the concatenated 2D positions
    # in three cameras
    world = BallWorldInfo()
    observer_list = [
        Observer(25, 0, 0),
        Observer(25, 90, 0),
        Observer(25, 0, -90),
    ]
    config = BallExperimentConfig(
        world,
        BallTransitionModel(world, np.ones(3)*0.1),
        BallTriangulationModel(np.ones(6)*0.25, observer_list),
        BallTriangulationModel(np.ones(6)*0.25, observer_list),
        Ball([0, 0, 0], [20, 9, 13]),
        np.zeros(6),
        np.eye(6)*10,
        observer_list,
        lambda m: m[0:3].tolist(),
        lambda m: observer_list
    )
    run_experiment(config)
    """
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
        lambda m: []
    )
    run_experiment(config)
    """
