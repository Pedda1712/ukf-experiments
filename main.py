from World import BallWorldInfo, Ball, BallTransition
from Display import Observer, DisplayConfig, Display

import numpy as np

from Filter import Smoother, TransitionModel, MeasurementModel
from ScenarioModels import BallTransitionModel, Ball3DMeasurementModel, BallTriangulationModel, BallUnknownCameraPositionsTriangulationModel

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt

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
    position_extractor: Callable[np.ndarray, list[float]]
    assumed_observers_extractor: Callable[np.ndarray, list[Observer]]

def run_experiment(config: BallExperimentConfig):
    world = config.ball_world
    ball = config.initial_true_state
    trans = BallTransition(world)

    smoother = Smoother(config.assumed_measurement_model, config.transition_model, config.prior_mean, config.prior_covariance)

    # record some measurments
    RECORD_FRAMES = 1500
    measurements = []
    measurement_covariances = []
    deltas = []
    balls = [ball]
    for r in range(RECORD_FRAMES):
        print(r, end = "\r")
        ball = trans.transition([ball], 1/60)[0]
        balls.append(ball)
        measurement_noise = config.measurement.get_noise()
        unfiltered = config.measurement.measure(np.concat((np.array(ball.pos), np.array(ball.velocity))).reshape(1,6), np.random.multivariate_normal(np.zeros(measurement_noise.shape[0]), measurement_noise))
        if len(unfiltered.shape) == 2:
            unfiltered = unfiltered[0]
        measurements.append(unfiltered)
        measurement_covariances.append(config.assumed_measurement_model.get_noise())
        deltas.append(1/60)

    # State Estimation!
    smoothed_means, _ = smoother.estimate_sequence(measurements, measurement_covariances, deltas)

    fov = []
    display = Display(DisplayConfig())
    for ball, mean in zip(balls,smoothed_means):
        observers = config.assumed_observers_extractor(mean)
        fov.append([o.camera_half_fov for o in observers])
        if not display.display(world, [ball, Ball(config.position_extractor(mean), np.zeros(3))], config.observer_list, observers):
            break
    
    fov = np.array(fov)
    x = np.arange(fov.shape[0])
    plt.plot(x, fov[:,0], c="r")
    plt.plot(x, fov[:,1], c="g")
    plt.plot(x, fov[:,2], c="b")
    plt.plot(x, fov[:,3], c="y")
    plt.show()

def get_pitch_yaws(mean: np.ndarray, known: Observer):
    result = [(known.camera_pitch, known.camera_yaw, known.camera_dist, known.camera_half_fov)]
    num_cams = int((mean.shape[0] - 6) / 4)
    for n in range(num_cams):
        s = 6 + n*4
        result.append((mean[s], mean[s+1], mean[s+2], mean[s+3]))
    return result

if __name__ == "__main__":
    world = BallWorldInfo()
    actual_observer_list = [
        Observer(25, 0, 0, 45),
        Observer(15, 90, 0, 50),
        Observer(35, 0, -90, 55),
        Observer(55, 0, -45, 40),
    ]
    ball_transition_variance = 0.01
    camera_transition_variance = 0.001
    assumed_measurement_variance = 0.01
    actual_measurement_variance = 0.01
    ball_prior_variance = 10
    camera_prior_variance = 45
    config = BallExperimentConfig(
        world,
        BallTransitionModel(world, np.hstack((np.ones(3)*ball_transition_variance, np.ones(12) * camera_transition_variance))),
        BallUnknownCameraPositionsTriangulationModel(np.ones(8)*assumed_measurement_variance, actual_observer_list[0]),
        BallTriangulationModel(np.ones(8)*actual_measurement_variance, actual_observer_list),
        Ball([0, 0, 0], [20, 9, 13]),
        np.hstack((np.zeros(6), np.array([85, -5, 20, 45, 5, -100, 30, 45, -5, -50, 60, 45]))),
        np.diag(np.hstack((np.ones(6)*ball_prior_variance,np.ones(12)*camera_prior_variance))),
        actual_observer_list,
        lambda m: m[0:3].tolist(),
        lambda m: [Observer(dist, pitch, yaw, fov) for (pitch, yaw, dist, fov) in get_pitch_yaws(m, actual_observer_list[0])]
    )
    run_experiment(config)
