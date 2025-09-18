from .UKFObserver import UKFObserver
from .UKFPredictor import UKFPredictor
from .MeasurementModel import MeasurementModel
from .TransitionModel import TransitionModel

import numpy as np

"""
Implements a Rauch-Tung-Striebel Smoother with the
unscented transform used to estimate propagated or
measured distributions as gaussians.
"""

class Smoother:
    predictor: UKFPredictor
    observer: UKFObserver
    prior_mean: np.ndarray
    prior_covariance: np.ndarray

    def __init__(self, measurement_model: MeasurementModel, transition_model: TransitionModel, prior_mean: np.ndarray, prior_covariance: np.ndarray):
        self.predictor = UKFPredictor(transition_model)
        self.observer = UKFObserver(measurement_model)
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance

    def estimate_sequence(self, measurements: list[np.ndarray], measurement_covariances: list[np.ndarray], deltas: list[float]):
        """
        Perform state estimation over a sequence of measurements
        using Unscented RTS Smoothing.

        Parameters:
        -----------
        measurements: list of state measurements
        measurement_covariances: each element here quantifies
          the sensor uncertainty of the corresponding element
          in the measurements list
        deltas: time delay between measurments
        Returns:
        --------
        smoothed_means: means of the estimated state distribution
          at every time step
        smoothed_covariances: covariance of the estimated state
          distribution at every time step
        """
        if len(measurements) != len(measurement_covariances):
            raise RuntimeError("need same amount of measurement and covariances")
        if len(measurements) != len(deltas):
            raise RuntimeError("need time delay between all measurements")
        
        # First, UKF pass, saving the filter distributions and predicted distributions
        mean = self.prior_mean
        cov = self.prior_covariance
        means = [self.prior_mean]
        covs = [self.prior_covariance]
        predicted_means = []
        predicted_covs = []
        Ds = [] # covariance between current filtered state and predicted next state

        for unfiltered, noise, dt in zip(measurements, measurement_covariances, deltas):
            predicted_mean, predicted_cov, D = self.predictor.predict(mean, cov, dt)
            predicted_means.append(predicted_mean)
            predicted_covs.append(predicted_cov)
            Ds.append(D)
            mean, cov = self.observer.observe(predicted_mean, predicted_cov, unfiltered, noise)
            means.append(mean)
            covs.append(cov)

        # Backward pass to integrate future information into past timesteps
        smoothed_means = [means.pop()] # last filter distribution already has all information
        smoothed_covs = [covs.pop()]
        for _ in range(len(means)):
            mk = means.pop()
            Pk = covs.pop()
            mk1_dash = predicted_means.pop()
            Pk1_dash = predicted_covs.pop()
            Dk1 = Ds.pop()
            mk1_s = smoothed_means[-1]
            Pk1_s = smoothed_means[-1]

            # Actual Smoothing equations are very simple:
            Gk = Dk1 @ np.linalg.pinv(Pk1_dash) # smoother gain
            mk_s = mk + Gk @ (mk1_s - mk1_dash)
            Pk_s = Pk + Gk @ (Pk1_s - Pk1_dash) @ Gk.T
            
            smoothed_means.append(mk_s)
            smoothed_covs.append(Pk_s)

        smoothed_means.reverse()
        smoothed_covs.reverse()

        return smoothed_means, smoothed_covs
