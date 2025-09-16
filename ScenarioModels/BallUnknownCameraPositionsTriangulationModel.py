import numpy as np
import glm

from Filter import MeasurementModel
from Display import Observer
from .BallTriangulationModel import BallTriangulationModel

"""
In this observation model:
- the position of one camera is known
- the viewing angle of all other cameras is unknown
- the distance of all other cameras is unknown
:: Relative camera viewing angles are to be inferred
   by the filter
"""

class BallUnknownCameraPositionsTriangulationModel(BallTriangulationModel):
    R: np.ndarray
    reference_point: Observer

    def __init__(self, observation_variance: np.ndarray, reference_viewpoint: Observer):
        self.R = np.diag(observation_variance)
        self.reference_point = reference_viewpoint

    def measure(self, states: np.ndarray, noises: np.ndarray) -> np.ndarray:
        project = glm.perspective(glm.radians(45), 1, 0.01, 500)
        num_estimated = int((states.shape[1] - 6) / 3)
        projected_points = []
        for s in states:
            pre_projection = np.concat((s[0:3], np.ones(1))).reshape(-1, 1)
            viewpoints = [self.reference_point]
            # construct the viewpoints of this state
            for n in range(num_estimated):
                start_index = 6 + n * 3
                pitch = s[start_index]
                yaw = s[start_index+1]
                dist = s[start_index+2]
                viewpoints.append(Observer(dist, pitch, yaw))
            # project the ball position into those viewpoints
            projected_point = []
            for viewpoint in viewpoints:
                viewmat = self.get_viewpoint(viewpoint)
                proj_view = np.array(project * viewmat) 
                projection = (proj_view @ pre_projection).T
                projected_point.append(projection[0, 0:2] / projection[0, 3])
            projected_points.append(np.hstack(projected_point))
        return np.array(projected_points) + noises

