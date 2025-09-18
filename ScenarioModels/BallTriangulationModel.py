import numpy as np
import glm

from Filter import MeasurementModel
from Display import Observer

"""
Produces observations by perspective projecting
the 3D observation into virtual cameras, producing a
6-dimensional observation (2D position for each camera).

Effectively, the Filter utilizing this model
will need to perform triangulation to infer the 3D state.
"""

class BallTriangulationModel(MeasurementModel):
    R: np.ndarray
    viewpoints: list[Observer]
    def __init__(self, observation_variance: np.ndarray, viewpoints: list[Observer]):
        self.R = np.diag(observation_variance)
        self.viewpoints = viewpoints
    def get_noise(self) -> np.ndarray:
        return self.R
    def _get_viewpoint(self, pitch, yaw, dist):
        mat1 = glm.rotate(glm.mat4(), glm.radians(yaw), glm.vec3(0, 1, 0))
        mat2 = glm.rotate(glm.mat4(), glm.radians(pitch), glm.vec3(1, 0, 0))
        mat3 = glm.translate(glm.mat4(), glm.vec3(0, 0, -dist))
        return mat3 * mat2 * mat1
    def get_viewpoint(self, observer: Observer):
        return self._get_viewpoint(observer.camera_pitch, observer.camera_yaw, observer.camera_dist)
    def measure(self, states: np.ndarray, noises: np.ndarray) -> np.ndarray:

        # projection matrix to use

        points = np.hstack((states[:, 0:3], np.ones((states.shape[0], 1))))
        projected_points = []
        
        for viewpoint in self.viewpoints:
            project = glm.perspective(glm.radians(viewpoint.camera_half_fov), 1, 0.01, 500)
            viewmat = self.get_viewpoint(viewpoint)
            proj_view = np.array(project * viewmat) # this operates on column vectors
            projection = (proj_view @ points.T).T # only keep x,y from cube
            projected_points.append(projection[:, 0:2] / projection[:, 3].reshape(-1, 1))

        return np.hstack(projected_points) + noises
