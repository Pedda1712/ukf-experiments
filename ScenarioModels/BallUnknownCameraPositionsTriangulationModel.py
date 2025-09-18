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
    reference_proj_view: np.ndarray

    def __init__(self, observation_variance: np.ndarray, reference_viewpoint: Observer):
        self.R = np.diag(observation_variance)
        self.reference_point = reference_viewpoint

    def measure(self, states: np.ndarray, noises: np.ndarray) -> np.ndarray:
        num_estimated = int((states.shape[1] - 6 - 1) / 3)
        projected_points = []
        # build maps of observers for each viewpoint
        viewpoint_maps = [dict() for _ in range(num_estimated)]
        for n in range(num_estimated):
            proto_observers = np.unique(np.concat((states[:, (6+1):][:, (n*3):((n+1)*3)], states[:, 6].reshape(-1,1)), axis=1), axis=0)
            for po in proto_observers:
                (pitch, yaw, dist, fov) = po
                glm_view = self._get_viewpoint(pitch, yaw, dist)
                glm_proj = glm.perspective(glm.radians(fov), 1, 0.01, 500)
                viewpoint_maps[n][po.tobytes()] = np.array(glm_proj * glm_view)

        already_projected_maps = [dict() for _ in range(num_estimated)]

        for s in states:
            self.reference_point.camera_half_fov = s[6]
            glm_view = self.get_viewpoint(self.reference_point)
            glm_proj = glm.perspective(glm.radians(self.reference_point.camera_half_fov), 1, 0.01, 500)
            self.reference_proj_view = np.array(glm_proj * glm_view)
            
            pre_projection = np.concat((s[0:3], np.ones(1))).reshape(-1, 1)
            reference = (self.reference_proj_view @ pre_projection).T
            projected_point = [reference[0, 0:2] / reference[0, 3]]
            for n in range(num_estimated):
                po = np.concat((s[(6+1):][(n*3):((n+1)*3)], [s[6]]))
                proj_view = viewpoint_maps[n][po.tobytes()]
                projection = None
                if (not po.tobytes() in already_projected_maps[n]) or (not pre_projection.tobytes() in already_projected_maps[n][po.tobytes()]):
                    projection = (proj_view @ pre_projection).T
                    already_projected_maps[n][po.tobytes()] = dict()
                    tmp = projection[0, 0:2] / projection[0, 3]
                    already_projected_maps[n][po.tobytes()][pre_projection.tobytes()] = tmp
                    projected_point.append(tmp)
                else:
                    projected_point.append(already_projected_maps[n][po.tobytes()][pre_projection.tobytes()])
            projected_points.append(np.hstack(projected_point))
        return np.array(projected_points) + noises

