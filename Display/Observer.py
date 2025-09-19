from dataclasses import dataclass

@dataclass
class Observer:
    camera_dist: float = 25
    camera_pitch: float = 0
    camera_yaw: float = 0
    camera_half_fov: float = 45
    camera_fine_pitch: float = 0
    camera_fine_yaw: float = 0
