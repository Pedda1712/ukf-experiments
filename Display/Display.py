import pygame
from pygame.locals import *

from .Observer import Observer
from .DisplayConfig import DisplayConfig
from .GLRenderer import GLRenderer

from World import BallWorldInfo, Ball

class Camera:
    def __init__(self):
        self.observer = Observer()
        self.camera_speed = 20
        self.camera_turn_speed = 100

    def update(self, keys, dt):
        if keys[pygame.K_a]:
            self.observer.camera_yaw += self.camera_turn_speed * dt
        if keys[pygame.K_d]:
            self.observer.camera_yaw -= self.camera_turn_speed * dt
        if keys[pygame.K_w]:
            self.observer.camera_pitch += self.camera_turn_speed * dt
        if keys[pygame.K_s]:
            self.observer.camera_pitch -= self.camera_turn_speed * dt
        if keys[pygame.K_q]:
            self.observer.camera_dist -= self.camera_speed * dt
        if keys[pygame.K_e]:
            self.observer.camera_dist += self.camera_speed * dt

class Display():

    def __init__(self, config: DisplayConfig):
        self.config = config
        self.clock = pygame.time.Clock()
        pygame.init()
        pygame.display.set_mode(self.config.dimensions, DOUBLEBUF|OPENGL)
        self.renderer = GLRenderer(self.config)
        self.camera = Camera()

    def display(self, world: BallWorldInfo, ball: Ball) -> bool:
        self.dt = self.clock.tick(60) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            
        keys = pygame.key.get_pressed()
        self.camera.update(keys, self.dt)

        self.renderer.beginPass()
        self.renderer.setObserver(self.camera.observer)
        self.renderer.drawBackWalls(world)
        for (b, c) in zip(ball, [(0, 0, 1), (0, 1, 0), (1, 0, 0)]):
            self.renderer.drawBall(b, c, world)
        self.renderer.endPass()

        pygame.display.flip()
        pygame.time.wait(10)
        return True

    def get_last_delta(self):
        return self.dt
