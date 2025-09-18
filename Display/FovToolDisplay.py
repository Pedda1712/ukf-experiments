import pygame
from pygame.locals import *

from .Observer import Observer
from .DisplayConfig import DisplayConfig
from .Camera import Camera
from .GLRenderer import GLRenderer

from World import BallWorldInfo, Ball


class FovToolDisplay():

    def __init__(self, config: DisplayConfig, initial_observer: Observer):
        self.config = config
        self.clock = pygame.time.Clock()
        pygame.init()
        pygame.display.set_mode(self.config.dimensions, DOUBLEBUF|OPENGL)
        self.renderer = GLRenderer(self.config)
        self.camera = Camera()
        self.camera.observer = initial_observer

    def display(self, world: BallWorldInfo) -> bool:
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
        self.renderer.endPass()

        pygame.display.flip()
        pygame.time.wait(10)
        return True

    def get_last_delta(self):
        return self.dt
