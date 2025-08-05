from .DisplayConfig import DisplayConfig
from .Shader import load_program, vertex, fragment
from .Observer import Observer
from World import BallWorldInfo, Ball

from OpenGL.GL import *
from OpenGL.GLU import *

import glm

"""
NOTE: this GL code is horrible, its a wild mix
of fixed function pipeline and shader usage
(e.g. the view matrix is assembled on the ff matrix
 stack but passed as a uniform to the shader :) )

It's ok since it's only prototypical though and the
focus here is not on performant (or nice to look at)
graphics.
"""

class GLRenderer():

    def __init__(self, config: DisplayConfig):
        self.config = config
        self.shad = load_program(vertex, fragment)
        self.view_loc = glGetUniformLocation(self.shad, "view")
        self.sphere = gluNewQuadric()

    def beginPass(self):
        glEnable(GL_DEPTH_TEST)
        glClearColor(*self.config.clear_color)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective(45, (self.config.dimensions[0]/self.config.dimensions[1]), 0.1, 500)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def endPass(self):
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
                    
    def setObserver(self, observer: Observer):
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(0, 0, -observer.camera_dist)
        glRotatef(observer.camera_pitch, 1, 0, 0)
        glRotatef(observer.camera_yaw, 0, 1, 0)
        view = glm.mat4(1.0)
        glGetFloatv(GL_MODELVIEW_MATRIX, glm.value_ptr(view))
        glPopMatrix()
        glUseProgram(self.shad)
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, glm.value_ptr(view))

    def drawCube(self, color: tuple[float, float, float]):
        glBegin(GL_QUADS)
        glColor3f(color[0], color[1], color[2])
        glVertex3f(-1, 1, -1)
        glVertex3f(1, 1, -1)
        glVertex3f(1, 1, 1)
        glVertex3f(-1, 1, 1)
        glVertex3f(-1, -1, -1)
        glVertex3f(1, -1, -1)
        glVertex3f(1, -1, 1)
        glVertex3f(-1, -1, 1)
        glVertex3f(-1,  -1, 1)
        glVertex3f( 1,  -1, 1)
        glVertex3f( 1,   1, 1)
        glVertex3f(-1,   1, 1)
        glVertex3f(-1,  -1, -1)
        glVertex3f( 1,  -1, -1)
        glVertex3f( 1,   1, -1)
        glVertex3f(-1,   1, -1)
        glVertex3f(1, -1, -1 )
        glVertex3f(1,  1, -1 )
        glVertex3f(1,  1,  1 )
        glVertex3f(1, -1,  1 )   
        glVertex3f(-1, -1, -1 )
        glVertex3f(-1,  1, -1 )
        glVertex3f(-1,  1,  1 )
        glVertex3f(-1, -1,  1 )
        glEnd()

    def drawCamera(self, observer: Observer, color: tuple[float, float, float], size: float):
        glPushMatrix()

        mat1 = glm.rotate(glm.mat4(), glm.radians(observer.camera_yaw), glm.vec3(0, 1, 0))
        mat2 = glm.rotate(glm.mat4(), glm.radians(observer.camera_pitch), glm.vec3(1, 0, 0))
        mat3 = glm.translate(glm.mat4(), glm.vec3(0, 0, -observer.camera_dist))
        mat = glm.inverse(mat3 * mat2 * mat1)
        glLoadMatrixf(mat.to_list())

        glScale(size, size, size)
        glRotate(180, 0, 1, 0)
        
        glPushMatrix()
        glTranslate(0, 0, -1)
        self.drawCube(color = color)
        glPopMatrix()
        glPushMatrix()
        glScale(1.3, 1.3, 0.3)
        glTranslate(0, 0, 1)
        self.drawCube(color = color)
        glPopMatrix()

        glPopMatrix()
    
    def drawBackWalls(self, world: BallWorldInfo):
        glPushMatrix()
        glLoadIdentity()
        glPushMatrix()
        glTranslate(0, -world.dims[1]/2, 0)
        glScale(world.dims[0]/2, 0.1, world.dims[2]/2)
        glTranslate(0, -1, 0)
        self.drawCube(color = (1, 1, 0))
        glPopMatrix()
        glPushMatrix()
        glTranslate(0, 0, -world.dims[2]/2)
        glScale(world.dims[0]/2, world.dims[1]/2, 0.1)
        glTranslate(0, 0, -1)
        self.drawCube(color = (0, 1, 1))
        glPopMatrix()
        glPushMatrix()
        glTranslate(-world.dims[0]/2, 0, 0)
        glScale(0.10, world.dims[1]/2, world.dims[2]/2)
        glTranslate(-1, 0, 0)
        self.drawCube(color = (1, 0, 1))
        glPopMatrix()

        glPopMatrix()

    def drawBall(self, ball: Ball, color: tuple[float, float, float], world: BallWorldInfo):
        glPushMatrix()
        glTranslate(*ball.pos)
        glColor3f(*color)
        gluSphere(self.sphere, world.ball_radius, 100, 100)
        glPopMatrix()
    

        
