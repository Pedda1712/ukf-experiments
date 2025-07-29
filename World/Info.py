class BallWorldInfo():
    def __init__(self, width = 20, height = 20, depth = 20, gravity = 9, ball_radius = 0.5):
        self.dims = [width, height, depth]
        self.gravity = gravity
        self.bounce_discount = 1
        self.air_discount = 1
        self.ground_discount = 1
        self.ball_radius = 1
