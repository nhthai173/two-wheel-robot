import pygame
import math
import numpy as np

class Enviroment(object):
    def __init__(self, dimensions: tuple):
        # colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue = (0, 0, 255)
        self.red = (255, 0, 0)
        self.yellow = (255, 255, 0)
        
        # map dims
        self.height = dimensions[0]
        self.width = dimensions[1]
        
        # window settings
        pygame.display.set_caption("Simulator Robot")
        self.map = pygame.display.set_mode((self.width, self.height))

        # write variables
        self.font = pygame.font.Font('freesansbold.ttf', 50)
        self.text = self.font.render('default', True, self.white, self.black)
        self.textRect = self.text.get_rect()
        self.textRect.center = (dimensions[1] - 900, dimensions[0] - 50)

        # trail set
        self.trailSet = []

    def write_info(self, vl, vr, theta):
        txt = f'vl = {vl:.2f} | vr = {vr:.2f} | theta = {int(math.degrees(theta))}'
        self.text = self.font.render(txt, True, self.white, self.black)
        self.map.blit(self.text, self.textRect)

    def trail(self, pos):
        for i in range(0, len(self.trailSet) - 1):
            pygame.draw.line(self.map, self.yellow, (self.trailSet[i][0], self.trailSet[i][1]), (self.trailSet[i + 1][0], self.trailSet[i + 1][1]))
        if self.trailSet.__sizeof__() > 10000:
            self.trailSet.pop(0)
        self.trailSet.append(pos)

    def frame(self, pos, rotation):
        n = 80
        cx, cy = pos
        x_axis = (cx + n * math.cos(-rotation), cy + n * math.sin(-rotation))
        y_axis = (cx + n * math.cos(-rotation + math.pi / 2), cy + n * math.sin(-rotation + math.pi / 2))
        pygame.draw.line(self.map, self.red, (cx, cy), x_axis, 3)
        pygame.draw.line(self.map, self.green, (cx, cy), y_axis, 3)


class Robot:
    def __init__(self, startPos, robotImg, width):
        self.m2p = 3779.52
        self.width = width * self.m2p
        self.x = startPos[0]
        self.y = startPos[1]
        self.theta = 0
        self.vl = self.m2p * 0.01e-2
        self.vr = self.m2p * -0.01e-2
        self.maxSpeed = self.m2p * 1e-2
        self.minSpeed = -self.m2p * 1e-2
        # robot variable
        self.l = 10e-2 * self.m2p
        self.r = 2.5e-2 * self.m2p
        # graphics
        self.img = pygame.image.load(robotImg)
        self.img = pygame.transform.scale(self.img, (int(self.width*2), int(self.width*2)))
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center = (self.x, self.y))
        # time
        self.dataTime = 0
        self.lastTime = pygame.time.get_ticks()

    def draw(self, map):
        map.blit(self.rotated, self.rect)

    def move(self, event = None):
        # Change velocity
        if event:
            if event.type == pygame.KEYDOWN:
                # change vl
                if event.key == pygame.K_1:
                    self.vl += 0.0001 * self.m2p
                if event.key == pygame.K_2:
                    self.vl -= 0.0001 * self.m2p
                # change vr
                if event.key == pygame.K_3:
                    self.vr += 0.0001 * self.m2p
                if event.key == pygame.K_4:
                    self.vr -= 0.0001 * self.m2p
        # update position
        self.vr = min(self.vr, self.maxSpeed)
        self.vl = min(self.vl, self.maxSpeed)
        self.vr = max(self.vr, self.minSpeed)
        self.vl = max(self.vl, self.minSpeed)
        self.dataTime = (pygame.time.get_ticks() - self.lastTime) / 1000
        self.lastTime = pygame.time.get_ticks()
        
        # kinematics
        R = np.array([
                [np.cos(self.theta), np.sin(self.theta), 0], 
                [-np.sin(self.theta), np.cos(self.theta), 0], 
                [0, 0, 1]])
        j1f = np.array([[1, 0, -self.l], 
                        [-1, 0, -self.l]])
        j2 = np.array([[self.r, 0], 
                    [0, self.r]])
        vv = np.linalg.inv(R) @ np.linalg.pinv(j1f) @ j2 @ np.array([[self.vl], [self.vr]])
        vx, vy, omega = vv.flatten()
        self.x += vx * self.dataTime
        self.y -= vy * self.dataTime
        self.theta += omega * self.dataTime

        # limit theta
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        
        self.rotated = pygame.transform.rotozoom(self.img, math.degrees(self.theta), 1)
        self.rect = self.rotated.get_rect(center = (self.x, self.y))



if __name__ == "__main__":
    pygame.init()

    # start position
    start = (200, 200)
    dims = (800, 1200)
    running = True

    # enviroment
    env = Enviroment(dims)

    # Robot
    robot = Robot(start, './2w-robot.png', 0.01)

    # loop
    while(running):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            robot.move(event)

        pygame.display.update()
        env.map.fill(env.black)
        robot.move()
        robot.draw(env.map)
        env.frame((robot.x, robot.y), robot.theta)
        env.trail((robot.x, robot.y))
        env.write_info(robot.vl, robot.vr, robot.theta)