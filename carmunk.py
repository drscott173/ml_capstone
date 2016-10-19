#
# Carmunk ported to PyMunk 5.0
#
# Original code
# https://github.com/harvitronix/reinforcement-learning-car/blob/master/flat_game/carmunk.py
#
# @scottpenberthy
# October, 2016
#

import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

class GameState:
    def __init__(self, game):
        # Global-ish.
        self.game = game
        self.crashed = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Create the car.
        self.create_car(100, 100, 0.5)

        # Record steps.
        self.num_steps = 0

        # Create walls.
        height = self.game.height
        width = self.game.width
        thick = 1

        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, thick), (0, height), thick),
            pymunk.Segment(
                self.space.static_body,
                (thick, height), (width, height), thick),
            pymunk.Segment(
                self.space.static_body,
                (width-thick, height), (width-thick, thick), thick),
            pymunk.Segment(
                self.space.static_body,
                (thick, thick), (width, thick), thick)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        self.obstacles.append(self.create_obstacle(200, 350, 100))
        self.obstacles.append(self.create_obstacle(700, 200, 125))
        self.obstacles.append(self.create_obstacle(600, 600, 35))

        # Create a cat and food and hud
        self.create_cat()
        self.hud = "HUD"
        self.show_hud()

    def show_hud(self):
        # Show a simple head-up display in the far corner
        font = pygame.font.Font(None, 24)
        survivedtext = font.render(self.hud.zfill(2), True, (255, 255, 255))
        textRect = survivedtext.get_rect()
        textRect.topright=[self.game.width-5,5]
        self.game.screen.blit(survivedtext, textRect)

    def create_obstacle(self, x, y, r):
        # Create a body in PyMunk to represent a big, heavy obstacle
        c_body = pymunk.Body(1000000, 1000000) # was pymunk.inf
        c_shape = pymunk.Circle(c_body, r*self.game.scale)
        c_shape.elasticity = 1.0
        c_body.position = x*self.game.scale, y*self.game.scale
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self):
        # Create a lighter body in PyMunk to represent a fast moving cat
        inertia = pymunk.moment_for_circle(1, 0, 14*self.game.scale, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50*self.game.scale, self.game.height - 100*self.game.scale
        self.cat_shape = pymunk.Circle(self.cat_body, 30*self.game.scale)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)

    def create_car(self, x, y, r):
        # moment has mass, inner_radius, outer_radius, offset=(0, 0))
        inertia = pymunk.moment_for_circle(1, 0, 14*self.game.scale, (0, 0))
        # mass, moment
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x*self.game.scale, y*self.game.scale
        # body, radius
        self.car_shape = pymunk.Circle(self.car_body, 25*self.game.scale)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse_at_local_point(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    def current(self):
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        # add x,y,theta
        readings.append(x/self.game.width)
        readings.append(y/self.game.height)
        readings.append(self.car_body.angle)
        state = np.array(readings)
        return state

    def update_screen(self, color):
        self.space.step(1./10)
        self.game.clock.tick()
        if self.game.draw_screen:
            self.game.screen.fill(THECOLORS[color])
            self.space.debug_draw(self.game.options)
            self.show_hud()
            pygame.display.flip()

    def frame_step(self, action):
        if action == 1:  # Turn left.
            self.car_body.angle -= .2
        elif action == 2:  # Turn right.
            self.car_body.angle += .2

        # Move obstacles.
        if self.num_steps % 100 == 0:
            self.move_obstacles()

        # Move cat.
        if self.num_steps % 5 == 0:
            self.move_cat()

        self.car_body.angle %= 2.0*math.pi
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = 100 * self.game.scale * driving_direction

        # Update the screen and stuff.
        self.update_screen("black")

        readings = self.current()
        # Set the reward.
        if self.car_is_crashed(readings):
            # Car crashed when any reading == 1
            self.crashed = True
            terminal = True
            reward = -100
            self.recover_from_crash(driving_direction)
        else:
            # Higher readings are better, so return the sum.
            # Reward based on the smallest sonar, e.g., having more clearance in general is better.
            # we penalize cars that are too close to other things (6 clicks or less)
            terminal = False
            reward = np.min(readings[:3])-6.0 
        self.num_steps += 1

        return reward, readings, terminal

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = self.game.scale*random.randint(1, 5)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cat(self):
        # zip saround
        speed = self.game.scale*random.randint(20, 200)
        self.cat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction

    def car_is_crashed(self, readings):
        if (readings[0] == 1) or (readings[1] == 1) or (readings[2] == 1):
            return True
        else:
            return False

    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
        """
        while self.crashed:
            # Go backwards.
            self.car_body.velocity = -100 * driving_direction * self.game.scale
            self.crashed = False
            for i in range(10):
                self.car_body.angle += .2  # Turn a little.
                self.update_screen("red")

    def sum_readings(self, readings):
        """Sum the number of non-zero readings."""
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm_left = self.make_sonar_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left

        # Rotate them and get readings.
        readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))

        """
        # add three more
        back = angle+math.pi
        readings.append(self.get_arm_distance(arm_left, x, y, back, 0.75))
        readings.append(self.get_arm_distance(arm_middle, x, y, back, 0))
        readings.append(self.get_arm_distance(arm_right, x, y, back, -0.75))
        """

        if self.game.show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                  or rotated_p[0] >= self.game.width or rotated_p[1] >= self.game.height:
                return i  # Sensor is off the screen.
            else:
                obs = self.game.screen.get_at(rotated_p)
                if self.get_track_or_not(obs, i == 1) != 0:
                    return i

            if self.game.show_sensors:
                pygame.draw.circle(self.game.screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i

    def make_sonar_arm(self, x, y):
        spread = 10
        distance = int(self.game.scale*20+0.5)  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = self.game.height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading, first):
        return (reading in [THECOLORS[x] for x in ['orange', 'blue', 'red']])*1

class Game:
    def __init__(self, scale):
        self.scale = scale
        self.width = int(1000*scale+0.5)
        self.height = int(700*scale+0.5)
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()
        self.screen.set_alpha(None)
        self.show_sensors = True
        self.draw_screen = True
        self.reset()
    def reset(self):
        self.state = GameState(self)
        self.total_reward = 0
        self.state.crashed = False
        self.state.move_cat()

    def step(self, action):
        reward, sensors, terminal = self.state.frame_step(action)
        self.total_reward += reward
        return reward, sensors, terminal

if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 3)))
