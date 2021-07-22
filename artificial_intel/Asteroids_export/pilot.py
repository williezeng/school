
from builtins import object
import random
import math
from matrix import matrix
import numpy as np

class Pilot(object):

    def __init__(self, min_dist, in_bounds):
        self.min_dist = min_dist
        self.in_bounds = in_bounds
        self.dt = 1
        self.last_moves = []

        self.u = np.array([0., 0., 0., 0., 0., 0.])

        self.H = np.array([[1., 0., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0., 0.]])

        self.F = np.array([[1.,0.,self.dt,0.,(self.dt**2)/2,0],
                           [0.,1.,0.,self.dt,0.,(self.dt**2)/2],
                           [0.,0.,1.,0.,self.dt,0.],
                           [0.,0.,0.,1.,0.,self.dt],
                           [0.,0.,0.,0.,1.,0.],
                           [0.,0.,0.,0.,0.,1.]])


        self.R = np.array([[1.,0.],[0.,1.]])

        self.I = np.array([[1.,0.,0.,0.,0.,0.],
                          [0.,1.,0.,0.,0.,0.],
                          [0.,0.,1.,0.,0.,0.],
                          [0.,0.,0.,1.,0.,0.],
                          [0.,0.,0.,0.,1.,0.],
                          [0.,0.,0.,0.,0.,1.]])
        self.asteroid_dict = {}

    def observe_asteroids(self, asteroid_locations):
        """ self - pointer to the current object.
           asteroid_locations - a list of asteroid observations. Each 
           observation is a tuple (i,x,y) where i is the unique ID for
           an asteroid, and x,y are the x,y locations (with noise) of the
           current observation of that asteroid at this timestep.
           Only asteroids that are currently 'in-bounds' will appear
           in this list, so be sure to use the asteroid ID, and not
           the position/index within the list to identify specific
           asteroids. (The list may change in size as asteroids move
           out-of-bounds or new asteroids appear in-bounds.)

           Return Values:
                    None
        """
        for asteroid in asteroid_locations:
            if asteroid[0] not in self.asteroid_dict.keys():
                x = np.array([asteroid[1], asteroid[2], 0., 0., 0., 0.])
                P = np.array([[10., 0., 0., 0., 0., 0.],  # covariance matrix (uncertainty)
                               [0., 10., 0., 0., 0., 0.],
                               [0., 0., 100., 0., 0., 0.],
                               [0., 0., 0., 100., 0., 0.],
                               [0., 0., 0., 0., 100., 0.],
                               [0., 0., 0., 0., 0., 100.]])
                self.asteroid_dict[asteroid[0]] = (x, P, asteroid[1], asteroid[2])
            else:
                (predicted_x, P, old_x_coord, old_y_coord) = self.asteroid_dict[asteroid[0]]

                self.asteroid_dict[asteroid[0]] = (predicted_x, P, asteroid[1], asteroid[2])

        # clean up any asteroids that aren't being reported
        for asteroid_key in list(self.asteroid_dict.keys()):
            if self.asteroid_dict[asteroid_key][2] > 1.1 or self.asteroid_dict[asteroid_key][2] < -1.1 or self.asteroid_dict[asteroid_key][3] > 1.1 or self.asteroid_dict[asteroid_key][3] < -1.1:
                del self.asteroid_dict[asteroid_key]

    def estimate_asteroid_locs(self):
        """ Should return an iterable (list or tuple for example) that
            contains data in the format (i,x,y), consisting of estimates
            for all in-bound asteroids. """
        updated_asteroid_tup = ()

        for asteroid in self.asteroid_dict.keys():
            x_coord = self.asteroid_dict[asteroid][2]
            y_coord = self.asteroid_dict[asteroid][3]

            # predict x
            matrix_multiply = self.F @ self.asteroid_dict[asteroid][0]
            predicted_x = np.add(matrix_multiply, self.u)
            # predict P
            matrix_multiply2 = self.F @ self.asteroid_dict[asteroid][1]
            predicted_P = np.dot(matrix_multiply2, self.F.transpose())
            # measurement update
            Z = np.array([x_coord, y_coord])
            matrix_multiply3 = self.H @ predicted_x
            y = np.subtract(Z, matrix_multiply3)
            S = np.add(self.H @ predicted_P @ self.H.transpose(), self.R)
            invert_S = np.linalg.inv(S)
            K = predicted_P @ self.H.transpose() @ invert_S

            new_x = np.add(predicted_x, (K @ y))
            new_P = (self.I - (K @ self.H)) @ predicted_P
            self.asteroid_dict[asteroid] = (new_x, new_P, x_coord, y_coord)
            updated_asteroid_tup = ((asteroid, new_x[0], new_x[1]),) + updated_asteroid_tup
        return updated_asteroid_tup

    def next_move(self, craft_state):
        """ self - a pointer to the current object.
            craft_state - implemented as CraftState in craft.py.

            return values: 
              angle change: the craft may turn left(1), right(-1), 
                            or go straight (0). 
                            Turns adjust the craft's heading by 
                             angle_increment.
              speed change: the craft may accelerate (1), decelerate (-1), or 
                            continue at its current velocity (0). Speed 
                            changes adjust the craft's velocity by 
                            speed_increment, maxing out at max_speed.
         """
        # prior this call, we observe the asteroid positions initially

        # self.asteroid_dict[asteroid] = (new_x, new_P, x_coord, y_coord)
        detection_buffer = 0.02
        construction_buffer = 0.04
        craft_right_x = craft_state.x + construction_buffer
        craft_top_y = craft_state.y + construction_buffer
        craft_left_x = craft_state.x - construction_buffer
        craft_bottom_y = craft_state.y - construction_buffer

        radar_right_x = craft_right_x + detection_buffer
        radar_top_y = craft_top_y + detection_buffer
        radar_left_x = craft_left_x - detection_buffer
        radar_bottom_y = craft_bottom_y - detection_buffer

        updated_asteroid_tup = self.estimate_asteroid_locs()
        L_weight = 0
        R_weight = 0
        angle = -1
        speed = 0
        U_weight = 0
        no_asteroids_detected_counter = 0

        for asteroid in list(self.asteroid_dict.keys()):
            # iterate through asteroids and create a weighted decision
            # after iteration check which decision has the most weight and then perform
            curr_roid_x = self.asteroid_dict[asteroid][2]
            curr_roid_y = self.asteroid_dict[asteroid][3]
            asteroid_predicted_x = self.asteroid_dict[asteroid][0][0]
            asteroid_predicted_y = self.asteroid_dict[asteroid][0][1]
            asteroid_predicted_x_velocity = self.asteroid_dict[asteroid][0][2]
            asteroid_predicted_y_velocity = self.asteroid_dict[asteroid][0][3]
            asteroid_predicted_x_acc = self.asteroid_dict[asteroid][0][4]
            asteroid_predicted_y_acc = self.asteroid_dict[asteroid][0][5]
            window = 0.001

            if craft_top_y <= asteroid_predicted_y <= radar_top_y and craft_right_x <= asteroid_predicted_x <= radar_right_x:
                # asteroid coming in the top right = make a left weighted
                distance_x = asteroid_predicted_x - craft_right_x
                distance_y = asteroid_predicted_y - craft_top_y
                # weighted more the closer they are and the faster they are
                L_weight += (1/distance_x + 1/distance_y) + (asteroid_predicted_x_velocity + asteroid_predicted_y_velocity)

            elif craft_top_y <= asteroid_predicted_y <= radar_top_y and craft_left_x >= asteroid_predicted_x >= radar_left_x:
                # asteroid detected in the top left = make a right
                distance_x = abs(asteroid_predicted_x - craft_left_x)
                distance_y = abs(asteroid_predicted_y - craft_top_y)
                R_weight += (1/distance_x + 1/distance_y) + (asteroid_predicted_x_velocity + asteroid_predicted_y_velocity)

            elif craft_bottom_y >= asteroid_predicted_y >= radar_bottom_y and craft_left_x >= asteroid_predicted_x >= radar_left_x:
                # asteroid detected in the bottom left = make a right
                distance_x = abs(asteroid_predicted_x - craft_left_x)
                distance_y = abs(asteroid_predicted_y - craft_bottom_y)
                R_weight += (1/distance_x + 1/distance_y) + (asteroid_predicted_x_velocity + asteroid_predicted_y_velocity)

            elif craft_bottom_y >= asteroid_predicted_y >= radar_bottom_y and craft_right_x >= asteroid_predicted_x >= radar_right_x:
                # asteroid detected in the bottom right = make a left
                distance_x = abs(asteroid_predicted_x - craft_right_x)
                distance_y = abs(asteroid_predicted_y - craft_bottom_y)
                L_weight += (1/distance_x + 1/distance_y) + (asteroid_predicted_x_velocity + asteroid_predicted_y_velocity)
            elif abs(curr_roid_y-craft_top_y) < window:
                L_weight = 900
                break
        # if self.last_move == 'right':
        #     angle = 1
        #     speed = 0
        #     self.last_move = 'straight'
        # elif self.last_move == 'left':
        #     angle = -1
        #     speed = 0
        #     self.last_move = 'straight'
        # el
        # if self.last_move == 'straight':
        if L_weight > R_weight:
            angle = 1
            speed = 0
            self.last_moves.append('left')
        elif R_weight > L_weight:
            angle = -1
            speed = 0
            self.last_moves.append('right')
        else:
            if not self.last_moves:
                # begins
                angle = 0
                speed = 1
            elif self.last_moves[0] == 'left':
                angle = -1
                speed = 0
                self.last_moves.pop(0)
            elif self.last_moves[0] == 'right':
                angle = 1
                speed = 0
                self.last_moves.pop(0)
        return 0, 1
