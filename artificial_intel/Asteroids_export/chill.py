
from builtins import object
import random
from matrix import matrix
import numpy as np

class Pilot(object):

    def __init__(self, min_dist, in_bounds):
        self.min_dist = min_dist
        self.in_bounds = in_bounds
        self.dt = 1
        self.current_state = 0
        self.u = np.matrix([[0.], [0.], [0.], [0.], [0.], [0.]])

        self.H = np.matrix([[1., 0., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0., 0.]])

        self.F = np.matrix([[1.,0.,self.dt,0.,(self.dt**2)/2,0],
                           [0.,1.,0.,self.dt,0.,(self.dt**2)/2],
                           [0.,0.,1.,0.,self.dt,0.],
                           [0.,0.,0.,1.,0.,self.dt],
                           [0.,0.,0.,0.,1.,0.],
                           [0.,0.,0.,0.,0.,1.]])


        self.R = np.matrix([[0.2,0.],[0.,0.2]])

        self.I = np.matrix([[1.,0.,0.,0.,0.,0.],
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
                # first time seeing asteroid with id
                x = np.matrix([[asteroid[1]], [asteroid[2]], [0.], [0.], [0.], [0.]])
                P = np.matrix([[0., 0., 0., 0., 0., 0.],  # covariance matrix (uncertainty)
                            [0., 0., 0., 0., 0., 0.],
                            [0., 0., 36., 0., 0., 0.],
                            [0., 0., 0., 36., 0., 0.],
                            [0., 0., 0., 0., 36., 0.],
                            [0., 0., 0., 0., 0., 36.]])
                self.asteroid_dict[asteroid[0]] = (x, P)
            else:
                # pass the values
                (predicted_x, P) = self.asteroid_dict[asteroid[0]]
                x = np.matrix([[asteroid[1]], [asteroid[2]], [0.], [0.], [0.], [0.]])
                self.asteroid_dict[asteroid[0]] = (x, P)


    def estimate_asteroid_locs(self):
        """ Should return an iterable (list or tuple for example) that
            contains data in the format (i,x,y), consisting of estimates
            for all in-bound asteroids. """
        updated_asteroid_tup = ()
        self.dt = 2
        self.F = np.matrix([[1., 0., self.dt, 0., (self.dt ** 2) / 2, 0],
                         [0., 1., 0., self.dt, 0., (self.dt ** 2) / 2],
                         [0., 0., 1., 0., self.dt, 0.],
                         [0., 0., 0., 1., 0., self.dt],
                         [0., 0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 0., 1.]])
        for asteroid in self.asteroid_dict.keys():
            # predict x
            predicted_x = (self.F * self.asteroid_dict[asteroid][0]) + self.u
            # predict P
            predicted_P = (self.F * self.asteroid_dict[asteroid][1]) * self.F.transpose()
            ok = self.asteroid_dict[asteroid][0][0]
            # Z = np.matrix([[ok], [self.asteroid_dict[asteroid][0][1]]])
            import pdb
            pdb.set_trace()
            y = Z - (self.H * predicted_x)

            S = self.H * predicted_P * self.H.transpose() + self.R
            K = predicted_P * self.H.transpose() * S.inverse()
            new_x = predicted_x + (K * y)
            new_P = (self.I - (K * self.H)) * predicted_P
            self.asteroid_dict[asteroid] = (new_x, new_P)
            import pdb
            pdb.set_trace()
            updated_asteroid_tup = ((asteroid,new_x[0][0], new_x[1][0]),) + updated_asteroid_tup
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

        return random.randint(-1,1), random.randint(-1,1)
