"""
 === Introduction ===

   The assignment is broken up into three parts.

   Part A:

        Create a SLAM implementation to process a series of landmark (gem) measurements and movement updates.
        The movements are defined for you so there are no decisions for you to make, you simply process the movements
        given to you.

        Hint: A planner with an unknown number of motions works well with an online version of SLAM.

    Part B:

        Here you will create the action planner for the robot.  The returned actions will be executed with the goal
        being to navigate to and extract a list of needed gems from the environment.  You will earn points by
        successfully extracting the list of gems from the environment.
        Example Actions (more explanation below):
            'move 3.14 1'
            'extract B'

    Part C:

        The only addition for part C will be to provide more information upon issuing the extract action.  The extra
        information will be the location of the gem you are attempting to extract.  The extra data will be compared
        to the real locations and you will be given points based on how close your estimates are to the real values.
        Example Action (more explanation below):
            'extract B 1.5 -0.2'

    Note: All of your estimates should be given relative to your robot's starting location.

    Details:

    - Start position
      - The robot will land at an unknown location on the map, however, you can represent this starting location
        as (0,0), so all future robot location estimates will be relative to this starting location.

    - Measurements
      - Measurements will come from gems located throughout the terrain.
        * The format is {'landmark id':{'distance':0.0, 'steering':0.0, 'type':'D'}, ...}
      - All gems on the terrain will return a measurement.

    - Movements
      - Action: 'move 1.570963 1.0'
        * The robot will turn counterclockwise 90 degrees and then move 1.0
      - Movements are stochastic due to, well, it being a robot.
      - If max distance or steering is exceeded, the robot will not move.

    - Needed Gems
      - Provided as list of gem types: ['A', 'B', 'L', ...]
      - Although the gem names aren't real, as a convenience there are 26 total names, each represented by an
        upper case letter of the alphabet (ABC...).
      - Action: 'extract'
        * The robot will attempt to extract a specified gem from the current location..
      - When a gem is extracted from the terrain, it no longer exists in the terrain, and thus won't return a
        measurement.
      - The robot must be with 0.25 distance to successfully extract a gem.
      - There may be gems in the environment which are not required to be extracted, which means you could extract
        them if you so desire, however, I advise you not to be greedy!  This is your warning.

    The robot will always execute a measurement first, followed by an action.

    The robot will have a time limit of 5 seconds to find and extract all of the needed gems.
"""

import math
from matrix import matrix
import numpy as np

class SLAM:
    """Create a basic SLAM module.
    """
    def __init__(self):
        """Initialize SLAM components here.
        """
        # self.omega = matrix()
        self.time = 0
        self.measurement_noise = 1
        self.num_landmarks = 0
        self.dimension = 0
        self.bearing = 0
        self.landmarks = []
        self.Omega = matrix()
        self.Xi = matrix()

    def print_stuff(self, stuff):
        for x in stuff:
            print(x)

    def process_measurements(self, measurements):
        """Process a new series of measurements.

        Args:
            measurements(dict): Collection of measurements
                in the format {'landmark id':{'distance':0.0, 'steering':0.0, 'type':'B'}, ...}

        Returns:
            x, y: current belief in location of the robot
        """
        self.dimension = (len(measurements) + 1) * 2
        if self.time == 0 :
            self.Omega.zero(self.dimension, self.dimension)
            self.Omega.value[0][0] = 1.0
            self.Omega.value[1][1] = 1.0
            self.Xi.zero(self.dimension, 1)
            self.Xi.value[0][0] = 0.0
            self.Xi.value[1][0] = 0.0

        for i, (key, value) in enumerate(measurements.items()):
            distance = value['distance']
            bearing = value['bearing']
            type = value['type']
            self.measurement_noise = math.e**distance
            bearing_noise = np.random.normal(bearing,0.00005)

            bearing = bearing_noise + self.bearing

            m = 2 * (1+i)
            # m is the index of the landmark coordinate in the matrix/vector
            dx = distance * math.cos(bearing)
            dy = distance * math.sin(bearing)
            z = [type, dx, dy]
            for b in range(0,2):
                self.Omega.value[b][b] += 1.0 / self.measurement_noise
                self.Omega.value[m + b][m + b] += 1.0 / self.measurement_noise
                self.Omega.value[b][m + b] += -1.0 / self.measurement_noise
                self.Omega.value[m + b][b] += -1.0 / self.measurement_noise

                self.Xi.value[b][0] += -z[1 + b] / self.measurement_noise
                self.Xi.value[m + b][0] += z[1 + b] / self.measurement_noise

        mu = self.Omega.inverse() * self.Xi
        x = mu[0][0]
        y = mu[1][0]
        self.time += 1

        return x, y

    def process_movement(self, steering, distance, motion_noise=0.01):
        """Process a new movement.

        Args:
            steering(float): amount to turn
            distance(float): distance to move
            motion_noise(float): movement noise

        Returns:
            x, y: current belief in location of the robot
        """

        self.Omega = self.Omega.expand(self.dimension + 2, self.dimension + 2, [0, 1] + list(range(4, self.dimension + 2)), [0, 1] + list(range(4, self.dimension + 2)))
        self.Xi = self.Xi.expand(self.dimension + 2, 1, [0, 1] + list(range(4, self.dimension + 2)), [0])
        bearing_to_point= steering + self.bearing
        # self.steering = ((bearing_to_point + math.pi) % (math.pi * 2)) - math.pi
        # update the information maxtrix/vector based on the robot motion
        bearing_noise = np.random.normal(bearing_to_point, 0.000002)
        self.bearing = bearing_noise

        dx = distance * math.cos(self.bearing)
        dy = distance * math.sin(self.bearing)

        z = [dx, dy]
        for b in range(4):
            self.Omega.value[b][b] += 1.0 / motion_noise
        for b in range(2):
            self.Omega.value[b][b + 2] += -1.0 / motion_noise
            self.Omega.value[b + 2][b] += -1.0 / motion_noise
            self.Xi.value[b][0] += -z[b] / motion_noise
            self.Xi.value[b + 2][0] += z[b] / motion_noise

        matrixB = self.Omega.take([0, 1], [0, 1])
        matrixA = self.Omega.take([0, 1], list(range(2, self.dimension + 2)))
        matrixC = self.Xi.take([0, 1], [0])
        OmegaPrime = self.Omega.take(list(range(2, self.dimension + 2)), list(range(2, self.dimension + 2)))
        XiPrime = self.Xi.take(list(range(2, self.dimension + 2)), [0])

        self.Omega = OmegaPrime - matrixA.transpose() * matrixB.inverse() * matrixA
        self.Xi = XiPrime - matrixA.transpose() * matrixB.inverse() * matrixC

        mu = self.Omega.inverse() * self.Xi
        x = mu[0][0]
        y = mu[1][0]
        # self.print_stuff(mu)

        self.time += 1

        return x, y


class GemExtractionPlanner:
    """
    Create a planner to navigate the robot to reach and extract all the needed gems from an unknown start position.
    """
    def __init__(self,  max_distance, max_steering):
        """Initialize your planner here.

        Args:
            max_distance(float): the max distance the robot can travel in a single move.
            max_steering(float): the max steering angle the robot can turn in a single move.
        """

        self.time = 0
        self.measurement_noise = 1
        self.num_landmarks = 0
        self.dimension = 0
        self.steering = 0
        self.landmarks = []
        self.Omega = matrix()
        self.Xi = matrix()
        self.max_distance = max_distance
        self.max_steering = max_steering
        self.min_steering = -max_steering
        self.planned_movement = 0
        self.x = 0
        self.y = 0

    def print_stuff(self, stuff):
        for x in stuff:
            print(x)

    def next_move(self, needed_gems, measurements):
        """Next move based on the current set of measurements.

        Args:
            needed_gems(list): List of gems remaining which still need to be found and extracted.
            measurements(dict): Collection of measurements from gems in the area.
                                {'landmark id': {
                                                    'distance': 0.0,
                                                    'steering' : 0.0,
                                                    'type'    :'B'
                                                },
                                ...}

        Return: action: str, points_to_plot: dict [optional]
            action (str): next command to execute on the robot.
                allowed:
                    'move 1.570963 1.0'  - Turn left 90 degrees and move 1.0 distance.
                    'extract D'          - [Part B] Attempt to extract a gem of type D from your current location.
                                           This will succeed if the specified gem is within the minimum sample distance.
                    'extract D 2.4 -1.6' - [Part C (also works for part B)] Attempt to extract a gem of type D
                                            from your current location.
                                           Specify the estimated location of gem D as (x=2.4, y=-1.6).
                                           This location is relative to your starting location (x=0, y=0).
                                           This will succeed if the specified gem is within the minimum sample distance.
            points_to_plot (dict): point estimates (x,y) to visualize if using the visualization tool [optional]
                            'self' represents the robot estimated position
                            <landmark_id> represents the estimated position for a certain landmark
                format:
                    {
                        'self': (x, y),
                        '<landmark_id_1>': (x1, y1),
                        '<landmark_id_2>': (x2, y2),
                        ....
                    }
        """
        # might need to sort closest first
        picked = False
        for needed_gem in needed_gems:
            for i, (key, value) in enumerate(measurements.items()):
                distance = value['distance']
                if 'bearing' in value:
                    bearing = value['bearing']
                else:
                    bearing = value['steering']
                type = value['type']
                if needed_gem == type:  # found the gem
                    # get gem x,y
                    if distance <= 0.25:
                        x,y = self.process_measurements({key:value})
                        action = 'extract {} {} {}'.format(needed_gem, x, y)
                        break
                    else:
                        x,y = self.process_measurements({key:value})
                        self.goal = [x,y]

                        if bearing > self.max_steering:
                            # remaining_steering = bearing - self.max_steering
                            if 'bearing' in value:
                                value['bearing'] = self.max_steering
                            else:
                                value['steering'] = self.max_steering
                        elif bearing < self.min_steering:
                            # remaining_steering = bearing - self.min_steering
                            if 'bearing' in value:
                                value['bearing'] = self.min_steering
                            else:
                                value['steering'] = self.min_steering
                        else:
                            if 'bearing' in value:
                                value['bearing'] = bearing
                            else:
                                value['steering'] = bearing

                        if distance > self.max_distance:
                            self.planned_movement = self.max_distance
                        else:
                            self.planned_movement = distance


                        if 'bearing' in value:
                            bruh = value['bearing']
                        else:
                            bruh = value['steering']

                        self.x,self.y = self.process_movement(bruh, self.planned_movement)
                        action = 'move {steering} {distance}'.format(steering=bruh, distance=self.planned_movement)

        return action


    def process_measurements(self, measurements):
        """Process a new series of measurements.

        Args:
            measurements(dict): Collection of measurements
                in the format {'landmark id':{'distance':0.0, 'steering':0.0, 'type':'B'}, ...}

        Returns:
            x, y: current belief in location of the robot
        """
        self.dimension = (len(measurements) + 1) * 2
        if self.time == 0 :
            self.Omega.zero(self.dimension, self.dimension)
            self.Omega.value[0][0] = 1.0
            self.Omega.value[1][1] = 1.0
            self.Xi.zero(self.dimension, 1)
            self.Xi.value[0][0] = 0.0
            self.Xi.value[1][0] = 0.0

        for i, (key, value) in enumerate(measurements.items()):
            distance = value['distance']
            if 'bearing' in value:
                bearing = value['bearing']
            else:
                bearing = value['steering']

            type = value['type']

            self.measurement_noise = 1
            # bearing = bearing + self.steering

            m = 2 * (1+i)
            # m is the index of the landmark coordinate in the matrix/vector
            dx = distance * math.cos(bearing)
            dy = distance * math.sin(bearing)
            z = [type, dx, dy]
            for b in range(0,2):
                self.Omega.value[b][b] += 1.0 / self.measurement_noise
                self.Omega.value[m + b][m + b] += 1.0 / self.measurement_noise
                self.Omega.value[b][m + b] += -1.0 / self.measurement_noise
                self.Omega.value[m + b][b] += -1.0 / self.measurement_noise

                self.Xi.value[b][0] += -z[1 + b] / self.measurement_noise
                self.Xi.value[m + b][0] += z[1 + b] / self.measurement_noise

        mu = self.Omega.inverse() * self.Xi
        x = mu[2][0]
        y = mu[3][0]
        self.time += 1
        return x, y

    def process_movement(self, steering, distance, motion_noise=0.01):
        """Process a new movement.

        Args:
            steering(float): amount to turn
            distance(float): distance to move
            motion_noise(float): movement noise

        Returns:
            x, y: current belief in location of the robot
        """

        self.Omega = self.Omega.expand(self.dimension + 2, self.dimension + 2, [0, 1] + list(range(4, self.dimension + 2)), [0, 1] + list(range(4, self.dimension + 2)))
        self.Xi = self.Xi.expand(self.dimension + 2, 1, [0, 1] + list(range(4, self.dimension + 2)), [0])
        bearing_to_point= steering + self.steering
        # self.steering = ((bearing_to_point + math.pi) % (math.pi * 2)) - math.pi
        # update the information maxtrix/vector based on the robot motion
        # bearing_noise = np.random.normal(bearing_to_point, 0.000002)
        self.steering = bearing_to_point
        dx = distance * math.cos(self.steering)
        dy = distance * math.sin(self.steering)

        z = [dx, dy]
        for b in range(4):
            self.Omega.value[b][b] += 1.0 / motion_noise
        for b in range(2):
            self.Omega.value[b][b + 2] += -1.0 / motion_noise
            self.Omega.value[b + 2][b] += -1.0 / motion_noise
            self.Xi.value[b][0] += -z[b] / motion_noise
            self.Xi.value[b + 2][0] += z[b] / motion_noise

        matrixB = self.Omega.take([0, 1], [0, 1])
        matrixA = self.Omega.take([0, 1], list(range(2, self.dimension + 2)))
        matrixC = self.Xi.take([0, 1], [0])
        OmegaPrime = self.Omega.take(list(range(2, self.dimension + 2)), list(range(2, self.dimension + 2)))
        XiPrime = self.Xi.take(list(range(2, self.dimension + 2)), [0])

        self.Omega = OmegaPrime - matrixA.transpose() * matrixB.inverse() * matrixA
        self.Xi = XiPrime - matrixA.transpose() * matrixB.inverse() * matrixC

        mu = self.Omega.inverse() * self.Xi
        x = mu[0][0]
        y = mu[1][0]
        self.time += 1

        return x, y