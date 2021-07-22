######################################################################
# This file copyright the Georgia Institute of Technology
#
# Permission is given to students to use or modify this file (only)
# to work on their assignments.
#
# You may NOT publish this file or make it available to others not in
# the course.
#
######################################################################

# These import statements give you access to library functions which you may
# (or may not?) want to use.
from math import *
from glider import *
from copy import deepcopy

# This is the function you will have to write for part A.
# -The argument 'height' is a floating point number representing
# the number of meters your glider is above the average surface based upon 
# atmospheric pressure. (You can think of this as height above 'sea level'
# except that Mars does not have seas.) Note that this sensor may be off
# a static  amount that will not change over the course of your flight.
# This number will go down over time as your glider slowly descends.
#
# -The argument 'radar' is a floating point number representing the
# number of meters your glider is above the specific point directly below
# your glider based off of a downward facing radar distance sensor. Note that
# this sensor has random Gaussian noise which is different for each read.

# -The argument 'mapFunc' is a function that takes two parameters (x,y)
# and returns the elevation above "sea level" for that location on the map
# of the area your glider is flying above.  Note that although this function
# accepts floating point numbers, the resolution of your map is 1 meter, so
# that passing in integer locations is reasonable.
#
#
# -The argument OTHER is initially None, but if you return an OTHER from
# this function call, it will be passed back to you the next time it is
# called, so that you can use it to keep track of important information
# over time.
#

def estimate_next_pos(height, radar, mapFunc, OTHER=None):
    """Estimate the next (x,y) position of the glider."""

    # example of how to find the actual elevation of a point of ground from the map:

    # You must return a tuple of (x,y) estimate, and OTHER (even if it is NONE)
    # in this order for grading purposes.
    #
    # OTHER = (PARTICLES, TIMESTAMP, SIGMA)
    SIGMA_SETTER = 100

    SIGMA_MAP = {0: SIGMA_SETTER,
                 5: SIGMA_SETTER*0.95,
                 10: SIGMA_SETTER*0.9,
                 15: SIGMA_SETTER*0.85,
                 20: SIGMA_SETTER*0.80,
                 25: SIGMA_SETTER*0.75,
                 30: SIGMA_SETTER*0.70,
                 35: SIGMA_SETTER*0.65,
                 40: SIGMA_SETTER*0.60,
                 45: SIGMA_SETTER*0.55,
                 50: SIGMA_SETTER*0.45,
                 55: SIGMA_SETTER*0.35,
                 60: SIGMA_SETTER*0.15,
                 65: SIGMA_SETTER*0.10,
                 70: SIGMA_SETTER*0.5
                 }

    N_MAP = {25000: 1500,
             1500: 1000,
             1000: 1000}

    if not OTHER:
        TIME_STAMP = 0
        world_bounds = 250  # 100000 meters
        N = 25000
        resample_N = N_MAP[N]
        SIGMA = SIGMA_MAP[TIME_STAMP]
        particles = []
        # initialize particles
        for i in range(N):
            p = glider()
            p.x = random.uniform(-1, 1) * world_bounds
            p.y = random.uniform(-1, 1) * world_bounds
            p.z = height
            p.heading = random.gauss(0,pi/4)
            p.mapFunc = mapFunc
            p.set_noise(0, 0, 0)
            particles.append(p)
    else:
        TIME_STAMP = OTHER[1]
        N = len(OTHER[0])
        resample_N = N_MAP[N]
        particles = OTHER[0]
        if TIME_STAMP in SIGMA_MAP.keys():
            SIGMA = SIGMA_MAP[TIME_STAMP]
        else:
            SIGMA = OTHER[2]
    # weigh each particle
    particle_weight = []
    for i in range(N):
        prob = 1
        prob *= Gaussian(0, SIGMA, particles[i].sense() - radar)
        particle_weight.append(prob)

    # wheel and Re-sample
    # from this point on, use resample_N
    weight_index = int(random.random() * N)
    beta = 0.0
    max_weighted_particle = max(particle_weight)
    particle_holder = []
    for i in range(resample_N):
        beta += random.random() * 2.0 * max_weighted_particle
        while beta > particle_weight[weight_index]:
            beta -= particle_weight[weight_index]
            weight_index = (weight_index + 1) % N
        particle_holder.append(deepcopy(particles[weight_index]))
    particles = particle_holder

    # add fuzzing
    for i in range(resample_N):
        if random.random() < 0.25:
            particles[i].x = random.uniform(-2.5, 2.5) + particles[i].x
            particles[i].y = random.uniform(-2.5, 2.5) + particles[i].y
            new_heading = particles[i].heading + random.uniform(-0.4, 0.4)
            adjusted_new_heading = angle_trunc(new_heading)
            particles[i].heading = adjusted_new_heading

    # glide resampled particles and obtain avg x,y
    # resample_N should equal len(particles) CHECK
    particle_holder = []
    optionalPointsToPlot = []
    total_x = 0
    total_y = 0
    for i in range(resample_N):
        particles[i].glide()
        total_x += particles[i].x
        total_y += particles[i].y
        particle_holder.append(particles[i])
        optionalPointsToPlot.append((particles[i].x, particles[i].y))

    particles = particle_holder
    predicted_x = total_x/resample_N
    predicted_y = total_y/resample_N

    xy_estimate = (predicted_x, predicted_y,)  # Sample answer, (X,Y) as a tuple.

    # You may optionally also return a list of (x,y,h) points that you would like
    # the PLOT_PARTICLES=True visualizer to plot for visualization purposes.
    # If you include an optional third value, it will be plotted as the heading
    # of your particle.

    TIME_STAMP += 1
    OTHER = (particles, TIME_STAMP, SIGMA)

    return xy_estimate, OTHER, optionalPointsToPlot


def Gaussian(mu, sigma, x):
    # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
    return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))


# This is the function you will have to write for part B. The goal in part B
# is to navigate your glider towards (0,0) on the map steering # the glider 
# using its rudder. Note that the Z height is unimportant.

#
# The input parameters are exactly the same as for part A.

def next_angle(height, radar, mapFunc, OTHER=None):
    # How far to turn this timestep, limited to +/-  pi/8, zero means no turn.
    #      OTHER = (current_xy_estimate, TIME_STEP, estimated_OTHER)
    MAX_TURN_ANGLE = pi/8.0
    new_holder = []
    g = None
    if not OTHER:
        TIME_STEP = 0
        current_xy_estimate, estimated_OTHER, optionalPointsToPlot = estimate_next_pos(height, radar, mapFunc)
        steering_angle = 0
    else: # check time step less than 50
        TIME_STEP = OTHER[1]
        if TIME_STEP <= 50:
            current_xy_estimate, estimated_OTHER, optionalPointsToPlot = estimate_next_pos(height, radar, mapFunc,
                                                                                           OTHER[2])
            steering_angle = 0
        elif 50 < TIME_STEP < 200:
            current_xy_estimate, estimated_OTHER, optionalPointsToPlot = estimate_next_pos(height, radar, mapFunc,
                                                                                           OTHER[2])

            previous_xy_estimate = OTHER[0]
            estimated_particles = estimated_OTHER[0]
            # atan2(y,x)
            Y = current_xy_estimate[1] - previous_xy_estimate[1]
            X = current_xy_estimate[0] - previous_xy_estimate[0]

            current_bearing = angle_trunc(atan2(Y, X))

            Y = 0 - current_xy_estimate[1]
            X = 0 - current_xy_estimate[0]
            goal_bearing = angle_trunc(atan2(Y, X))

            estimated_bearing = angle_trunc(goal_bearing - current_bearing)
            estimated_bearing = max(-MAX_TURN_ANGLE, estimated_bearing)
            estimated_bearing = min(MAX_TURN_ANGLE, estimated_bearing)
            steering_angle = estimated_bearing
            for particle in estimated_particles:
                particle.heading += steering_angle * 0.99
                particle.heading = angle_trunc(particle.heading)

        else:
            current_xy_estimate, estimated_OTHER, optionalPointsToPlot = estimate_next_pos(height, radar, mapFunc,
                                                                                           OTHER[2])
            steering_angle = 0


    TIME_STEP += 1
    NEW_OTHER = (current_xy_estimate, TIME_STEP, estimated_OTHER)


    return steering_angle, NEW_OTHER
