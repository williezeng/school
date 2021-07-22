from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
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

#!/usr/bin/python

import math
import random
import unittest
import glider
import multiprocessing as mproc
import queue
import traceback
from opensimplex import OpenSimplex


TIME_LIMIT = 12  # seconds - Note, if you turn on Verbose Logging
                 # or Plotting of Particles, you probably need to increase this
                 # number.

VERBOSE = True # False for grading
PLOT_PARTICLES = True # False for grading  (Set to True for Vizualization!)
PART_A = False # Enable/disable Part A (Estimation) - True for grading
PART_B = True # Enable/disable Part B (Steering) - True for grading

########################################################################
# If your debugger does not handle multiprocess debugging very easily
# then when debugging set the following flag true.
########################################################################
DEBUGGING_SINGLE_PROCESS = False

WINDOW_SIZE = 300   #Size of the window in "units" (actually 2x this...)


#Note for Mac OS High Sierra users having problems with "an error occurred while attempting to obtain endpoint for listener" errors:
#Running with the following environment variable fixed this issue for one student. 
#OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
#Looks like a problem specific to Mac OS High Sierra and internal threading libraries.

PI = math.pi
CREDIT_PER_PASS = 7  # points per test case pass.

#10 test cases, ran in both parts A & B for 20 total.
# Max score if you get all test cases is potentially 140, but capped at 101 
# This means you do not need to get all test cases for a full score.


# The real test cases will be generated using generate_parms_marsglidder.py 
# You are welcome to make your own test cases
# and share them on Piazza to expose issues that these test cases may miss.
#
#


GLOBAL_PARAMETERS = [None,

		     #Case 1 has no noise to make things easy for you!
                     {'test_case': 1,
                      'target_x': 39.84595717195,
                      'target_y': -13.82584680823,
                      'target_heading': 1.335,
                      'map_seed': 1,
                      'map_freq': 4.0,
                      'measurement_noise': 0.0,
                      'turning_noise': 0.0,
                      'altitude_offset': 0.0,
                      'max_steps': 4500 },



                     #Case 2 has no measurement noise, so hopefully you can 
                     #locate the glider, even with turning noise.
                     
                     #
                     {'test_case': 2,
                      'target_x': 199.95,
                      'target_y': -199.23,
                      'target_heading': -0.75598927002,
                      'map_seed': 2,
                      'map_freq': 4.0,
                      'measurement_noise': 0.0,
                      'turning_noise': 0.01,
                      'altitude_offset': 0.0,
                      'max_steps': 4500 },


                     #Case 3 has only measurement noise, and no turning noise!
                     {'test_case': 3,
                      'target_x': -99.6,
                      'target_y': 199.23,
                      'target_heading': 0.598927002,
                      'map_seed': 2,
                      'map_freq': 2.0,
                      'measurement_noise': 0.1,
                      'turning_noise': 0.0,
                      'altitude_offset': 0.0,
                      'max_steps': 4500 },

	             #Case 4 has both measurement AND (more) turning noise!
                     {'test_case': 4,
                      'target_x': 242.8,
                      'target_y': 139.450053,
                      'target_heading': -0.65272,
                      'map_seed': 1,
                      'map_freq': 2.0,
                      'measurement_noise': 0.1,
                      'turning_noise': 0.050,
                      'altitude_offset': 0.0,
                      'max_steps': 4500 },

		     #Case 5 has no noise, But the altitude offset may add
                     #a new dimension to your search space!
                     # Also note that the heading is on the extreme end of
                     # the random range for possible headings!
                     {'test_case': 5,
                      'target_x': 19.84595717195,
                      'target_y': -23.82584680823,
                      'target_heading': 1.5703,
                      'map_seed': 5,
                      'map_freq': 3.0,
                      'measurement_noise': 0.0,
                      'turning_noise': 0.0,
                      'altitude_offset': 8.0,
                      'max_steps': 4500 },

                     
                     {'test_case': 6,
                      'target_x': 199.95,
                      'target_y': 166.0,
                      'target_heading': 0.1152,
                      'map_seed': 6,
                      'map_freq': 8.0,
                      'measurement_noise': 0.5,
                      'turning_noise': 0.02,
                      'altitude_offset': -8.0,
                      'max_steps': 4500 },


                     {'test_case': 7,
                      'target_x': -99.6,
                      'target_y': 199.23,
                      'target_heading': 0.298927002,
                      'map_seed': 2,
                      'map_freq': 4.0,
                      'measurement_noise': 1.0,
                      'turning_noise': 0.05,
                      'altitude_offset': 5.0,
                      'max_steps': 4500 },

                     {'test_case': 8,
                      'target_x': 232.8,
                      'target_y': 139.53,
                      'target_heading': -0.65272,
                      'map_seed': 3,
                      'map_freq': 2.0,
                      'measurement_noise': 4.0,
                      'turning_noise': 0.000,
                      'altitude_offset': 0.0,
                      'max_steps': 4500 },


                     {'test_case': 9,
                      'target_x': 49.84595717195,
                      'target_y': -83.82584680823,
                      'target_heading': -0.340218,
                      'map_seed': 9,
                      'map_freq': 4.0,
                      'measurement_noise': 1.0,
                      'turning_noise': 0.00,
                      'altitude_offset': 8.0,
                      'max_steps': 4500 },

                     {'test_case': 10,
                      'target_x': 19.84595717195,
                      'target_y': -23.82584680823,
                      'target_heading': -0.3703,
                      'map_seed': 5,
                      'map_freq': 4.0,
                      'measurement_noise': 0.50,
                      'turning_noise': 0.09,
                      'altitude_offset': 0.0,
                      'max_steps': 4500 },


                      ]


#Function that generates the map function....
def getMapFunc(Seed, Freq):
   #initialize OpenSimplex one time, use it many times.
   gen = OpenSimplex(Seed)

   def mapFunc(nx,ny):
      #Force 1 unit resolution by flooring nx,ny to nearest integer.
      nx = int(nx)
      ny = int(ny)
      nx = nx / 5000.0
      ny = ny / 5000.0

      #Generate noise from both low and high frequency noise:
      e0 = 1 * gen.noise2d(Freq * nx, Freq * ny)
      e1 = 0.5 * gen.noise2d(Freq*4*nx, Freq*4*ny)
      e2 = 0.25 * gen.noise2d(Freq*16*nx, Freq*16*ny)
      e = e0 + e1 + e2

      return e * 500  #500 meters above/below average...

   return mapFunc


#Try importing the student code here:

try:
    import marsglider 
    marsglider1Exc=None
except Exception as e:
    print("Error importing marsglider.py:", e)
    marsglider1Exc=e

class GliderSimulator(object):
    """Run student submission code.

    Attributes:
        glider_steps(Queue): synchronized queue to store glider steps.
        glider_found(Queue): synchronized queue to store if glider located.
        glider_error(Queue): synchronized queue to store exception messages.
    """
    def __init__(self):

        if DEBUGGING_SINGLE_PROCESS:

            self.glider_steps = queue.Queue(1)
            self.glider_found = queue.Queue(1)
            self.glider_error = queue.Queue(1)

        else:

            self.glider_steps = mproc.Queue(1)
            self.glider_found = mproc.Queue(1)
            self.glider_error = mproc.Queue(1)

    def _reset(self):
        """Reset submission results.
        """
        while not self.glider_steps.empty():
            self.glider_steps.get()

        while not self.glider_found.empty():
            self.glider_found.get()

        while not self.glider_error.empty():
            self.glider_found.get()

    @staticmethod
    def distance(p, q):
        """Calculate the distance between two points.

        Args:
            p(tuple): point 1.
            q(tuple): point 2.

        Returns:
            distance between points.
        """
        x1, y1 = p[0],p[1]
        x2, y2 = q

        dx = x2 - x1
        dy = y2 - y1

        return math.sqrt(dx**2 + dy**2)

    @staticmethod
    def truncate_angle(t):
        """Truncate angle between pi and -pi.

        Args:
            t(float): angle to truncate.

        Returns:
            truncated angle.
        """
        return ((t + PI) % (2 * PI)) - PI

    def simulate_without_steering(self, estimate_next_pos, params):
        """Run simulation only to locate glider.

        Args:
            estimate_next_pos(func): Student submission function to estimate next glider position.
            params(dict): Test parameters.

        Raises:
            Exception if error running submission.
        """
        self._reset()

        #make the test somewhat repeatable by seeding the RNG.
        random.seed(params['map_seed'])

        ourMapFunc = getMapFunc( params['map_seed'], params['map_freq'])
        #Student function is separate, so they can mess it up if they want.
        studentMapFunc = getMapFunc( params['map_seed'], params['map_freq'] )

        target = glider.glider(params['target_x'],
                             params['target_y'],
                             5000 + random.randint(-50,50), #Altitude
                             params['target_heading'],
                             ourMapFunc)
        target.set_noise(params['measurement_noise'],
                         params['altitude_offset'],
                         params['turning_noise'] )

        tolerance = 5.0   
        other_info = None
        steps = 0


        #Set up the particle plotter if requested
        if PLOT_PARTICLES == True:
           import turtle		#Only import if plotting is on.
           X1,Y1,X2,Y2 = (-WINDOW_SIZE, -WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE)
           turtle.setup(width=800, height=800) #800 pixels is screensize
           turtle.setworldcoordinates(X1,Y1, X2,Y2 )
           turtle.hideturtle()
           turtle.penup()

           turtleList = []
           target_turtle = None
           estimate_turtle = None

           
        try:
            while steps < params['max_steps']:
                target_meas = target.sense()
                target_height = target.get_height()

                result = estimate_next_pos(target_height,target_meas, studentMapFunc, other_info)

                if len(result) == 3:
                   estimate, other_info,extra_points = result
                elif len(result) == 2: 
                   estimate, other_info = result
                   extra_points =  None
                else:
                   print("estimate_next_pos did not return correct number of return values!")

		#Calculate the actual position of the target next timestep.
                target.glide()
                target_pos = (target.x, target.y)

                if PLOT_PARTICLES == True and extra_points != None :

                   #If the target goes outside the window coordinates,
                   #recenter the window on the target.
                   if target_pos[0] < X1 or target_pos[0] > X2 or target_pos[1] < Y1 or target_pos[1] > Y2:
                       #calculate new bounding rectangle:
                       X1 = target_pos[0] - WINDOW_SIZE 
                       X2 = X1 + 2000
                       Y1 = target_pos[1] - WINDOW_SIZE 
                       Y2 = Y1 + 2000
                       turtle.setworldcoordinates(X1,Y1, X2,Y2 )
        
                   s = turtle.Screen()
                   s.tracer(0,1)

                   # Add turtles if needed.
                   while len(extra_points) > len(turtleList):
                       newTurtle = turtle.Turtle()
                       newTurtle.penup()
                       turtleList.append(newTurtle)

                   # remove turtles if needed.
                   while len(extra_points) < len(turtleList):
                       turtleList[-1].hideturtle()
                       turtleList = turtleList[0:-1]

                   for i in range(len(extra_points)):
                      t = turtleList[i]
                      p = extra_points[i]
                      #Optionally plot heading if provided by student
                      if len(p) > 2:
                         t.shape("triangle")
                         t.shapesize(0.2,0.4)
                         t.settiltangle( p[2] * 180 / math.pi )
                      else:
                         t.shape("circle")
                         t.shapesize(0.1,0.1)
                      t.setposition(p[0],p[1])

                   #Draw the actual glider.
                   if target_turtle is not None:
                       target_turtle.hideturtle()

                   if target_turtle is None:
                       target_turtle = turtle.Turtle()
                       target_turtle.shape("triangle")
                       target_turtle.shapesize(0.2, 0.4)
                       target_turtle.pencolor("red")
                       target_turtle.fillcolor("red")
                       target_turtle.penup()

                   target_turtle.setposition(target_pos[0], target_pos[1])
                   target_turtle.settiltangle( target.heading * 180 / math.pi )
                   target_turtle.showturtle()

                   #Draw the student estimate of the glider
                   if estimate_turtle is not None:
                       estimate_turtle.hideturtle()

                   if estimate_turtle is None:
                       estimate_turtle = turtle.Turtle()
                       estimate_turtle.shape("circle")
                       estimate_turtle.shapesize(0.5,0.5)
                       estimate_turtle.fill = False
                       estimate_turtle.color("purple")
                       estimate_turtle.penup()

                   estimate_turtle.setposition(estimate[0], estimate[1])
                   estimate_turtle.showturtle()

                   s.update()

 
                separation = self.distance(estimate, target_pos)
                if separation < tolerance:
                    self.glider_found.put(True)
                    self.glider_steps.put(steps)
                    return

                steps += 1

                if VERBOSE == True:
                   actual_height = target.z
                   ground_height = ourMapFunc(target.x, target.y)
                   actual_dist_to_ground = actual_height - ground_height 
                   print("\nStep: {} Actual ({})  predicted: ({})\n  Difference = {}\n  Height={}, Ground Height = {} Dist To Ground = {}".format( steps, target_pos, estimate, separation, actual_height, ground_height, actual_dist_to_ground)) 
                   if extra_points != None and len(extra_points) > 0:
                      particle_dist = []
                      for p in extra_points:
                        dist = self.distance(p,target_pos)
                        particle_dist.append(dist)
                      pMin = min(particle_dist)
                      pMax = max(particle_dist)
                      pAvg = sum(particle_dist) / float( len(particle_dist))
                      print("{} Particles, Min dist: {}, Avg dist: {}, Max Dist: {}".format(len(extra_points), pMin, pAvg, pMax))


            self.glider_found.put(False)
            self.glider_steps.put(steps)

        except:
            self.glider_error.put(traceback.format_exc())

    def simulate_with_steering(self, next_angle, params):
        """Run simulation to allow glider to be steered.

        Args:
            next_angle(func): Student submission function for gliders next turn angle.
            params(dict): Test parameters.

        Raises:
            Exception if error running submission.
        """
        self._reset()
        
        #make the test somewhat repeatable by seeding the RNG.
        random.seed(params['map_seed'])

        ourMapFunc = getMapFunc( params['map_seed'], params['map_freq'])
        #Student function is separate, so they can mess it up if they want.
        studentMapFunc = getMapFunc( params['map_seed'], params['map_freq'] )

        target = glider.glider(params['target_x'],
                               params['target_y'],
                               5000 + random.randint(-50,50), #Altitude
                               params['target_heading'],
                               ourMapFunc)
        target.set_noise(params['measurement_noise'],
                         params['altitude_offset'],
                         params['turning_noise'] )

        other_info = None
        steps = 0
        tolerance = 10.0	#Double tolerance for steering task.



        # Set up the plotter if requested
        if PLOT_PARTICLES == True:
           import turtle
           s = turtle.Screen()
           s.tracer(0,1)
           turtle.setup(width=800,height=800)
           turtle.setworldcoordinates(-WINDOW_SIZE,-WINDOW_SIZE,WINDOW_SIZE,WINDOW_SIZE)

           #Draw a cross at (0,0)
           turtle.penup()
           turtle.setposition(0,-5)
           turtle.pendown()
           turtle.setposition(0,5)
           turtle.penup()
           turtle.setposition(-5,0)
           turtle.pendown()
           turtle.setposition(5,0)
           turtle.penup()


           turtle.setposition(target.x, target.y)
           turtle.pendown()
           turtle.shape("triangle")
           turtle.shapesize(0.2, 0.4)
           turtle.settiltangle(target.heading * 180 / math.pi )
           turtle.pencolor("red")

           turtleList = []

 
        try:
            while steps < params['max_steps']:
                target_meas = target.sense()
                target_height = target.get_height()
                result = next_angle(target_height, target_meas, studentMapFunc, other_info)
                if len(result) == 3:
                  steering, other_info, extra_points = result
                elif len(result) == 2:
                    steering, other_info = result
                    extra_points = None
                else:
                    print("next_angle did not return 2 or 3 return values!") 

                steering = max( -PI/8.0, steering)
                steering = min( steering, PI/8.0)

                target.glide(steering)

                target_pos = (target.x, target.y)
                separation = self.distance( (0,0) , target_pos)

                if PLOT_PARTICLES == True:

                   if extra_points != None:
                      #Add turtles if needed.
                      while len(extra_points) > len(turtleList):
                         newTurtle = turtle.Turtle()
                         newTurtle.penup()
                         turtleList.append( newTurtle )

                      #remove turtles if needed.
                      while len(extra_points) < len(turtleList):
                         turtleList[-1].hideturtle()
                         turtleList = turtleList[0:-1]


                      # Draw the particles, set angle and position of each turtle.
                      for i in range( len(extra_points)) :
                         t = turtleList[i]
                         p = extra_points[i]
                         if len(p) > 2:
                             t.shape("triangle")
                             t.shapesize(0.2, 0.4)
                             t.settiltangle( p[2] * 180 / math.pi)
                         else:
                             t.shape("circle")
                             t.shapesize(0.1, 0.1)
                         t.setposition(p[0],p[1])
                         
                         
                   #Always show the actual glider.   
                   turtle.setposition(target.x, target.y)
                   turtle.settiltangle(target.heading * 180 / math.pi)
                  
                   s = turtle.Screen()
                   s.update()
                  
                if VERBOSE == True:
                   actual_height = target.z
                   ground_height = ourMapFunc(target.x, target.y)
                   actual_dist_to_ground = actual_height - ground_height
                   print("Step: {} Actual Heading ({}) \n  Dist To (0,0) = {}\n ".format(steps, target.heading, separation ))

                if separation < tolerance:
                    self.glider_found.put(True)
                    self.glider_steps.put(steps)
                    return


                steps += 1

            self.glider_found.put(False)
            self.glider_steps.put(steps)

        except:
            self.glider_error.put(traceback.format_exc())


NOT_FOUND = "Part {} - Test Case {}: glider took {} step(s) which exceeded the {} allowable step(s)."


class CaseRunner(unittest.TestCase):
    """Run test case using specified parameters.

    Attributes:
        simulator(GliderSimulator): Simulation.
    """
    @classmethod
    def setUpClass(cls):
        """Setup test class.
        """
        cls.simulator = GliderSimulator()

    def run_with_params(self, k, test_params, test_method, student_method):
        """Run test case with parameters.

        Args:
            k(int): Test case global parameters.
            test_params(dict): Test parameters.
            test_method(func): Test function.
            student_method(func): Student submission function.
        """
        test_params.update(GLOBAL_PARAMETERS[k])

        error_message = ''
        steps = None
        glider_found = False

        if DEBUGGING_SINGLE_PROCESS:
            test_method( student_method, test_params )
        else:
            test_process = mproc.Process(target=test_method, args=(student_method, test_params))

            try:
                test_process.start()
                test_process.join(TIME_LIMIT)
            except Exception as exp:
                error_message += str(exp) + ' '

            if test_process.is_alive():
                test_process.terminate()
                error_message = ('Test aborted due to timeout. ' +
                                 'Test was expected to finish in fewer than {} second(s).'.format(TIME_LIMIT))

        if not error_message:
            if not self.simulator.glider_error.empty():
                error_message += self.simulator.glider_error.get()

            if not self.simulator.glider_found.empty():
                glider_found = self.simulator.glider_found.get()

            if not self.simulator.glider_steps.empty():
                steps = self.simulator.glider_steps.get()

        self.assertFalse(error_message, error_message)
        self.assertTrue(glider_found, NOT_FOUND.format(test_params['part'],
                                                      test_params['test_case'],
                                                      steps,
                                                      test_params['max_steps']))



class PartATestCase(CaseRunner):
    """Test Part A (localization only, no steering)

    Attributes:
        test_method(func): Test function.
        student_method(func): Student submission function.
        params(dict): Test parameters.
    """
    def setUp(self):
        """Setup for each test case.
        """

        if marsglider1Exc:
            raise marsglider1Exc

        self.test_method = self.simulator.simulate_without_steering
        self.student_method = marsglider.estimate_next_pos

        self.params = dict()
        self.params['part'] = 'A'

    def test_case01(self):
        self.run_with_params(1, self.params, self.test_method, self.student_method)

    def test_case02(self):
        self.run_with_params(2, self.params, self.test_method, self.student_method)

    def test_case03(self):
        self.run_with_params(3, self.params, self.test_method, self.student_method)

    def test_case04(self):
        self.run_with_params(4, self.params, self.test_method, self.student_method)

    def test_case05(self):
        self.run_with_params(5, self.params, self.test_method, self.student_method)

    def test_case06(self):
        self.run_with_params(6, self.params, self.test_method, self.student_method)

    def test_case07(self):
        self.run_with_params(7, self.params, self.test_method, self.student_method)

    def test_case08(self):
        self.run_with_params(8, self.params, self.test_method, self.student_method)

    def test_case09(self):
        self.run_with_params(9, self.params, self.test_method, self.student_method)

    def test_case10(self):
        self.run_with_params(10, self.params, self.test_method, self.student_method)

class PartBTestCase(CaseRunner):
    """Test Part B (localization and steering back to (0,0) )

    Attributes:
        test_method(func): Test function.
        student_method(func): Student submission function.
        params(dict): Test parameters.
    """


    def setUp(self):
        """Setup for each test case.
        """

        if marsglider1Exc:
            raise marsglider1Exc

        self.test_method = self.simulator.simulate_with_steering
        self.student_method = marsglider.next_angle

        self.params = dict()
        self.params['part'] = 'B'

    def test_case01(self):
        self.run_with_params(1, self.params, self.test_method, self.student_method)

    def test_case02(self):
        self.run_with_params(2, self.params, self.test_method, self.student_method)

    def test_case03(self):
        self.run_with_params(3, self.params, self.test_method, self.student_method)

    def test_case04(self):
        self.run_with_params(4, self.params, self.test_method, self.student_method)

    def test_case05(self):
        self.run_with_params(5, self.params, self.test_method, self.student_method)

    def test_case06(self):
        self.run_with_params(6, self.params, self.test_method, self.student_method)

    def test_case07(self):
        self.run_with_params(7, self.params, self.test_method, self.student_method)

    def test_case08(self):
        self.run_with_params(8, self.params, self.test_method, self.student_method)

    def test_case09(self):
        self.run_with_params(9, self.params, self.test_method, self.student_method)

    def test_case10(self):
        self.run_with_params(10, self.params, self.test_method, self.student_method)



# Only run all of the test automatically if this file was executed from the command line.
# Otherwise, let Nose/py.test do it's own thing with the test cases.
if __name__ == "__main__":
    cases = []
    if PART_A is True: cases.append(PartATestCase)
    if PART_B is True: cases.append(PartBTestCase)
    suites = [unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(case)) for case in cases]

    total_passes = 0

    for i, suite in zip(list(range(1, 1+len(suites))), suites):
        print("====================\nTests for Part {}:".format(i))

        result = unittest.TestResult()
        suite.run(result)

        for x in result.errors:
            print(x[0], x[1])
        for x in result.failures:
            print(x[0], x[1])

        num_errors = len(result.errors)
        num_fails = len(result.failures)
        num_passes = result.testsRun - num_errors - num_fails
        total_passes += num_passes

        print("Successes: {}\nFailures: {}\n".format(num_passes, num_errors + num_fails))

    #We cap the maximum score to 101 if they pass more than 12.5 test cases. 
    overall_score = total_passes * CREDIT_PER_PASS
    if overall_score > 100:
       print("Score above 100:", overall_score, " capped to 101!")
       overall_score = 101
       
    print("====================\nOverall Score: {}".format(overall_score))
