#!/usr/bin/python

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

import unittest
import numpy as np
import random
import math
import time
import traceback
import hashlib
import copy
import string

try:
  import gem_finder
  studentExc=None
except Exception as e:
  studentExc=e
import robot



PI = math.pi

########################################################################
# for debugging set the time limit to a big number
########################################################################
TIME_LIMIT = 5  # seconds

########################################################################
# set to True for lots-o-output, also passed into robot under test
########################################################################
VERBOSE_FLAG = False

########################################################################
# set to True to disable multiprocessing while running in a debugger
########################################################################
DEBUGGING_SINGLE_PROCESS = True

########################################################################
# TODO: you can set NOISE_FLAG to false during development
# but be sure to run and test with noise = True
# before submitting your solution.
########################################################################
NOISE_FLAG = True
NOISE_MOVE = 0.01


if DEBUGGING_SINGLE_PROCESS:
    import multiprocessing.dummy as mproc
else:
    import multiprocessing as mproc

########################################################################
# used to generate unique ids for landmarks.  will change for grader
########################################################################
HASH_SEED = 'some_seed'


PART_A_CREDIT = 0.40
PART_B_CREDIT = 0.30
PART_C_CREDIT = 0.30

# DO NOT REMOVE THESE VARIABLES.
PART_A_SCORE = None
PART_B_SCORE = None
PART_C_SCORE = None


class Submission:
    """Student Submission.

    Attributes:
        submission_action_plan(Queue): Student score of executed action plan.
        submission_error(Queue): Error messages generated during executed action plan.
        submission_reported_gem_locations(Queue): log of gem locations reported by the extract action (used for grading).
    """

    def __init__(self):

        # if DEBUGGING_SINGLE_PROCESS:
        #     import queue
        #     self.submission_action_plan = queue.Queue(1)
        #     self.submission_error = queue.Queue(1)
        #     self.submission_reported_gem_locations = queue.Queue(1)
        # else:
        self.submission_action_plan = mproc.Manager().Queue(1)
        self.submission_error = mproc.Manager().Queue(1)
        self.submission_reported_gem_locations = mproc.Manager().Queue(1)

    def _reset(self):
        """Reset submission results.
        """
        while not self.submission_action_plan.empty():
            self.submission_action_plan.get()

        while not self.submission_error.empty():
            self.submission_error.get()

        while not self.submission_reported_gem_locations.empty():
            self.submission_reported_gem_locations.get()

    def execute_student_plan(self, area_map, gem_checklist, max_distance=1.0, max_steering=PI/2.+0.01,
                             robot_distance_noise=0.05, robot_bearing_noise=0.02):
        """Execute student plan and store results in submission.

        Args:
            area_map(list(list)): the area map to test against.
            gem_checklist(list): the list of gems to extract.
            max_distance(float): maximum distance per move.
            max_steering(float): maximum steering per move.
        """
        self._reset()

        state = State(area_map,
                      gem_checklist,
                      max_distance,
                      max_steering,
                      robot_dist_noise=robot_distance_noise,
                      robot_bearing_noise=robot_bearing_noise)

        if VERBOSE_FLAG:
            print('Initial State:')
            print(state)

        try:
            student_planner = gem_finder.GemExtractionPlanner(max_distance, max_steering)

            state_output = ''

            while len(state.collected_gems) < len(gem_checklist):
                state_output += str(state)
                ret = student_planner.next_move(copy.deepcopy(state.gem_checklist),
                                                state.generate_measurements())
                if isinstance(ret, str):
                    action = ret
                else:
                    action,locs = ret

                state.update_according_to(action)
                if VERBOSE_FLAG:
                    print(state)
            if VERBOSE_FLAG:
                print('Final State:')
                print(state)

            self.submission_action_plan.put(state.collected_gems)
            self.submission_reported_gem_locations.put(state.reported_gem_locations)

        except Exception as exc:
            self.submission_error.put(traceback.format_exc())
            self.submission_action_plan.put([])

class State:
    """Current State.

    Args:
        area_map(list(list)): the area map.
        gem_checklist(list):  the list of gems you need to collect
        max_distance(float):  the max distance the robot can travel in a single move.
        max_steering(float):  the max steering angle the robot can turn in a single move.

    Attributes:
        gem_checklist(list):   the list of needed gems
        collected_gems(list):  gems successfully extracted.
        max_distance(float):   max distance the robot can travel in one move.
        max_steering(float):   the max steering angle the robot can turn in a single move.
        _start_position(dict): location of initial robot placement
    """
    EXTRACTION_DISTANCE = 0.25
    WAIT_PENALTY = 0.1  # seconds

    def __init__(self,
                 area_map,
                 gem_checklist=[],
                 max_distance=1.0,
                 max_steering=PI/2.+0.01,
                 robot_dist_noise=0.05,
                 robot_bearing_noise=0.02):

        self.orig_gem_checklist = list(gem_checklist)
        self.gem_checklist = list(gem_checklist)
        self.collected_gems = []
        self.max_distance = max_distance
        self.max_steering = max_steering
        self.gem_locs_on_map = []
        self.reported_gem_locations = {}

        rows = len(area_map)
        cols = len(area_map[0])

        self._start_position = dict()

        # Now process the interior of the provided map
        for i in range(rows):
            for j in range(cols):
                this_square = area_map[i][j]
                x, y = float(j), -float(i)

                # Process gems
                if this_square in string.ascii_uppercase:
                    gem = {
                        'x'     : x,
                        'y'     : y,
                        'type'  : this_square,
                    }

                    self.gem_locs_on_map.append(gem)

                # Process start
                elif this_square == '@':
                    self._start_position['x'] = x + 0.5
                    self._start_position['y'] = y - 0.5

        # initialize the robot at the start position and at a steering pointing due east
        self.robot = robot.Robot(x=self._start_position['x'],
                                 y=self._start_position['y'],
                                 bearing=0.0,
                                 max_distance=self.max_distance,
                                 max_steering=self.max_steering,
                                 dist_noise=robot_dist_noise,
                                 bearing_noise=robot_bearing_noise,
                                 )

    def generate_measurements(self, noise=NOISE_FLAG):
        """Generate measurements of gems on map.

        Args:
            noise(bool): Move with noise if True.
                Default: NOISE_FLAG

        Returns:
            Measurements to gems in the format:
                {'unique gem id':{'distance': 0.0, 'steering': 0.0, 'type': 'A'}, ...}
        """
        measurements = dict()

        # process gems
        for location in self.gem_locs_on_map:
            distance, bearing = self.robot.measure_distance_and_bearing_to((location['x'], location['y']), noise=noise)
            measurements[int(hashlib.md5((str(location) + HASH_SEED)
                                         .encode('utf-8')).hexdigest(), 16)] = {'distance': distance,
                                                                                 'steering': bearing,
                                                                                 'type'   : location['type']}

        return measurements

    def update_according_to(self, action, noise=NOISE_FLAG):
        """Update state according to action.

        Args:
            action(str): action to execute.
            noise(bool): Move with noise if True.
                Default: NOISE_FLAG

        Raises:
            Exception: if improperly formatted action.
        """
        action = action.split()
        action_type = action[0]

        if action_type == 'move':
            steering, distance = action[1:]
            self._attempt_move(float(steering), float(distance), noise=noise)

        elif action_type == 'extract':
            gem_type = action[1]
            if len(action) == 4:               # if x,y provided (part C) then log values
                current_x = float(action[2])
                current_y = float(action[3])
                self._attempt_extraction(gem_type, current_x, current_y)
            else:
                self._attempt_extraction(gem_type)   # if x,y not provided (part B) then don't log

        else:
            # improper move format: kill test
            raise Exception('improperly formatted action: {}'.format(' '.join(action)))

    def _attempt_move(self, steering, distance, noise=NOISE_FLAG):
        """Attempt move action if valid.

        The robot may move between 0 and max_distance
        The robot may turn between -max_steering and +max_steering

        Illegal moves - the robot will not move
        - Moving a distance outside of [0,max_distance]
        - Steering angle outside [-max_steering, max_steering]

        Args:
            steering(float): Angle to turn before moving.
            distance(float): Distance to travel.

        Raises:
            ValueError: if improperly formatted move destination.
        """
        try:
            distance_ok = 0.0 <= distance <= self.max_distance
            steering_ok = (-self.max_steering) <= steering <= self.max_steering

            if noise:
                steering += random.uniform(-NOISE_MOVE, NOISE_MOVE)
                distance *= random.uniform(1.0 - NOISE_MOVE, 1.0 + NOISE_MOVE)

            if distance_ok and steering_ok:
                self.robot.move(steering, distance)

        except ValueError:
            raise Exception('improperly formatted move command : {} {}'.format(steering, distance))

    def _attempt_extraction(self, gem_type, current_x=None, current_y=None):
        """Attempt to extract a gem from the current x,y location.

        Extract gem if current location is within EXTRACTION_DISTANCE of specified gem_type.
        Otherwise, pause for WAIT_PENALTY
        """

        for gem_location in self.gem_locs_on_map:
            if gem_location['type'] == gem_type:
                distance = np.sqrt((self.robot.x - gem_location['x']) ** 2 + (self.robot.y - gem_location['y']) ** 2)
                if distance < self.EXTRACTION_DISTANCE:
                    self.collected_gems.append(gem_location)
                    self.gem_locs_on_map.remove(gem_location)
                    self.gem_checklist.remove(gem_location['type'])
                    self.reported_gem_locations[gem_type] = {
                        'x': current_x,
                        'y': current_y,
                    }
                    return

        time.sleep(self.WAIT_PENALTY)

        if VERBOSE_FLAG:
            print("*** Location ({}, {}) does not contain a gem type <{}> within the extraction distance.".format(
                self.robot.x,
                self.robot.y,
                gem_type))

    def __repr__(self):
        """Output state object as string.
        """
        output = '\n'
        output += 'Robot State:\n'
        output += '\t x = {:6.2f}, y = {:6.2f}, hdg = {:6.2f}\n'.format(self.robot.x, self.robot.y,
                                                                        self.robot.bearing * 180. / PI)
        output += 'Gems Extracted: {}\n'.format(self.collected_gems)
        output += 'Remaining Gems Needed: {}\n'.format(self.gem_checklist)

        return output

class GemFinderTestResult(unittest.TestResult):

    def __init__(self, stream=None, descriptions=None, verbosity=None):
        super(GemFinderTestResult, self).__init__(stream, verbosity, descriptions)
        self.stream = stream
        self.credit=[]
        self.results=[]

    def stopTest(self, test):
        super(GemFinderTestResult, self).stopTest(test)
        try:
            self.credit.append(test.last_credit)
            self.results.append(test.last_result)
            self.stream.write(test.last_result + '\n')
        except AttributeError as e:
            self.stream.write(str(e))

    @property
    def avg_credit(self):
        try:
            return sum(self.credit) / len(self.credit)
        except Exception as e:
            return 0.0

class PartATestCase(unittest.TestCase):
    """Test PartA
    """
    results_file = 'results_partA.txt'

    results = ['', 'PART A TEST CASE RESULTS']
    SCORE_TEMPLATE = "\n".join((
        "\n-----------",
        "Part A Test Case {test_case}",
        "  Expected Location:\t{expected}",
        "  SLAM Location:\t{location}",
        "  Credit: {score:.0%}"
    ))
    FAIL_TEMPLATE = "\n".join((
        "\n-----------",
        "Part A Test Case {test_case}",
        "  Failed: {message}",
        "  Expected Location:\t{expected}",
        "  SLAM Location:\t{location}",
        "  Credit: 0.0"
    ))

    credit = []

    def setUp(self):

        self.last_result=''
        self.last_credit=0.0

        if studentExc:
            self.last_result=str(studentExc)
            raise studentExc

    def run_with_params(self, params):
        """Run test case using desired parameters.
        Args:
            params(dict): a dictionary of test parameters.
        """

        state = State(params['area_map'],
                      robot_dist_noise=params['robot_distance_noise'],
                      robot_bearing_noise=params['robot_bearing_noise'])
        dist_error = params['test_tolerance'] * 2.0
        try:
            rover_slam = gem_finder.SLAM()

            for move in params['move']:
                meas = state.generate_measurements()
                rover_slam.process_measurements(meas)

                action = move.split()
                state.update_according_to(move)
                belief = rover_slam.process_movement(float(action[1]),
                                                     float(action[2]), NOISE_MOVE)
                truth = (state.robot.x - state._start_position['x'],
                         state.robot.y - state._start_position['y'])

                dist_error = robot.compute_distance(belief, truth)
                if VERBOSE_FLAG:
                    print("Current Belief:", belief)
                    print("True Position:", truth)
                    print("Error:", dist_error, "\n")
        except Exception as exc:
            self.last_result = self.FAIL_TEMPLATE.format(message=traceback.format_exc(),
                                                         expected="exception",
                                                         location="exception",
                                                         **params)
            self.last_credit = 0.0
            self.fail(str(exc))

        if dist_error < params['test_tolerance']:
            result = self.SCORE_TEMPLATE.format(expected=truth,
                                                location=belief,
                                                score=1.0, **params)
            score = 1.0
        else:
            result = self.FAIL_TEMPLATE.format(message='Distance greater than tolerance {}'.format(params['test_tolerance']),
                                               expected=truth,
                                               location=belief, **params)
            score = 0.0

        self.last_result = result
        self.last_credit = score

        self.assertTrue(dist_error < params['test_tolerance'],
                        'Location error {} as a distance must be less than {}'.format(dist_error,
                                                                                      params['test_tolerance']))

    def test_case1(self):
        params = {'test_case': 1,
                  'area_map': ['...B..',
                               '......',
                               'A..@.C'],
                  'move': ['move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0'],
                  'test_tolerance': 0.1,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                  }

        self.run_with_params(params)

    def test_case2(self):
        params = {'test_case': 2,
                  'area_map': ['AB.DE..A',
                               'M......K',
                               'N......L',
                               'O@......',
                               'F.HIJ.TR'],
                  'move': ['move 0.0 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0',
                           'move 1.570963 1.0'],
                  'test_tolerance': 0.1,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)

    def test_case3(self):
        params = {'test_case': 3,
                  'area_map': ['.QRHID.V',
                               '....G..X',
                               '.A..C..Y',
                               '...V.@.Z',
                               'O.B.FBCM'],
                  'move': ['move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0'],
                  'test_tolerance': 0.1,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                  }

        self.run_with_params(params)

    def test_case4(self):
        params = {'test_case': 4,
                  'area_map': ['........',
                               '...L....',
                               '..A.....',
                               '........',
                               '...D....',
                               '.@......',
                               '......M.'],
                  'move': ['move 1.570963 0.0',
                           'move 1.570963 0.0',
                           'move 0.785481 1.0',
                           'move 0.0 0.5',
                           'move 0.0 0.5',
                           'move 1.0 0.5',
                           'move 0.0 0.5'],
                  'test_tolerance': 0.1,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                  }

        self.run_with_params(params)

    def test_case5(self):
        params = {'test_case': 5,
                  'area_map': ['J.A.MNOC',
                               'D@..L..P',
                               'G.B....K',
                               'H.....Q.',
                               'I.E.R...'],
                  'move': ['move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0'],
                  'test_tolerance': 0.15,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                  }

        self.run_with_params(params)

    def test_case6(self):
        params = {'test_case': 6,
                  'area_map': ['A......DEFG',
                               '..........J',
                               'Z..........',
			                   'Y..........',
                               'X.@........',
                               'W.........N',
                               'STUV...OPQR'],
                  'move': ['move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.2 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.2 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move -1.570963 1.0',
                           'move 0.0 1.0',
                           'move 0.0 1.0'],
                  'test_tolerance': 0.15,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                  }

        self.run_with_params(params)


class PartBTestCase(unittest.TestCase):
    """ Test PartB.
    """
    results_file = 'results_partB.txt'

    results = ['', 'PART B TEST CASE RESULTS']
    SCORE_TEMPLATE = "\n".join((
        "\n-----------",
        "Part B Test Case {test_case}",
        "  Needed Gems:\t {needed_gems}",
        "  Collected Gems:{collected}",
        "  Credit: {score:.0%}"
    ))
    FAIL_TEMPLATE = "\n".join((
        "\n-----------",
        "Part B Test Case {test_case}",
        "  Failed: {message}",
        "  Credit: 0.0"
    ))

    credit = []

    def setUp(self):
        """Initialize test setup.
        """
        self.last_result=''
        self.last_credit=0.0
        if studentExc:
            self.last_result=str(studentExc)
            raise studentExc
        self.student_submission = Submission()


    def check_results(self, params, error_message):

        extracted_gems = 0
        score = 0.0

        # Get number of gems collected
        if not self.student_submission.submission_action_plan.empty():
            extracted_gems = self.student_submission.submission_action_plan.get()

        extracted_gem_types = sorted([g['type'] for g in extracted_gems])
        extracted_gems_actually_needed = sorted(list(set(extracted_gem_types).intersection(set(params['needed_gems']))))
        score = len(extracted_gems_actually_needed) / float(len(params['needed_gems']))

        if not self.student_submission.submission_error.empty():
            error_message = self.student_submission.submission_error.get()
            result = self.FAIL_TEMPLATE.format(message=error_message, **params)
        else:
            result = self.SCORE_TEMPLATE.format(collected=extracted_gem_types, score=score, **params)

        return result, score, error_message, extracted_gems_actually_needed


    def run_with_params(self, params):
        """Run test case using desired parameters.
        Args:
            params(dict): a dictionary of test parameters.
        """

        error_message = ''
        score = 0.0

        if DEBUGGING_SINGLE_PROCESS:
            try:
                self.student_submission.execute_student_plan(params['area_map'],
                                                             params['needed_gems'],
                                                             params['max_distance'],
                                                             params['max_steering'],
                                                             params['robot_distance_noise'],
                                                             params['robot_bearing_noise'],
                                                             )
            except Exception as exp:
                error_message = exp.message

            result,score,error_message, extracted_gems_actually_needed = self.check_results( params, error_message )

        else:
            test_process = mproc.Process(target=self.student_submission.execute_student_plan,
                                     args=(params['area_map'],
                                           params['needed_gems'],
                                           params['max_distance'],
                                           params['max_steering'],
                                           params['robot_distance_noise'],
                                           params['robot_bearing_noise'],
                                           ))

            try:
                test_process.start()
                test_process.join(TIME_LIMIT)
            except Exception as exp:
                error_message = exp.message

            # If test still running then terminate
            if test_process.is_alive():
                test_process.terminate()
                error_message = ('Test aborted due to timeout. ' +
                                'Test was expected to finish in fewer than {} second(s).'.format(TIME_LIMIT))
                result = self.FAIL_TEMPLATE.format(message=error_message, **params)
                score = 0.0
            else:
                result,score,error_message, extracted_gems_actually_needed  = self.check_results( params, error_message )

        self.last_result = result
        self.last_credit = score

        self.assertFalse(error_message, error_message)
        self.assertTrue(round(score, 7) == 1.0,
                        "Only {} gems were extracted out of the {} requested types.".format(len(extracted_gems_actually_needed),len(params['needed_gems'])))



    def test_case1(self):
        params = {'test_case': 1,
                  'area_map': ['....',
                               '..@A',
                               '..B.'],
                  'needed_gems': list('A'),
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                  }

        self.run_with_params(params)

    def test_case2(self):
        params = {'test_case': 2,
                  'area_map': ['AXCZ...S',
                               'M.....@.',
                               'O...V..R',
                               '.B..FGH.',
                               'T......E'],
                  'needed_gems': list('ABC'),
                  'horizon_distance': 4.0,
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                  }

        self.run_with_params(params)

    def test_case3(self):
        params = {'test_case': 3,
                  'area_map': ['A......B',
                               '.F..@...',
                               '.E......',
                               'C......D',],
                  'needed_gems': list('ABCD'),
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                  }

        self.run_with_params(params)

    def test_case4(self):
        params = {'test_case': 4,
                  'area_map': ['O....A...',
                               '.@....C..',
                               '........M',
                               'PB......N'],
                  'needed_gems': list('ABC'),
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                  }

        self.run_with_params(params)

    def test_case5(self):
        params = {'test_case': 5,
                  'area_map': ['ABCDEF',
                               'GHIJKL',
                               'M.@PQR',
                               'S.UVWX',
                               '..YNOT'],
                  'needed_gems': list('ABCDEFGHIJKLMNOPQRSTUVWXY'),
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                  }

        self.run_with_params(params)

    def test_case6(self):
        params = {'test_case': 6,
                  'area_map':  ['..TOP.S',
                                '......I',
                                '......D',
                                '...@..E',
                                '.......'],
                  'needed_gems': list('OPT'),
                  'max_distance': 1.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                  }

        self.run_with_params(params)

    def test_case7(self):
        params = {'test_case': 7,
                  'area_map': ['.......C..........................................',
                               '..............................O...................',
                               '....R..........................................O..',
                               '...............N.........................A........',
                               'S..............................E..................',
                               '.............L.................................F..',
                               '....I..................S................O.........',
                               '.............L..................A.................',
                               '......T..................I....................O...',
                               'N......................@..........................',
                               '..................................................'],
                  'needed_gems': list('CFIRT'),
                  'max_distance': 3.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)

    def test_case8(self):
        params = {'test_case': 8,
                  'area_map': ['ABCDEFG',
                               'QR.@.HI',
                               'JKLMNOP'],
                  'needed_gems': list('ACEGIK'),
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)

    def test_case9(self):
        params = {'test_case': 9,
                  'area_map': ['@A....F',
                               '.....MP',
                               'G....ZD'],
                  'needed_gems': list('ADM'),
                  'max_distance': 1.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)


class PartCTestCase(unittest.TestCase):
    """ Test PartC.
    """
    results_file = 'results_partC.txt'

    results = ['', 'PART C TEST CASE RESULTS']
    SCORE_TEMPLATE = "\n".join((
        "\n-----------",
        "Part C Test Case {test_case}",
        "  Needed Gems:\t {needed_gems}",
        "  Collected Gems:{collected}",
        "  Total Error: {total_error}",
        "  Error Thresh: {thresh}",
        "  Credit: {score:.0%}"
    ))
    FAIL_TEMPLATE = "\n".join((
        "\n-----------",
        "Part C Test Case {test_case}",
        "  Failed: {message}",
        "  Credit: 0.0"
    ))

    credit = []

    def setUp(self):
        """Initialize test setup.
        """
        self.last_result=''
        self.last_credit=0.0
        if studentExc:
            self.last_result=str(studentExc)
            raise studentExc
        self.student_submission = Submission()


    def check_results(self, params, error_message):

        extracted_gems = 0
        score = 0.0

        state = State(params['area_map'])

        # Get number of gems collected
        if not self.student_submission.submission_action_plan.empty():
            extracted_gems = self.student_submission.submission_action_plan.get()

        extracted_gem_types = sorted([g['type'] for g in extracted_gems])
        extracted_gems_actually_needed = set(extracted_gem_types).intersection(set(params['needed_gems']))

        reported_gem_locations = {}
        if not self.student_submission.submission_reported_gem_locations.empty():
            reported_gem_locations = self.student_submission.submission_reported_gem_locations.get()

        all_collected = len(extracted_gems_actually_needed) == float(len(params['needed_gems']))
        if not all_collected:
            score = 0
            self.assertTrue(round(score, 7) == 1.0,
                    "Only {} gems were extracted out of the {} requested types.".format(len(extracted_gems_actually_needed),
                                                                            len(params['needed_gems'])))
        else:
            map_errors = []
            for gem in reported_gem_locations:

                reported_relative_x = reported_gem_locations[gem]['x']
                reported_relative_y = reported_gem_locations[gem]['y']
                reported_absolute_x = state._start_position['x'] + reported_relative_x
                reported_absolute_y = state._start_position['y'] + reported_relative_y

                gem_actual_location = list(filter(lambda g: g['type'] == gem, state.gem_locs_on_map))[0]
                actual_x = gem_actual_location['x']
                actual_y = gem_actual_location['y']

                error_dist = robot.compute_distance((reported_absolute_x, reported_absolute_y), (actual_x, actual_y))

                map_errors.append(error_dist)

            total_dist_error = sum(map_errors)

            kindness_factor = 3   # the larger this value the easier it is to get points in part C
            num_gems = len(extracted_gems)
            thresh = num_gems * State.EXTRACTION_DISTANCE * kindness_factor
            score = 1 if total_dist_error < thresh else 0

        if not self.student_submission.submission_error.empty():
            error_message = self.student_submission.submission_error.get()
            result = self.FAIL_TEMPLATE.format(message=error_message, **params)
        else:
            result = self.SCORE_TEMPLATE.format(collected=extracted_gem_types,
                                                total_error=round(total_dist_error,2),
                                                thresh = round(thresh,2),
                                                score=score, **params)

        return result, score, error_message



    def run_with_params(self, params):
        """Run test case using desired parameters.
        Args:
            params(dict): a dictionary of test parameters.
        """

        error_message = ''
        score = 0.0

        if DEBUGGING_SINGLE_PROCESS:
            try:
                self.student_submission.execute_student_plan(params['area_map'],
                                                             params['needed_gems'],
                                                             params['max_distance'],
                                                             params['max_steering'],
                                                             params['robot_distance_noise'],
                                                             params['robot_bearing_noise'],
                                                             )
            except Exception as exp:
                error_message = exp.message

            result, score, error_message = self.check_results(params, error_message)

        else:
            test_process = mproc.Process(target=self.student_submission.execute_student_plan,
                                     args=(params['area_map'],
                                           params['needed_gems'],
                                           params['max_distance'],
                                           params['max_steering'],
                                           params['robot_distance_noise'],
                                           params['robot_bearing_noise'],
                                           ))

            try:
                test_process.start()
                test_process.join(TIME_LIMIT)
            except Exception as exp:
                error_message = exp.message

            # If test still running then terminate
            if test_process.is_alive():
                test_process.terminate()
                error_message = ('Test aborted due to timeout. ' +
                                'Test was expected to finish in fewer than {} second(s).'.format(TIME_LIMIT))
                result = self.FAIL_TEMPLATE.format(message=error_message, **params)
                score = 0.0
            else:
                result, score, error_message = self.check_results(params, error_message)


        self.last_result = result
        self.last_credit = score

        self.assertFalse(error_message, error_message)
        self.assertTrue(round(score, 7) == 1.0, "Reported locations of gems not within threshold of actual locations.")


    def test_case1(self):
        params = {'test_case': 1,
                  'area_map': ['....',
                               '..@A',
                               '..B.'],
                  'needed_gems': list('A'),
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)

    def test_case2(self):
        params = {'test_case': 2,
                  'area_map': ['AXCZ...S',
                               'M.....@.',
                               'O...V..R',
                               '.B..FGH.',
                               'T......E'],
                  'needed_gems': list('ABC'),
                  'horizon_distance': 4.0,
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01,
                   'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)

    def test_case3(self):
        params = {'test_case': 3,
                  'area_map': ['A......B',
                               '.F..@..G',
                               '.E.....H',
                               'C...I..D',],
                  'needed_gems': list('ABCD'),
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)

    def test_case4(self):
        params = {'test_case': 4,
                  'area_map': ['O....A...',
                               '.@....C..',
                               '........M',
                               'PB......N'],
                  'needed_gems': list('ABC'),
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)

    def test_case5(self):
        params = {'test_case': 5,
                  'area_map': ['ABCDEF',
                               'GHIJKL',
                               'M.@PQR',
                               'S.UVWX',
                               '..YNOT'],
                  'needed_gems': list('ABCDEFGHIJKLMNOPQRSTUVWXY'),
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)

    def test_case6(self):
        params = {'test_case': 6,
                  'area_map':  ['..TOP.S',
                                '......I',
                                '......D',
                                '...@..E',
                                '.......'],
                  'needed_gems': list('OPT'),
                  'max_distance': 1.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)

    def test_case7(self):
        params = {'test_case': 7,
                  'area_map': ['...C.....O....',
                               '....R......O..',
                               '..N....A......',
                               'S...E.........',
                               '.......L...F..',
                               '..I....S..O...',
                               '....L.....A...',
                               '.....T...I..O.',
                               'N.......@.....',
                               'ZZZZZZZZZZZZZZ'],
                  'needed_gems': list('EFT'),
                  'max_distance': 3.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)

    def test_case8(self):
        params = {'test_case': 8,
                  'area_map': ['ABCDEFG',
                               'QR.@.HI',
                               'JKLMNOP'],
                  'needed_gems': list('ACEGIK'),
                  'max_distance': 2.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)

    def test_case9(self):
        params = {'test_case': 9,
                  'area_map': ['@A....F',
                               '.....MP',
                               'G....ZD'],
                  'needed_gems': list('ADM'),
                  'max_distance': 1.0,
                  'max_steering': PI / 2. + 0.01,
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,}

        self.run_with_params(params)


def run_all(stream):

    suites = map(lambda case: unittest.TestSuite(unittest.TestLoader().loadTestsFromTestCase(case)),
                 [PartATestCase,
                 PartBTestCase,
                 PartCTestCase])

    avgs = ()
    for suite in suites:
        result = GemFinderTestResult(stream=stream)
        suite.run(result)
        avgs += (result.avg_credit,)

    stream.write('part A score: %.02f\n' % (avgs[0] * 100) )
    stream.write('part B score: %.02f\n' % (avgs[1] * 100) )
    stream.write('part C score: %.02f\n' % (avgs[2] * 100) )

    weights = (PART_A_CREDIT, PART_B_CREDIT, PART_C_CREDIT)
    total_score = round(sum(avgs[i] * weights[i] for i in (0,1,2)) * 100)
    stream.write('score: %.02f\n' % total_score)
      
# Only run all of the test automatically if this file was executed from the command line.
# Otherwise, let Nose/py.test do it's own thing with the test cases.
if __name__ == "__main__":
    import sys
    run_all(sys.stdout)


