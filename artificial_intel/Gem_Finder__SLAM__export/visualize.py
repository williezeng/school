
import argparse
import copy
import math
import time
import turtle

import testing_suite_gem_finder
import gem_finder

PI = math.pi

# In future semesters,
# share these test cases between files
# rather than copy-pasting

CASES = { 'A': { 1: {'test_case': 1,
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
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                     'test_tolerance': 0.1},
                 2: {'test_case': 2,
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
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                     'test_tolerance': 0.1},
                 3: {'test_case': 3,
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
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                     'test_tolerance': 0.1},
                 4: {'test_case': 4,
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
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                     'test_tolerance': 0.1},
                 5: {'test_case': 5,
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
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                     'test_tolerance': 0.15},
                 6: {'test_case': 6,
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
                  'robot_distance_noise': 0.05,
                  'robot_bearing_noise': 0.02,
                     'test_tolerance': 0.15}},
          'B': { 1: {'test_case': 1,
                     'area_map': ['....',
                                  '..@A',
                                  '..B.'],
                     'needed_gems': list('A'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 2.0,
                     'max_steering': PI / 2. + 0.01},
                 2:  {'test_case': 2,
                      'area_map': ['AXCZ...S',
                                   'M.....@.',
                                   'O...V..R',
                                   '.B..FGH.',
                                   'T......E'],
                      'needed_gems': list('ABC'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                      'horizon_distance': 4.0,
                      'max_distance': 2.0,
                      'max_steering': PI / 2. + 0.01},
                 3: {'test_case': 3,
                     'area_map': ['A......B',
                                  '.F..@...',
                                  '.E......',
                                  'C......D',],
                     'needed_gems': list('ABCD'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 2.0,
                     'max_steering': PI / 2. + 0.01},
                 4: {'test_case': 4,
                     'area_map': ['O....A...',
                                  '.@....C..',
                                  '........M',
                                  'PB......N'],
                     'needed_gems': list('ABC'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 2.0,
                     'max_steering': PI / 2. + 0.01},
                 5: {'test_case': 5,
                     'area_map': ['ABCDEF',
                                  'GHIJKL',
                                  'M.@PQR',
                                  'S.UVWX',
                                  '..YNOT'],
                     'needed_gems': list('ABCDEFGHIJKLMNOPQRSTUVWXY'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 2.0,
                     'max_steering': PI / 2. + 0.01},
                 6: {'test_case': 6,
                     'area_map':  ['..TOP.S',
                                   '......I',
                                   '......D',
                                   '...@..E',
                                   '.......'],
                     'needed_gems': list('OPT'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 1.0,
                     'max_steering': PI / 2. + 0.01},
                 7: {'test_case': 7,
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
                     'max_steering': PI / 2. + 0.01},
                 8: {'test_case': 8,
                     'area_map': ['ABCDEFG',
                                  'QR.@.HI',
                                  'JKLMNOP'],
                     'needed_gems': list('ACEGIK'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 2.0,
                     'max_steering': PI / 2. + 0.01},
                 9: {'test_case': 9,
                     'area_map': ['@A....F',
                                  '.....MP',
                                  'G....ZD'],
                     'needed_gems': list('ADM'),
                     'max_distance': 1.0,
                     'max_steering': PI / 2. + 0.01}},
          'C': { 1: {'test_case': 1,
                     'area_map': ['....',
                                  '..@A',
                                  '..B.'],
                     'needed_gems': list('A'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 2.0,
                     'max_steering': PI / 2. + 0.01},
                 2: {'test_case': 2,
                     'area_map': ['AXCZ...S',
                                  'M.....@.',
                                  'O...V..R',
                                  '.B..FGH.',
                                  'T......E'],
                     'needed_gems': list('ABC'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'horizon_distance': 4.0,
                     'max_distance': 2.0,
                     'max_steering': PI / 2. + 0.01},
                 3: {'test_case': 3,
                     'area_map': ['A......B',
                                  '.F..@..G',
                                  '.E.....H',
                                  'C...I..D',],
                     'needed_gems': list('ABCD'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 2.0,
                     'max_steering': PI / 2. + 0.01},
                 4: {'test_case': 4,
                     'area_map': ['O....A...',
                                  '.@....C..',
                                  '........M',
                                  'PB......N'],
                     'needed_gems': list('ABC'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 2.0,
                     'max_steering': PI / 2. + 0.01},
                 5: {'test_case': 5,
                     'area_map': ['ABCDEF',
                                  'GHIJKL',
                                  'M.@PQR',
                                  'S.UVWX',
                                  '..YNOT'],
                     'needed_gems': list('ABCDEFGHIJKLMNOPQRSTUVWXY'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 2.0,
                     'max_steering': PI / 2. + 0.01},
                 6: {'test_case': 6,
                     'area_map':  ['..TOP.S',
                                   '......I',
                                   '......D',
                                   '...@..E',
                                   '.......'],
                     'needed_gems': list('OPT'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 1.0,
                     'max_steering': PI / 2. + 0.01},
                 7: {'test_case': 7,
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
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 3.0,
                     'max_steering': PI / 2. + 0.01},
                 8: {'test_case': 8,
                     'area_map': ['ABCDEFG',
                                  'QR.@.HI',
                                  'JKLMNOP'],
                     'needed_gems': list('ACEGIK'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 2.0,
                     'max_steering': PI / 2. + 0.01},
                 9: {'test_case': 9,
                     'area_map': ['@A....F',
                                  '.....MP',
                                  'G....ZD'],
                     'needed_gems': list('ADM'),
                     'robot_distance_noise': 0.05,
                     'robot_bearing_noise': 0.02,
                     'max_distance': 1.0,
                     'max_steering': PI / 2. + 0.01}} }


class TurtleDisplay(object):

    WIDTH  = 800
    HEIGHT = 800

    def __init__(self, xbounds, ybounds):
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.gem_turtles = {}
        self.gem_estimate_turtles = {}
        self.robot_turtle = None
        self.robot_estimate_turtle = None

    def setup(self):
        xmin,xmax = self.xbounds
        ymin,ymax = self.ybounds
        dx = xmax - xmin
        dy = ymax - ymin
        margin = 0.3
        turtle.setup( width  = self.WIDTH,
                      height = self.HEIGHT )
        turtle.setworldcoordinates(xmin - (dx * margin),
                                   ymin - (dy * margin),
                                   xmax + (dx * margin),
                                   ymax + (dy * margin))
        turtle.tracer(0,1)
        turtle.hideturtle()
        turtle.penup()

    def start_time_step(self):

        for gem_id,trtl in self.gem_turtles.items():
            trtl.clear()

    def _new_turtle(self, shape='circle', color='gray', shapesize=(0.1, 0.1)):

        trtl = turtle.Turtle()
        trtl.shape(shape)
        trtl.color(color)
        trtl.shapesize(*shapesize)
        trtl.penup()
        return trtl

    def gem_at_location(self, gem_id, x, y):

        key = (gem_id, x, y)
        
        if key not in self.gem_turtles:

            self.gem_turtles[key] = self._new_turtle( shape     = 'square',
                                                      color     = 'gray',
                                                      shapesize = (0.5, 0.5) )

        self.gem_turtles[key]._write( str(gem_id)[:2], 'center', 'arial' )
        self.gem_turtles[key].setposition(x, y)

    def gem_estimate_at_location(self, gem_id, x, y):

        if gem_id not in self.gem_estimate_turtles:

            trtl = turtle.Turtle()
            trtl.shape("circle")
            trtl.color("black" if gem_id != 'self' else 'red')
            trtl.shapesize(0.2,0.2)
            trtl.penup()
            self.gem_estimate_turtles[gem_id] = trtl

        self.gem_estimate_turtles[gem_id].setposition(x, y)
        
    def robot_at_location(self, x, y):

        if self.robot_turtle is None:

            self.robot_turtle = self._new_turtle( color     = 'red',
                                                  shapesize = (0.5, 0.5) )
            
        self.robot_turtle.setposition(x, y)

    def robot_estimate_at_location(self, x, y):

        if self.robot_estimate_turtle is None:

            self.robot_estimate_turtle = self._new_turtle( color     = '#cc0000',
                                                           shapesize = (0.3, 0.3) )

        self.robot_estimate_turtle.setposition(x, y)
        
    def end_time_step(self):
        turtle.update()
        time.sleep(1.0)

    def done(self):
        turtle.done()
        
def bounds(state):

    robot_init_x = state.robot.x
    robot_init_y = state.robot.y

    xs = [robot_init_x] + [ gem['x'] for gem in state.gem_locs_on_map ]
    ys = [robot_init_y] + [ gem['y'] for gem in state.gem_locs_on_map ]

    return ((min(xs),max(xs)),
            (min(ys),max(ys)))
    
def part_A( params ):

    area_map = params['area_map']
    state = testing_suite_gem_finder.State( area_map = area_map )

    robot_init_x = state.robot.x
    robot_init_y = state.robot.y

    xbounds,ybounds = bounds(state)

    display = TurtleDisplay( xbounds = xbounds,
                             ybounds = ybounds )
    display.setup()

    display.start_time_step()
    display.robot_at_location( robot_init_x, robot_init_y )
    display.end_time_step()

    try:

        rover_slam = gem_finder.SLAM()

        for move in params['move']:

            display.start_time_step()

            meas = state.generate_measurements()
            rover_slam.process_measurements(meas)

            action = move.split()
            state.update_according_to(move)
            belief_x, belief_y = rover_slam.process_movement(float(action[1]),
                                                 float(action[2]),
                                                 testing_suite_gem_finder.NOISE_MOVE)
            belief = (belief_x + robot_init_x, belief_y +robot_init_y)
            truth = (state.robot.x - state._start_position['x'] + robot_init_x,
                     state.robot.y - state._start_position['y'] + robot_init_y)

            for gem in state.gem_locs_on_map:
                display.gem_at_location( gem['type'], gem['x'], gem['y'] )

            display.robot_at_location( *truth )
            display.robot_estimate_at_location( *belief )

            display.end_time_step()

    except Exception as e:
        print(e)

    turtle.bye()

def part_B_or_C( params ):

    area_map      = params['area_map']
    gem_checklist = params['needed_gems']
    max_distance  = params['max_distance']
    max_steering  = params['max_steering']
    state = testing_suite_gem_finder.State( area_map      = area_map,
                                            gem_checklist = gem_checklist,
                                            max_distance  = max_distance,
                                            max_steering  = max_steering )
    robot_init_x = state.robot.x
    robot_init_y = state.robot.y

    xbounds,ybounds = bounds(state)

    display = TurtleDisplay( xbounds = xbounds,
                             ybounds = ybounds )
    display.setup()

    try:

        student_planner = gem_finder.GemExtractionPlanner(max_distance, max_steering)

        while len(state.collected_gems) < len(gem_checklist):

            display.start_time_step()

            for gem in state.gem_locs_on_map:
                display.gem_at_location( gem['type'], gem['x'], gem['y'] )

            display.robot_at_location(state.robot.x, state.robot.y)

            ret = student_planner.next_move(copy.deepcopy(state.gem_checklist),
                                            state.generate_measurements())
            if isinstance(ret, str):
                action = ret
            else:
                action,locs = ret
                for locid,xy in locs.items():
                    x,y = xy
                    if locid == 'self':
                        display.robot_estimate_at_location( x + robot_init_x,
                                                            y + robot_init_y )
                    else:
                        display.gem_estimate_at_location( locid,
                                                          x + robot_init_x,
                                                          y + robot_init_y )

            display.end_time_step()

            state.update_according_to(action)

        # One more time step to show final state.
            
        display.start_time_step()
        
        for gem in state.gem_locs_on_map:
            display.gem_at_location( gem['type'], gem['x'], gem['y'] )

        display.robot_at_location(state.robot.x, state.robot.y)
        display.end_time_step()

    except Exception as e:
        print(e)

    turtle.bye()

def main(part, case):

    params = CASES[part][case]

    if part == 'A':
        part_A( params )
    elif part in ('B', 'C'):
        part_B_or_C( params )

def parser():
    prsr = argparse.ArgumentParser()
    prsr.add_argument( '--part',
                       help="test part",
                       type=str,
                       choices = ('A', 'B', 'C') )

    prsr.add_argument( '--case',
                       help="test case",
                       type=int,
                       default=1,
                       choices = (1,2,3,4,5,6,7,8,9) )
    return prsr


if __name__ == '__main__':
    args = parser().parse_args()
    main( part = args.part,
          case = args.case )
