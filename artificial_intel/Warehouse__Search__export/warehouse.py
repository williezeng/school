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


import math
import numpy as np

class DeliveryPlanner_PartA:

    """
    Required methods in this class are:
    
      plan_delivery(self, debug = False) which is stubbed out below.  
        You may not change the method signature as it will be called directly 
        by the autograder but you may modify the internals as needed.
    
      __init__: which is required to initialize the class.  Starter code is 
        provided that intializes class variables based on the definitions in 
        testing_suite_partA.py.  You may choose to use this starter code
        or modify and replace it based on your own solution
    
    The following methods are starter code you may use for part A.  
    However, they are not required and can be replaced with your
    own methods.
    
      _set_initial_state_from(self, warehouse): creates structures based on
          the warehouse and todo definitions and intializes the robot
          location in the warehouse
    
      _search(self, debug=False): Where the bulk of the A* search algorithm
          could reside.  It should find an optimal path from the robot
          location to a goal.  Hint:  you may want to structure this based
          on whether looking for a box or delivering a box.
  
    """

    ## Definitions taken from testing_suite_partA.py
    ORTHOGONAL_MOVE_COST = 2
    DIAGONAL_MOVE_COST = 3
    BOX_LIFT_COST = 4
    BOX_DOWN_COST = 2
    ILLEGAL_MOVE_PENALTY = 100

    def __init__(self, warehouse, todo):
        
        self.todo = todo
        self.boxes_delivered = []
        self.total_cost = 0
        self._set_initial_state_from(warehouse)

        self.delta = [[-1, 0 ], # north
                      [ 0,-1 ], # west
                      [ 1, 0 ], # south
                      [ 0, 1 ], # east
                      [-1,-1 ], # northwest (diag)
                      [-1, 1 ], # northeast (diag)
                      [ 1, 1 ], # southeast (diag)
                      [ 1,-1 ]] # southwest (diag)

        self.delta_directions = ["n","w","s","e","nw","ne","se","sw"]

        # Can use this for a visual debug
        self.delta_name = [ '^', '<', 'v', '>','\\','/','[',']' ]

        # Costs for each move
        self.delta_cost = [  self.ORTHOGONAL_MOVE_COST, 
                             self.ORTHOGONAL_MOVE_COST, 
                             self.ORTHOGONAL_MOVE_COST, 
                             self.ORTHOGONAL_MOVE_COST,
                             self.DIAGONAL_MOVE_COST,
                             self.DIAGONAL_MOVE_COST,
                             self.DIAGONAL_MOVE_COST,
                             self.DIAGONAL_MOVE_COST ]

    ## state parsing and initialization function from testing_suite_partA.py
    def _set_initial_state_from(self, warehouse):
        """Set initial state.

        Args:
            warehouse(list(list)): the warehouse map.
        """
        rows = len(warehouse)
        cols = len(warehouse[0])

        self.warehouse_state = [[None for j in range(cols)] for i in range(rows)]
        self.dropzone = None
        self.boxes = dict()

        for i in range(rows):
            for j in range(cols):
                this_square = warehouse[i][j]

                if this_square == '.':
                    self.warehouse_state[i][j] = '.'

                elif this_square == '#':
                    self.warehouse_state[i][j] = '#'

                elif this_square == '@':
                    self.warehouse_state[i][j] = '*'
                    self.dropzone = (i, j)

                else:  # a box
                    box_id = this_square
                    self.warehouse_state[i][j] = box_id
                    self.boxes[box_id] = (i, j)

        self.robot_position = self.dropzone
        self.box_held = None

    
    def _search(self, box, current_position, heuristic_map, goal, debug=False):
        """
        This method should be based on Udacity Quizes for A*, see Lesson 12, Section 10-12.
        The bulk of the search logic should reside here, should you choose to use this starter code.
        Please condition any printout on the debug flag provided in the argument.  
        You may change this function signature (i.e. add arguments) as 
        necessary, except for the debug argument which must remain with a default of False
        """

        # get a shortcut variable for the warehouse (note this is just a view no copying)
        grid = self.warehouse_state
        action_list = []
        # Find and fill in the required moves per the instructions - example moves for test case 1
        # moves = [ 'move w',
        #           'move nw',
        #           'lift 1',
        #           'move se',
        #           'down e',
        #           'move ne',
        #           'lift 2',
        #           'down s']

        closed = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]
        expand = [[-1 for col in range(len(grid[0]))] for row in range(len(grid))]
        action_map = [['' for col in range(len(grid[0]))] for row in range(len(grid))]
        points = [[99 for col in range(len(grid[0]))] for row in range(len(grid))]
        direction_map = [['' for col in range(len(grid[0]))] for row in range(len(grid))]

        x = current_position[0]
        y = current_position[1]
        closed[x][y] = 1
        g = 0
        f = g + heuristic_map[x][y]
        points_holder = 99
        found_min_points = False
        open = [[f, g, x, y]]

        found = False  # flag that is set when search is complete
        resign = False  # flag set if we can't find expand
        count = 0

        while not found and not resign:
            if len(open) == 0:
                resign = True
                return "Fail"
            else:
                open.sort()
                open.reverse()
                # self.print_stuff(heuristic_map, True)
                # self.print_stuff(expand, True)
                # self.print_stuff(action_map, True)
                # self.print_stuff(points, True)
                # if box == '2':
                #     import pdb
                #     pdb.set_trace()
                next_up = open.pop()

                x = next_up[2]
                y = next_up[3]
                f = next_up[1]
                f = next_up[0]
                expand[x][y] = count
                direction_map[x][y] = action_map[x][y]
                count += 1


                if action_map[x][y] :

                    action_list.append("move {}".format(action_map[x][y]))

                if (x + 1 == goal[0] and y == goal[1]) or ( x - 1 == goal[0] and y == goal[1]) or \
                        (x == goal[0] and y + 1 == goal[1]) or (x == goal[0] and y - 1 == goal[1]) or \
                        (x + 1 == goal[0] and y + 1 == goal[1]) or ( x - 1 == goal[0] and y - 1 == goal[1]) or \
                        (x + 1 == goal[0] and y - 1 == goal[1]) or (x - 1 == goal[0] and y + 1 == goal[1]):
                    found = True
                else:
                    for i in range(len(self.delta)):
                        x2 = x + self.delta[i][0]
                        y2 = y + self.delta[i][1]
                        if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):
                            if closed[x2][y2] == 0 and (grid[x2][y2] == '.' or grid[x2][y2] == '*'):
                                g2 = g + self.delta_cost[i]
                                f2 = g2 + heuristic_map[x2][y2]
                                if f2 <= points_holder:
                                    points_holder = f2
                                    found_min_points = True
                                    points[x2][y2] = f2
                                    open.append([points_holder, g2, x2, y2])
                                closed[x2][y2] = 1
                                action_map[x2][y2] = self.delta_directions[i]


        return expand, action_list, direction_map

    def print_stuff(self, list_of_stuff, debug):
        if debug:
            print('----')
            for stuff in list_of_stuff:
                print(stuff)

    def compute_value(self, grid, goal):
        value = [[self.ILLEGAL_MOVE_PENALTY for x in range(len(grid[0]))] for y in range(len(grid))]
        # make sure your function returns a grid of values as
        # demonstrated in the previous video.
        value[goal[0]][goal[1]] = 0
        openList = []
        openList.append([0, goal[0], goal[1]])
        while len(openList) != 0:
            openList.sort()
            currentCell = openList.pop(0)
            for i in range(len(self.delta)):
                targetX = currentCell[2] - self.delta[i][1]
                targetY = currentCell[1] - self.delta[i][0]
                if self.inGrid(grid, targetX, targetY):
                    if (grid[targetY][targetX] == '*' or grid[targetY][targetX] == '.') and value[targetY][targetX] == self.ILLEGAL_MOVE_PENALTY:
                        openList.append([currentCell[0] + self.delta_cost[i], targetY, targetX])
                        value[targetY][targetX] = currentCell[0] + self.delta_cost[i]
        return value

    def inGrid(self, grid, x, y):
        if x >= 0 and x < len(grid[0]) and y >= 0 and y < len(grid):
            return True
        else:
            return False

    # def policy(self, goal, action, start_location):
    #     grid = self.warehouse_state
    #     policy = [['' for col in range(len(grid[0]))] for row in range(len(grid))]
    #     x = goal[0]
    #     y = goal[1]
    #     policy[x][y] = '*'
    #     while x != start_location[0] and y != start_location[1]:



    def plan_delivery(self, debug = False):
        """
        plan_delivery() is required and will be called by the autograder directly.  
        You may not change the function signature for it.
        Add logic here to find the moves.  You may use the starter code provided above
        in any way you choose, but please condition any printouts on the debug flag
        """

        # Find the moves - you may add arguments and change this logic but please leave
        # the debug flag in place and condition all printouts on it.

        # You may wish to break the task into one-way paths, like this:
        #
        #    moves_to_1   = self._search( ..., debug=debug )
        #    moves_from_1 = self._search( ..., debug=debug )
        #    moves_to_2   = self._search( ..., debug=debug )
        #    moves_from_2 = self._search( ..., debug=debug )
        #    moves        = moves_to_1 + moves_from_1 + moves_to_2 + moves_from_2
        #
        # If you use _search(), you may need to modify it to take some
        # additional arguments for starting location, goal location, and
        # whether to pick up or deliver a box.
        list_of_actions = []



        current_position = self.robot_position
        for box_number_letter in self.todo:
            grid = self.warehouse_state
            self.print_stuff(grid, debug)
            box_coord = self.boxes[str(box_number_letter)]
            # pick up the box
            heuristic_map = self.compute_value(grid, box_coord)
            moves, pickup_list, direction_map = self._search(box_number_letter, current_position, heuristic_map, box_coord, debug=debug)
            self.print_stuff(moves, debug)
            current_position = self.update_position(moves)
            pickup_list.append('lift {}'.format(box_number_letter))
            self.print_stuff(pickup_list, debug)
            self.warehouse_state[box_coord[0]][box_coord[1]] = '.' # this spot is now moveable

            # drop the box
            grid = self.warehouse_state
            heuristic_map = self.compute_value(grid, self.dropzone)
            # self.print_stuff(heuristic_map, debug)
            moves, drop_list, direction_map = self._search(box_number_letter, current_position, heuristic_map, self.dropzone, debug=debug)
            self.print_stuff(moves, debug)
            current_position = self.update_position(moves)

            if current_position[0] == self.dropzone[0] + 1 and current_position[1] == self.dropzone[1]: # robot is on bottom of the drop
                drop_list.append('down n')
            elif current_position[0] == self.dropzone[0] - 1 and current_position[1] == self.dropzone[1]: # robot is on top of the drop
                drop_list.append('down s')
            elif current_position[0] == self.dropzone[0] and current_position[1] == self.dropzone[1] + 1: # robot is on right of the drop
                drop_list.append('down w')
            elif current_position[0] == self.dropzone[0] and current_position[1] == self.dropzone[1] - 1:  # robot is on left of the drop
                drop_list.append('down e')
            elif current_position[0] == self.dropzone[0] + 1 and current_position[1] == self.dropzone[1] + 1: # robot is on bottom right (ne)
                drop_list.append('down nw')
            elif current_position[0] == self.dropzone[0] - 1 and current_position[1] == self.dropzone[1] + 1:  # robot is on top right (ne)
                drop_list.append('down sw')
            elif current_position[0] == self.dropzone[0] + 1 and current_position[1] == self.dropzone[1] - 1: # robot is on bottom left (sw)
                drop_list.append('down ne')
            elif current_position[0] == self.dropzone[0] - 1 and current_position[1] == self.dropzone[1] - 1:  # robot is on top left (se)
                drop_list.append('down se')
            self.print_stuff(drop_list, debug)
            list_of_actions.extend(pickup_list + drop_list)
        return list_of_actions


    def update_position(self, moves):
        dropoff_location = max([max(move) for move in moves])
        array_of_moves = np.array(moves)
        coord = np.where(array_of_moves == dropoff_location)
        current_position = (coord[0][0], coord[1][0])
        return current_position

class DeliveryPlanner_PartB:
    """
    Required methods in this class are:

        plan_delivery(self, debug = False) which is stubbed out below.
        You may not change the method signature as it will be called directly
        by the autograder but you may modify the internals as needed.

        __init__: required to initialize the class.  Starter code is
        provided that intializes class variables based on the definitions in
        testing_suite_partB.py.  You may choose to use this starter code
        or modify and replace it based on your own solution

    The following methods are starter code you may use for part B.
    However, they are not required and can be replaced with your
    own methods.

        _set_initial_state_from(self, warehouse): creates structures based on
            the warehouse and todo definitions and intializes the robot
            location in the warehouse

        _find_policy(self, debug=False): Where the bulk of the A* search algorithm
            could reside.  It should find an optimal path from the robot
            location to a goal.  Hint:  you may want to structure this based
            on whether looking for a box or delivering a box.

    """

    # Definitions taken from testing_suite_partA.py
    ORTHOGONAL_MOVE_COST = 2
    DIAGONAL_MOVE_COST = 3
    BOX_LIFT_COST = 4
    BOX_DOWN_COST = 2
    ILLEGAL_MOVE_PENALTY = 100

    def __init__(self, warehouse, warehouse_cost, todo):

        self.todo = todo
        self.boxes_delivered = []
        self.total_cost = 0
        self._set_initial_state_from(warehouse)
        self.warehouse_cost = warehouse_cost

        self.delta = [[-1, 0],  # go up
                        [0, -1],  # go left
                        [1, 0],  # go down
                        [0, 1],  # go right
                        [-1, -1],  # up left (diag)
                        [-1, 1],  # up right (diag)
                        [1, 1],  # dn right (diag)
                        [1, -1]]  # dn left (diag)

        self.delta_directions = ["n", "w", "s", "e", "nw", "ne", "se", "sw"]

        # Use this for a visual debug
        self.delta_name = ['^', '<', 'v', '>', '\\', '/', '[', ']']

        # Costs for each move
        self.delta_cost = [self.ORTHOGONAL_MOVE_COST,
                        self.ORTHOGONAL_MOVE_COST,
                        self.ORTHOGONAL_MOVE_COST,
                        self.ORTHOGONAL_MOVE_COST,
                        self.DIAGONAL_MOVE_COST,
                        self.DIAGONAL_MOVE_COST,
                        self.DIAGONAL_MOVE_COST,
                        self.DIAGONAL_MOVE_COST]

    # state parsing and initialization function from testing_suite_partA.py
    def _set_initial_state_from(self, warehouse):
        """Set initial state.

        Args:
            warehouse(list(list)): the warehouse map.
        """
        rows = len(warehouse)
        cols = len(warehouse[0])

        self.warehouse_state = [[None for j in range(cols)] for i in range(rows)]
        self.dropzone = None
        self.boxes = dict()

        for i in range(rows):
            for j in range(cols):
                this_square = warehouse[i][j]

                if this_square == '.':
                    self.warehouse_state[i][j] = '.'

                elif this_square == '#':
                    self.warehouse_state[i][j] = '#'

                elif this_square == '@':
                    self.warehouse_state[i][j] = '*'
                    self.dropzone = (i, j)

                else:  # a box
                    box_id = this_square
                    self.warehouse_state[i][j] = box_id
                    self.boxes[box_id] = (i, j)


    def _find_policy(self, box,current_position ,heuristic_map, goal, debug=False):
        """
        This method should be based on Udacity Quizes for Dynamic Progamming,
        see Lesson 12, Section 14-20 and Problem Set 4, Question 5.  The bulk of
        the logic for finding the policy should reside here should you choose to
        use this starter code.  Please condition any printout on the debug flag
        provided in the argument. You may change this function signature
        (i.e. add arguments) as necessary, except for the debug argument which
        must remain with a default of False
        """

        ##############################################################################
        # insert code in this method if using the starter code we've provided
        ##############################################################################


        # get a shortcut variable for the warehouse (note this is just a view it does not make a copy)
        grid = self.warehouse_state
        grid_costs = self.warehouse_cost

        # action_list = []
        # # Find and fill in the required moves per the instructions - example moves for test case 1
        # # moves = [ 'move w',
        # #           'move nw',
        # #           'lift 1',
        # #           'move se',
        # #           'down e',
        # #           'move ne',
        # #           'lift 2',
        # #           'down s']
        #
        # closed = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]
        # expand = [[-1 for col in range(len(grid[0]))] for row in range(len(grid))]
        # action_map = [['' for col in range(len(grid[0]))] for row in range(len(grid))]
        # points = [[99 for col in range(len(grid[0]))] for row in range(len(grid))]
        # direction_map = [['' for col in range(len(grid[0]))] for row in range(len(grid))]
        # sources = {}
        # x = current_position[0]
        # y = current_position[1]
        # closed[x][y] = 1
        # g = 0
        # f = g + heuristic_map[x][y]
        # z = 0
        # points_holder = 100
        # found_min_points = False
        # open = [[f, g, x, y]]
        # cumulative_sum = 0
        # found_sources = False
        # found = False  # flag that is set when search is complete
        # resign = False  # flag set if we can't find expand
        # count = 0
        # open_tracker = []
        # while not len(open) != 0:
        #     # if len(open) == 0:
        #     #     resign = True
        #     #     return "Fail"
        #     # else:
        #         open.sort()
        #         open.reverse()
        #         open_tracker.extend(open)
        #         self.print_stuff(grid_costs, debug)
        #
        #         self.print_stuff(heuristic_map, True)
        #         self.print_stuff(expand, True)
        #         self.print_stuff(action_map, True)
        #         self.print_stuff(points, True)
        #         import pdb
        #         pdb.set_trace()
        #         next_up = open.pop()
        #
        #         x = next_up[2]
        #         y = next_up[3]
        #         f = next_up[1]
        #         f = next_up[0]
        #         expand[x][y] = count
        #         count += 1
        #         # if action_map[x][y] :
        #         #     action_list.append("move {}".format(action_map[x][y]))
        #         #
        #         # if (x + 1 == goal[0] and y == goal[1]) or ( x - 1 == goal[0] and y == goal[1]) or \
        #         #         (x == goal[0] and y + 1 == goal[1]) or (x == goal[0] and y - 1 == goal[1]) or \
        #         #         (x + 1 == goal[0] and y + 1 == goal[1]) or ( x - 1 == goal[0] and y - 1 == goal[1]) or \
        #         #         (x + 1 == goal[0] and y - 1 == goal[1]) or (x - 1 == goal[0] and y + 1 == goal[1]):
        #         #     found = True
        #         # else:
        #         possible_actions_counter = 0
        #
        #         for i in range(len(self.delta)):
        #
        #             x2 = x + self.delta[i][0]
        #             y2 = y + self.delta[i][1]
        #             if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):
        #                 if closed[x2][y2] == 0 and (grid[x2][y2] == '.' or grid[x2][y2] == '*'):
        #                     g2 = g + self.delta_cost[i]
        #
        #                     if grid_costs[x2][y2] == 'inf':
        #                         grid_costs[x2][y2] = 999
        #                     f2 = g2 + heuristic_map[x2][y2] + grid_costs[x2][y2]
        #                     if f2 <= points_holder:
        #                         action_map[x2][y2] = self.delta_directions[i]
        #                     if not found_sources: # if no possible route sources found yet, create them
        #                         sources[possible_actions_counter] = f2
        #                     # else:
        #                         # f2 = f2 + points[x][y]
        #
        #                     # points_holder = f2
        #                     open.append([f2, grid_costs[x2][y2], x2, y2])
        #                     closed[x2][y2] = 1
        #                     action_map[x2][y2] = self.delta_directions[i]
        #                     points[x2][y2] = f2
        #                     possible_actions_counter += 1
        #             found_sources = True
        #
        # # You will need to fill in the algorithm here to find the policy
        # # The following are what your algorithm should return for test case 1
        # # if pickup_box:
        # #     # To box policy
        # #     policy = [['B',  'lift 1','move w'],
        # #               ['lift 1', '-1', 'move nw'],
        # #               ['move n', 'move nw', 'move n']]
        # #
        # # else:
        # #     # Deliver policy
        # #     policy = [['move e', 'move se', 'move s'],
        # #               ['move ne', '-1', 'down s'],
        # #               ['move e', 'down e', 'move n']]
        #
        #
        # return expand, action_map, points


    def compute_value(self, grid, goal):
        value = [[self.ILLEGAL_MOVE_PENALTY for x in range(len(grid[0]))] for y in range(len(grid))]
        # make sure your function returns a grid of values as
        # demonstrated in the previous video.
        value[goal[0]][goal[1]] = 0
        openList = []
        openList.append([0, goal[0], goal[1]])
        while len(openList) != 0:
            openList.sort()
            currentCell = openList.pop(0)

            for i in range(len(self.delta)):
                targetX = currentCell[2] - self.delta[i][1]
                targetY = currentCell[1] - self.delta[i][0]
                if self.inGrid(grid, targetX, targetY):
                    if (grid[targetY][targetX] == '*' or grid[targetY][targetX] == '.') and value[targetY][targetX] == self.ILLEGAL_MOVE_PENALTY:
                        openList.append([currentCell[0] + self.delta_cost[i], targetY, targetX])
                        value[targetY][targetX] = currentCell[0] + self.delta_cost[i] + self.warehouse_cost[targetY][targetX]
        return value

    def inGrid(self, grid, x, y):
        if x >= 0 and x < len(grid[0]) and y >= 0 and y < len(grid):
            return True
        else:
            return False


    def pickup(self, box_number_letter, true_goal, heuristic_map):
        opposite_directions_map={'nw':'se',
                                 'ne':'sw',
                                 'sw':'ne',
                                 'se':'nw',
                                 'w':'e',
                                 'e':'w',
                                 'n':'s',
                                 's':'n'}

        grid = self.warehouse_state
        found_action_map = [['' for x in range(len(grid[0]))] for y in range(len(grid))]
        found = False
        openList = []
        count = 0
        action_list = []
        openList.append([0, true_goal[0], true_goal[1]])
        while len(openList) != 0:
            openList.sort()
            openList.reverse()
            currentCell = openList.pop()
            x = currentCell[2]
            y = currentCell[1]
            if count == 0:
                found_action_map[y][x] = 'B'

            count += 1
            if (x + 1 == true_goal[1] and y == true_goal[0]) or (x - 1 == true_goal[1] and y == true_goal[0]) or (
                    x == true_goal[1] and y + 1 == true_goal[0]) or (x == true_goal[1] and y - 1 == true_goal[0]) or (
                    x + 1 == true_goal[1] and y + 1 == true_goal[0]) or (
                    x - 1 == true_goal[1] and y - 1 == true_goal[0]) or (
                    x + 1 == true_goal[1] and y - 1 == true_goal[0]) or (
                    x - 1 == true_goal[1] and y + 1 == true_goal[0]):
                found_action_map[y][x] = 'lift {}'.format(box_number_letter)
            # self.print_stuff(found_action_map, True)

            for i in range(len(self.delta)):
                targetX = currentCell[2] - self.delta[i][1]
                targetY = currentCell[1] - self.delta[i][0]
                if self.inGrid(grid, targetX, targetY):
                    if (grid[targetY][targetX] == '*' or grid[targetY][targetX] == '.') and \
                            found_action_map[targetY][targetX] == '':
                        openList.append([heuristic_map[targetY][targetX], targetY, targetX])
                        found_action_map[targetY][targetX] = 'move {}'.format(self.delta_directions[i])
                        # self.print_stuff(found_action_map)
                    elif grid[targetY][targetX] == '#': # found a wall
                        found_action_map[targetY][targetX] = '-1'
        return found_action_map

    def dropoff(self, box_number_letter, true_goal, heuristic_map):

        grid = self.warehouse_state
        direction_map = [['' for x in range(len(grid[0]))] for y in range(len(grid))]
        found_action_map = [['' for x in range(len(grid[0]))] for y in range(len(grid))]
        found = False
        openList = []
        count = 0
        action_list = []
        openList.append([0, true_goal[0], true_goal[1]])
        while len(openList) != 0:
            openList.sort()
            openList.reverse()
            currentCell = openList.pop()
            x = currentCell[2]
            y = currentCell[1]
            # if count == 0:
                # found_action_map[y][x] = 'B'

            count += 1
            if (x + 1 == true_goal[1] and y == true_goal[0]) or (x - 1 == true_goal[1] and y == true_goal[0]) or (
                    x == true_goal[1] and y + 1 == true_goal[0]) or (
                    x == true_goal[1] and y - 1 == true_goal[0]) or (
                    x + 1 == true_goal[1] and y + 1 == true_goal[0]) or (
                    x - 1 == true_goal[1] and y - 1 == true_goal[0]) or (
                    x + 1 == true_goal[1] and y - 1 == true_goal[0]) or (
                    x - 1 == true_goal[1] and y + 1 == true_goal[0]):
                found_action_map[y][x] = 'down {}'.format(direction_map[y][x])
            # self.print_stuff(found_action_map, True)

            for i in range(len(self.delta)):
                targetX = currentCell[2] - self.delta[i][1]
                targetY = currentCell[1] - self.delta[i][0]
                if self.inGrid(grid, targetX, targetY):
                    if (grid[targetY][targetX] == '*' or grid[targetY][targetX] == '.' or grid[targetY][targetX] == box_number_letter) and \
                            found_action_map[targetY][targetX] == '':
                        openList.append([heuristic_map[targetY][targetX], targetY, targetX])
                        found_action_map[targetY][targetX] = 'move {}'.format(self.delta_directions[i])
                        direction_map[targetY][targetX] = self.delta_directions[i]
                        # if targetX == true_goal[1]
                        # self.print_stuff(found_action_map)
                    elif grid[targetY][targetX] == '#':  # found a wall
                        found_action_map[targetY][targetX] = '-1'
        return found_action_map
        # self.print_stuff(action_list)


            #             found_a_minimum_point = True
            # if found_a_minimum_point:
            #     minimum_point = points_map[targetY][targetX]
            # minimum_point_holder = points_map[targetY][targetX]
            #             lowest_path_map[targetY][targetX] =
    def plan_delivery(self, debug=False):
        """
        plan_delivery() is required and will be called by the autograder directly.  
        You may not change the function signature for it.
        Add logic here to find the policies:  First to the box from any grid position
        then to the dropzone, again from any grid position.  You may use the starter
        code provided above in any way you choose, but please condition any printouts
        on the debug flag
        """
        ###########################################################################
        # Following is an example of how one could structure the solution using
        # the starter code we've provided.
        ###########################################################################


        # Start by finding a policy to direct the robot to the box from any grid position
        # The last command(s) in this policy will be 'lift 1' (i.e. lift box 1)

        array_of_map = np.array(self.warehouse_state)
        starting_point = np.where(array_of_map == '*')
        pickup_policy = None
        dropoff_policy = None

        current_position = (starting_point[0][0], starting_point[1][0])
        for box_number_letter in self.todo:
            grid = self.warehouse_state
            self.print_stuff(grid, debug)
            box_coord = self.boxes[str(box_number_letter)]
            # pick up the box
            pickup_heuristic_map = self.compute_value(grid, box_coord)
            self.print_stuff(pickup_heuristic_map, debug)
            pickup_policy = self.pickup(box_number_letter, box_coord, pickup_heuristic_map)
            self.print_stuff(pickup_policy, debug)
            dropoff_heuristic_map = self.compute_value(grid, self.dropzone)
            self.print_stuff(dropoff_heuristic_map, debug)
            dropoff_policy = self.dropoff(box_number_letter, self.dropzone, dropoff_heuristic_map)
            self.print_stuff(dropoff_policy, debug)

            # dropoff_policy = self.find_lowest_route()

            # self.print_stuff(moves, debug)
            # current_position = self.update_position(moves)
            # pickup_list.append('lift {}'.format(box_number_letter))
            # self.print_stuff(pickup_list, debug)
            # self.warehouse_state[box_coord[0]][box_coord[1]] = '.'  # this spot is now moveable


        # Now that the robot has the box, transition to the deliver policy.  The
        # last command(s) in this policy will be 'down x' where x = the appropriate
        # direction to set the box into the dropzone
        # goal = self.dropzone
        # deliver_policy = self._find_policy(goal, pickup_box=False, debug=debug)

        # if debug:
        #     print("\nTo Box Policy:")
        #     for i in range(len(to_box_policy)):
        #         print(to_box_policy[i])
        #
        #     print("\nDeliver Policy:")
        #     for i in range(len(deliver_policy)):
        #         print(deliver_policy[i])
        #
        #     print("\n\n")

        return (pickup_policy, dropoff_policy)

    def print_stuff(self, list_of_stuff, debug=True):
        if debug:
            print('----')
            for stuff in list_of_stuff:
                print(stuff)


if __name__ == "__main__":
    """ 
    You may execute this file to develop and test the search algorithm prior to running 
    the delivery planner in the testing suite.  Copy any test cases from the
    testing suite or make up your own.
    Run command:  python warehouse.py
    """

    # Test code in here will not be called by the autograder

    # Testing for Part A
    # testcase 1
    warehouse = ['1#2',
                '.#.',
                '..@']

    todo =  ['1','2'] 

    partA = DeliveryPlanner_PartA(warehouse, todo)
    partA.plan_delivery(debug=True)

    # Testing for Part B
    # testcase 1
    warehouse = ['1..',
                 '.#.',
                 '..@']

    warehouse_cost = [[0, 5, 2],
                      [10, math.inf, 2],
                      [2, 10, 2]]

    todo = ['1']

    partB = DeliveryPlanner_PartB(warehouse, warehouse_cost, todo)
    partB.plan_delivery(debug=True)





