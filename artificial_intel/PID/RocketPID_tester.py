
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

#
# If matplotlib doesn't work on your mac, try uncommenting this:
#plt.use('tkagg')
#
# or type the following in a terminal:
# echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc

from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
import matplotlib.pyplot as plt
import numpy as np
import unittest
from scipy.integrate import odeint


class PressurePD(object):

    def __init__(self):
        # Standard values and variables
        self.max_flow_delta = 10.0
        self.consumption_rate = 5.0
        self.initial_level = 10.0
        self.max_level = 105.0
        self.min_level = 0.0
        self.ideal_level = 100.0

        # Workspace
        self.time_final = 300
        self.time_steps = 301

    def start_pumps(self, pressure_pd_solution, show_graph=True):
        """Start turbopump to begin feeding LOX to engine for liftoff

        :
            pressure_pd_solution (func): Pump pressure PD function to control pumps
            show_graph (bool): Show graphed results

        Returns:
              Final grade, Output of launch
        """
        output = '\n-------------------------------------------\n'
        output += 'Maintaining Turbopump Pressure PD \n'
        current_pressure = self.initial_level
        adjust_log = np.zeros(self.time_steps)
        pressure_log = np.zeros(self.time_steps)
        pressure_change = 0.0
        target_pressure = 100.0

        # deltaT is 1 if time + 1 = time_step
        delta_t = old_div(self.time_final, (self.time_steps - 1))
        time = np.linspace(0, self.time_final, self.time_steps)

        # Initialize data
        data = {'ErrorP': 0,
                'ErrorI': 0,
                'ErrorD': 0}

        for instance in range(len(time)):
            pressure_adjust, data = pressure_pd_solution(
                delta_t, current_pressure, target_pressure, data)

            pressure_adjust = min(pressure_adjust, 1.0)
            pressure_adjust = max(pressure_adjust, -1.0)

            adjust_log[instance] = pressure_adjust

            pressure_change += pressure_adjust
            pressure_change = min(pressure_change, self.max_flow_delta)
            pressure_change = max(pressure_change, -self.max_flow_delta)

            current_pressure += pressure_change
            current_pressure -= self.consumption_rate

            pressure_log[instance] = current_pressure

        # Plotting for testing purposes
        if show_graph:
            try:
                plt.figure()
                plt.subplot(5, 1, 1)
                plt.title('LOX Turbopump Pressure')
                plt.plot(time, pressure_log, 'b',
                         linewidth=2, label='Pump Pressure')
                plt.ylabel('Output (%)')
                plt.axhline(y=self.ideal_level, color='y',
                            linewidth=2, label='Optimal level')
                plt.legend(loc='best')
                plt.subplot(5, 1, 2)
                plt.plot(time, adjust_log, 'r', linewidth=2,
                         label='Input adjustments')
                plt.legend(loc='best')
                plt.show()
            except Exception as exp:
                output += 'Error with plotting results:' + str(exp)
                output += '\n'

        # Generate scoring
        min_pressure_level = np.min(pressure_log)
        max_pressure_level = np.max(pressure_log)

        if min_pressure_level < self.min_level:
            output += 'Turbopump pressure level dropped below safe minimum values.\n'
            score = 0.0

        elif max_pressure_level > self.max_level:
            output += 'Turbopump pressure exceeded maximum design limits.\n'
            score = 0.0

        else:
            start_position = 55
            tolerance = 1.0
            lower_bounds = self.ideal_level - tolerance
            upper_bounds = self.ideal_level + tolerance
            pressure_data = pressure_log[start_position:self.time_steps]
            correct = len(np.where(np.logical_and(
                pressure_data <= upper_bounds, pressure_data >= lower_bounds))[0])
            total_positions = self.time_steps - start_position
            score = (correct / float(total_positions)) * 100

        final_score = score*0.25
        output += '\n'
        output += 'Part A Completion: {}%\n'.format(score)
        output += 'Points Earned: {}\n'.format(final_score)
        return final_score, output


class RocketPID(object):
    """
    Attributes:
        force_propulsion (int): kn maximum thrust of engines in N
        rho (int): density of air in kg/km3
        cd (float): air drag coefficient in unit-less
        area (float): area of rocket cross section in km2
        vehicle (int): nominal weight of rocket in kg
        standard_gravity (float): gravity w/r to altitude in km/s2
        time_final (int): final number of time steps
        time_steps (int): total number of time steps
        thrust (ndarray): thrust log for each time step
        gravity (ndarray): gravity log for each time step
        drag (ndarray): drag log for each time step
    """

    def __init__(self):
        # Standard values and variables
        self.force_propulsion = 4000
        self.rho = 1225000000  #
        self.cd = 0.5  #
        self.area = 0.000016  #
        self.vehicle = 50000  #
        self.standard_gravity = 0.00981  #

        # Workspace
        self.time_final = 600
        self.time_steps = 601

        self.thrust = np.zeros(self.time_steps)
        self.gravity = np.zeros(self.time_steps)
        self.drag = np.zeros(self.time_steps)

    def rocket(self, velocity, full_time, instance_count, throttle, fuel):
        """Models rocket velocity.

        Args:
            velocity (float): Current rocket velocity.
            full_time (ndarray): Full list of time steps.
            instance_count (int): Current time step in launch.
            throttle (float): Current throttle value.
            fuel (float): Current fuel value.

        Returns:
            Change in velocity as float.
        """

        # Force Equations
        mass = self.vehicle + max(0, fuel)
        thrust_force = self.force_propulsion * throttle
        gravity_force = self.standard_gravity * mass
        drag_force = 0.5 * self.rho * self.cd * self.area * velocity ** 2

        if velocity < 0:
            drag_force = -drag_force
        if fuel < 0:
            thrust_force = 0

        # Store for plotting
        self.thrust[instance_count + 1] = thrust_force
        self.gravity[instance_count + 1] = abs(gravity_force)
        self.drag[instance_count + 1] = abs(drag_force)

        # First Order Equation for Solving Change in Velocity
        d_vdt = old_div((thrust_force - gravity_force - drag_force), mass)

        return d_vdt

    def launch_rocket(self, rocket_pid_solution, show_graph=True):
        """Launch rocket to attempt to fly optimal flight path

        Args:
            rocket_pid_solution (func): Rocket PID function to control launch
            show_graph (bool): Show graphed results

        Returns:
              Final grade, Output of launch
        """
        output = '\n-------------------------------------------\n'
        output += 'Rocket Launch with PID controlled Throttle\n\n'

        init_fuel = 35000  # fuel load in kg
        # initial velocity level (height = 0 at base) in km/2
        init_velocity = 0
        # initial engine position (shutoff = 0, max thrust = 1) in percent
        init_throttle = 0
        fuel_consumption = 480  # kerosene RG-1 consumption in kg/s
        landed = 0  # status indicator for landing
        fuel_empty = 0  # status indicator for fuel tank
        good_landing = 0  # status indicator for successful landing

        # deltaT is 1 if time + 1 = time_step
        delta_t = old_div(self.time_final, (self.time_steps - 1))
        time = np.linspace(0, self.time_final, self.time_steps)

        throttle_set = np.zeros(self.time_steps)
        velocity_log = np.zeros(self.time_steps)
        optimal_velocity_log = np.zeros(self.time_steps)
        height = np.zeros(self.time_steps)
        fuel_level = np.zeros(self.time_steps)

        # Initialize data
        data = {'ErrorP': 0,
                'ErrorI': 0,
                'ErrorD': 0}

        # Rocket Altitude ODE solver
        for instance in range(len(time) - 1):
            if landed > 0:
                break

            if instance < 100:
                optimal_velocity = 0.25
            elif instance < 150:
                optimal_velocity = 0.5
            elif instance == 150:
                optimal_velocity = -0.5
            elif height[instance] < 3:
                optimal_velocity = -0.1

            init_throttle, data = rocket_pid_solution(
                delta_t, velocity_log[instance], optimal_velocity, data)
            init_throttle = max(0, min(1, init_throttle))

            # simulate air density drop with altitude
            self.rho = 1225000000 * np.exp(old_div(-height[instance], 1000))

            # shutoff engines if fuel empty
            if fuel_empty == 1:
                output += 'Out of Fuel!\n'
                init_throttle = 0

            # ODE solver to simulate rocket velocity change
            rocket_velocity = odeint(self.rocket, init_velocity, [time[instance], time[instance + 1]],
                                     args=(instance, init_throttle, init_fuel))

            # update velocity with ODE value
            init_velocity = rocket_velocity[1][0]
            velocity_log[instance + 1] = init_velocity  # log current velocity
            throttle_set[instance + 1] = init_throttle  # log throttle
            init_fuel = init_fuel - fuel_consumption * \
                init_throttle  # reduce fuel per consumption rate
            # log optimal velocity
            optimal_velocity_log[instance + 1] = optimal_velocity

            # Altitude and Fuel Checks
            if height[instance] < 0 and abs(init_velocity) > 0.11:
                height[instance + 1] = 0
                landed = instance
                output += 'YOU CRASHED!\n'
            elif height[instance] < 0 and abs(init_velocity) <= 0.11 and instance > 10:
                height[instance + 1] = 0
                landed = instance
                good_landing = 1
                output += 'You Landed in the student tester!\n'
            elif height[instance] >= 0:
                height[instance + 1] = height[instance] + \
                    init_velocity * delta_t

            if fuel_empty == 1:
                fuel_level[instance + 1] = 0
            elif init_fuel < 0:
                fuel_level[instance + 1] = 0
                fuel_empty = 1
            else:
                fuel_level[instance + 1] = init_fuel

        # Plotting for testing purposes
        if show_graph:
            try:
                plt.figure()
                plt.subplot(5, 1, 1)
                plt.title('Rocket Launch')
                plt.plot(time, optimal_velocity_log, 'c--',
                         linewidth=2, label='Optimum Velocity')
                plt.plot(time, velocity_log, 'b:',
                         linewidth=2, label='Current Velocity')
                plt.axvline(x=landed)
                plt.ylabel('Velocity (km/s)')
                plt.legend(loc='best')
                plt.subplot(5, 1, 2)
                plt.plot([0, self.time_final], [1, 1], 'm--',
                         linewidth=2, label='Maximum Thrust')
                plt.plot(time, throttle_set, 'r-',
                         linewidth=2, label='Current Throttle')
                plt.ylabel('Throttle (%)')
                plt.axvline(x=landed)
                plt.legend(loc='best')
                plt.subplot(5, 1, 3)
                plt.plot(time, height, 'g-', linewidth=2,
                         label='Current Height')
                plt.ylabel('Height (km)')
                plt.axvline(x=landed)
                plt.legend(loc='best')
                plt.subplot(5, 1, 4)
                plt.plot(time, fuel_level, 'y-',
                         linewidth=2, label='Current Fuel')
                plt.ylabel('Fuel (kg)')
                plt.axvline(x=landed)
                plt.legend(loc='best')
                plt.xlabel('Time (sec)')
                plt.subplot(5, 1, 5)
                plt.plot(time, self.thrust, 'b:', linewidth=2, label='Thrust')
                plt.plot(time, self.gravity, 'g:',
                         linewidth=2, label='Gravity')
                plt.plot(time, self.drag, 'r:', linewidth=2, label='Drag')
                plt.ylabel('Force (N)')
                plt.axvline(x=landed)
                plt.legend(loc='best')
                plt.xlabel('Time (sec)')
                plt.show()
            except Exception as exp:
                output += 'Error plotting results:' + str(exp)
                output += '\n'

        # Score for following optimal course
        # 90 seconds on 0.25 km/s course, 40 seconds for 0.5 km/s. full score 65 points
        score_one = 0
        score_two = 0
        for instance in range(len(velocity_log)):
            if 0.24 < velocity_log[instance] < 0.26:
                score_one += 1
            if 0.49 < velocity_log[instance] < 0.51:
                score_two += 1

        # Score for making a successful landing
        # 35 points for good landing
        if good_landing == 1:
            landing_score = 35
        else:
            landing_score = 0

        flight_score = (min(score_one, 90) + min(score_two, 40)) / 2.0

        rocket_score = min(flight_score + landing_score, 100)

        final_score = rocket_score * 0.65
        output += '\n'
        output += 'Optimal Flight Score: {}\nLanding Score: {}\n'.format(
            flight_score, landing_score)
        output += '\n'
        output += 'Part B Completion: {}%\n'.format(rocket_score)
        output += 'Points Earned: {}\n'.format(final_score)

        return final_score, landing_score, flight_score, output

class BipropellantRocketPID(object):
    """
    Attributes:
        force_propulsion (int): kn maximum thrust of engines in N
        rho (int): density of air in kg/km3
        cd (float): air drag coefficient in unit-less
        area (float): area of rocket cross section in km2
        vehicle (int): nominal weight of rocket in kg
        standard_gravity (float): gravity w/r to altitude in km/s2
        time_final (int): final number of time steps
        time_steps (int): total number of time steps
        thrust (ndarray): thrust log for each time step
        gravity (ndarray): gravity log for each time step
        drag (ndarray): drag log for each time step
    """

    def __init__(self):
        # Standard values and variables
        self.force_propulsion = 4000
        self.oxidizer_propulsion = 2500
        self.rho = 1225000000  #
        self.cd = 0.5  #
        self.area = 0.000016  #
        self.vehicle = 50000  #
        self.standard_gravity = 0.00981  #

        # Workspace
        self.time_final = 600
        self.time_steps = 601

        self.thrust = np.zeros(self.time_steps)
        self.gravity = np.zeros(self.time_steps)
        self.drag = np.zeros(self.time_steps)

    def rocket(self, velocity, full_time, instance_count, throttle, oxidizer_throttle, fuel, oxidizer):
        """Models rocket velocity.

        Args:
            velocity (float): Current rocket velocity.
            full_time (ndarray): Full list of time steps.
            instance_count (int): Current time step in launch.
            throttle (float): Current throttle value.
            fuel (float): Current fuel value.

        Returns:
            Change in velocity as float.
        """

        # Force Equations
        mass = self.vehicle + max(0, fuel) + max(0, oxidizer)
        thrust_force = self.force_propulsion * throttle + self.oxidizer_propulsion * oxidizer_throttle
        gravity_force = self.standard_gravity * mass
        drag_force = 0.5 * self.rho * self.cd * self.area * velocity ** 2

        if velocity < 0:
            drag_force = -drag_force
        if fuel < 0:
            thrust_force = 0

        # Store for plotting
        self.thrust[instance_count + 1] = thrust_force
        self.gravity[instance_count + 1] = abs(gravity_force)
        self.drag[instance_count + 1] = abs(drag_force)

        # First Order Equation for Solving Change in Velocity
        d_vdt = old_div((thrust_force - gravity_force - drag_force), mass)

        return d_vdt

    def launch_rocket(self, bipropellant_rocket_pid_solution, show_graph=True):
        """Launch rocket to attempt to fly optimal flight path

        Args:
            bipropellant_rocket_pid_solution (func): Rocket PID function to control launch
            show_graph (bool): Show graphed results

        Returns:
              Final grade, Output of launch
        """
        output = '\n-------------------------------------------\n'
        output += 'Bipropellant Rocket Launch with PID controlled Throttle\n\n'

        init_fuel = 35000  # fuel load in kg
        init_oxidizer = 85000 # oxidizer load in kg
        # initial velocity level (height = 0 at base) in km/2
        init_velocity = 0
        # initial engine position (shutoff = 0, max thrust = 1) in percent
        fuel_throttle = 0
        oxidizer_throttle = 0
        fuel_consumption = 480  # kerosene RG-1 consumption in kg/s
        oxidizer_consumption = 480 # oxidizer consumption in kg/s
        oxidizer_fuel_ratio = 2.77
        
        landed = 0  # status indicator for landing
        fuel_empty = False  # status indicator for fuel tank
        oxidizer_empty = False
        good_landing = 0  # status indicator for successful landing

        # deltaT is 1 if time + 1 = time_step
        delta_t = old_div(self.time_final, (self.time_steps - 1))
        time = np.linspace(0, self.time_final, self.time_steps)

        throttle_set = np.zeros(self.time_steps)
        velocity_log = np.zeros(self.time_steps)
        optimal_velocity_log = np.zeros(self.time_steps)
        height = np.zeros(self.time_steps)
        fuel_level = np.zeros(self.time_steps)
        oxidizer_level = np.zeros(self.time_steps)

        # Initialize data
        data = {'ErrorP': 0,
                'ErrorI': 0,
                'ErrorD': 0}

        # Rocket Altitude ODE solver
        for instance in range(len(time) - 1):
            if landed > 0:
                break
                
            if instance < 150:
                optimal_velocity = 0.25
            elif instance == 150:
                optimal_velocity = -0.25
            elif height[instance] < 3:
                optimal_velocity = -0.1

            fuel_throttle, oxidizer_throttle, data = bipropellant_rocket_pid_solution(
                delta_t, velocity_log[instance], optimal_velocity, data)
            fuel_throttle = effective_fuel_throttle = max(0, min(1, fuel_throttle))
            oxidizer_throttle = effective_oxidizer_throttle = max(0, min(1, oxidizer_throttle))
            
            # Check for oxidizer : fuel ratio
            if fuel_throttle != 0:
                current_oxidizer_fuel_ratio = oxidizer_consumption * oxidizer_throttle / (fuel_consumption * fuel_throttle)
                if current_oxidizer_fuel_ratio < oxidizer_fuel_ratio - 0.1: # oxidizer is less, so use fuel according to oxidizer
                    effective_fuel_throttle = max(0, min(1, oxidizer_consumption * oxidizer_throttle / (fuel_consumption * oxidizer_fuel_ratio)))
                    
                elif current_oxidizer_fuel_ratio > oxidizer_fuel_ratio + 0.1: # oxidizer is more, so use oxidizer according to fuel
                    effective_oxidizer_throttle =  oxidizer_fuel_ratio * fuel_consumption * fuel_throttle / oxidizer_consumption
                    
            else:
                effective_oxidizer_throttle = 0
                effective_fuel_throttle = 0
                
            
            # simulate air density drop with altitude
            self.rho = 1225000000 * np.exp(old_div(-height[instance], 1000))

            # shutoff engines if fuel empty
            if fuel_empty :
                output += 'Bipropellant rocket: Out of Fuel !\n'
                effective_fuel_throttle = 0
                effective_oxidizer_throttle = 0
            
            if oxidizer_empty:
                output += 'Bipropellant rocket: Out of Oxidizer !\n'
                effective_fuel_throttle = 0
                effective_oxidizer_throttle = 0

            # ODE solver to simulate rocket velocity change
            rocket_velocity = odeint(self.rocket, init_velocity, [time[instance], time[instance + 1]],
                                     args=(instance, effective_fuel_throttle, effective_oxidizer_throttle, init_fuel, init_oxidizer))

            # update velocity with ODE value
            init_velocity = rocket_velocity[1][0]
            velocity_log[instance + 1] = init_velocity  # log current velocity
            throttle_set[instance + 1] = fuel_throttle  # log throttle
            init_fuel = init_fuel - fuel_consumption * \
                fuel_throttle  # reduce fuel per consumption rate
            init_oxidizer = init_oxidizer - oxidizer_consumption * oxidizer_throttle
            
           
            # log optimal velocity
            optimal_velocity_log[instance + 1] = optimal_velocity

            # Altitude and Fuel Checks
            if height[instance] < 0 and abs(init_velocity) > 0.11:
                height[instance + 1] = 0
                landed = instance
                output += 'Bipropellant rocket: YOU CRASHED!\n'
            elif height[instance] < 0 and abs(init_velocity) <= 0.11 and instance > 10:
                height[instance + 1] = 0
                landed = instance
                good_landing = 1
                output += 'Bipropellant rocket: You Landed in the student tester!\n'
            elif height[instance] >= 0:
                height[instance + 1] = height[instance] + \
                    init_velocity * delta_t

            if fuel_empty:
                fuel_level[instance + 1] = 0
            if oxidizer_empty:
                oxidizer_level[instance + 1] = 0
            elif init_fuel < 0:
                fuel_level[instance + 1] = 0
                fuel_empty = True
            elif init_oxidizer < 0:
                oxidizer_level[instance + 1] = 0
                oxidizer_empty = True
            else:
                oxidizer_level[instance + 1] = init_oxidizer
                fuel_level[instance + 1] = init_fuel
            
            

        # Plotting for testing purposes
        if show_graph:
            try:
                plt.figure()
                plt.subplot(6, 1, 1)
                plt.title('Bipropellant Rocket Launch')
                plt.plot(time, optimal_velocity_log, 'c--',
                         linewidth=2, label='Optimum Velocity')
                plt.plot(time, velocity_log, 'b:',
                         linewidth=2, label='Current Velocity')
                plt.axvline(x=landed)
                plt.ylabel('Velocity (km/s)')
                plt.legend(loc='best')
                plt.subplot(6, 1, 2)
                plt.plot([0, self.time_final], [1, 1], 'm--',
                         linewidth=2, label='Maximum Thrust')
                plt.plot(time, throttle_set, 'r-',
                         linewidth=2, label='Current Throttle')
                plt.ylabel('Throttle (%)')
                plt.axvline(x=landed)
                plt.legend(loc='best')
                plt.subplot(6, 1, 3)
                plt.plot(time, height, 'g-', linewidth=2,
                         label='Current Height')
                plt.ylabel('Height (km)')
                plt.axvline(x=landed)
                plt.legend(loc='best')
                plt.subplot(6, 1, 4)
                plt.plot(time, fuel_level, 'y-',
                         linewidth=2, label='Current Fuel')
                plt.ylabel('Fuel (kg)')
                plt.axvline(x=landed)
                plt.legend(loc='best')
                plt.xlabel('Time (sec)')
                plt.subplot(6, 1, 5)
                plt.plot(time, oxidizer_level, 'y-',
                         linewidth=2, label='Current Oxidizer')
                plt.ylabel('Fuel (kg)')
                plt.axvline(x=landed)
                plt.legend(loc='best')
                plt.xlabel('Time (sec)')
                plt.subplot(6, 1, 6)
                plt.plot(time, self.thrust, 'b:', linewidth=2, label='Thrust')
                plt.plot(time, self.gravity, 'g:',
                         linewidth=2, label='Gravity')
                plt.plot(time, self.drag, 'r:', linewidth=2, label='Drag')
                plt.ylabel('Force (N)')
                plt.axvline(x=landed)
                plt.legend(loc='best')
                plt.xlabel('Time (sec)')
                plt.show()
            except Exception as exp:
                output += 'Error plotting results:' + str(exp)
                output += '\n'

        # Score for following optimal course
        # 130 seconds on 0.25 km/s course. full score 65 points
        score_one = 0
        score_two = 0
        for instance in range(len(velocity_log)):
            if 0.24 < velocity_log[instance] < 0.26:
                score_one += 1
            

        # Score for making a successful landing
        # 35 points for good landing
        if good_landing == 1:
            landing_score = 35
        else:
            landing_score = 0

        flight_score = min(score_one, 130) / 2.0

        rocket_score = min(flight_score + landing_score, 100)

        final_score = rocket_score * 0.10
        output += '\n'
        output += 'Bipropellant rocket: Optimal Flight Score: {}\nLanding Score: {}\n'.format(
            flight_score, landing_score)
        output += '\n'
        output += 'Part C Completion: {}%\n'.format(rocket_score)
        output += 'Points Earned: {}\n'.format(final_score)

        return final_score, landing_score, flight_score, output


class PIDTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # We initalize these to zero to start.
        cls.running_pump_score = 0.0
        cls.rocket_flight_score = 0.0
        cls.bipropellant_rocket_flight_score = 0.0

    # Ran before each individual test.

    def setUp(self):

        try:
            import RocketPIDStudent_submission
            self.pump_pd_solution = RocketPIDStudent_submission.pressure_pd_solution
            self.rocket_pid_solution = RocketPIDStudent_submission.rocket_pid_solution
            self.bipropellant_rocket_pid_solution = RocketPIDStudent_submission.bipropellant_rocket_pid_solution

        except Exception as e:
            print('Error importing RocketPIDStudent:' + str(e))

    def test_running_pumps(self):
        """Test the rocket maintains constant pressure rate in the turbopumps
        """

        my_pumps = PressurePD()
        pressure_score, pressure_output = my_pumps.start_pumps(
            self.pump_pd_solution)

        print(pressure_output)
        PIDTest.running_pump_score = pressure_score

        self.assertTrue(pressure_score >= 25,
                        'Turbopump pressure was not maintained or exceeded design limits.')

    def test_rocket_flight(self):
        """Test rocket flight using PID solution
        """
        my_rocket = RocketPID()
        final_score, landing_score, flight_score, rocket_output = my_rocket.launch_rocket(
            self.rocket_pid_solution)

        print(rocket_output)
        PIDTest.rocket_flight_score = final_score

        self.assertTrue(landing_score == 35,
                        'Rocket did not land successfully.')
        self.assertTrue(flight_score >= 65,
                        'Rocket flight did not follow optimal flight path.')

        
        
    def test_bipropellant_rocket_flight(self):
        """Test bipropellant rocket flight using PID solution
        """
        my_rocket = BipropellantRocketPID()
        final_score, landing_score, flight_score, rocket_output = my_rocket.launch_rocket(
            self.bipropellant_rocket_pid_solution)

        print(rocket_output)
        PIDTest.bipropellant_rocket_flight_score = final_score

        self.assertTrue(landing_score == 35,
                        'Bipropellant Rocket did not land successfully.')
        self.assertTrue(flight_score >= 65,
                        'Bipropellant Rocket flight did not follow optimal flight path.')

        

    @classmethod
    def tearDownClass(cls):
        # Prints the entire score for the project at the very end.
        print("\nOverall total score :", int(cls.running_pump_score + cls.rocket_flight_score + cls.bipropellant_rocket_flight_score))


if __name__ == "__main__":
    unittest.main()
