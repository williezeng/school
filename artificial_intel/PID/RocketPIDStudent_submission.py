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

# Optimize your PID parameters here:
pressure_tau_p = 0.3
pressure_tau_d = 0.6

rocket_tau_p = 20
rocket_tau_i = 0.2
rocket_tau_d = 0.8


bipropellant_rocket_fuel_tau_p = 63
bipropellant_rocket_fuel_tau_i = 0.36
bipropellant_rocket_fuel_tau_d = 6.3

bipropellant_rocket_oxidizer_tau_p = 36
bipropellant_rocket_oxidizer_tau_i = 0.8
bipropellant_rocket_oxidizer_tau_d = 6.3


def pressure_pd_solution(delta_t, current_pressure, target_pressure, data):
	"""Student solution to maintain LOX pressure to the turbopump at a level of 100.

    Args:
        delta_t (float): Time step length.
        current_pressure (float): Current pressure level of the turbopump.
        target_pressure (float): Target pressure level of the turbopump.
        data (dict): Data passed through out run.  Additional data can be added and existing values modified.
            'ErrorP': Proportional error.  Initialized to 0.0
            'ErrorD': Derivative error.  Initialized to 0.0
    """

	pressure_error = current_pressure - target_pressure
	if 'previous_pressure' in data:
		differential_pressure_error = pressure_error - data['previous_pressure']
	else:
		differential_pressure_error = 0
	P = (-pressure_tau_p * pressure_error) - (pressure_tau_d * differential_pressure_error)
	adjust_pressure = P
	data['previous_pressure'] = pressure_error

	return adjust_pressure, data


def rocket_pid_solution(delta_t, current_velocity, optimal_velocity, data):
	"""Student solution for maintaining rocket throttle through out the launch based on an optimal flight path

    Args:
        delta_t (float): Time step length.
        current_velocity (float): Current velocity of rocket.
        optimal_velocity (float): Optimal velocity of rocket.
        data (dict): Data passed through out run.  Additional data can be added and existing values modified.
            'ErrorP': Proportional error.  Initialized to 0.0
            'ErrorI': Integral error.  Initialized to 0.0
            'ErrorD': Derivative error.  Initialized to 0.0

    Returns:
        Throttle to set, data dictionary to be passed through run.
    """
	throttle = 0
	velocity_error = current_velocity - optimal_velocity
	if 'previous_velocity' in data and 'summation' in data:
		differential_velocity_error = velocity_error - data['previous_velocity']
		summation_of_errors = data['summation'] + velocity_error
	else:
		differential_velocity_error = 0
		summation_of_errors = 0
	if optimal_velocity == 0.25 or optimal_velocity == 0.5:
		throttle = (-rocket_tau_p * velocity_error) - (rocket_tau_d * differential_velocity_error) - (rocket_tau_i * summation_of_errors)
	elif optimal_velocity == -0.1:
		if 'first_time_land' not in data:
			differential_velocity_error = 0
			summation_of_errors = 0
			velocity_error = 0
			data['first_time_land'] = True
		throttle = (-rocket_tau_p * velocity_error) - (rocket_tau_d * differential_velocity_error) - (rocket_tau_i * summation_of_errors)
	# else:
	# 	throttle = 0

	data['previous_velocity'] = velocity_error
	data['summation'] = summation_of_errors
	return throttle, data


def bipropellant_rocket_pid_solution(delta_t, current_velocity, optimal_velocity, data):
	"""Student solution for maintaining fuel and oxidizer throttles through out the launch based on an optimal flight
	path

    Args:
        delta_t (float): Time step length.
        current_velocity (float): Current velocity of rocket.
        optimal_velocity (float): Optimal velocity of rocket.
        data (dict): Data passed through out run.  Additional data can be added and existing values modified.
            'ErrorP': Proportional error.  Initialized to 0.0
            'ErrorI': Integral error.  Initialized to 0.0
            'ErrorD': Derivative error.  Initialized to 0.0

    Returns:
        Fuel Throttle, Oxidizer Throttle to set, data dictionary to be passed through run.
    """

	# TODO: remove naive solution
	# fuel_throttle = optimal_velocity - current_velocity

	fuel_throttle = 0
	oxidizer_throttle = 0
	velocity_error = current_velocity - optimal_velocity
	if 'previous_velocity' in data and 'summation' in data:
		differential_velocity_error = velocity_error - data['previous_velocity']
		summation_of_errors = data['summation'] + velocity_error
	else:
		differential_velocity_error = 0
		summation_of_errors = 0

	if optimal_velocity == 0.25 or optimal_velocity == -0.1:
		fuel_throttle = (-bipropellant_rocket_fuel_tau_p * velocity_error) - (bipropellant_rocket_fuel_tau_d * differential_velocity_error) - (
					bipropellant_rocket_fuel_tau_i * summation_of_errors)
		oxidizer_throttle = (-bipropellant_rocket_oxidizer_tau_p * velocity_error) - (bipropellant_rocket_oxidizer_tau_d * differential_velocity_error) - (
					bipropellant_rocket_oxidizer_tau_i * summation_of_errors)
	# elif :
	# 	# if 'first_time_land' not in data:
	# 	# 	differential_velocity_error = 0
	# 	# 	summation_of_errors = 0
	# 	# 	velocity_error = 0
	# 	# 	data['first_time_land'] = True
	# 	fuel_throttle = (-fp * velocity_error) - (fd * differential_velocity_error) - (
	# 				fi * summation_of_errors)
	# 	oxidizer_throttle = (-op * velocity_error) - (od * differential_velocity_error) - (
	# 				oi * summation_of_errors)
	# else:
	# 	fuel_throttle = 0
	# 	oxidizer_throttle = 0
	data['previous_velocity'] = velocity_error
	data['summation'] = summation_of_errors
	# TODO: implement PID Solution here

	return fuel_throttle, oxidizer_throttle, data
