dim = 2 * (1 + num_landmarks)

# make the constraint information matrix and vector
Omega = matrix()
Omega.zero(dim, dim)
Omega.value[0][0] = 1.0
Omega.value[1][1] = 1.0

Xi = matrix()
Xi.zero(dim, 1)
Xi.value[0][0] = world_size / 2.0
Xi.value[1][0] = world_size / 2.0

# process the data

for k in range(len(data)):

	# n is the index of the robot pose in the matrix/vector
	# n = k * 2    #.... do not need

	measurement = data[k][0]
	motion = data[k][1]

	# integrate the measurements
	for i in range(len(measurement)):

		# m is the index of the landmark coordinate in the matrix/vector
		m = 2 * (1 + measurement[i][0])

		# update the information maxtrix/vector based on the measurement
		for b in range(2):
			Omega.value[b][b] += 1.0 / measurement_noise
			Omega.value[m + b][m + b] += 1.0 / measurement_noise
			Omega.value[b][m + b] += -1.0 / measurement_noise
			Omega.value[m + b][b] += -1.0 / measurement_noise
			Xi.value[b][0] += -measurement[i][1 + b] / measurement_noise
			Xi.value[m + b][0] += measurement[i][1 + b] / measurement_noise

	# add zero rows for the new position
	Omega = Omega.expand(dim + 2, dim + 2, [0, 1] + range(4, dim + 2), [0, 1] + range(4, dim + 2))
	Xi = Xi.expand(dim + 2, 1, [0, 1] + range(4, dim + 2), [0])

	# update the information maxtrix/vector based on the robot motion
	for b in range(4):
		Omega.value[b][b] += 1.0 / motion_noise
	for b in range(2):
		Omega.value[b][b + 2] += -1.0 / motion_noise
		Omega.value[b + 2][b] += -1.0 / motion_noise
		Xi.value[b][0] += -motion[b] / motion_noise
		Xi.value[b + 2][0] += motion[b] / motion_noise

	matrixB = Omega.take([0, 1], [0, 1])
	matrixA = Omega.take([0, 1], range(2, dim + 2))
	matrixC = Xi.take([0, 1], [0])
	OmegaPrime = Omega.take(range(2, dim + 2), range(2, dim + 2))
	XiPrime = Xi.take(range(2, dim + 2), [0])

	Omega = OmegaPrime - matrixA.transpose() * matrixB.inverse() * matrixA
	Xi = XiPrime - matrixA.transpose() * matrixB.inverse() * matrixC

# compute best estimate
mu = Omega.inverse() * Xi

# return the result
return mu, Omega  # make sure you return both of these matrices to be marked correct.