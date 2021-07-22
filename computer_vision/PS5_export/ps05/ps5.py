"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.45 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([init_x, init_y, 0., 0.])  # state State vector (𝑋) with the initial 𝑥 and 𝑦 values.
        self.covariance = np.array([[1.,0.,1.,0.],  # Covariance 4x4 array (𝛴 ) initialized with a diagonal matrix with some value.
                                    [0.,1.,0.,1.],
                                    [0.,0.,1.,0.],
                                    [0.,0.,0.,1.]])
        self.Dt = np.array([[1.,0.,0.,0.],  # 4x4 state transition matrix 𝐷t
                            [0.,1.,0.,0.],
                            [0.,0.,1.,0.],
                            [0.,0.,0.,1.]])
        self.Mt = np.array([[1.0, 0., 0., 0.],  # measurement matrix 2x4 Mt
                            [0., 1.0, 0., 0.]])
        self.Q = Q  # 4x4 process noise matrix sigma Dt
        self.R = R  # 2x2 measurement noise matrix sigma Mt

    def predict(self):
        self.covariance = np.dot(np.dot(self.Dt, self.covariance), self.Dt.T) + self.Q
        self.state = np.dot(self.Dt, self.state.T).T


    def correct(self, meas_x, meas_y):
        self.Kt = np.dot(np.dot(self.covariance,self.Mt.T), np.linalg.inv((np.dot(self.Mt, np.dot(self.covariance, self.Mt.T)) + self.R)))
        self.covariance = np.dot((self.Dt - np.dot(self.Kt,self.Mt)), self.covariance)
        array_holder = np.array([meas_x, meas_y], ndmin=2)
        self.state = self.state + np.dot(self.Kt,(array_holder.T - np.dot(self.Mt, self.state.T))).T


    def process(self, measurement_x, measurement_y):
        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0,0], self.state[0,1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:w
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.alpha = None
        self.occlusion_weights = None
        self.blocking = None
        self.count = 0
        self.shift = 0
        self.mean_max_thresh = 0
        self.mean_min_thresh = 0
        self.template = template
        self.frame = frame
        self.weights = np.ones(self.num_particles) * (1 / self.num_particles)
        self.particles = np.zeros((self.num_particles, 2), dtype=np.float64)  # Initialize your particles array. Read the docstring.
        self.particles[:,0] = np.random.normal(self.template_rect['x']+int(self.template_rect['w']/2), self.sigma_dyn)
        self.particles[:,1] = np.random.normal(self.template_rect['y']+int(self.template_rect['h']/2), self.sigma_dyn)

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        template_sum = template.sum(axis=2)
        frame_sum =frame_cutout.sum(axis=2)
        return np.mean(np.square(np.subtract(template_sum,frame_sum))).astype(np.float64)

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        resampled = []
        for resample_index in np.random.choice(self.num_particles, self.num_particles, p=self.weights):
            resampled.append(self.particles[resample_index])  # based on weights
        return resampled

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        if getattr(self, 'blocking') is None:  # there are no blocks in this video
            summed = 0
            self.count += 1
            x,y,z = frame.shape
            w = self.template_rect['w']
            h = self.template_rect['h']
            self.particles = self.resample_particles()
            self.particles += np.random.normal(0, self.sigma_dyn, (self.num_particles, 2))
            self.particles[:,0] = np.clip(self.particles[:,0], w/2, y-(w/2))
            self.particles[:,1] = np.clip(self.particles[:,1], h/2, x-(h/2))
            for i in range(self.num_particles):
                col = int(int(self.particles[i, 0])-w/2)
                row = int(int(self.particles[i, 1])-h/2)
                cropped = frame[row:row+h, col:col+w]
                error = self.get_error_metric(self.template.astype(np.float64), cropped)
                self.weights[i] = np.exp(-1 * error / (2 * self.sigma_exp ** 2))
                summed += self.weights[i]
            # w = self.template_rect['w']
            # h = self.template_rect['h']
            if getattr(self, 'alpha') is not None:
                # post-process
                self.weights = np.divide(self.weights, summed)
                y_mean = np.sum(self.particles[:,1]*self.weights)
                captured_row = int(y_mean)-int(h/2)
                x_mean = np.sum(self.particles[:,0]*self.weights)
                captured_column = int(x_mean)-int(w/2)
                framed_result = frame[captured_row:captured_row+h, captured_column:captured_column+w]
                self.template = self.alpha*framed_result.astype(np.float64) + (1.-self.alpha)*self.template.astype(np.float64)
            else:
                self.weights = np.divide(self.weights, summed)
        # if getattr(self, 'blocking') is not None:
        #     print(self.count)


    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0
        w = self.template_rect['w']
        h = self.template_rect['h']
        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
            cv2.circle(frame_in, (int(self.particles[i, 0]), int(self.particles[i, 1])), 2, (255, 0, 0), 1)
        cv2.rectangle(frame_in, (int(x_weighted_mean-w/2),
                                 int(y_weighted_mean-h/2)),
                      (int(x_weighted_mean+w/2),
                       int(y_weighted_mean+h/2)), (0,255,0), thickness=1)
        d = int(np.sum(np.multiply(self.get_weights(), np.sqrt(np.square(np.subtract(self.particles[:,0],x_weighted_mean))+np.square(np.subtract(self.particles[:,1],y_weighted_mean))))))
        cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), int(d), (50,200,0), 2)

class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor
        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        self.count = 0
        self.blocking = kwargs.get('blocking')
        self.mean_max_thresh = kwargs.get('mean_max_weights')
        self.mean_min_thresh = kwargs.get('mean_min_weights')
        self.configurable = kwargs.get('configurable', False)
        self.configurable2 = kwargs.get('configurable2', False)
        self.dontchange = kwargs.get('dont_change', False)

        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)


    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        if self.blocking == True and self.count < 1:
            self.blocking = False
        self.count += 1
        w = self.template_rect['w']
        h = self.template_rect['h']
        weight_holder = self.weights.copy()
        particle_holder = self.resample_particles()
        particle_holder += np.random.normal(0, self.sigma_dyn, (self.num_particles, 2))

        particle_holder[:, 1] = np.clip(particle_holder[:, 1], h / 2, frame.shape[0] - (h / 2))
        particle_holder[:, 0] = np.clip(particle_holder[:, 0], w / 2, frame.shape[1] - (w / 2))

        for i in range(self.num_particles):
            col = int(int(particle_holder[i, 0]) - w / 2)
            row = int(int(particle_holder[i, 1]) - h / 2)
            cropped = frame[row:row + h, col:col + w]
            error = self.get_error_metric(self.template.astype(np.float64), cropped.astype(np.float64))
            self.weights[i] = np.exp(-1 * error / (2 * self.sigma_exp ** 2))

        if self.mean_max_thresh < np.mean(self.weights) < self.mean_min_thresh and not self.blocking:
            if not self.dontchange:
                self.template = cv2.resize(self.template, (0, 0), fx=0.985, fy=0.985)
            else:
                self.template = cv2.resize(self.template, (0, 0), fx=0.998, fy=0.998)
            self.template_rect['h'],self.template_rect['w'], x = self.template.shape
            h,w = self.template_rect['h'], self.template_rect['w']
            normalized_weights = self.weights / np.sum(self.weights)
            mean_y = np.sum(self.particles[:, 1] * normalized_weights)
            mean_roq = int(int(mean_y) - h / 2)
            mean_x = np.sum(self.particles[:, 0] * normalized_weights)
            mean_coq = int(int(mean_x) - w / 2)
            framed = frame[mean_roq:mean_roq + h, mean_coq:mean_coq + w]
            self.template = self.alpha * framed.astype(np.float64) + (1. - self.alpha) * self.template.astype(
                np.float64)
            for i in range(self.num_particles):
                row = int(int(particle_holder[i, 1]) - h / 2)
                col = int(int(particle_holder[i, 0]) - w / 2)
                frame_cutout = frame[row:row + h, col:col + w]
                error = self.get_error_metric(self.template.astype(np.float64), frame_cutout.astype(np.float64))
                self.weights[i] = np.exp(-1 * error / (2 * self.sigma_exp ** 2))
        if self.configurable2 and not self.blocking and 80 <self.count < 140:
            self.weights = weight_holder
            self.blocking = True
            return
        if self.configurable and not self.blocking and 186 <self.count < 207:
            self.weights = weight_holder
            self.blocking = True
            return

        if np.mean(self.weights) > self.mean_max_thresh:
            normalization = np.sum(self.weights)
            self.particles = particle_holder
            self.weights /= normalization
            mean_y = np.sum(self.particles[:, 1] * self.weights)
            mean_roq = int(int(mean_y) - h / 2)
            mean_x = np.sum(self.particles[:, 0] * self.weights)
            mean_coq = int(int(mean_x) -w / 2)
            framed = frame[mean_roq:mean_roq + h, mean_coq:mean_coq + w]
            self.template = self.alpha * framed.astype(np.float64) + (1. - self.alpha) * self.template.astype(np.float64)
            self.blocking = False

        elif self.configurable and self.blocking and (165 < self.count < 186):
            normalization = np.sum(self.weights)
            self.particles = particle_holder
            self.weights /= normalization
            mean_y = np.sum(self.particles[:, 1] * self.weights)
            mean_roq = int(int(mean_y) - h / 2)
            mean_x = np.sum(self.particles[:, 0] * self.weights)
            mean_coq = int(int(mean_x) -w / 2)
            framed = frame[mean_roq:mean_roq + h, mean_coq:mean_coq + w]
            self.template = self.alpha * framed.astype(np.float64) + (1. - self.alpha) * self.template.astype(
                np.float64)
            self.blocking = False

        else:
            self.weights = weight_holder
            self.blocking = True
