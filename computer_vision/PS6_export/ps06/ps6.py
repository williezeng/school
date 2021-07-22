"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os
from helper_classes import WeakClassifier, VJ_Classifier

# def show_image(img):
#     cv2.imshow('temp',img)
#     cv2.waitKey(0)

# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of digit_locations (int).
    """
    regex = r"[0-9]+$"
    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    array_of_labels = np.empty(len(images_files))
    flattened_imgs = np.empty(len(images_files))
    X = []
    for i in range(0,len(images_files)):
        image_name = images_files[i]
        img = cv2.imread(os.path.join(folder, image_name))
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_small = cv2.resize(grayed, tuple(size))

        X.append(img_small.flatten())
        prefix_name = str(image_name.split('.')[0])
        found_label = prefix_name[-2:]
        array_of_labels[i] = str(int(found_label))

    flattened_imgs = np.asarray(X)
    return flattened_imgs, array_of_labels

def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the digit_locations
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of digit_locations (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data digit_locations.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data digit_locations.
    """

    num_of_images = len(X)
    random_indices = np.random.permutation(num_of_images)
    ytrain = y[random_indices[:int(p*X.shape[0])]]
    Xtrain = X[random_indices[:int(p*X.shape[0])],:]
    ytest = y[random_indices[int(p*X.shape[0]):]]
    Xtest = X[random_indices[int(p*X.shape[0]):], :]
    return Xtrain, ytrain, Xtest, ytest

def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """

    return np.mean(x, axis=0)

def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """

    mean_face = get_mean_face(X)
    x_minus_mean = np.subtract(X, mean_face)
    summation = np.dot(x_minus_mean.T, x_minus_mean)
    eigenvalues, eigenvectors = np.linalg.eigh(summation)
    sorted_eigenvalue_indices = np.argsort(eigenvalues)[::-1][0:k]
    list_to_array = []
    for index in sorted_eigenvalue_indices:
        list_to_array.append(eigenvalues[index])
    sorted_eigenvalue = np.asarray(list_to_array)
    sorted_eigenvectors = eigenvectors[:,sorted_eigenvalue_indices]
    return sorted_eigenvectors, sorted_eigenvalue


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for iteration in range(self.num_iterations):
            self.weights = self.weights/np.sum(self.weights)
            h = WeakClassifier(self.Xtrain, self.ytrain, self.weights, self.eps)
            h.train()
            hx_holder = []
            for num_ob in range(self.num_obs):
                hx_holder.append(h.predict(self.Xtrain[num_ob]))
            self.weakClassifiers.append(h)
            Ej = np.sum(self.weights[hx_holder != self.ytrain])
            alpha = 0.5*np.log((1-Ej)/Ej)
            self.alphas.append(alpha)
            if Ej > self.eps:
                self.weights = self.weights*np.exp(-1.*self.ytrain*self.alphas[iteration]*hx_holder)
            else:
                return

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training digit_locations (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        predictions = self.predict(self.Xtrain)
        correct = np.sum(predictions == self.ytrain)
        incorrect = np.sum(predictions != self.ytrain)
        return correct, incorrect


    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        sums = np.zeros((X.shape[0]))
        for j in range(len(self.weakClassifiers)):
            wk_results = np.asarray([self.weakClassifiers[j].predict(X[i]) for i in range(X.shape[0])])
            scaled_results = self.alphas[j] * wk_results
            sums += scaled_results
        predictions = np.sign(sums)
        return np.asarray(predictions)


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        canvas = np.zeros(shape)
        row_start = self.position[0]
        col_start = self.position[1]

        added_row = int(self.size[0]/2)
        added_col = self.size[1]
        canvas[row_start:row_start + added_row, col_start:col_start + added_col] = 255 # top half
        canvas[row_start + added_row:row_start + added_row + added_row, col_start: col_start + added_col] = 126 # bottom half
        return canvas


    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        canvas = np.zeros(shape)
        row_start = self.position[0]
        col_start = self.position[1]

        added_row = self.size[0]
        added_col = int(self.size[1]/2)
        canvas[row_start:row_start + added_row, col_start: col_start + added_col] = 255 # left half
        canvas[row_start:row_start + added_row, col_start + added_col:col_start + added_col + added_col] = 126 # right half
        return canvas

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        canvas = np.zeros(shape)
        row_start = self.position[0]
        col_start = self.position[1]

        added_row = int(self.size[0]/3)
        added_col = self.size[1]
        canvas[row_start:row_start + added_row, col_start: col_start + added_col] = 255 # top third
        canvas[row_start + added_row: row_start + added_row + added_row, col_start: col_start + added_col] = 126 # middle third
        canvas[row_start + added_row + added_row: row_start + added_row + added_row + added_row, col_start: col_start + added_col] = 255 # bottom third
        return canvas

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        canvas = np.zeros(shape)
        row_start = self.position[0]
        col_start = self.position[1]

        added_row = self.size[0]
        added_col = int(self.size[1]/3)
        canvas[row_start:row_start + added_row, col_start: col_start + added_col] = 255 # left third
        canvas[row_start:row_start + added_row, col_start + added_col:col_start + added_col + added_col] = 126 # middle third
        canvas[row_start:row_start + added_row, col_start + added_col + added_col: col_start + added_col + added_col + added_col] = 255 # right third

        return canvas


    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        canvas = np.zeros(shape)
        row_start = self.position[0]
        col_start = self.position[1]

        added_row = int(self.size[0]/2)
        added_col = int(self.size[1]/2)
        canvas[row_start:row_start + added_row, col_start: col_start + added_col] = 126 # left top
        canvas[row_start:row_start + added_row, col_start + added_col:col_start + added_col + added_col] = 255 # right top
        canvas[row_start + added_row: row_start + added_row + added_row, col_start: col_start + added_col] = 255 # left bottom
        canvas[row_start + added_row: row_start + added_row + added_row, col_start + added_col:col_start + added_col + added_col] = 126 # right bottom

        return canvas


    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        # show_image(X)
        if filename is None:

            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """

        final_score = None
        white_area = None
        grey_area = None
        row_start = self.position[0]
        col_start = self.position[1]
        canvas = np.zeros((ii.shape[0]+1, ii.shape[1]+1))
        canvas[1:, 1:] = ii
        if self.feat_type == (2, 1):  # top white, bottom grey
            # split white into a,b,c,d points
            added_row = int(self.size[0] / 2)
            added_col = self.size[1]

            white_a = (row_start, col_start)
            white_b = (row_start, col_start + added_col)
            white_c = (row_start + added_row, col_start)
            white_d = (row_start + added_row, col_start + added_col)
            white_area = canvas[white_d] - canvas[white_b] - canvas[white_c] + canvas[white_a]


            grey_a = (row_start + added_row, col_start)
            grey_b = (row_start + added_row, col_start + added_col)
            grey_c = (row_start + added_row + added_row, col_start)
            grey_d = (row_start + added_row + added_row, col_start+ added_col)

            grey_area = canvas[grey_d] - canvas[grey_b] - canvas[grey_c] + canvas[grey_a]
            final_score = white_area - grey_area

        if self.feat_type == (1, 2): # left white , right grey
            # split white into a,b,c,d points
            added_row = self.size[0]
            added_col = int(self.size[1] / 2)

            white_a = (row_start, col_start)
            white_b = (row_start, col_start + added_col)
            white_c = (row_start + added_row, col_start)
            white_d = (row_start + added_row, col_start + added_col)
            white_area = canvas[white_d] - canvas[white_b] - canvas[white_c] + canvas[white_a]


            grey_a = (row_start, col_start + added_col)
            grey_b = (row_start, col_start + added_col + added_col)
            grey_c = (row_start + added_row, col_start + added_col)
            grey_d = (row_start + added_row, col_start + added_col + added_col)
            grey_area = canvas[grey_d] - canvas[grey_b] - canvas[grey_c] + canvas[grey_a]
            final_score = white_area - grey_area

        elif self.feat_type == (3, 1): # top white, middle grey, bottom white
            # split white into a,b,c,d points
            added_row = int(self.size[0]/3)
            added_col = self.size[1]

            # top white
            white_a = (row_start, col_start)
            white_b = (row_start, col_start + added_col)
            white_c = (row_start + added_row, col_start)
            white_d = (row_start + added_row, col_start + added_col)
            top_white_area = canvas[white_d] - canvas[white_b] - canvas[white_c] + canvas[white_a]

            # middle grey
            grey_a = (row_start + added_row, col_start)
            grey_b = (row_start + added_row, col_start + added_col)
            grey_c = (row_start + added_row + added_row, col_start)
            grey_d = (row_start + added_row + added_row, col_start+ added_col)
            grey_area = canvas[grey_d] - canvas[grey_b] - canvas[grey_c] + canvas[grey_a]

            # bottom_white_area
            white_a = (row_start + added_row + added_row, col_start)
            white_b = (row_start + added_row + added_row, col_start + added_col)
            white_c = (row_start + added_row + added_row + added_row, col_start)
            white_d = (row_start + added_row + added_row + added_row, col_start + added_col)
            bottom_white_area = canvas[white_d] - canvas[white_b] - canvas[white_c] + canvas[white_a]
            final_score = top_white_area + bottom_white_area - grey_area

        elif self.feat_type == (1, 3):  # left white , middle grey , right white
            # split white into a,b,c,d points
            added_row = self.size[0]
            added_col = int(self.size[1] / 3)

            # LEFT WHITE
            white_a = (row_start, col_start)
            white_b = (row_start, col_start + added_col)
            white_c = (row_start + added_row, col_start)
            white_d = (row_start + added_row, col_start + added_col)
            left_white_area = canvas[white_d] - canvas[white_b] - canvas[white_c] + canvas[white_a]

            # MIDDLE GREY
            grey_a = (row_start, col_start + added_col)
            grey_b = (row_start, col_start + added_col + added_col)
            grey_c = (row_start + added_row, col_start + added_col)
            grey_d = (row_start + added_row, col_start + added_col + added_col)
            grey_area = canvas[grey_d] - canvas[grey_b] - canvas[grey_c] + canvas[grey_a]

            # RIGHT WHITE
            white_a = (row_start, col_start + added_col + added_col)
            white_b = (row_start, col_start + added_col + added_col + added_col)
            white_c = (row_start + added_row, col_start + added_col + added_col)
            white_d = (row_start + added_row, col_start + added_col + added_col + added_col)
            right_white_area = canvas[white_d] - canvas[white_b] - canvas[white_c] + canvas[white_a]
            final_score = left_white_area + right_white_area - grey_area

        elif self.feat_type == (2, 2):  # top left grey, top right white, bottom left white, bottom right grey
            # split white into a,b,c,d points
            added_row = int(self.size[0] / 2)
            added_col = int(self.size[1] / 2)

            grey_a = (row_start, col_start)
            grey_b = (row_start, col_start + added_col)
            grey_c = (row_start + added_row, col_start)
            grey_d = (row_start + added_row, col_start + added_col)
            top_left_grey = canvas[grey_d] - canvas[grey_b] - canvas[grey_c] + canvas[grey_a]

            white_a = (row_start, col_start + added_col)
            white_b = (row_start, col_start + added_col + added_col)
            white_c = (row_start + added_row, col_start + added_col)
            white_d = (row_start + added_row, col_start + added_col + added_col)
            top_right_white = canvas[white_d] - canvas[white_b] - canvas[white_c] + canvas[white_a]

            white_a = (row_start + added_row, col_start)
            white_b = (row_start + added_row, col_start + added_col)
            white_c = (row_start + added_row + added_row, col_start)
            white_d = (row_start + added_row + added_row, col_start+ added_col)
            bottom_left_white = canvas[white_d] - canvas[white_b] - canvas[white_c] + canvas[white_a]

            grey_a = (row_start + added_row, col_start + added_col)
            grey_b = (row_start + added_row, col_start + added_col + added_col)
            grey_c = (row_start + added_row + added_row, col_start + added_col)
            grey_d = (row_start + added_row + added_row, col_start + added_col + added_col)
            bottom_right_grey = canvas[grey_d] - canvas[grey_b] - canvas[grey_c] + canvas[grey_a]

            final_score = top_right_white + bottom_left_white - (top_left_grey + bottom_right_grey)
        return final_score


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    list_of_integrals = []
    for i in range(len(images)):
        height, width = images[i].shape
        sum_over_rows = np.cumsum(images[i], axis=0)
        sum_over_cols = np.cumsum(sum_over_rows, axis=1)
        list_of_integrals.append(sum_over_cols)
    return list_of_integrals


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative digit_locations.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print(" -- select classifiers --")
        predictions_holder = []
        for i in range(num_classifiers):
            ei = np.ones(len(weights), dtype='float')
            ones_holder = np.ones(len(weights), dtype='float')
            weights = weights/np.sum(weights)
            hj = VJ_Classifier(scores, self.labels, weights)
            hj.train()
            self.classifiers.append(hj)
            error = hj.error
            Beta = error/(1.-error)
            for i in range(len(self.integralImages)):
                predictions_holder.append(hj.predict(scores[i]))
            predictions = np.asarray(predictions_holder)
            for i in range(len(self.labels)):
                if predictions[i] == self.labels[i]:
                    ei[i] = -1.0
            exponent = np.subtract(ones_holder, ei)
            weights = weights * np.power(Beta, exponent)
            self.alphas.append(np.log(1.0/Beta))

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        result = []
        classifier= None
        range_maker = range(len(self.classifiers))
        for z in range(len(ii)):
            for classifier_index in range_maker:
                classifier = self.classifiers[classifier_index]
                f_index = classifier.feature
                haar = self.haarFeatures[f_index]
                scores[z, f_index] = haar.evaluate(ii[z])

        for x in range(len(scores)):
            holder = 0
            for z in range_maker:
                holder += self.alphas[z] * self.classifiers[z].predict(scores[x])
            temp_alpha = (1/2) * np.sum(self.alphas)
            if holder <= temp_alpha:
                result.append(-1)
            else:
                result.append(1)

        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """

        image_copy = np.copy(image)
        grayed = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        rowss, columnss = grayed.shape
        if rowss > 24 and columnss > 24:
            x_list = []
            y_list = []
            for rows in range(rowss-24):
                for columns in range(columnss-24):
                    if self.predict([grayed[rows:rows+24, columns:columns+24]]) == [1]:
                        x_list.append(rows)
                        y_list.append(columns)
            x = int(np.mean(np.asarray(y_list)))
            y = int(np.mean(np.asarray(x_list)))
            cv2.rectangle(image_copy, (x-4,y),(x+20, y+24),(0,255,0), thickness=2)
            cv2.imwrite("output/{}.png".format(filename), image_copy)