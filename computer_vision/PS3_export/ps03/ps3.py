"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


def show_image(image, name=None):
    if name:
        cv2.imshow(name, image)
        cv2.waitKey(0)
    else:
        cv2.imshow('', image)
        cv2.waitKey(0)



def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    return np.linalg.norm((p0 - p1))

def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    image_copy = np.copy(image)

    top_left = (0,0)
    top_right = (image_copy.shape[1]-1,0)
    bottom_left = (0,image_copy.shape[0]-1)
    holder = image_copy.shape[0:2]
    bottom_right = (holder[1]-1, holder[0]-1)
    return [top_left, bottom_left, top_right, bottom_right]


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    ksize = (11,13)
    image_copy = np.copy(image)
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, ksize)
    # Blur and use corner detection with kmeans
    corners_found = cv2.cornerHarris(gray_blurred, 5, 19, 0, 1)
    X, Y = np.nonzero(corners_found > 0.01 * corners_found.max())
    corner_points = np.array([[x, y] for x, y in zip(X, Y)], dtype=np.float32)
    ret, label, centers = cv2.kmeans(corner_points, 4, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
    # sorted from left to right
    centers_sorted = centers[centers[:, 1].argsort()]
    saved_index = None
    top_left = None
    top_right = None
    if len(centers_sorted) == 4:
        # go through left points to find top left and bottom left
        toppest = 99999
        for counter in range(0,2):
            center_yx = centers_sorted[counter]
            y = center_yx[0]
            x = center_yx[1]
            if y < toppest:
                toppest = y
                top_left = (int(x),int(y))
                saved_index = counter
        # delete the points from our list after saving them
        missing_top_left = np.delete(centers_sorted, saved_index, 0)
        bottom_left = (int(missing_top_left[0][1]), int(missing_top_left[0][0]))
        missing_left = np.delete(missing_top_left, 0, 0)
        # go through remaining right points to find top right and bottom right
        toppest = 99999
        for counter in range(0,2):
            center_yx = missing_left[counter]
            y = center_yx[0]
            x = center_yx[1]
            if y < toppest:
                toppest = y
                top_right = (int(x),int(y))
                saved_index = counter
        missing_left_and_top_right = np.delete(missing_left, saved_index, 0)
        bottom_right = (int(missing_left_and_top_right[0][1]), int(missing_left_and_top_right[0][0]))
    else:
        # we cant determine the center
        return None, None, None, None
    return top_left, bottom_left, top_right, bottom_right


def draw_box(image, markers, thickness=3):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """

    image_copy = image.copy()
    first_marker = markers[0]
    second_marker = markers[1]
    third_marker = markers[2]
    fourth_marker = markers[3]
    cv2.line(image_copy, first_marker, second_marker, (0,0,0), thickness)
    cv2.line(image_copy, first_marker, third_marker, (0,0,0), thickness)
    cv2.line(image_copy, third_marker, fourth_marker, (0,0,0), thickness)
    cv2.line(image_copy, fourth_marker, second_marker, (0,0,0), thickness)
    return image_copy



def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    # import pdb
    # pdb.set_trace()
    A_copy = np.copy(imageA)
    B_copy = np.copy(imageB)
    projected_x = B_copy.shape[0]
    projected_y = B_copy.shape[1]
    matrix_holder = np.zeros(shape=(3,projected_x * projected_y))

    grid = np.indices((projected_x, projected_y)).astype(np.float32)
    pixelsX = grid[1].shape[0]*grid[1].shape[1]
    # grab x
    matrix_holder[0] = np.reshape(grid[1], (1, pixelsX))
    # grab y
    matrix_holder[1] = np.reshape(grid[0], (1, pixelsX))
    matrix_holder[2,:] = 1

    projected_matrix=None
    if np.count_nonzero(matrix_holder[2]) == pixelsX:
        projected_matrix = np.dot(np.linalg.inv(homography), matrix_holder)
    xx = (projected_matrix[0]/projected_matrix[2]).tolist()
    yy = (projected_matrix[1]/projected_matrix[2]).tolist()
    xx = np.reshape(xx, (projected_x, projected_y))
    yy = np.reshape(yy, (projected_x, projected_y))
    retImage = np.copy(B_copy).astype(np.float32)
    retImage = cv2.remap(A_copy, xx.astype(np.float32), yy.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return retImage



def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    if len(src_points) == len(dst_points):
        mholder = np.zeros((len(src_points), 2, 9))
        for i in range(len(src_points)):
            Sx, Sy = src_points[i]
            Dx, Dy = dst_points[i]
            mholder[i] = [[-Sx, -Sy, -1, 0, 0, 0, Sx*Dx, Sy*Dx, Dx],
                     [0, 0, 0, -Sx, -Sy, -1, Sx*Dy, Sy*Dy, Dy]]
        rowwise = np.vstack((mholder[0], mholder[1], mholder[2], mholder[3]))
        matrix_holder = np.linalg.svd(rowwise)[2]
        Homography = matrix_holder[-1, :] / matrix_holder[-1, -1]
        Homography = Homography.reshape(3, 3)
    else:
        print('the lengths are not equal')
        return None
    return Homography


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    if filename:
        video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None
