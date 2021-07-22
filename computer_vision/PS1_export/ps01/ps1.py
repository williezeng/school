import math
import numpy as np
import cv2
import sys

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    img1_red = image[:,:,2]
    return img1_red

def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    # create monochrome images using the green and red channels of the orig. img
    img1_green = image[:,:,1]
    return img1_green


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    raise NotImplementedError


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    img1_swapped = image.copy()
    img1_swapped[:,:,1], img1_swapped[:,:,0] = image[:,:,0], image[:,:,1]
    return img1_swapped


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    image1 = src.copy()
    image2 = dst.copy()
    img1_center = []
    img2_center = []
    if shape[0] % 2 == 0:  # if even shape
        x = int(shape[0] / 2)
        y = int(shape[1] / 2)
        img1_center = [dim//2 for dim in image1.shape[:2]]
        img2_center = [dim//2 for dim in image2.shape[:2]]
    elif shape[0] % 2 != 0:  # if odd shape, even subtracts 1
        x = shape[0] / 2
        y = shape[1] / 2
        img1_center = find_center(image1)
        img2_center = find_center(image2)
    if len(image1.shape) == 3:
        image1_mono = image1[:, :, 1]
    else:
        image1_mono = image1.copy()
    image2[int(img2_center[0]-x):int(img2_center[0]+x), int(img2_center[1]-y):int(img2_center[1]+y)] = \
        image1_mono[int(img1_center[0]-x):int(img1_center[0]+x), int(img1_center[1]-y):int(img1_center[1]+y)]
    return image2


def find_center(image_shape):
    img_center = []
    holder_center = image_shape.shape[:2]
    if holder_center[0] % 2 == 0:  # if x is even
        img_center.append((holder_center[0] - 1) / 2)
    else:  # if x is odd
        img_center.append((holder_center[0]) / 2)

    if holder_center[1] % 2 == 0:  # if y is even
        img_center.append((holder_center[1] - 1)/2)
    else:  # if y is odd
        img_center.append((holder_center[1])/2)
    return img_center


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    image_copy = image.copy()
    min = float(image_copy.min())
    max = float(image_copy.max())
    mean = float(image_copy.mean())
    std = float(image_copy.std())
    return min, max, mean, std


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """

    image_copy = image.copy()
    mean = image_copy.mean()
    std = image_copy.std()

    subtracted = image_copy - mean
    divided = subtracted/std
    multiplied = divided * scale
    added = multiplied + mean

    return added


def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    image_copy = image.copy()
    num_rows, num_cols = image.shape

    shortened_image = image_copy[:, 0:num_cols - shift]
    shortened_image[:, 0:num_cols - shift] = image_copy[:, shift:]
    shifted_image = cv2.copyMakeBorder(shortened_image, top=0, bottom=0, left=0, right=shift, borderType=cv2.BORDER_REPLICATE)
    return shifted_image

def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    img1 = img1.astype('float64')
    img2 = img2.astype('float64')
    diff = img1 - img2
    normalized = diff.copy()
    normalized = cv2.normalize(normalized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return normalized

def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    image_copied = image.copy()
    gauss = np.random.normal(0, sigma, (image.shape[0], image.shape[1]))
    image_copied[:, :, channel] = image_copied[:, :, channel] + gauss
    return image_copied
