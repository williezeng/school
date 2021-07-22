"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2



def show_image(image, name=None):
    if name:
        cv2.imshow(name, image)
        cv2.waitKey(0)
    else:
        cv2.imshow('', image)
        cv2.waitKey(0)


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    img_copy = np.copy(image)
    # ddepth = If you use -1, the result (destination) image will have the same depth as the input (source) image.
    return cv2.Sobel(img_copy, ddepth=-1, dx=1, dy=0, ksize=3, scale=0.125)

def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    img_copy = np.copy(image)
    # ddepth = If you use -1, the result (destination) image will have the same depth as the input (source) image.
    return cv2.Sobel(img_copy, ddepth=-1, dx=0, dy=1, ksize=3, scale=0.125)

def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    img_a_copy = np.copy(img_a)
    img_b_copy = np.copy(img_b)

    if k_type == 'gaussian':
        blurred_a = cv2.GaussianBlur(img_a_copy, ksize=(k_size, k_size), sigmaX=sigma, sigmaY=sigma)
        blurred_b = cv2.GaussianBlur(img_b_copy, ksize=(k_size, k_size), sigmaX=sigma, sigmaY=sigma)
        Ix = gradient_x(blurred_a)
        Iy = gradient_y(blurred_a)
        It = (blurred_a - blurred_b).astype(np.float64)
    else:
        Ix = gradient_x(img_a_copy)
        Iy = gradient_y(img_a_copy)
        It = (img_a_copy - img_b_copy).astype(np.float64)
    xt = cv2.boxFilter(-Ix*It, -1, ksize=(k_size, k_size))
    yt = cv2.boxFilter(-Iy*It, -1, ksize=(k_size, k_size))

    xy = cv2.boxFilter(-Ix*Iy, -1, ksize=(k_size, k_size))
    xx = cv2.boxFilter(Ix**2, -1, ksize=(k_size, k_size))
    yy = cv2.boxFilter(Iy**2, -1, ksize=(k_size, k_size))

    U = -(yy * xt + xy * yt)
    V = -(xy * xt + xx * yt)
    determinants = (xx * yy) - (xy**2)
    #  check for edge case values
    U = np.nan_to_num(U)
    V = np.nan_to_num(V)
    U[np.isinf(U)] = 0.0
    V[np.isinf(V)] = 0.0
    U = np.where(determinants == 0, 0, U / determinants).astype(np.float64)
    V = np.where(determinants == 0, 0, V / determinants).astype(np.float64)
    return (U, V)


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    image_copy = image.copy()
    separable_filter = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])
    reduced_layer = cv2.sepFilter2D(image_copy, -1, separable_filter, separable_filter)
    return reduced_layer[::2, ::2]

def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    list_of_pyramids = []
    image_copy = np.copy(image)
    for level in range(0, levels):
        list_of_pyramids.append(image_copy)
        image_copy = reduce_image(image_copy)
    return list_of_pyramids

def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    toggle = True
    final = None
    original_height = None
    for image in img_list:
        image_copy = image.copy()
        if toggle:
            original_height = image_copy.shape[0]
            final = normalize_and_scale(image_copy)
            toggle = False
        else:
            new_height, new_width = image_copy.shape
            image_holder = np.zeros((original_height, new_width))
            image_holder[:new_height, :] = image_copy
            final = np.hstack((final, normalize_and_scale(image_holder)))
    return final

def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    image_copy = np.copy(image)
    doubled_image = np.zeros((image_copy.shape[0] * 2, image_copy.shape[1] * 2))
    doubled_image[::2,::2]=image_copy[:,:]
    separable_filter = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])
    doubled_image = cv2.sepFilter2D(doubled_image, -1, separable_filter, separable_filter)
    return 4.0 * doubled_image


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """

    lap_pyramids = []
    for i in range(len(g_pyr)-1):
        # expand the next image
        pyramid = g_pyr[i]
        pyramid_h, pyramid_w = pyramid.shape
        expanded = expand_image(g_pyr[i+1])
        # expanded_h, expanded_w = expanded.shape
        if expanded.shape != pyramid.shape:
            expanded = expanded[0:pyramid_h, 0:pyramid_w]
        lap_pyramids.append(pyramid - expanded)
    lap_pyramids.append(g_pyr[-1])

    return lap_pyramids

def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    image_copy = image.copy()
    height, width = image_copy.shape
    X, Y = np.meshgrid(range(width), range(height))
    mapy = (Y + V)
    mapx = (X + U)
    warped = cv2.remap(image_copy, mapx.astype(np.float32), mapy.astype(np.float32), interpolation=interpolation, borderMode=border_mode)
    return warped

def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    image_a_copy = img_a.copy()
    image_b_copy = img_b.copy()
    pyramid_a = gaussian_pyramid(image_a_copy, levels)
    pyramid_b = gaussian_pyramid(image_b_copy, levels)
    try:
        U, V = optic_flow_lk(pyramid_a[levels-1], pyramid_b[levels-1], k_size, k_type, sigma)
    except:
        U, V = None, None

    for i in range(levels-2,-1,-1):
        base_u = 2 * expand_image(U)
        base_v = 2 * expand_image(V)
        WARPED = warp(pyramid_b[i], base_u, base_v, interpolation, border_mode)
        flow_u, flow_v = optic_flow_lk(pyramid_a[i], WARPED, k_size, k_type, sigma)
        U = base_u + flow_u
        V = base_v + flow_v
    return U,V
