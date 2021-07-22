"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np
import math

#
# def show_image(name, img):
#     cv2.imshow(name, img)
#     cv2.waitKey(0)

BLUR = False

def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """

    image_copy = img_in.copy()
    state = None
    yellow_light_tuple = None
    gray = image_copy[:,:,2]  # filter out red
    ksize = (6,6)
    gray_blurred = cv2.blur(gray, ksize)
    circles_found = cv2.HoughCircles(image=gray_blurred, method=cv2.HOUGH_GRADIENT, dp=1, minDist=min(radii_range), param1=70, param2=20, minRadius=min(radii_range), maxRadius=max(radii_range))
    found_light = False
    r_light_y =0
    r_light_x = 0

    g_light_x =0
    g_light_y = 0
    if circles_found is not None:
        detected_circles = np.uint16(np.around(circles_found))[0]

        if detected_circles is not None:
            # g_light_y = detected_circles.max(axis=0)[1]  # look at the max y
            # r_light_y = detected_circles.min(axis=0)[1]  # look at the min y
            for (x, y, r) in detected_circles:
                #  the yellow light is in between green and red

                if image_copy[y, x][0] < 35 and image_copy[y, x][1] > 105 and image_copy[y, x][2] < 30:
                    if image_copy[y+r+3, x+r+3][0] < 80 and image_copy[y+r+3, x+r+3][1] < 80 and image_copy[y+r+3, x+r+3][2] < 80:

                        g_light_y = y
                        g_light_x = x
                # elif y == detected_circles.min(axis=0)[1]:  # look at the min y (red)
                elif image_copy[y, x][0] < 35 and image_copy[y, x][1] < 35 and image_copy[y, x][2] > 100:
                    if image_copy[y+r+3, x+r+3][0] < 80 and image_copy[y+r+3, x+r+3][1] < 80 and image_copy[y+r+3, x+r+3][2] < 80:
                        r_light_y = y
                        r_light_x = x
            potential_yellow = []
            yellow_light_tuple = (0,0)
            for (x, y, r) in detected_circles:
                for ranger in range(0, 5, 1):
                    if x == (g_light_x + ranger) or x == (g_light_x - ranger):
                        potential_yellow.append((x,y))
                        found_light = True

            if not found_light:
                for (x, y, r) in detected_circles:
                    for ranger in range(0, 5, 1):
                        if x == (r_light_x + ranger) or x == (r_light_x - ranger):
                            potential_yellow.append((x, y))
                            found_light = True

            # if r_light_y - 10 < y < g_light_y + 10:
            #     cv2.circle(image_copy, ((x + 12, y + 12)), 1, (255, 255, 255), 10)
            #
            #         # import pdb
            #         # pdb.set_trace()
            #         if image_copy[y, x][0] < 30 and image_copy[y, x][1] > 100 and image_copy[y, x][2] > 100:

            if found_light:
                for yellow in potential_yellow:
                    donex = yellow[0]
                    doney = yellow[1]
                    if image_copy[doney, donex][0] < 35 and image_copy[doney, donex][1] > 50 and image_copy[doney, donex][2] > 50:
                        yellow_light_tuple = yellow

                if image_copy[r_light_y, r_light_x][0] < 5 and image_copy[r_light_y, r_light_x][1] < 5 and image_copy[r_light_y, r_light_x][2] > 250:
                    state = 'red'

                if image_copy[g_light_y, g_light_x][0] < 5 and image_copy[g_light_y, g_light_x][1] > 250 and image_copy[g_light_y, g_light_x][2] < 5:
                    state = 'green'
                if image_copy[yellow_light_tuple[1], yellow_light_tuple[0]][1] > 220 and \
                        image_copy[yellow_light_tuple[1], yellow_light_tuple[0]][2] > 200 and \
                        image_copy[yellow_light_tuple[1], yellow_light_tuple[0]][0] < 25:
                    state = "yellow"

    return (yellow_light_tuple, state)


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    image_copy = img_in.copy()
    best_guess = (0,0)
    point_dict = {}
    potential_bottom = []
    counter = 1
    list_of_points = []
    potential_top = []
    bottom_middle = None
    found_it = False
    gray = image_copy[:, :, 1]  # filter out
    if BLUR:
        ksize = (3,3)
    else:
        ksize = (2,2)
    gray_blurred = cv2.blur(gray, ksize)
    edges = cv2.Canny(gray_blurred, 160, 220)
    # lines_found = cv2.HoughLinesP(image=edges, rho=2, theta=np.pi / 6, threshold=4, minLineLength=18, maxLineGap=2)
    lines_found = cv2.HoughLinesP(image=edges, rho=2, theta=np.pi / 8, threshold=1, minLineLength=1, maxLineGap=0)

    if lines_found is not None:
        for line in lines_found:
            (x1, y1, x2, y2) = line[0]
            distance = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


            if distance not in point_dict:
                point_dict[distance] = (x1, y1, x2, y2)
            else:  # compile all the lines with the same distance
                ref_x1, ref_y1, ref_x2, ref_y2 = point_dict[distance]
                if (x1,y1) not in list_of_points:
                    list_of_points.append((x1, y1))
                if (x2, y2) not in list_of_points:
                    list_of_points.append((x2, y2))
                if (ref_x1, ref_y1) not in list_of_points:
                    list_of_points.append((ref_x1, ref_y1))
                if (ref_x2, ref_y2) not in list_of_points:
                    list_of_points.append((ref_x2, ref_y2))
    potential_bottom_middle = (0,0)
    largest_y_value = 0
    for x,y in list_of_points:

        # find bottom middle first
        # look at close top = red
        if image_copy[y - 5, x][0] < 30 and image_copy[y - 5, x][1] < 30 and image_copy[y - 5, x][2] > 220:
            # cv2.circle(image_copy, (x, y), 1, (0, 0, 0), 2)
            # look at far top = white

            if image_copy[y - 35, x][0] > 200 and image_copy[y - 35, x][1] > 200 and image_copy[y - 35, x][2] > 200:
                if image_copy[y - 20, x + 10][0] < 30 and image_copy[y - 30, x + 10][1] < 30 and image_copy[y - 30, x + 10][2] > 200:
                    if image_copy[y - 20, x + 10][0] < 30 and image_copy[y - 30, x + 10][1] < 30 and image_copy[y - 30, x + 10][2] > 200:
                        if image_copy[y - 20, x - 10][0] < 30 and image_copy[y - 30, x - 10][1] < 30 and image_copy[y - 30, x - 10][2] > 200:
                            # found bottom middle
                            if y > largest_y_value:
                                largest_y_value = y
                                potential_bottom_middle = (x,y)

    # cv2.circle(image_copy, potential_bottom_middle, 1, (0, 0, 0), 2)

    # we want the smalledst potential middle

    # find top
    potential_points = []
    second_string = []
    if potential_bottom_middle:
        potential_points = []
        for x,y in list_of_points:
            for second_x,second_y in list_of_points[counter:]:
                if y == second_y:
                    filteredx = int((second_x + x)/2)
                    filteredy = int((second_y + y)/2)
                    #  the bottom middle y point needs to be same as middle y point
                    for x in range(0, 2, 1):
                        if filteredx == int(potential_bottom_middle[0]) + x or filteredx == int(potential_bottom_middle[0]) - x:
                            # point needs to be white
                            if image_copy[filteredy, filteredx][0] > 200 and image_copy[filteredy, filteredx][1] > 200 and image_copy[filteredy, filteredx][2] > 200:
                                # surrounding needs to be white
                                if image_copy[filteredy-5, filteredx][0] > 200 and image_copy[filteredy - 5, filteredx][1] > 200 and image_copy[filteredy - 5, filteredx][2] > 200:
                                    if image_copy[filteredy+15, filteredx][0] > 200 and image_copy[filteredy + 15, filteredx][1] > 200 and image_copy[filteredy + 15, filteredx][2] > 200:
                                        if image_copy[filteredy - 15, filteredx][0] > 200 and image_copy[filteredy - 15, filteredx][1] > 200 and image_copy[filteredy - 15, filteredx][2] > 200:
                                            if image_copy[filteredy , filteredx+ 15][0] > 200 and image_copy[filteredy, filteredx+ 15][1] > 200 and image_copy[filteredy, filteredx+ 15][2] > 200:
                                                # looking way top = red
                                                if image_copy[filteredy - 35, filteredx][0] < 40 and image_copy[filteredy - 35, filteredx][1] < 40 and image_copy[filteredy - 35, filteredx][2] > 200:
                                                    # if image_copy[filteredy + 30, filteredx][0] < 30 and image_copy[filteredy + 30, filteredx][1] < 30 and image_copy[filteredy + 30, filteredx][2] > 220:
                                                    potential_points.append((filteredx, filteredy))
                                                elif image_copy[filteredy - 30, filteredx][0] < 40 and image_copy[filteredy - 30, filteredx][1] < 40 and image_copy[filteredy - 30, filteredx][2] > 200:
                                                    second_string.append((filteredx, filteredy))
                                                # best_guess = (filteredx, filteredy)
            counter += 1
    if len(potential_points) > 0:
        potential_points.sort()
        z = len(potential_points)
        best_guess = potential_points[int(z/2)]
    elif len(second_string) > 0:
        second_string.sort()
        z = len(second_string)
        best_guess = second_string[int(z/2)]
        best_guess = (best_guess[0],best_guess[1]+5)
    #     show_image('d', image_copy)

    return best_guess


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    image_copy = img_in.copy()
    best_guess = None
    left = (0,0)
    found_it = False
    bottom = (0,0)
    list_of_points = []
    potential_middles = []
    top = (0,0)
    gray = image_copy[:, :, 1]  # filter out
    if BLUR:
        ksize = (3,3)
    else:
        ksize = (2,2)
    gray_blurred = cv2.blur(gray, ksize)
    edges = cv2.Canny(gray_blurred, 100, 200)
    lines_found = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 2, threshold=1, minLineLength=20, maxLineGap=1)
    if lines_found is not None:
        for line in lines_found:
            (x1, y1, x2, y2) = line[0]


            med_x = int((x1 + x2) / 2)
            med_y = int((y1 + y2) / 2)
            list_of_points.append(med_x)
            try:
                # look bottom =  red
                if image_copy[med_y + 10, med_x][0] < 30 and image_copy[med_y + 10, med_x,][1] < 30 and image_copy[med_y + 10, med_x][2] > 170:
                    # look a little right = red
                    if image_copy[med_y, med_x+5][0] < 50 and image_copy[med_y, med_x + 5][1] < 50 and image_copy[med_y, med_x + 5][2] > 170:
                        # either top left or top will make it this far
                        # cv2.circle(image_copy, (x2, y2), 1, (0, 0, 0), 2)
                        potential_middles.append((med_x, med_y))
            except:
                continue  # dont care about points outside the map

        for potential_middle in potential_middles:
            potX = potential_middle[0]
            potY = potential_middle[1]

            # to find top middle, look a little left = red
            if image_copy[potY, potX - 10][0] < 30 and image_copy[potY, potX - 10][1] < 30 and 170 < image_copy[potY, potX - 10][2] < 254:
                if image_copy[potY + 80, potX ][0] < 30 and image_copy[potY + 80, potX ][1] < 30 and 170 < image_copy[potY+ 80, potX ][2] < 254:
                    top = potX, potY
                    # cv2.circle(image_copy, top, 1, (0, 0, 0), 2)


            # to find left middle, look a little top = red
            elif image_copy[potY - 6, potX][0] < 50 and image_copy[potY - 6, potX][1] < 50 and 150 < image_copy[potY - 6, potX][2] < 254:
                if image_copy[potY, potX + 8][0] < 50 and image_copy[potY , potX + 8][1] < 50 and 150 < image_copy[potY, potX + 8][2] < 254:
                    left = potX, potY
                    cv2.circle(image_copy, left, 1, (0, 0, 0), 2)

        if top != (0,0) and left == (0,0):
            for line in lines_found:
                (x1, y1, x2, y2) = line[0]
                bottomX = int((x1 + x2) / 2)
                bottomY = int((y1 + y2) / 2)
                # potential bottom look up = red
                # cv2.circle(image_copy, (bottomX, bottomY), 1, (0, 0, 0), 2)
                if image_copy[bottomY - 10, bottomX][0] < 30 and image_copy[bottomY - 10, bottomX][1] < 30 and image_copy[bottomY - 10, bottomX][2] > 170:
                    if bottomY > int(top[1]) + 10:  # the bottom Y value must be greater than + 10
                        centerYY = int((bottomY + top[1]) / 2)
                        best_guess = bottomX, centerYY
                        left = (0,0)

                        # cv2.circle(image_copy, best_guess, 1, (0, 0, 0), 2)

                        break

        if best_guess is None or left != (0,0):
            for line in lines_found:
                (x1, y1, x2, y2) = line[0]
                rightX = int((x1 + x2) / 2)
                rightY = int((y1 + y2) / 2)
                # potential right , look left = red
                if image_copy[rightY, rightX-5][0] < 50 and image_copy[rightY, rightX - 5][1] < 50 and image_copy[rightY, rightX - 5][2] > 160:
                    # potential right, look up &  right = red
                    if image_copy[rightY - 3, rightX - 3][0] < 50 and image_copy[rightY - 3, rightX - 3][1] < 50 and image_copy[rightY - 3, rightX - 3][2] > 160:

                        if rightX > int(left[0]) + 10:  # the right X value must be greater than + 10
                            centerXX = int((rightX + left[0]) / 2)
                            best_guess = centerXX, rightY
                            # cv2.circle(image_copy, best_guess, 1, (0, 0, 0), 5)

                            break

        return best_guess


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """


    image_copy = img_in.copy()
    best_guess = (0,0)
    point_dict = {}
    potential_bottom = []
    list_of_points = []
    potential_top = []
    found_it = False
    gray = image_copy[:, :, 2]  # filter out red
    if BLUR:
        ksize = (4,4)
    else:
        ksize = (2,2)
    gray_blurred = cv2.blur(gray, ksize)
    edges = cv2.Canny(gray_blurred, 190, 200)
    lines_found = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 4, threshold=40, minLineLength=19, maxLineGap=2)
    if lines_found is not None:
        for line in lines_found:
            (x1, y1, x2, y2) = line[0]
            distance = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
            if distance not in point_dict:
                point_dict[distance] = (x1, y1, x2, y2)
            else:  # compile all the lines with the same distance
                ref_x1, ref_y1, ref_x2, ref_y2 = point_dict[distance]
                if (x1,y1) not in list_of_points:
                    list_of_points.append((x1, y1))
                if (x2, y2) not in list_of_points:
                    list_of_points.append((x2, y2))
                if (ref_x1, ref_y1) not in list_of_points:
                    list_of_points.append((ref_x1, ref_y1))
                if (ref_x2, ref_y2) not in list_of_points:
                    list_of_points.append((ref_x2, ref_y2))


    # go through list of all points from lines that have the same distance
    for x,y in list_of_points:
        if not found_it:
            for second_x,second_y in list_of_points[1:]:
                # The top and bottom points must have the same x value
                if x == second_x:
                    if second_y > y:

                        potential_bottom = (second_x,second_y)
                        potential_top = (x,y)

                        try:
                            if image_copy[potential_top[1] + 10, potential_top[0]][0] < 40 and image_copy[potential_top[1] + 10, potential_top[0]][1] > 200 and image_copy[potential_top[1] + 10, potential_top[0]][2] > 200:
                                if image_copy[potential_bottom[1] - 10, potential_bottom[0]][0] < 40 and image_copy[potential_bottom[1] - 10, potential_bottom[0]][1] > 200 and image_copy[potential_bottom[1] - 10, potential_bottom[0]][2] > 200:
                                    centerY = int((potential_top[1] + potential_bottom[1]) / 2)
                                    centerX = int((potential_top[0] + potential_bottom[0]) / 2)
                                    best_guess = (centerX, centerY)
                                    found_it = True
                                    cv2.circle(image_copy, best_guess, 1, (0, 0, 0), 3)

                        except:
                            continue # don't care if it's off the map

                    else:
                        potential_bottom = (x,y)
                        potential_top = (second_x,second_y)
                        try:
                            if image_copy[potential_top[1] + 10, potential_top[0]][0] < 40 and image_copy[potential_top[1] + 10, potential_top[0]][1] > 200 and image_copy[potential_top[1] + 10, potential_top[0]][2] > 200:
                                if image_copy[potential_bottom[1] - 10, potential_bottom[0]][0] < 40 and \
                                        image_copy[potential_bottom[1] - 10, potential_bottom[0]][1] > 200 and \
                                        image_copy[potential_bottom[1] - 10, potential_bottom[0]][2] > 200:
                                    centerY = int((potential_top[1] + potential_bottom[1]) / 2)
                                    centerX = int((potential_top[0] + potential_bottom[0]) / 2)
                                    best_guess = (centerX, centerY)
                                    cv2.circle(image_copy, best_guess, 1, (0, 0, 0), 3)
                                    found_it = True

                        except:
                            continue # don't care if it's off the map
    # show_image('d', image_copy)
    return best_guess

def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """

    image_copy = img_in.copy()
    best_guess = (0,0)
    point_dict = {}
    potential_bottom = []
    list_of_points = []
    potential_top = []
    found_it = False
    gray = image_copy[:, :, 2]  # filter out red
    if BLUR:
        ksize = (3,3)
    else:
        ksize = (2,2)
    gray_blurred = cv2.blur(gray, ksize)
    edges = cv2.Canny(gray_blurred, 100, 200)
    lines_found = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 4, threshold=5, minLineLength=10, maxLineGap=2)
    if lines_found is not None:
        for line in lines_found:
            (x1, y1, x2, y2) = line[0]
            distance = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
            if distance not in point_dict:
                point_dict[distance] = (x1, y1, x2, y2)
            else:  # compile all the lines with the same distance
                ref_x1, ref_y1, ref_x2, ref_y2 = point_dict[distance]
                if (x1,y1) not in list_of_points:
                    list_of_points.append((x1, y1))
                if (x2, y2) not in list_of_points:
                    list_of_points.append((x2, y2))
                if (ref_x1, ref_y1) not in list_of_points:
                    list_of_points.append((ref_x1, ref_y1))
                if (ref_x2, ref_y2) not in list_of_points:
                    list_of_points.append((ref_x2, ref_y2))


    # go through list of all points from lines that have the same distance
    for x,y in list_of_points:
        if not found_it:
            for second_x,second_y in list_of_points[1:]:
                # The top and bottom points must have the same x value
                if x == second_x:
                    if second_y > y:

                        potential_bottom = (second_x,second_y)
                        potential_top = (x,y)
                        try:
                            if image_copy[potential_top[1] + 10, potential_top[0]][0] < 30 and 100 < image_copy[potential_top[1] + 10, potential_top[0]][1] < 150 and image_copy[potential_top[1] + 10, potential_top[0]][2] > 200:
                                if image_copy[potential_bottom[1] - 10, potential_bottom[0]][0] < 30 and 100 < image_copy[potential_bottom[1] - 10, potential_bottom[0]][1] < 150 and image_copy[potential_bottom[1] - 10, potential_bottom[0]][2] > 200:
                                    centerY = int((potential_top[1] + potential_bottom[1]) / 2)
                                    centerX = int((potential_top[0] + potential_bottom[0]) / 2)
                                    cv2.circle(image_copy, (centerX,centerY), 1, (0, 0, 0), 3)
                                    best_guess = (centerX, centerY)
                                    found_it = True

                        except:
                            continue # don't care if it's off the map

                    else:
                        potential_bottom = (x, y)
                        potential_top = (second_x, second_y)
                        try:
                            if image_copy[potential_top[1] + 10, potential_top[0]][0] < 30 and 100 < image_copy[potential_top[1] + 10, potential_top[0]][1] < 150 and image_copy[potential_top[1] + 10, potential_top[0]][2] > 200:
                                if image_copy[potential_bottom[1] - 10, potential_bottom[0]][0] < 30 and 100 < image_copy[potential_bottom[1] - 10, potential_bottom[0]][1] < 150 and image_copy[potential_bottom[1] - 10, potential_bottom[0]][2] > 200:
                                    centerY = int((potential_top[1] + potential_bottom[1]) / 2)
                                    centerX = int((potential_top[0] + potential_bottom[0]) / 2)
                                    cv2.circle(image_copy, (centerX,centerY), 1, (0, 0, 0), 3)
                                    best_guess = (centerX, centerY)
                                    found_it = True

                        except:
                            continue # don't care if it's off the map

                    break
    return best_guess


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    image_copy = img_in.copy()
    radii_range = range(10, 50, 1)
    sign_center = (0,0)
    gray = image_copy[:,:,1]  # filter out
    if BLUR:
        ksize = (4,4)
    else:
        ksize = (2,2)
    found = False
    gray_blurred = cv2.blur(gray, ksize)
    for radius in radii_range:
        circles_found = cv2.HoughCircles(image=gray_blurred, method=cv2.HOUGH_GRADIENT, dp=1, minDist=radius, param1=100, param2=20, minRadius=radius, maxRadius=radius)
        if circles_found is not None and not found:
            detected_circles = np.uint16(np.around(circles_found))[0]
            for (x, y, r) in detected_circles:
                try:
                    if image_copy[y, x][0] > 220 and image_copy[y, x][1] > 220 and image_copy[y, x][2] > 220:
                        if image_copy[y, x+5][0] > 200 and image_copy[y, x+5][1] > 200 and image_copy[y, x+5][2] > 200:
                            if image_copy[y , x - 5][0] > 200 and image_copy[y, x - 5 ][1] > 200 and image_copy[y, x - 5][2] > 200:
                                sign_center = (x,y)
                                found = True
                                break
                except:
                    continue
    return sign_center

def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """

    compiled_results = {}
    dne = do_not_enter_sign_detection(img_in)
    if dne:
        compiled_results['no_entry'] = dne
    stop = stop_sign_detection(img_in)

    if stop:
        compiled_results['stop'] = stop

    construction = construction_sign_detection(img_in)
    if construction != (0,0):
        compiled_results['construction'] = construction

    warning = warning_sign_detection(img_in)
    if warning != (0,0):
        compiled_results['warning'] = warning

    yld = yield_sign_detection(img_in)
    if yld != (0,0):
        compiled_results['yield'] = yld

    radii_range = range(5, 30, 1)
    traffic_light = traffic_light_detection(img_in, radii_range)
    if traffic_light != (None, None) and traffic_light[1] != None:
        compiled_results['traffic_light'] = traffic_light[0]
    # import pdb
    # pdb.set_trace()
    return compiled_results

def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    BLUR = True
    compiled_results = {}
    dne = do_not_enter_sign_detection(img_in)
    if dne:
        compiled_results['no_entry'] = dne
    #
    stop = stop_sign_detection(img_in)
    if stop:
        compiled_results['stop'] = stop
    #
    construction = construction_sign_detection(img_in)
    if construction != (0,0):
        compiled_results['construction'] = construction
    #
    warning = warning_sign_detection(img_in)
    if warning != (0,0):
        compiled_results['warning'] = warning
    #
    yld = yield_sign_detection(img_in)
    if yld != (0,0):
        compiled_results['yield'] = yld

    radii_range = range(5, 20, 1)
    traffic_light = traffic_light_detection(img_in, radii_range)
    if traffic_light != (None, None) and traffic_light[0] != (0,0):
        compiled_results['traffic_light'] = traffic_light[0]

    return compiled_results

def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    return {'stop': (0,0)}
