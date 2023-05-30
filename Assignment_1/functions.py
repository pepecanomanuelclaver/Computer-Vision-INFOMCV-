import numpy as np
import cv2 as cv


def index2d(data, value):
    """
    The function index 2d obtains the index from
    a 2d array
    :param: data
    :param: value
    """
    for i, j in enumerate(data):
        if value in j:
            return [i, j.index(value)]


def line_intersection(line1, line2):
    """
    The function line_intersection finds the intersected points
    between two lines.
    :param line1: line one
    :param line2: line two
    :return: intersection coordinates
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def interpolation(interp_corner, side1, side2):
    """
    The function interpolation finds the positions with respect to the
    x,y axis of the corners.
    Then, depending the side1 and side2 values are chosen depending on
    the lenght of the parallel and vertical lines: the ones that are
    longer would contain more corners (squares in the chessboard).
    Finally, the interpolation is done to obtain the grid with all
    the corner points of the chessboard.
    :param interp_corner: four corner points positions
    :param side1: side 1 length
    :param side2: side 2 length
    :return: corner_int_f and corners_int_print correspond to the corner
    locations in float and int.
    """
    a = interp_corner

    # Left
    left = []
    minx = min(a, key=lambda x: x[0])[0]
    position = index2d(interp_corner, minx)
    left.append(a.pop(position[0]))

    minx = min(a, key=lambda x: x[0])[0]
    position = index2d(interp_corner, minx)
    left.append(a.pop(position[0]))

    maxx = max(left, key=lambda x: x[1])[1]
    position = index2d(left, maxx)
    top_left = left.pop(position[0])
    bottom_left = left[0]

    # Right
    right = a

    maxx = max(right, key=lambda x: x[1])[1]
    position = index2d(right, maxx)
    top_right = right.pop(position[0])
    bottom_right = right[0]

    # Choose whether the upper line is longer or shorter than the horizontal left line.
    if abs(bottom_right[0] - bottom_left[0]) < abs(top_left[1] - bottom_left[1]):
        x1 = np.linspace(top_right[0], top_left[0], side2)
        y1 = np.linspace(top_right[1], top_left[1], side2)

        x2 = np.linspace(bottom_left[0], top_left[0], side1)
        y2 = np.linspace(bottom_left[1], top_left[1], side1)

        x3 = np.linspace(bottom_right[0], top_right[0], side1)
        y3 = np.linspace(bottom_right[1], top_right[1], side1)

        x4 = np.linspace(bottom_right[0], bottom_left[0], side2)
        y4 = np.linspace(bottom_right[1], bottom_left[1], side2)

        corners_int_f = []
        corners_int_print = []
        for i in range(0, side2):
            for j in range(0, side1):
                line1 = [[x1[i], y1[i]], [x4[i], y4[i]]]
                line2 = [[x2[j], y2[j]], [x3[j], y3[j]]]
                x_int, y_int = line_intersection(line1, line2)
                corners_int_f.append([[x_int, y_int]])
                x_int = int(x_int)
                y_int = int(y_int)
                corners_int_print.append((x_int, y_int))

    else:
        x1 = np.linspace(top_right[0], top_left[0], side1)
        y1 = np.linspace(top_right[1], top_left[1], side1)

        x2 = np.linspace(bottom_left[0], top_left[0], side2)
        y2 = np.linspace(bottom_left[1], top_left[1], side2)

        x3 = np.linspace(bottom_right[0], top_right[0], side2)
        y3 = np.linspace(bottom_right[1], top_right[1], side2)

        x4 = np.linspace(bottom_right[0], bottom_left[0], side1)
        y4 = np.linspace(bottom_right[1], bottom_left[1], side1)

        corners_int_f = []
        corners_int_print = []
        for i in range(0, side1):
            for j in range(0, side2):
                line1 = [[x1[i], y1[i]], [x4[i], y4[i]]]
                line2 = [[x2[j], y2[j]], [x3[j], y3[j]]]
                x_int, y_int = line_intersection(line1, line2)
                corners_int_f.append([[x_int, y_int]])
                x_int = int(x_int)
                y_int = int(y_int)
                corners_int_print.append((x_int, y_int))

    return corners_int_f, corners_int_print


def canny(image):
    """
        The canny function computes an edge detection of the image, obtaining
        an image of the edges and the new image with enhanced edges
        :param image: input RGB image
        :return: edge image and sharpened image
        """
    # split image in its three RBG channels
    r, g, b = cv.split(image)

    # compute canny fot each channel
    edger = cv.Canny(r, 50, 200, None, 3)
    edgeg = cv.Canny(g, 50, 200, None, 3)
    edgeb = cv.Canny(b, 50, 200, None, 3)

    # add all channels
    edge = cv.merge([edger, edgeg, edgeb])

    # add the edges to the original image
    sharpened = edge + image

    return edge, sharpened


def calibration(realworld_points, pixel_points, gray_img):
    """
    The calibration function calibrates geometrically your camera using the images
    from the offline phase and calculates the error of calibration.
    :param realworld_points: real world object points
    :param pixel_points: pixel image points
    :param gray_img: gray image
    :return: new array of real world object and pixel image points
    """

    # calibrate the camera with the real world object points and the pixel points from the image
    patternfound, k_matrix, distortion, rotation_vecs, translation_vecs = cv.calibrateCamera(realworld_points,
                                                                                             pixel_points,
                                                                                             gray_img.shape[::-1], None,
                                                                                             None)
    mean_error = 0  # mean calibration error
    tot_error = []  # images calibration error

    # compare the new points and the old points to obtain the error
    for i in range(len(realworld_points)):
        # calculate new pixel points from every image
        pixel_points2, _ = cv.projectPoints(realworld_points[i], rotation_vecs[i], translation_vecs[i], k_matrix,
                                            distortion)
        # calculate error
        error = cv.norm(pixel_points[i], pixel_points2, cv.NORM_L2) / len(pixel_points2)
        tot_error.append(error)
        mean_error += error
    print(mean_error / len(realworld_points))

    return k_matrix, distortion, rotation_vecs, translation_vecs, tot_error


def reject_outliers(data, realworld_points, pixel_points):
    """
    The reject_outliers function takes an array of numbers to calculate its
    outliers based on its mean and standard deviation and removes those detected
    from the real world object points and the pixel image points.
    :param data: number list
    :param realworld_points: real world object points
    :param pixel_points: pixel image points
    :return: new array of real world object and pixel image points
    """
    # calculate mean and standard deviation from list
    mean = np.mean(data)
    std = np.std(data)

    # create a list of the founded outlier's indexes
    outliers = [data.index(e) for e in data if not (mean - 2 * std < e < mean + 2 * std)]

    # remove outliers from list real world object and pixel image points
    if outliers:
        for index in outliers:
            realworld_points.pop(index)
            pixel_points.pop(index)

    return outliers, realworld_points, pixel_points


def online_phase(img, new_camera_matrix, distortion, side1, side2, square_size):
    """
    The online_phase function takes a new input image to draw the x, y, z axis and a cube
    using the estimated camera parameters, obtained in the offline phase, with the origin
    at the center of the world coordinates.
    :param:img: input image
    :param:new_camara_matrix: optimal k matrix of new image based on estimated intrinsic parameters'
    distortion.
    :param:side1: grid side 1
    :param:side2: grid side 2
    :param:square_size: length of square in mm
    :return: returns an image with a cube and the axis drawn on it
    """
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # create an array of the real world points in mm starting from the top left
    obj_points = np.zeros((side1 * side2, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:side1, 0:side2].T.reshape(-1, 2)
    obj_points[:, :2] = obj_points[:, :2] * square_size  # in mm

    # arrays to store object points and image points from all the images.
    realworld_points = []  # 3d point in real world space
    pixel_points = []  # 2d points in image plane.

    # apply canny filter, edge detection for a sharpened image
    _, img = canny(img)
    # convert image to gray
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    patternfound, corners = cv.findChessboardCorners(gray_img, (side1, side2), None)
    # If found, add object points, image points (after refining them)
    if patternfound:
        realworld_points.append(obj_points)
        # Detect corners location in subpixels
        corners2 = cv.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
        # Obtain extrinsic param by estimated intrinsic param for cube and axis
        patternfound, rotation_vecs, translation_vecs = cv.solvePnP(obj_points, corners2, new_camera_matrix, distortion)
        pixel_points.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (side1, side2), corners2, patternfound)

        # Obtain the axis
        # determine x,y,z axis
        points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1], [0, 0, 0]]).reshape(-1, 3)
        # calculate ending axis points
        axis_pts, _ = cv.projectPoints(points * square_size * 4, rotation_vecs,
                                       translation_vecs, new_camera_matrix, distortion)
        axis_pts = np.round(axis_pts)
        axis_pts = axis_pts.astype(int)
        # convert to int each pixel coordinate so cv.line can take it as input
        corners2 = np.round(corners2)
        corners2 = corners.astype(int)

        # draw lines on image
        img = cv.line(img, tuple(corners2[0].ravel()), tuple(axis_pts[0].ravel()), (255, 255, 0), 4)
        img = cv.line(img, tuple(corners2[0].ravel()), tuple(axis_pts[1].ravel()), (255, 0, 255), 4)
        img = cv.line(img, tuple(corners2[0].ravel()), tuple(axis_pts[2].ravel()), (9, 2, 255), 4)

        # Obtain the cube
        # determine x,y,z axis
        points = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                             [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
        # calculate ending axis points
        axis_pts, _ = cv.projectPoints(points * square_size * 2, rotation_vecs,
                                       translation_vecs, new_camera_matrix, distortion)
        # convert to int each pixel coordinate so cv.line can take it as input
        image_pts = np.int32(axis_pts).reshape(-1, 2)

        # draw the edges of the cube axis z
        for i, j in zip(range(4), range(4, 8)):
            img = cv.line(img, tuple(image_pts[i]), tuple(image_pts[j]), (0, 255, 250), 2)
        # top and bottom edges (axis x, y)
        img = cv.drawContours(img, [image_pts[4:]], -1, (0, 255, 250), 2)
        img = cv.drawContours(img, [image_pts[:4]], -1, (0, 255, 250), 2)

    return img
