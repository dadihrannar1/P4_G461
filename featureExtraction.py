'''

This file contains functions used for extracting features of BLOBs (binary large objects).
Each of the functions is independent and can therefore be used for other purposes as well.

The file uses OpenCV Version 4.0.0, Python 3.8
Contact: Diana-Valeria Vacaru, dvacar21@student.aau.dk

'''

import numpy as np
import cv2 as cv
import math

# <editor-fold desc="UNCLASSIFIED EXTRACTION FUNCTIONS">

# Computes the distance, given two points
def computeDistanceBetweenTwoPoints(pt1_x, pt1_y, pt2_x, pt2_y):
    return math.sqrt(math.pow(pt2_x - pt1_x, 2) + math.pow(pt2_y - pt1_y, 2))


# The angle is in radians
def computeAngleBetweenTwoPoints(pt1_x, pt1_y, pt2_x, pt2_y):
    return math.atan2((pt2_y - pt1_y), (pt2_x - pt1_x))


# Get object center of mass coordinates
def getCentroidCoordinates(cnt):
    M = cv.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


# Given a contour of an object, a line is fit through the longest part of the contour
def fitLineThroughCnt(inputImg, cnt):
    rows, cols = inputImg.shape[:2]
    [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    # The line has to have the thickness two to make sure when the line is intersected with
    # a contour, it will return more than one intersection point
    # This is used to avoid having 0 intersection points, when the line perfectly passes through
    # another without intersecting it
    return cv.line(inputImg, (cols - 1, righty), (0, lefty), (255, 255, 255), 2)


# This function takes a point (e.g. one from the image) and returns its neighbouring point coordinates
# depending on the radius size, e.g. a radius with size 1 it will cover the area of a (3x3) kernel, 2 => (5x5)
# The type of combinations are ordered with replacement with the total number calculated as n^k where n is the number
# of options and k is the number of positions (in the case of points, k is equal to two and if radius = 2 => n = 5)
def combinations(point, radius, imageShape):
    pointList = []
    countRemoved = 0
    for i in range(radius+1):
        for j in range(radius+1):
            pointList.append((point[0] + i, point[1] + j))
            pointList.append((point[0] - i, point[1] - j))
            pointList.append((point[0] - i, point[1] + j))
            pointList.append((point[0] + i, point[1] - j))
    # Remove repetitive entries
    pointList = np.unique(pointList, axis=0)
    # Remove points that go out of the image bounds
    for index, point in enumerate(pointList):
        if (point[0] > imageShape[0] or point[0] < 0) or (point[1] > imageShape[1] or point[1] < 0):
                pointList = np.delete(pointList, index - countRemoved, 0)
                countRemoved += 1
    return pointList


# This function takes a point, finds its neighbours, extracts their depth data and computes the average
# Then it uses that and intrinsic camera parameters to transform the point from 2D to 3D
def transform2Dto3DSpace(point, depthArray):
    # Smoothen the depth map to attempt removing 0 depth results
    # depthArray = cv.medianBlur(depthArray.astype(np.uint8), 3)
    # Get the coordinates of the surrounding points given the radius and image size
    surroundingPts = combinations(point, 1, depthArray.shape)
    # Get the depth at those points
    depthOfSurroundingPts = []
    for pt in surroundingPts:
            # Get 2D depth data from pixel coordinate (i.e. point)
            # Because the depth array has the pixel coordinates reversed,
            # we need to first input the y and then the x
            depthOfSurroundingPts.append(depthArray[pt[1], pt[0]])  # first indexed by row - matrices
    # Get the indexes of the nonzero values
    index = np.nonzero(depthOfSurroundingPts)
    # If there is no nonzero value then just assign the depth to zero
    if len(index[0]) == 0:
        averageDepthOfNonZeroSurrPts = depthArray[point[1], point[0]]
    else:
        # Transform to numpy array to be able to apply numpy functions
        depthOfSurroundingPts = np.asarray(depthOfSurroundingPts)
        # Get average of non-zero depths around a point
        # This is used to diminish the chance of having depth = 0
        averageDepthOfNonZeroSurrPts = np.average(depthOfSurroundingPts[index])

    # Get intrinsic of RGB camera data
    # Why RGB? because we aligned the depth to rgb size
    # The intrinsic values will vary depending on the image resolution
    # When transforming from 2D to 3D it is not necessary to transform
    # intrinsic data from px to mm because the depth
    # data makes sure to scale the X and Y results into the right unit
    # focal lengths: 1.88mm, Color sensor type is OV2740 with size 1.4 micrometers per px
    fx = 607.417724609375 # in pixels
    fy = 607.579345703125
    # principal points
    ppx = 324.2457580566406 # in pixels
    ppy = 237.2323455810547

    # Calculate 3D world coordinate from 2D coordinate and depth data
    Z = averageDepthOfNonZeroSurrPts # in mm
    X = round((Z * (point[0] - ppx)) / fx, 2)
    Y = round((Z * (point[1] - ppy)) / fy, 2)

    return averageDepthOfNonZeroSurrPts, np.array([X, Y, Z])


# Given a blob mask and a line mask, the intersection points of the two are computed
def intersectionPoints(objectMask, inputLineMask):
    # Perform binary and operation to only keep the points of intersection and discard the rest
    intersectionMask = cv.bitwise_and(objectMask, inputLineMask)
    # Get the coordinates of the points of intersection (i.e. the white pixels)
    intersectionPts = np.transpose(np.nonzero(intersectionMask))
    # Remove the third useless channel (i.e from [x y a] to [x y]) by slicing pt[0:2]
    # and reverses [::-1] the points to match the real intersected points
    # e.g. a = [1,2,3], then a[-1] = 3, a[-2] = 2, a[-3] = 1, this is how the reverting works
    # then it also removes the repetitive points with the function np.unique
    return np.unique([pt[0:2][::-1] for pt in intersectionPts], axis=0)


# Given a list of multiple points, find the pair that returns the longest lengths
def findPointsWithLongestLength(pointList):
    maximum = 0
    point1 = 0
    point2 = 0
    for i in range(0, len(pointList)):
        for j in range(i, len(pointList)):
            dist = computeDistanceBetweenTwoPoints(pointList[i][0], pointList[i][1], pointList[j][0], pointList[j][1])
            if dist > maximum:
                maximum = dist
                point1 = (pointList[i][0], pointList[i][1])
                point2 = (pointList[j][0], pointList[j][1])
    return point1, point2

# Given two points on a line, calculate its perpendicular line coordinates and find the coordinates of the points
# of intersection of that line with the object contour. Next step is to keep only two points that return the
# longest distance. The distance between these points will represent the width of an object part, but can be using in
# other usecases as well.
def getTwoPointCoordinatesOnContour(heightLinePoints, intersecTwoLinesPt, blobMask, imageToDrawOn):
    fitLineDistance = computeDistanceBetweenTwoPoints(heightLinePoints[0][0], heightLinePoints[0][1], heightLinePoints[1][0],
                                                      heightLinePoints[1][1])
    # Here where imageToDrawOn.shape[0] - 100 is, (fitlineDistance // 2) was meant to be
    # However it was decided to make a line long enough that will guarantee full intersection with the contour
    perpendicularLinePts = getPerpCoord(intersecTwoLinesPt, tuple(heightLinePoints[0]), imageToDrawOn.shape[0] - 100)
    pt1 = (int(perpendicularLinePts[0]), int(perpendicularLinePts[1]))
    pt2 = (int(perpendicularLinePts[2]), int(perpendicularLinePts[3]))
    blackMask = np.zeros(imageToDrawOn.shape, np.uint8)
    widthLineMask = cv.line(blackMask, pt1, pt2, (255, 255, 255), 2)
    # Find the points of intersection of the contour with the width line
    widthIntersecPts = intersectionPoints(blobMask, widthLineMask)
    # Keep the two points making up the longest distance
    widthIntersecPts2 = np.asarray(findPointsWithLongestLength(widthIntersecPts))
    # cv.circle(imageToDrawOn, (widthIntersecPts2[0][0], widthIntersecPts2[0][1]), 2, (255, 50, 50), 2)
    # cv.circle(imageToDrawOn, (widthIntersecPts2[1][0], widthIntersecPts2[1][1]), 2, (255, 50, 40), 2)
    # cv.line(imageToDrawOn, pt1, pt2, (255, 255, 255), 2)
    return widthIntersecPts2

# </editor-fold>

# <editor-fold desc="CALCULATE PERPENDICULAR LINE">

# This function calculates the slope of a line given two points on that line
def calculateSlope(point1, point2):
    # If the x-coordinates are the same for both points
    if np.subtract(point2[0], point1[0]) == 0.0:
        # return a number "close" to infinity
        return 10000000000000000000
    else:
        return round(np.subtract(point2[1], point1[1]) / np.subtract(point2[0], point1[0]), 1)

# In an equation of the form slope-intercept y = mx + b, find b
# given that the rest of the variable values are known
def findYInterceptForStraightLine(point, slope):
    return point[1] - slope * point[0]


# Get the new line points that make up a line perpendicular to the input line
def getPerpCoord(point1, point2, length):
    slope1 = calculateSlope(point1, point2)
    # Special case, when we have a line  of the form y = b, and we want to find the line of the
    # form x = a, the new line will have slope = inf, so to avoid that we account for it in a special case
    # here the y-coordinate will change while the x-coordinate will remain the same
    if slope1 == 0.0 or slope1 == -0.0:
        cx = point1[0]
        cy = point1[1] + length
        dx = point1[0]
        dy = point1[1] - length
    else:
        # Replicate slope equal to infinity
        if slope1 == 10000000000000000000:
            # For a line to be perpendicular to a line with an infinite slope,
            # it needs to have the slope equal to zero
            slope2 = 0
        else:
            # else the new line slope is the negative inverse of the input line's slope
            slope2 = -1 / slope1
        # Find b from straight line equation y = mx + b
        b = findYInterceptForStraightLine(point1, slope2)
        # Given that we know where we want our x-coordinate for the new
        # parallel line to pass through, and we know the new perpendicular
        # line's slope and the equation, we can also find its y-coordinate
        cx = point1[0] + length
        cy = slope2 * cx + b
        dx = point1[0] - length
        dy = slope2 * dx + b
    return int(cx), int(cy), int(dx), int(dy)

# </editor-fold>

# <editor-fold desc="CALCULATE BLOB TOP and BOTTOM WIDTH RATIO FEATURE">

# This function returns the new X or Y coordinate on a line determined by two input points
# and whose location is determined by the given percentage. Whether it is the X or Y coordinate
# it is determined by which of the X and Y projections resulted from the subtraction of the two points is the largest
def findLargestMagnitudeOfVectorProjections(vector1, vector2, percentage):
    vector3 = np.absolute(np.subtract(vector1, vector2))
    maxMagnitude = max(vector3[0], vector3[1])
    # If the projection on X is larger
    if maxMagnitude == vector3[0]:
        # Find which point is furthest on X
        maxXVal = max(vector1[0], vector2[0])
        if maxXVal == vector1[0]:
            # Get the X value of the point placed at the chosen percentage with respect
            # to the line between the two vectors length
            newXCoord = percentage * (vector1[0] - vector2[0]) + vector2[0]
        else:
            newXCoord = percentage * (vector2[0] - vector1[0]) + vector1[0]
        xIsLarger = True
        return xIsLarger, newXCoord
    # If the projection on Y is larger
    elif maxMagnitude == vector3[1]:
        # Find which point is furthest on Y
        maxYVal = max(vector1[1], vector2[1])
        if maxYVal == vector1[1]:
            newYCoord = percentage * (vector1[1] - vector2[1]) + vector2[1]
        else:
            newYCoord = percentage * (vector2[1] - vector1[1]) + vector1[1]
        xIsLarger = False
        return xIsLarger, newYCoord


# Implemented linear interpolation that takes two data points, say (x0,y0) and (x1,y1), and the interpolant
# is given by the formula presented in the function at the point (x, y). That is if known x or y, we can find the other
# because of the linearity properties (i.e. 10 % increase in x will have a 10 % increase in y)
def findYWithXOrXWithYAndInterpolation(isXLarger, y1, y0, x1, x0, largestMagnitude):
    if isXLarger == True:
        # Calculate y given x, here largest magnitude = x
        y = y0 + (y1 - y0) * ((largestMagnitude - x0) / (x1 - x0))
        return y
    else:
        # Calculate x given y, here largest magnitude = y
        x = x0 + (x1 - x0) * ((largestMagnitude - y0) / (y1 - y0))
        return x


# This function returns the coordinates of a point placed at a specified percentage on a line, given the two
# points that make up that line
def getPointCoordinatesOnLine(points, percentage):
    xIsLarger, newCoordBasedOnChosenPercentage = findLargestMagnitudeOfVectorProjections(points[0], points[1], percentage)
    if xIsLarger == True:
        newYCoord = findYWithXOrXWithYAndInterpolation(xIsLarger, points[1][1], points[0][1], points[1][0], points[0][0], newCoordBasedOnChosenPercentage)
        return math.floor(newCoordBasedOnChosenPercentage), math.floor(newYCoord)
    else:
        newXCoord = findYWithXOrXWithYAndInterpolation(xIsLarger, points[1][1], points[0][1], points[1][0], points[0][0], newCoordBasedOnChosenPercentage)
        return math.floor(newXCoord), math.floor(newCoordBasedOnChosenPercentage)

# </editor-fold>