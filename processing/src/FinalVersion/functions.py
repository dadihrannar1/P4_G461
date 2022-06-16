"""

The file uses OpenCV Version 4.0.0, Python 3.8

Contact: Diana-Valeria Vacaru, dvacar21@student.aau.dk

"""

import cv2 as cv
import numpy as np
import math
import copy

# <editor-fold desc="SEGMENTATION FUNCTIONS">

def resizeImage(inputImage):
    resizedDataLocal = []
    for indexImage in inputImage:
        resizedDataLocal.append(cv.resize(indexImage, (848, 480)))
    return resizedDataLocal

# Show multiple images from file at once
def showImgs(figName, imageList):
    for i, image in enumerate(imageList):
        cv.imshow(figName + ' {}'.format(i), image)

# Threshold the RGB image to get the parts of interest
def bgrThreshold(inputImage, lowerColor, upperColor):
    localMasks = []
    for localImg in inputImage:
        localMasks.append(cv.inRange(localImg, lowerColor, upperColor))
    return localMasks

def grayThreshold(grayscale, T):
    binary = np.ndarray((grayscale.shape[0], grayscale.shape[1]), dtype='uint8')
    for i in range(0, grayscale.shape[0]):
        for j in range(0, grayscale.shape[1]):
            if grayscale[i, j] <= T:
                binary[i, j] = 0
            else:
                binary[i, j] = 255
    return binary


# Morphology
def closing(kernel, inputMasks, itr):
    localClosing = []
    for localMask in inputMasks:
        localClosing.append(cv.morphologyEx(localMask, cv.MORPH_CLOSE, kernel, iterations=itr))
    return localClosing


def opening(kernel, inputMasks, itr):
    localOpening = []
    for localMask in inputMasks:
        localOpening.append(cv.morphologyEx(localMask, cv.MORPH_OPEN, kernel, iterations=itr))
    return localOpening


def dilation(kernel, inputMasks, itr):
    localDilation = []
    for localMask in inputMasks:
        localDilation.append(cv.morphologyEx(localMask, cv.MORPH_DILATE, kernel, iterations=itr))
    return localDilation


def erosion(kernel, inputMasks, itr):
    localErosion = []
    for localMask in inputMasks:
        localErosion.append(cv.morphologyEx(localMask, cv.MORPH_ERODE, kernel, iterations=itr))
    return localErosion


# Edge detection Canny
def edgeDetection(inputImages, threshold1, threshold2, apertureSize):
    localEdges = []
    for localImg in inputImages:
        localEdges.append(cv.Canny(localImg, threshold1, threshold2, None, apertureSize))
    return localEdges

def edgeDetectionGrad(inputImage):
    scale = 0.3 # put 0.1 for test2, 0.3 for test1
    delta = 0
    ddepth = cv.CV_16S

    grayObj = []
    for obj in inputImage:
        grayObj.append(cv.cvtColor(obj, cv.COLOR_BGR2GRAY))

    grad_x = []
    grad_y = []
    abs_grad_x = []
    abs_grad_y = []
    grad = []
    for i, gray in enumerate(grayObj):
        grad_x.append(cv.Sobel(gray, ddepth, 1, 0, ksize=5, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT))
        grad_y.append(cv.Sobel(gray, ddepth, 0, 1, ksize=5, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT))

        abs_grad_x.append(cv.convertScaleAbs(grad_x[i]))
        abs_grad_y.append(cv.convertScaleAbs(grad_y[i]))

        grad.append(cv.addWeighted(abs_grad_x[i], 0.5, abs_grad_y[i], 0.5, 0))
    return grad

# Line detection/ get slopes
def lineDetection(inputEdges, imageToGetLinesFrom):
    allSlopes = []
    for j, edgeImg in enumerate(inputEdges):
        # the slope lists will be emptied when looping through the edge data of a new image
        slopeP = []
        slope = []
        lines = cv.HoughLines(edgeImg, 1, np.pi / 180, 150, None, 0, 0)
        linesP = cv.HoughLinesP(edgeImg, 1, np.pi / 180, 65, None, 50, 10)

        # Use the probabilistic approach to get lines
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(imageToGetLinesFrom[j], (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
                if (l[2] - l[0]) != 0:
                    slopeP.append((l[3] - l[1]) / (l[2] - l[0]))

        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))  # pt1 = (x1, y1)
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))  # pt2 = (x2, y2)
                cv.line(imageToGetLinesFrom[j], pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
                # Calculate the slopes of each line and add them to a list
                slope.append(
                    (int(y0 - 1000 * (a)) - int(y0 + 1000 * (a))) / (int(x0 - 1000 * (-b)) - int(x0 + 1000 * (-b))))
        # Add the slope lists for each image into a list of lists containing all images
        # Concatenate them with the lines obtained from the probabilistic approach
        allSlopes.append(slope + slopeP)
    return allSlopes


# Get the number of parallel line pairs
def findParallelLines(inputImages, slopes):
    # Create a list that will contain the number of pairs of parallel lines in each image
    # i.e. the size of the list will be the same as the amount of images from the training data
    parallelPairs = []
    # Loop through each row containing line slopes lists for one image each
    for i in range(0, len(inputImages)):
        count = 0  # reset the count of parallel lines for each image
        # Loop through the inner list of slopes and compare them to each other to find pairs
        for j in range(0, len(slopes[i])):
            slope = slopes[i][j]
            for k in range(j + 1, len(slopes[i])):
                # Check whether there are any slopes that are equal with a difference of +- 0.1
                if abs(slope - slopes[i][k]) <= 0.1:
                    count += 1
        parallelPairs.append(count)
    return parallelPairs


# Find contours of each BLOB
def findContours(inputImages):
    contours = []
    hierarchys = []
    for binaryImg in inputImages:
        iThContour, hierarchy = cv.findContours(binaryImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours.append(iThContour)
        hierarchys.append(hierarchy)
    return contours, hierarchys


# This function checks if a contour has an area smaller than 4000 px and returns True or False
def check_area(ctr, area):
    return cv.contourArea(ctr) < area


# Remove contours with very small area, considered noise
def removeSmallAreaContours(inputContours, inputHierarchies, area):
    for ith in range(0, len(inputContours)):
        countRemoved = 0
        for contour in range(0, len(inputContours[ith])):
            if check_area(inputContours[ith][contour - countRemoved], area):
                inputContours[ith].pop(contour - countRemoved)
                inputHierarchies[ith] = np.delete(inputHierarchies[ith], contour - countRemoved, 1)
                countRemoved += 1


# Leave only contours from 1st hierarchy (because in this case they define the main BLOBs)
def leave1StHierarchy(localContours, localHierarchy):
    hierarchy1Cont = copy.copy(localContours)
    countRemoved2 = 0
    for iThContour in range(0, localHierarchy.shape[1]):
        if localHierarchy[0, iThContour, 3] != -1:
            hierarchy1Cont.pop(iThContour - countRemoved2)
            countRemoved2 += 1
    return hierarchy1Cont


# Draw contours
def drawContours(inputContours, imageToDrawOn, inputHierarchy1Contour, color, thickness):
    drawContoursAllImg = []
    for i in range(0, len(inputContours)):
        # -1 is for drawing all contours at once
        drawContoursAllImg.append(cv.drawContours(imageToDrawOn[i], inputHierarchy1Contour[i], -1, color, thickness))
    return drawContoursAllImg


# Make background with even illumination
def consistentBackground(inputImage, kernel):
    mean = cv.blur(inputImage, kernel)
    clean_background = cv.subtract(mean, inputImage)
    return clean_background


# </editor-fold>

# <editor-fold desc="FEATURE EXTRACTION FUNCTIONS">

# 1. Check for circularity

def circularity(ctr):
    # Circularity = (4 * pi * Area) / Perimeter^2
    return (4 * math.pi * cv.contourArea(ctr)) / pow(cv.arcLength(ctr, True), 2)


# 2. Check for elongation (this function works only if the box lines are parallel to the image plane

def parallelBoundingBoxRatio(ctr, img):
    rect = cv.minAreaRect(ctr)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img, [box], 0, (25, 100, 50), 2)
    yMax = np.amax(box, 0)[1]
    xMax = np.amax(box, 0)[0]
    yMin = np.amin(box, 0)[1]
    xMin = np.amin(box, 0)[0]
    width = xMax - xMin
    height = yMax - yMin
    return height / width
    # print('Max', np.amax(box, 0)[0])
    # print('Min', np.amin(box, 0)[0])


def boundingBoxRatio(ctr):
    rect = cv.minAreaRect(ctr)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    # cv.drawContours(img, [box], 0, (25, 100, 50), 2)
    maxEdge = max(rect[1][1], rect[1][0])
    if maxEdge == rect[1][0]:
        return rect[1][1] / rect[1][0]
    else:
        return rect[1][0] / rect[1][1]

# 3. Check for convexhull defects
def convexityDefects(ctr, i, img):
    # hullList = []
    depth = []
    # To be able to draw the convex hull, the coordinates of the hull points are needed
    # hull2 = cv.convexHull(ctr)
    # hullList.append(hull2)
    # cv.drawContours(img, hullList, i, (100, 200, 25), 1)
    # To be able to compute the convexity defects, the indices of contour points
    # corresponding to the hull points are needed
    hull = cv.convexHull(ctr, clockwise=True, returnPoints=False)

    defects = cv.convexityDefects(ctr, hull)
    if defects is not None:
        for j in range(defects.shape[0]):
            s, e, f, d = defects[j, 0]
            far = tuple(ctr[f][0])
            depth.append(d)
            # cv.circle(img, far, 4, [0, 0, 255], -1)

    return depth


# Calculate approximate polygon and convex hull (draw them)
def convexHull(cnt, imageToDrawOn):
    # calculate epsilon base on contour's perimeter
    # contour's perimeter is returned by cv2.arcLength
    epsilon = 0.01 * cv.arcLength(cnt, True)
    # get approx polygons
    approx = cv.approxPolyDP(cnt, epsilon, True)
    # draw approx polygons
    # cv.drawContours(imageToDrawOn, [approx], -1, (0, 255, 0), 1)

    # hull is convex shape as a polygon
    hull = cv.convexHull(cnt, returnPoints=True)
    # cv.drawContours(imageToDrawOn, [hull], -1, (0, 0, 255))

    return hull, approx

# 4. Check for compactness

def compactnessRatio(ctr, localImg):
    rect = cv.minAreaRect(ctr)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    # cv.drawContours(localImg, [box], 0, (25, 100, 50), 2)
    # compactness = Area of BLOB / width * height of bounding box

    return cv.contourArea(ctr) / (rect[1][1] * rect[1][0])


# 5. Average color of objects of interest (without the background), looks at whole image
# Useful when same type of objects are in the same image or only one object

def averageColor(colorImg, field2dMask):
    # Turn 2D mask into 3D channel mask
    field3dMask = np.stack((field2dMask,) * 3, axis=-1)
    maskAndImg = cv.bitwise_and(colorImg, field3dMask)

    # filter black color and fetch color values
    data = []
    for x in range(3):
        channel = maskAndImg[:, :, x]
        indices = np.where(channel != 0)[0]
        color = np.mean(channel[indices])
        data.append(int(color))

    return data


# Average color per object of interest

def averageColorPerObject(colorImg, contours):
    meanOfColours = []
    for contour in contours:
        # Finds the average of each colour channel inside the contour
        mask = np.zeros((colorImg.shape[0], colorImg.shape[1]), np.uint8)
        cv.drawContours(mask, [contour], 0, 255, -1)
        meanOfColours = cv.mean(colorImg, mask=mask)
    return meanOfColours

# 6. Different areas within BLOB pixel count ratio

# Count number of white pixels per row per contour and create a 3-level list
# with data from contours from multiple images
def countWhitePxPerRow3LayerList(originalImg, allImgContours):
    allImagesCountWhilePx = []
    allImagesNrRowsUntilCnt = []
    # Loop through all images' contour data
    for cnts in allImgContours:
        oneImgContoursWhitePxCount = []
        oneImgNrRowsUntilCnt = []
        # Loop through all contours of one image
        for oneCnt in cnts:
            blackMaskPerCnt = np.zeros((originalImg[0].shape[0], originalImg[0].shape[1]), np.uint8)
            # Draw a mask with only one contour at a time
            cntDraw = cv.drawContours(blackMaskPerCnt, [oneCnt], 0, 255, -1)
            # Count the amount of pixels per row of each contour (without the background)
            countWhitePxPerRow = [np.count_nonzero(row) for row in cntDraw if np.count_nonzero(row) != 0]
            # Get nr of rows until the contour starts
            nrRowsUntilCnt = 0
            for row in cntDraw:
                # If the rows contain only black pixels
                if np.count_nonzero(row) == 0:
                    nrRowsUntilCnt += 1
                # Else stop counting because the contour started
                elif np.count_nonzero(row) != 0:
                    break
            # Show the singled contours
            # cv.imshow('Grad', cntDraw)
            # cv.waitKey(0)
            oneImgContoursWhitePxCount.append(countWhitePxPerRow)
            oneImgNrRowsUntilCnt.append(nrRowsUntilCnt)
        allImagesCountWhilePx.append(oneImgContoursWhitePxCount)
        allImagesNrRowsUntilCnt.append(oneImgNrRowsUntilCnt)
    return allImagesCountWhilePx, allImagesNrRowsUntilCnt


def countWhitePxPerRow(originalImg, oneCnt):
    blackMaskPerCnt = np.zeros((originalImg[0].shape[0], originalImg[0].shape[1]), np.uint8)
    # Draw a mask with only one contour at a time
    cntDraw = cv.drawContours(blackMaskPerCnt, [oneCnt], 0, 255, -1)
    # Count the amount of pixels per row of each contour (without the background)
    countWhitePxPerRow = [np.count_nonzero(row) for row in cntDraw if np.count_nonzero(row) != 0]
    # Get nr of rows until the contour starts
    nrRowsUntilCnt = 0
    for row in cntDraw:
        # If the rows contain only black pixels
        if np.count_nonzero(row) == 0:
            nrRowsUntilCnt += 1
        # Else stop counting because the contour started
        elif np.count_nonzero(row) != 0:
            break
    return countWhitePxPerRow, nrRowsUntilCnt


def countWhitePxPerCol3LayerList(originalImg, allImgContours):
    allImagesCountWhilePx = []
    # Loop through all images' contour data
    for cnts in allImgContours:
        oneImgContoursWhitePxCount = []
        # Loop through all contours of one image
        for oneCnt in cnts:
            blackMaskPerCnt = np.zeros((originalImg[0].shape[0], originalImg[0].shape[1]), np.uint8)
            # Draw a mask with only one contour at a time
            cntDraw = cv.drawContours(blackMaskPerCnt, [oneCnt], 0, 255, -1)
            # Count the amount of pixels per row of each contour (without the background)
            countWhitePxPerCol = [np.count_nonzero(col) for col in np.transpose(cntDraw) if np.count_nonzero(col) != 0]
            # Show the singled contours
            # cv.imshow('Grad', cntDraw)
            # cv.waitKey(0)
            oneImgContoursWhitePxCount.append(countWhitePxPerCol)
        allImagesCountWhilePx.append(oneImgContoursWhitePxCount)
    return allImagesCountWhilePx


def countWhitePxPerCol(originalImg, oneCnt):
    blackMaskPerCnt = np.zeros((originalImg[0].shape[0], originalImg[0].shape[1]), np.uint8)
    # Draw a mask with only one contour at a time
    cntDraw = cv.drawContours(blackMaskPerCnt, [oneCnt], 0, 255, -1)
    # Count the amount of pixels per row of each contour (without the background)
    countWhitePxPerCol = [np.count_nonzero(col) for col in np.transpose(cntDraw) if np.count_nonzero(col) != 0]

    return countWhitePxPerCol

# This function returns the row number inside the BLOB of the top and bottom parts
# The result of the row location varies depending on the percentage
# E.g. if the BLOB is containing 100 rows, and percentage = 0.2, then the
# top row nr will be equal to 20 and the bottom row equal to 80
def getTopAndBottomRowBasedOnPercentage(nrOfRowsPerCnt, percentage):
    rowTop = math.floor(nrOfRowsPerCnt * percentage)
    rowBottom = nrOfRowsPerCnt - math.floor(nrOfRowsPerCnt * percentage)
    return rowTop, rowBottom


# Get the amount of white pixels at the top and bottom of a BLOB
# useful for example for recognizing bottles, because the top has less pixels than the bottom
def sumOfWhitePxTopAndBottomOfContour(listOfAllImgCountWhitePx, nrOfRowsToExtractInfo):
    sumsTop = []
    sumsBottom = []
    for imageData in listOfAllImgCountWhitePx:
        sumsPerImgTop = []
        sumsPerImgBottom = []
        for oneCntData in imageData:
            sumOfWhitePxTop = 0
            sumOfWhitePxBottom = 0
            # Calculate the total amount of white pixels in the first (e.g. 20) rows of the BLOB
            for countWhitePxT in range(0, nrOfRowsToExtractInfo):
                sumOfWhitePxTop += oneCntData[countWhitePxT]
            # Calculate the total amount of white pixels in the last (e.g. 20) rows of the BLOB
            for countWhitePxB in range(len(oneCntData) - nrOfRowsToExtractInfo, len(oneCntData)):
                sumOfWhitePxBottom += oneCntData[countWhitePxB]
            sumsPerImgTop.append(sumOfWhitePxTop)
            sumsPerImgBottom.append(sumOfWhitePxBottom)
        sumsTop.append(sumsPerImgTop)
        sumsBottom.append(sumsPerImgBottom)
    return sumsTop, sumsBottom


def sumOfWhitePxLeftAndRightOfContour(oneCntData, nrOfColsToExtractInfo):
    sumOfWhitePxLeft = 0
    sumOfWhitePxRight = 0
    # Calculate the total amount of white pixels in the first (e.g. 20) cols of the BLOB
    for countWhitePxL in range(0, nrOfColsToExtractInfo):
        sumOfWhitePxLeft += oneCntData[countWhitePxL]
    # Calculate the total amount of white pixels in the last (e.g. 20) cols of the BLOB
    for countWhitePxR in range(len(oneCntData) - nrOfColsToExtractInfo, len(oneCntData)):
        sumOfWhitePxRight += oneCntData[countWhitePxR]
    return sumOfWhitePxLeft, sumOfWhitePxRight


# Get the ratio of the amount of white pixels at the top of a BLOB vs at the bottom of it
def ratioWhitePxTopVsBottomPerContour3LayerList(sumListTop, sumListBottom):
    ratioAllImgs = []
    for sumsPerImgTop, sumsPerImgBottom in zip(sumListTop, sumListBottom):
        ratioOfTopVsBottomOfContours = []
        for sumOfWhitePxTop, sumOfWhitePxBottom in zip(sumsPerImgTop, sumsPerImgBottom):
            maximum = max(sumOfWhitePxTop, sumOfWhitePxBottom)
            if sumOfWhitePxBottom == 0 or sumOfWhitePxTop == 0:
                return 0
            if maximum == sumOfWhitePxBottom:
                ratioOfTopVsBottomOfContours.append(round(sumOfWhitePxTop / sumOfWhitePxBottom, 2))
            else:
                ratioOfTopVsBottomOfContours.append(round(sumOfWhitePxBottom / sumOfWhitePxTop, 2))
        ratioAllImgs.append(ratioOfTopVsBottomOfContours)
    return ratioAllImgs


def ratioWhitePxTopVsBottomPerContour(sumOfWhitePxTop, sumOfWhitePxBottom):
    maximum = max(sumOfWhitePxTop, sumOfWhitePxBottom)
    if sumOfWhitePxBottom == 0 or sumOfWhitePxTop == 0:
        return 0.0
    if maximum == sumOfWhitePxBottom:
        ratioOfTopVsBottomOfContour = round(sumOfWhitePxTop / sumOfWhitePxBottom, 2)
    else:
        ratioOfTopVsBottomOfContour = round(sumOfWhitePxBottom / sumOfWhitePxTop, 2)

    return ratioOfTopVsBottomOfContour

# </editor-fold>

