'''

This is the main file of the project, where the algorithm loops through every contour and extracts a feature
vector for it. That is passed to the k-nearest neighbour classifier that uses pre-made training data
and detects the class of which the new object belongs to.

The training data can also be obtained by uncommenting the lines at the beginning and end of the main function and
by passing in the desired lists of images and depth data files.

The file uses OpenCV Version 4.0.0, Python 3.8
Contact: Diana-Valeria Vacaru, dvacar21@student.aau.dk

'''


import glob
import copy
import math
import cv2 as cv
import numpy as np
import segmentation as s
import segmentation3 as seg3
import featureExtraction as f
import functions as f2
import KNearestNeighbour as knn
import PCA as pca


# Get color data
trainingBottles = [np.load(file) for file in glob.glob("SavedImagesCleaned/Training/Bottles/Color/*.npy")]
trainingRandomObj = [np.load(file) for file in glob.glob("SavedImagesCleaned/Training/RandomObjects/Color/*.npy")]
testImages = [np.load(file) for file in glob.glob("SavedImagesCleaned/Test images/Color/*.npy")]
#
# # Get the depth data
trainingBottlesDepth = [np.load(file) for file in glob.glob("SavedImagesCleaned/Training/Bottles/Depth/*.npy")]
trainingRandomObjDepth = [np.load(file) for file in glob.glob("SavedImagesCleaned/Training/RandomObjects/Depth/*.npy")]
testImagesDepth = [np.load(file) for file in glob.glob("SavedImagesCleaned/Test images/Depth/*.npy")]
#
# # Change the data used here
colorData = copy.copy(testImages)
depthData = copy.copy(testImagesDepth)
#
# Load ready-made training data (6 feature vector)
bottleTrainingData = np.loadtxt('FeatureData/DAT/trainingBottlesNoCilindricality.dat', dtype=float, delimiter=',')
randomObjTrainingData = np.loadtxt('FeatureData/DAT/trainingRandomObjNoCilindricality.dat', dtype=float, delimiter=',')
#
allFileData = np.concatenate((bottleTrainingData, randomObjTrainingData))
combinedLists = [bottleTrainingData, randomObjTrainingData]

# Load ready-made 2D training data after performing PCA on the 6D data
twoDTrainingData = np.loadtxt('FeatureData/DAT/2DTrainingSetAfterPCAFrom5D.dat', dtype=float, delimiter=',')

# Load top two eigen-vectors that can only be applied to 6D feature vectors
eigenVectors = np.loadtxt('FeatureData/DAT/eigenVectors5D.dat', dtype=float, delimiter=',')

def imageProcessing(colorData, depthData, colorDepthData):

    # contours, hierarchies = s.segmentation([colorData])
    contours, hierarchies = seg3.segmentation([colorData], colorDepthData)
    f2.drawContours(contours, colorData, contours, (0, 255, 0), 2)

    # Prepare the feature data files
    # np.savetxt('trainingRandomObj.dat',
    #            [('topAndBottomWidthRatio', 'elongation', 'cilindricality', 'compactness', 'areaRatioAllImg', 'circularity')],
    #            delimiter=',', header='', fmt=['%s', '%s', '%s', '%s', '%s', '%s'], comments='')
    # np.savetxt('trainingRandomObj.csv',
    #            [('topAndBottomWidthRatio', 'elongation', 'cilindricality', 'compactness', 'areaRatioAllImg', 'circularity')],
    #            delimiter=',', header='', fmt=['%s', '%s', '%s', '%s', '%s', '%s'], comments='')

    for i, oneImgContours in enumerate(contours):
        #This list will contain the coordiantes
        allPoints = []


        for k, cnt in enumerate(oneImgContours):

            # Get centroid coordinates
            comx, comy = f.getCentroidCoordinates(cnt)
            # Display the centroid on the original image
            cv.circle(colorData[i], (comx, comy), 2, (0, 0, 255), 2)

            # Fit a line through the contour
            line = f.fitLineThroughCnt(colorData[i], cnt)

            # Create a black mask where the contour will be drawn on
            # The contours are drawn each at a time
            objectMask = np.zeros(colorData[i].shape, np.uint8)
            objectMask = cv.drawContours(objectMask, oneImgContours, k, (255, 250, 255), 2)

            # Create a black mask where the new line will be drawn on
            blackMask1 = np.zeros(colorData[i].shape, np.uint8)
            heightLineMask = f.fitLineThroughCnt(blackMask1, cnt)

            # Find the points of intersection of the contour with the height line
            lineFitIntersecPts = f.intersectionPoints(objectMask, heightLineMask)

            # Choose the two points that make up the highest distance
            # This is necessary for cases when the blob has weird contour shape
            hIntersecPts = np.asarray(f.findPointsWithLongestLength(lineFitIntersecPts))

            # Get points of the line intersection with the contour, perpendicular to the fit line
            # through the object center point
            wIntersecPts = f.getTwoPointCoordinatesOnContour(hIntersecPts, (comx, comy), objectMask, colorData[i])

            # ______________________________FEATURE 1: Relative percent-wise width at top and bottom__________________________________
            # Get the ratio between the first and last 15% of the contours widths
            # The width at the top should be smaller than the width at the bottom
            pointLineGoesThroughTop = f.getPointCoordinatesOnLine(hIntersecPts, 0.15)
            fifteenPercentTopPts = f.getTwoPointCoordinatesOnContour(hIntersecPts, pointLineGoesThroughTop, objectMask, colorData[i])
            widthFifteenPercentTop = f.computeDistanceBetweenTwoPoints(fifteenPercentTopPts[0][0], fifteenPercentTopPts[0][1], fifteenPercentTopPts[1][0], fifteenPercentTopPts[1][1])
            pointLineGoesThroughBottom = f.getPointCoordinatesOnLine(hIntersecPts, 0.85)
            eightyfivePercentBottomPts = f.getTwoPointCoordinatesOnContour(hIntersecPts, pointLineGoesThroughBottom, objectMask, colorData[i])
            widthEightyfivePercentBottom = f.computeDistanceBetweenTwoPoints(eightyfivePercentBottomPts[0][0], eightyfivePercentBottomPts[0][1],
                                                                           eightyfivePercentBottomPts[1][0], eightyfivePercentBottomPts[1][1])
            minimum = min(widthFifteenPercentTop, widthEightyfivePercentBottom)
            maximum = max(widthFifteenPercentTop, widthEightyfivePercentBottom)
            topAndBottomWidthRatio = round(minimum / maximum, 2)

            # ______________________________FEATURE 2: ELONGATION__________________________________
            # Figure out how much to move the found points inside the contour to avoid picking up depth data from
            # the background instead of the BLOB
            pointTopInside = f.getPointCoordinatesOnLine(hIntersecPts, 0.1)
            pointBottomInside = f.getPointCoordinatesOnLine(hIntersecPts, 0.85)
            pointLeftInside = f.getPointCoordinatesOnLine(wIntersecPts, 0.2)
            pointRightInside = f.getPointCoordinatesOnLine(wIntersecPts, 0.8)
            cv.circle(colorData[i], (pointTopInside[0], pointTopInside[1]), 2, (200, 50, 50), 2)
            cv.circle(colorData[i], (pointBottomInside[0], pointBottomInside[1]), 2, (200, 50, 40), 2)
            cv.circle(colorData[i], (pointLeftInside[0], pointLeftInside[1]), 2, (200, 50, 50), 2)
            cv.circle(colorData[i], (pointRightInside[0], pointRightInside[1]), 2, (200, 50, 40), 2)

            # Get distances and 3D coordinates from 2D coordinates of the 5 main points on the BLOB:
            # top most point, bottom most point, center of mass and left and right edge points on the same line as the COM
            distanceToCOM, com3D = f.transform2Dto3DSpace((comx, comy), depthData[i])
            distanceToTop, top3D = f.transform2Dto3DSpace((pointTopInside[0], pointTopInside[1]), depthData[i])
            distanceToBottom, bottom3D = f.transform2Dto3DSpace((pointBottomInside[0], pointBottomInside[1]), depthData[i])
            distanceToLeft, left3D = f.transform2Dto3DSpace((pointLeftInside[0], pointLeftInside[1]), depthData[i])
            distanceToRight, right3D = f.transform2Dto3DSpace((pointRightInside[0], pointRightInside[1]), depthData[i])
            allPoints.append(list(com3D))

            # Calculate real width and height in millimeters
            # Vector norm = norm(point 1 - point 2) = sqrt(x^2 + y^2 + z^2) = distance between point 1 and point 2
            height = round(np.linalg.norm(np.subtract(top3D, bottom3D)), 2)
            width = round(np.linalg.norm(np.subtract(left3D, right3D)), 2)
            elongation = float(round(width / height, 2))

            # ______________________________FEATURE 3: CYLINDRICALITY__________________________________
            # Check if the BLOB is cylindrical
            # Calculate radii on two axes
            # Axis 1: subtract the average depth data from the width points from the one at the centroid
            averageDepthOfWidthPts = (distanceToLeft + distanceToRight) / 2
            radius1 = distanceToCOM - averageDepthOfWidthPts
            # Axis 2: divide width by two
            radius2 = width / 2
            # Check if the two radii are very close to each other, that would mean object is cylindrical
            # If the ratio is bigger than 0.9 then it is considered cylindrical
            minRadius = min(radius1, radius2)
            maxRadius = max(radius1, radius2)
            cilindricality = float(round(minRadius / maxRadius, 2))

            # ______________________________FEATURE 4: COMPACTNESS__________________________________
            compactness = round(f2.compactnessRatio(cnt, colorData[i]), 2)

            # ______________________________FEATURE 5: Relative percent-wise area at top and bottom__________________________________

            oneCntCountWhitePxRow, nrRowsUntilCnt = f2.countWhitePxPerRow(colorData, cnt)
            oneCntCountWhitePxCol = f2.countWhitePxPerCol(colorData, cnt)

            rowNrTop, rowNrBottom = f2.getTopAndBottomRowBasedOnPercentage(len(oneCntCountWhitePxRow), 0.15)
            colNrLeft, colNrRight = f2.getTopAndBottomRowBasedOnPercentage(len(oneCntCountWhitePxCol), 0.15)

            sumWhitePxTop, sumWhitePxBottom = f2.sumOfWhitePxLeftAndRightOfContour(oneCntCountWhitePxRow, rowNrTop)
            sumWhitePxLeft, sumWhitePxRight = f2.sumOfWhitePxLeftAndRightOfContour(oneCntCountWhitePxCol, colNrLeft)

            # The area ratio of the real top and bottom regions of the object is calculated
            # Depending on the orientation of the object, different parts of the object in
            # the image will be compared to each other
            # E.g. if the blob is in a vertical position, then the image will be analyzed by rows
            # i.e. by rows would mean the area of the first X rows of the blob vs the area of the last X rows
            # if the blob is in a horizontal position, the image will be analyzed by columns
            # and if the blob is in an oblique position, the image will be analyzed by both rows and columns
            # The object's orientation is checked based on its fit height line's slope

            slopeOfHeightLine = f.calculateSlope(hIntersecPts[0], hIntersecPts[1])
            areaRatioAllImg = 0

            # If the line is relatively vertical, i.e. slope = inf
            # There are many cases when the slope is close to vertical
            if slopeOfHeightLine == 10000000000000000000 or (-24.0 <= slopeOfHeightLine <= -3.0) or (3.0 <= slopeOfHeightLine <= 24.0) \
                    or (-86.0 <= slopeOfHeightLine <= -5.0) or (5.0 <= slopeOfHeightLine <= 86.0):
                areaRatioAllImg = f2.ratioWhitePxTopVsBottomPerContour(sumWhitePxTop, sumWhitePxBottom)

            # If the line is relatively horizontal, i.e. slope is between [-0.2, 0.2]
            elif -0.2 <= slopeOfHeightLine <= 0.2:
                areaRatioAllImg = f2.ratioWhitePxTopVsBottomPerContour(sumWhitePxLeft, sumWhitePxRight)

            # If the slope is positive, but since Y axis is down, then the slope is also going down: \
            elif (0.3 <= slopeOfHeightLine < 3.0) or (0.2 < slopeOfHeightLine < 5.0):
                sumFromTheLeft = sumWhitePxLeft + sumWhitePxTop
                sumFromTheRight = sumWhitePxRight + sumWhitePxBottom
                areaRatioAllImg = f2.ratioWhitePxTopVsBottomPerContour(sumFromTheLeft, sumFromTheRight)

            # If the slope is negative, but since Y axis is down, then the slope is going up: /
            elif (-5.0 < slopeOfHeightLine <= -0.3) or (-3.0 < slopeOfHeightLine < -0.2):
                sumFromTheLeft = sumWhitePxLeft + sumWhitePxBottom
                sumFromTheRight = sumWhitePxRight + sumWhitePxTop
                areaRatioAllImg = f2.ratioWhitePxTopVsBottomPerContour(sumFromTheLeft, sumFromTheRight)

            # ______________________________FEATURE 6: CIRCULARITY__________________________________
            circularityValue = round(f2.circularity(cnt), 2)

            # Create the feature vector
            featureVector = [(topAndBottomWidthRatio, elongation, compactness, areaRatioAllImg, circularityValue)]

            # # Run the new object data through the K-nearest neighbour classifier
            # classNrList = knn.getClassNrArr(combinedLists)
            # classType = knn.KNN(5, featureVector, allFileData, classNrList)[2]

            # With 2D classifier after PCA
            classNrList = knn.getClassNrArr(combinedLists)
            twoDFeatureVector = pca.fromXDimentionTo2D(np.asarray(featureVector)[0, :], eigenVectors)
            classType = knn.KNN(5, [twoDFeatureVector], twoDTrainingData[:, 0:2], classNrList)[2]
            #
            # # Show the class type of each object, 0 - for bottle, 1 - for random objects
            coordinates = (cnt[math.floor(cnt.shape[0] / 2), 0, 0], cnt[math.floor(cnt.shape[0] / 2), 0, 1])
            # cv.putText(colorData[i], format(com3D), (100,200), cv.FONT_HERSHEY_PLAIN,
            #            2, (0, 0, 255), 2)

            cv.putText(colorData[i], format(classType), (coordinates[0], coordinates[1]-100), cv.FONT_HERSHEY_PLAIN,
                       2, (0, 0,255), 2)


            # Save all the samples' feature vectors
            # with open('testImages.dat', 'a') as dataFile:
            #     np.savetxt(dataFile, featureVector, delimiter=',', fmt=['%f', '%f', '%f', '%f', '%f', '%f'], comments='')
            # with open('trainingRandomObj.csv', 'a') as csvFile:
            #     np.savetxt(csvFile, featureVector, delimiter=',', fmt=['%f', '%f', '%f', '%f', '%f', '%f'], comments='')
            # cv.imwrite('contoursAndClass.png', colorData[i])
    f2.showImgs('numpy', colorData)
    return allPoints


    # print("")


# if __name__ == '__main__':
#     main()