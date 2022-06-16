'''

This file contains steps used for detecting objects in an image or multiple images, a process called segmentation.
The result of the segmentation function is a three layered list, with the first layer
for each image, the second for each contour in that image and the third is for each point
within a contour. This list is later used for the next step in the image processing
pipeline which is feature extraction or representation.

Contact: Rasmus Lilholt Jacobsen, rlja20@student.aau.dk

'''

import cv2 as cv
import numpy as np
import glob


def segmentation(allData, rangeMask):

    #These two number variables might be redundant but i think I used it to save class of object for the KNN
    #NumberTraining = 0

    # Here we make two loops, first loop goes through all the lists with lists of images
    # These lists represent different classes, so a list with all the bottles and a list with random objects
    for imageSet in allData:

        #number = 0
        # Initialize lists to save contour and hierarchy lists for all images
        allImgContours = []
        allImgHierarchies = []
        # Here the for loop goes through every single image in the current list
        for image in imageSet:
            # Initialize lists to save all contours and hierarchy lists for one image at a time
            oneImgContours = []
            oneImgHierarchies = []

            # Some images are in grayscale and others are not
            # This code tries to convert the image to grayscale and if
            # it is already in grayscale it will continue
            try:
                grayscaleImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            except:
                print("")

            # Here the image is getting blurred, it will blur depending on whether the original image was grayscale
            try:
                gBlur = cv.GaussianBlur(grayscaleImage, (11,11), 0)
            except:
                gBlur = cv.GaussianBlur(image, (11, 11), 0)

            # Edge detection (Sobel Derivatives)
            scale = 1.5
            delta = 0
            ddepth = cv.CV_32F

            grad_x = cv.Sobel(gBlur, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
            grad_y = cv.Sobel(gBlur, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

            abs_grad_x = cv.convertScaleAbs(grad_x)
            abs_grad_y = cv.convertScaleAbs(grad_y)

            grad = cv.addWeighted(abs_grad_x, 0.7, abs_grad_y, 0.3, 0)


            # Thresholding on the output of the edge detection
            thresh = cv.threshold(grad,50, 190, cv.THRESH_BINARY)[1]

            # A bunch of different kernels for morphology are declared here
            roundKernel5 = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(5, 5))
            roundKernel11 = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(11, 11))
            roundKernel31 = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(31, 31))
            # Vertical kernel can be used to remove horizontal lines, and vice-versa
            verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, 6))
            horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (5, 1))
            verticalStructure2 = cv.getStructuringElement(cv.MORPH_RECT, (1, 12))

            # Morphology
            dilation = cv.morphologyEx(thresh, cv.MORPH_DILATE, roundKernel11, iterations=3)
            horizontalDilation = cv.morphologyEx(dilation, cv.MORPH_DILATE, horizontalStructure, iterations=2)

            horizontalClose = cv.morphologyEx(horizontalDilation, cv.MORPH_CLOSE, horizontalStructure, iterations=1)

            # dilation2 = cv.morphologyEx(closing, cv.MORPH_ERODE, roundKernel5, iterations=1)


            # Filtering
            medianFilter = cv.medianBlur(horizontalClose, 9)


            # This section if for the depth color map,
            # Currently the colorscheme from the color depthmap is the OCEAN color scheme from opencv
            # First the image is converted to the HSV color scheme
            HSV_CM = cv.cvtColor(rangeMask, cv.COLOR_BGR2HSV)
            # Here we are looking for a blue color with a certain intensity
            blueMask = cv.inRange(HSV_CM, (100, 250, 0), (140, 255, 34))
            medianBlueMask = cv.medianBlur(blueMask, 3)

            medianBlue= cv.morphologyEx(medianBlueMask, cv.MORPH_OPEN, roundKernel11, iterations=3)
            overlapOfMasks = cv.bitwise_and(medianFilter, medianBlue)

            maskDialation = cv.morphologyEx(overlapOfMasks, cv.MORPH_DILATE, roundKernel5, iterations=1)

            horizontalMaskErosion = cv.morphologyEx(maskDialation, cv.MORPH_ERODE, horizontalStructure, iterations=1)


            # This function increases the amount of space needed between objects
            # If they are too close the contours will connect
            bigMaskClose = cv.morphologyEx(horizontalMaskErosion, cv.MORPH_CLOSE, roundKernel31, iterations=1)

            verticalMaskErosion = cv.morphologyEx(bigMaskClose, cv.MORPH_ERODE, verticalStructure2, iterations=3)
            outputBlobs = cv.medianBlur(verticalMaskErosion, 9)
            # cv.imwrite('Blobs.png', outputBlobs)

            # Find all hierarchy types contours
            contours, hierarchy = cv.findContours(outputBlobs, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

            # The reason for using try-except is to avoid the code from breaking, if there were no contours in the image
            # In this part of the code, contours inside other contours and contours that are too small are removed
            try:

                # Making an array here to save the contour ID for a later the user input
                contourIDs = []

                # Just some variable declaration
                hierarchy = hierarchy[0]
                contourNumber = 0

                # Here the all the contours and hierarchy belonging to the contour is saved
                # in the following format: (contour, hierarchy)
                # The for loop runs through every contour and hierarchy
                for cnt in zip(contours, hierarchy):


                    # Some temporary variable declaration
                    currentContour = cnt[0]
                    currentHierarchy = cnt[1]

                    if currentHierarchy[3] == -1 and cv.contourArea(currentContour) > 4700:
                        # Everytime a contour passes the if statement we add to the counter
                        contourNumber += 1

                        # This is not a feature calculation, I'm using this to draw the contour ID on the image
                        (x, y), (w, h), angle = cv.minAreaRect(currentContour)

                        # The current contour is drawn here
                        cv.drawContours(image, currentContour, -1, (0, 0, 255), 2)

                        # The contournumber is added to the earlier declared list for the user input
                        contourIDs.append(contourNumber)

                        # Put text on the image with the number of the contour
                        # Now it is placed in the middle of the object, but can be moved
                        cv.putText(image, "CI {}".format(round(contourNumber, 1)), (int(x), int(y)), cv.FONT_HERSHEY_PLAIN,
                                   1, (100, 200, 0), 2)
                        # cv.imshow("testing",image)
                        # Get all contours and hierarchies for one image
                        oneImgContours.append(currentContour)
                        oneImgHierarchies.append(currentHierarchy)

                # Get all contours and hierarchies for all images, the final result will be a list of images,
                # where each list will contain a list of contours, where each contour list will contain a list of points
                # Same applies for the hierarchy lists
                allImgContours.append(oneImgContours)
                allImgHierarchies.append(oneImgHierarchies)
                # cv.imshow("output", image)

            except:
                print("")

                # If there is no contours in the image, this will be printed instead
                # and the next image will load after the next key press
                # print("No contours in the image")

            # Show each original image and the contour(s) painted on it
            # cv.imshow("New", image)
            #number += 1

            # This code looks at what is being pressed
            # If it's one of the mapped keys below it goes through the code
            # This could potentially be replaced with some communication function later on
            # keyReg = cv.waitKey(0)
            # try:
            #     if keyReg == ord('a'):
            #         print("You have selected:", contourStuff[0])
            #     elif keyReg == ord('1'):
            #         print("You have selected:", contourStuff[1])
            # except:
            #     print("There are still no contours in the image")
            # # End of the image

        #NumberTraining += 1
        return allImgContours, allImgHierarchies