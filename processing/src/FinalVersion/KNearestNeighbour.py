'''

This file contains the functions needed to implement the k-nearest neighbour classifier.
The classifier is the last step in our image processing pipeline.

How to use:

import the file using: import KNearestNeighbour as knn
get the list of class identification numbers for each entry in the training class files using:
            classNrList = knn.getClassNrArr(combinedLists)
get the class type number for each BLOB, here the feature vector is a list of features for one sample
            classType = knn.KNN(10, featureVector, allFileData, classNrList)[2]

Contact: Rasmus Lilholt Jacobsen, rlja20@student.aau.dk

'''

import numpy as np
import math

# Given the lists of different classes,
# creates an array with the class numbers
# E.g. if there are two classes only, then
# fist class will have nr. 0, second will
# have nr. 1 and so on
def getClassNrArr(combinedListsLocal):
    classNrArr = []
    # Get the array of class type numbers,
    # here we have two classes, bottle and not a bottle
    # so the class numbers will be 0 and 1
    classNumber = 0
    for list in combinedListsLocal:
        for i in range(len(list)):
            classNrArr.append([classNumber])
        classNumber += 1
    return classNrArr


# Given a list with the k-nearest neighbour classes, it
# returns the number of the class that is most frequent
def most_frequent(list):
    return max(set(list), key=list.count)


# Get a list will all distances from the unknown data point
# to the training data points
def EucledianDistance(trainingSet, unknownDataSet):
    distanceList = []
    for i in range(len(trainingSet)):
        temp = math.dist(trainingSet[i], np.array(unknownDataSet[0]))
        distanceList.append(temp)

    return distanceList


# Takes as input the distance list to all the classes' points
# and finds the top k-smallest distances
# and the class numbers for those points
# k is the amount of neighbours with the smallest distance
def ShortestDistance(distanceList, k, classNr):
    kDistance = []
    kClassNr = []
    for i in range(k):
        minDistance = min(distanceList)
        pop = distanceList.index(minDistance)
        tempClassNr = classNr[pop][0]
        distanceList.pop(pop)
        np.delete(classNr, pop)

        kDistance.append(minDistance)
        kClassNr.append(tempClassNr)
    return kDistance, kClassNr



# Given the nr. of neighbours, the unknown data,
# the training data and the class number,
# get the list of distances to the k-nearest
# neighbours, their class numbers and the
# estimated class number of the unknown entry(s)
def KNN(k, unknownDataSet, trainingSet, classNrSet):
    distanceList = EucledianDistance(trainingSet, unknownDataSet)
    kDistanceList, kClassNr = ShortestDistance(distanceList, k, classNrSet)
    judge = most_frequent(kClassNr)
    # print("Distance to nearest neighbors:", dist)
    # print("Class of the nearest neighbors", index)
    # print("Object is most likely to be of class:", judge)
    return kDistanceList, kClassNr, judge


