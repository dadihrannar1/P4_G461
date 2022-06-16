'''

How to use:

This file is meant to be used separately for transforming a dataset of one dimension to a dataset of 2D
dimension. Thus, it is necessary to upload the training files for each class. Each file is a text file,
with the data saved as a matrix, where each row represents a sample (i.e. an object) and each column
represents one feature.
To save the new dataset, uncomment the lines 104-105, 116-117 and make sure to input the right file paths

It is possible to get a dataset of a dimension other than two. That would mean the end result will have more than the
x and y coordinate. To get that, simply calculate for N biggest eigenvectors. Keep in mind that the functions will have
to be adapted.

The file uses OpenCV Version 4.0.0, Python 3.8
Contact: Diana-Valeria Vacaru, dvacar21@student.aau.dk
Contact: Hector Gabriel Fabricius, hfabri20@student.aau.dk

'''

import numpy as np

# Class 0
bottleTrainingData = np.loadtxt('FeatureData/DAT/trainingBottlesNoCilindricality.dat', dtype=float, delimiter=',')
# Class 1
randomObjTrainingData = np.loadtxt('FeatureData/DAT/trainingRandomObjNoCilindricality.dat', dtype=float, delimiter=',')

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

# Calculate the mean for each feature in the data matrix,
# where each row is a sample, and each column is a feature
def meanOfArray(arrayIn):
    mean = np.array([0.0]*(len(arrayIn[0, :]) - 1), dtype=float)
    for i in range(arrayIn.shape[1] - 1):
        mean[i] = sum(arrayIn[:, i]) / len(arrayIn[:, i])
    return mean

# Transform data matrix from any dimension to 2D dimension
def fromXDimentionTo2D(oneCntFeatureVector, inputEigenVectors):
    x = np.dot(oneCntFeatureVector[0: oneCntFeatureVector.shape[0]], inputEigenVectors[0])
    y = np.dot(oneCntFeatureVector[0: oneCntFeatureVector.shape[0]], inputEigenVectors[1])
    return x, y


# Returns the two biggest numbers in an array
def eigenValuesBiggestTwo(eigenValues):
    biggestEigenValueIndex1 = 0
    biggestEigenValueIndex2 = 0
    for x in range(len(eigenValues)):
        if eigenValues[x] == max(eigenValues):
            biggestEigenValueIndex1 = x

    # Set the max to zero, to be able to find the next biggest number
    eigenValues[biggestEigenValueIndex1] = 0

    for y in range(len(eigenValues)):
        if eigenValues[y] == max(eigenValues):
            biggestEigenValueIndex2 = y

    return biggestEigenValueIndex1, biggestEigenValueIndex2


# Performs Principal component analysis (PCA) for an entire dataset
def PCA():
    combinedClasses = [bottleTrainingData, randomObjTrainingData]
    classNr = np.asarray(getClassNrArr(combinedClasses))
    trainingData = np.concatenate((bottleTrainingData, randomObjTrainingData))
    trainingDataWithLabels = np.append(trainingData, classNr, axis=1)

    data = trainingDataWithLabels

    standardized = np.zeros((data.shape[0], data.shape[1] - 1), dtype=data.dtype)

    mean = meanOfArray(data)
    for i in range(data.shape[1] - 1):
        standardized[:, i] = data[:, i] - mean[i]

    covariance = np.cov(np.transpose(standardized))

    # Because the vectors are most of the time sorted by the length of the eigenvalues available in the first list: eigenvectors[0]
    # we can just pick up the first two vectors in the second list: eigenvectors[1] to be the eigenvectors we work with
    # eigenVectors = np.transpose(np.linalg.eig(covariance)[1][:, 0:2])
    eigenValues, eigenVectors = np.linalg.eig(covariance)
    # The documentation for the numpy np.linalg.eig function says that "The eigenvalues are not necessarily ordered."
    # So, as a safety measure, the eigenvectors are selected are according to their corresponding biggest eigenvalues
    index1, index2 = eigenValuesBiggestTwo(eigenValues)

    # Save eigenVectors
    # with open('eigenVectors5D.dat', 'a') as dataFile:
    #     np.savetxt(dataFile, eigenVectors, delimiter=',', fmt=['%f', '%f', '%f', '%f', '%f'], comments='')

    # Get the new dimension coordinates using the many dimension data and the two longest eigenvectors
    # x = np.reshape(np.dot(data[:, 0: data.shape[1] - 1], eigenVectors[0]), (len(data), -1))
    # y = np.reshape(np.dot(data[:, 0: data.shape[1] - 1], eigenVectors[1]), (len(data), -1))
    x = np.reshape(np.dot(data[:, 0: data.shape[1] - 1], eigenVectors[index1]), (len(data), -1))
    y = np.reshape(np.dot(data[:, 0: data.shape[1] - 1], eigenVectors[index2]), (len(data), -1))

    twoDimentionFeatures = np.concatenate((x, y, classNr), axis=1)

    # Save the new training data, transformed to two dimensions
    # with open('2DTrainingSetAfterPCAFrom5D.dat', 'a') as dataFile:
    #     np.savetxt(dataFile, twoDimentionFeatures, delimiter=',', fmt=['%f', '%f', '%f'], comments='')

if __name__ == '__main__':
    PCA()
