# -*- coding: utf-8 -*-
"""
借鉴自http://blog.csdn.net/zouxy09/article/details/16955347
"""

from numpy import *
import operator
import time
import read_train_data as data

def classify(inputPoint, dataSet, labels, k):
    dataSetSize = len(dataSet)
    diffMat = tile(inputPoint,(dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[ sortedDistIndicies[i] ]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def img2vector(trainingData):
    returnVect = []
    for i in range(28):
        for j in range(28):
            returnVect.append(trainingData[i,j])
    return returnVect

def trainingDataSet():
    hwLabels = data.train_labels1
    trainingDataList = data.train_data1
    m = len(trainingDataList)
    trainingMat = zeros((m, 28*28))
    for i in range(m):
        trainingMat[i,:] = img2vector(trainingDataList[i])
    return hwLabels,trainingMat

def Test(testData, k):
    hwLabels,trainingMat = trainingDataSet()
    mTest = len(testData)
    classifierResult = []
    for i in range(mTest):
        vectorUnderTest = img2vector(testData[i])
        classifierResult.append(classify(vectorUnderTest, trainingMat, hwLabels, k))
    return classifierResult

def handwritingTest():
    for k in range(8,11):
        t1 = time.time()
        errorCount = 0.0
        mTest = len(data.valid_labels1)
        classifierResult = Test(data.valid_data1,k)
        for i in range(mTest):
            if (data.valid_labels1[i] != classifierResult[i]): errorCount += 1.0
        print "knn method k = %d" % k
        print "the total error rate is: %f" % (errorCount / float(mTest))
        t2 = time.time()
        print "Cost time: %.2fmin, %.4fs." % ((t2 - t1) // 60, (t2 - t1) % 60)

if __name__ == "__main__":
    handwritingTest()