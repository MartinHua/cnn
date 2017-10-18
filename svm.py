# -*- coding: utf-8 -*-
"""
学习自https://xacecask2.gitbooks.io/scikit-learn-user-guide-chinese-version/content/sec1.4.html
"""
from numpy import zeros
import time
from sklearn import svm
import read_train_data as data
from sklearn.externals import joblib

def img2vector(trainingData):
    returnVect = []
    for i in range(28):
        for j in range(28):
            returnVect.append(trainingData[i,j]/255.)
    return returnVect

def trainingDataSet():
    hwLabels = data.train_labels1
    trainingDataList = data.train_data1
    m = len(trainingDataList)
    trainingMat = zeros((m, 28*28))
    for i in range(m):
        trainingMat[i,:] = img2vector(trainingDataList[i])
    return hwLabels,trainingMat

def Test(hwLabels,trainingMat, testingMat, kernel, gamma):
    clf = svm.SVC(kernel=kernel, C=100, gamma=gamma)
    clf.fit(trainingMat, hwLabels)
    filename = kernel + "_model_C=100_" + str(gamma) + ".m"
    joblib.dump(clf, filename)
    classifierResult = clf.predict(testingMat)
    return classifierResult

def handwritingTest():
    t1 = time.time()
    hwLabels, trainingMat = trainingDataSet()
    mTest = len(data.valid_data1)
    testingMat = zeros((mTest, 28 * 28))
    for i in range(mTest):
        testingMat[i, :] = img2vector(data.valid_data1[i])
    t2 = time.time()
    print "Cost time: %.2fmin, %.4fs." % ((t2 - t1) // 60, (t2 - t1) % 60)
    for gamma in [0.001, 0.05, 0.1, 0.0001]:
        for kernel in ['rbf']:
            t1 = time.time()
            errorCount = 0.0
            mTest = len(data.valid_labels1)
            classifierResult = Test(hwLabels, trainingMat, testingMat, kernel, gamma)
            for i in range(mTest):
                if (data.valid_labels1[i] != classifierResult[i]): errorCount += 1.0
            print "svm method kernel = %s, C = %.2f, gamma = %.4f"% (kernel, 100, gamma)
            print "the total error rate is: %f" % (errorCount / float(mTest))
            t2 = time.time()
            print "Cost time: %.2fmin, %.4fs." % ((t2 - t1) // 60, (t2 - t1) % 60)


if __name__ == "__main__":
    handwritingTest()