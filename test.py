# -*- coding: utf-8 -*-
import knn
import svm
import logistic_sgd
import mlp
import cnn
import numpy
from numpy import zeros
from sklearn.externals import joblib
import cPickle as pickle

def txtprint(classifierResult):
    fo = open('predict.txt', 'w')
    for i in range(len(classifierResult)):
        line = str(classifierResult[i]) + '\n'
        fo.write(line)
    fo.close()

def knnTest(testData):
    classifierResult = knn.Test(testData, 3)
    txtprint(classifierResult)
    # print float(numpy.sum(classifierResult - test_labels1 != 0)) / len(test_labels1)

def svmTest(testData):
    clf = joblib.load("SVM_model.m")
    mTest = len(testData)
    testingMat = zeros((mTest, 28 * 28))
    for i in range(mTest):
        testingMat[i, :] = svm.img2vector(testData[i])
    classifierResult = clf.predict(testingMat)
    txtprint(classifierResult)
    # print float(numpy.sum(classifierResult - test_labels1 != 0)) / len(test_labels1)

def softmaxTest(testData):
    predicted_values= logistic_sgd.predict(testData)
    txtprint(predicted_values)
    # print float(numpy.sum(predicted_values - test_labels1 != 0)) / len(test_labels1)

def mlpTest(testData):
    predicted_values= mlp.predict(testData)
    txtprint(predicted_values)
    # print float(numpy.sum(predicted_values - test_labels1 != 0)) / len(test_labels1)

def cnnTest(testData):
    predicted_values= cnn.predict(testData)
    txtprint(predicted_values)
    # print float(numpy.sum(predicted_values - test_labels1 != 0)) / len(test_labels1)


# 示例
# fo = open('test.pkl', 'rb')
# entry = pickle.load(fo)
# test_data = entry['test']
# cnnTest(test_data)
