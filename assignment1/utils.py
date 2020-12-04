import numpy as np
import csv
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from DecisionTree import decisionTreeClassifier
from kNN import knnClassifier
from scipy.stats import t
from statistics import mean, stdev
from math import sqrt

def read_data(address, targetColumnName):
    with open(address, 'r') as file:
        reader = csv.reader(file)
        data = [data for data in reader]
        data_array = np.array(data)
        columns = data_array[0, :]
        targetColumnIndex = np.where(columns == targetColumnName)
        file.close()
        X = data_array[1:, :targetColumnIndex[0][0]]
        X = np.asarray(X, dtype=float)
        Y = data_array[1:, targetColumnIndex[0][0]]
        Y = np.asarray(Y, dtype=int)
        return X, Y

def shuffle_data(X, Y):
    numOfRow, numOfColumn = np.shape(X)
    mask = np.random.permutation(numOfRow)
    np.random.shuffle(mask)
    newX, newY = X[mask], Y[mask]
    return newX, newY

def split_data(X, Y, ratio):
    numOfRow, numOfColumn = np.shape(X)
    numOfTrainingData = int(numOfRow * ratio)
    trainData = X[:numOfTrainingData, :], Y[:numOfTrainingData]
    testData = X[numOfTrainingData:, :], Y[numOfTrainingData:]
    return trainData, testData

def accuracy(predicted_label, true_label):
    numOfSamples = np.size(predicted_label)
    numOfSamples2 = np.size(true_label)
    if numOfSamples is not numOfSamples2:
        return "invalid state to compute accuracy"
    numOfTrue = 0
    for i in range(numOfSamples):
        if predicted_label[i] == true_label[i]:
            numOfTrue += 1
    return numOfTrue / numOfSamples

def confusion_matrix(predicted_label, true_label):
    numOfSamples = np.size(predicted_label)
    numOfSamples2 = np.size(true_label)
    if numOfSamples is not numOfSamples2:
        return "invalid state to compute accuracy"
    FP, FN, TN, TP = 0, 0, 0, 0
    for i in range(numOfSamples):
        if predicted_label[i] == 1 and true_label[i] == 1:
            TP += 1
        elif predicted_label[i] == 1 and true_label[i] == 0:
            FP += 1
        elif predicted_label[i] == 0 and true_label[i] == 1:
            FN += 1
        elif predicted_label[i] == 0 and true_label[i] == 0:
            TN += 1
    return FP, FN, TN, TP
def classification_report(predicted_label, true_label):
    FP, FN, TN, TP = confusion_matrix(predicted_label, true_label)
    acc = accuracy(predicted_label, true_label)
    recall = TP / (TP + FN)
    spec = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1score = 2 * (precision * recall) / (precision + recall)
    return acc, recall, spec, precision, f1score

def ROC_curve(predicted_label, true_label):
    #acc, recall, spec, precision, f1score = classification_report(predicted_label, true_label)
    #tpr, fpr = recall, 1 - spec
    fpr, tpr, thresholds = metrics.roc_curve(true_label, predicted_label)
    print(fpr, tpr, thresholds)
    roc_auc = metrics.auc(tpr, fpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()



def k_foldCrossVal(k, classifier, trainData):
    trainX, trainY = trainData
    numOfRows, _ = np.shape(trainX)
    result = 0
    for i in range(k):
        validX, validY = trainX[i * numOfRows//k:(i+1) * numOfRows//k, :], trainY[i * numOfRows//k:(i+1) *numOfRows//k]
        train_X = np.concatenate((trainX[:i * numOfRows//k + 1, :], trainX[(i+1) * numOfRows//k:, :]), axis=0)
        train_Y = np.concatenate((trainY[:i * numOfRows//k + 1], trainY[(i+1) * numOfRows//k:]))
        classifier.fit(train_X, train_Y)
        labels = classifier.predict(validX)
        result += accuracy(labels, validY)
    result /= k
    return result
def t_test(treeprediction, knnprediction, labels):
    numOfSamples = np.size(labels)
    tree , knn = [], []
    m , SE = 0 , 0
    for i in range(numOfSamples):
        if(treeprediction[i] == labels[i]):
            tree += [1]
        if(treeprediction[i] != labels[i]):
            tree += [0]
        if knnprediction[i] == labels[i]:
            knn += [1]
        if knnprediction[i] != labels[i]:
            knn += [0]
    diff = []
    for i in range(numOfSamples):
        diff += [tree[i] - knn[i]]
    m = mean(diff)
    SE = stdev(diff) / sqrt(numOfSamples)
    tc = m / SE
    ta = t.ppf(1-0.05, df=numOfSamples - 1)
    print("t-value calculated", tc)
    print("t-value expected to reject", ta)
    if tc > ta:
        print("there is improvement in using decision tree rather than knn")
    else:
        print("no significant difference between two models")
    
    

