import numpy as np
import math
import sys
np.set_printoptions(threshold=sys.maxsize)

class decisionTreeClassifier:
    def __init__(self, max_depth, threshold):
        self.max_depth = max_depth
        self.threshold = threshold
        self.root = None

    def fit(self, X, Y):
        root = node(X, Y, depth=1, state=None, div=[])
        root.is_leaf = False
        root.split(X, Y, root.depth, self.max_depth, self.threshold)
        self.root = root

    def predict(self, X):
        numOfRows,  numOfColumns = np.shape(X)
        labels = -1 * np.ones(numOfRows)
        for i in range(numOfRows):
            node = self.root
            while node.is_leaf is not True and node.depth <= self.max_depth:
                x = 0
                attr = node.attribute
                #print(X[i, attr])
                if node.value is None:
                    for child in node.children:
                        if child.state == X[i, attr]:
                            x = 1
                            node = child
                            break
                else:
                    if X[i, attr] < node.value:
                        x = 1
                        node = node.children[0]
                    else:
                        x = 1
                        node = node.children[1]
                #print("label", node.label, "depth", node.depth)
                if x == 0:
                    labels[i] = labels[i-1]
                    break
            labels[i] = node.label
        return labels


class node:
    def __init__(self, X, Y, depth, state, div):
        self.X = X
        self.Y = Y
        self.is_leaf = None
        self.entropy = None
        self.attribute = None
        self.value = None
        self.state = state
        self.depth = depth
        self.label = None
        self.children = []
        self.div = div

    def split(self, X, Y, depth, max_depth, threshold):
        self.entropy = entropy(Y)
        if depth == max_depth:
            self.is_leaf = True
            self.label = np.argmax(np.bincount(Y))
            self.children = None
        elif np.count_nonzero(Y == 0) / np.size(Y) > threshold:
            #print("threshold0")
            self.is_leaf = True
            self.label = 0
            self.children = None
        elif np.count_nonzero(Y == 1) / np.size(Y) > threshold:
            #print("threshold1")
            self.is_leaf = True
            self.label = 1
            self.children = None
        else:
            best_attr, Value = self.best_attribute(X, Y)
            self.divide(X, Y, best_attr, Value)
            for child in self.children:
                child.split(child.X, child.Y, child.depth, max_depth, threshold)

    def best_attribute(self, X, Y):
        numOfRows, numOfColumns = np.shape(X)
        bestAttrIG = 0
        attr = -1
        Value = None
        for index in range(numOfColumns):
            x = False
            for j in self.div:
                if j == index:
                    x = True
                    break
            if x == False:
                if (index == 1 or index == 2 or index == 5 or index == 6 or index == 8 or index == 10 or index == 11 or index == 12):
                    tempAttrIG = self.information_gain1(X, Y, index)
                    if tempAttrIG > bestAttrIG:
                        bestAttrIG = tempAttrIG
                        attr = index
                else:
                    value, tempAttrIG = self.information_gain2(X, Y, index)
                    if tempAttrIG > bestAttrIG:
                        bestAttrIG = tempAttrIG
                        attr = index
                        Value = value
        
        return attr, Value

    def information_gain1(self, X, Y, index):
        numOfRows, _ = np.shape(X)
        u = np.unique(X[:, index])
        IG = self.entropy
        for i in range(len(u)):
            indicies = np.where(X[:, index] == u[i])
            IG -= len(indicies) / np.size(Y) * (entropy(Y[indicies]))
        return IG

    def information_gain2(self, X, Y, index):
        maxIG = 0
        Value = None
        for value in X[:, index]:
            big_Indicies = np.where(X[:,index] >= value)
            lit_Indicies = np.where(X[:,index] <= value)
            tempIG = entropy(Y) - len(big_Indicies) / np.size(Y) * entropy(Y[big_Indicies]) - len(lit_Indicies) / np.size(Y) * entropy(Y[lit_Indicies])
            if tempIG > maxIG:
                maxIG = tempIG
                Value = value
        return  Value, maxIG

    def divide(self, X, Y, best_attr, Value):
        if Value is None:
            self.attribute = best_attr
            for v in np.unique(X[:, best_attr]):
                indicies = np.where(X[:, best_attr] == v)
                self.children += [node(X=X[indicies], Y=Y[indicies], depth=self.depth + 1, state=v, div=self.div + [self.attribute])]
        else:
            self.attribute = best_attr
            self.value = Value
            big_indicies = np.where(X[:, best_attr] >= Value)
            lit_indicies = np.where(X[:, best_attr] <= Value)
            self.children += [node(X=X[lit_indicies], Y=Y[lit_indicies], depth=self.depth + 1, state=None, div=self.div + [self.attribute])]
            self.children += [node(X=X[big_indicies], Y=Y[big_indicies], depth=self.depth + 1, state=None, div=self.div + [self.attribute])]

def entropy(Y):
    if np.size(Y) == 0:
        return 0
    pone = np.count_nonzero(Y == 1) / np.size(Y)
    pzero = np.count_nonzero(Y == 0) / np.size(Y)
    if pone == 0:
        return -1 * pzero * math.log(pzero, 2)
    elif pzero == 0:
        return -1 * pone * math.log(pone, 2)
    else:
        return -1 * pone * math.log(pone, 2) - pzero * math.log(pzero, 2)
