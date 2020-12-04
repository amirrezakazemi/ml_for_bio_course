import numpy as np

class knnClassifier:
    def __init__(self, k):
        self.k = k
        self.X, self.Y = None, None

    def fit(self, X, Y):
        self.X, self.Y = X, Y

    @staticmethod
    def distance(a, b):
        return np.linalg.norm(np.subtract(a, b))

    def predict(self, testData):
        testRowsCount, testColumnsCount = np.shape(testData)
        trainRowsCount, _ = np.shape(self.X)
        labels = -1 * np.ones(testRowsCount)
        for i in range(testRowsCount):
            distances = np.array([(np.inf, -1 * np.inf)])
            for j in range(trainRowsCount):
                dist = self.distance(testData[i,:], self.X[j,:])
                distances = np.vstack((distances, (dist, self.Y[j])))
            distances = distances[np.argsort(distances[:, 0])]
            k_labels = distances[:self.k, 1]
            k_labels = np.asarray(k_labels, dtype=int)
            labels[i] = np.argmax(np.bincount(k_labels))
        return labels

