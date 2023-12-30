from knn import KNN_Classifier
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import defaultdict


iris = datasets.load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.7) #70% training, 30% testing

def evaluate_knn_classifier(k):
    knn_instance = KNN_Classifier(k)
    knn_instance.fit(x_train, y_train)
    predictions = knn_instance.predict(x_test)
    y_test_array = np.array(y_test)
    TP_dict, FP_dict, TN_dict, FN_dict = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)

    for pred, actual in zip(predictions, y_test_array):
        for label in range(3): #dataset has 3 classes (0, 1, 2)
            if actual == label:
                if pred == label:
                    TP_dict[label] += 1
                else:
                    FN_dict[label] += 1
            else:
                if pred == label:
                    FP_dict[label] += 1
                else:
                    TN_dict[label] += 1

    TP, FP, TN, FN = sum(TP_dict.values()), sum(FP_dict.values()), sum(TN_dict.values()), sum(FN_dict.values())
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


for i in range(1, 20,2):
    accuracy, precision, recall, f1_score = evaluate_knn_classifier(i)
    print(f'\nMetrics for k = {i}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')






