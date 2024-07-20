import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metric(ABC):
    @abstractmethod
    def calculate(self, y, y_pred):
        pass

class AccuracyMetric(Metric):

    def calculate(self, y: np.ndarray, y_pred: np.ndarray):
        name = 'accuracy'
        accuracy = accuracy_score(y, y_pred)
        return name, accuracy

class PrecisionMetric(Metric):
    def calculate(self, y: np.ndarray, y_pred: np.ndarray):
        name = 'precision'
        precision = precision_score(y, y_pred)
        return name, precision

class RecallMetric(Metric):
    def calculate(self, y: np.ndarray, y_pred: np.ndarray):
        name = 'recall'
        recall = recall_score(y, y_pred)
        return name, recall

class FScoreMetric(Metric):
    def calculate(self, y: np.ndarray, y_pred: np.ndarray):
        name = 'f_score'
        f1 = f1_score(y, y_pred)
        return name, f1
