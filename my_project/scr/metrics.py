from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score

class Metric(ABC):
    @abstractmethod
    def calculate(self, y, y_pred):
        pass

class AccuracyMetric(Metric):

    def calculate(self, y, y_pred):
        name = 'accuracy'
        accuracy = accuracy_score(y, y_pred)
        return name, accuracy
