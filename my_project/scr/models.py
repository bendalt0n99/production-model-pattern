import lightgbm as lgb
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Any

class AbstractModel(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class LightGBMModel(AbstractModel):
    def __init__(self, params: Optional[dict] = None) -> None:
        self.params = params
        self.trained_model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        data = lgb.Dataset(X, label=y)
        self.trained_model = lgb.train(self.params, data)
        return self.trained_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.trained_model.predict(X)
