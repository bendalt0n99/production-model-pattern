import mlflow
from typing import List
from load_data import DataLoader
from metrics import Metric
from models import AbstractModel

class ModelRunner:
    def __init__(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: AbstractModel,
        target: str,
        features: List[str],
        metrics: List[Metric],
        model_name: str = 'model',
    ) -> None:
        self.model_name = model_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.target = target
        self.features = features
        self.metrics = metrics

    def run(self):
        with mlflow.start_run() as run:
            train_data = self.train_loader.load_data()
            test_data = self.test_loader.load_data()
            X_train = train_data[self.features]
            y_train = train_data[self.target]
            X_test = test_data[self.features]
            y_test = test_data[self.target]

            mlflow.log_params(self.model.params)

            trained_model = self.model.fit(X_train, y_train)

            y_pred = trained_model.predict(X_test)

            for metric in self.metrics:
                name, value = metric.calculate(y_test, y_pred)
                mlflow.log_metric(name, value)

            mlflow.lightgbm.log_model(trained_model, self.model_name)

            return trained_model

