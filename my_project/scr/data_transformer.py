import pandas as pd
from load_data import DataLoader

class DataPipeline:

    def __init__(
            self,
            data: DataLoader,
            target,
            test_size = 0.2,
            random_state = 42
        ) -> None:
        self.data: DataLoader
        self.target = target
        self.train_path = 'train_output_path'
        self.test_path = 'test_output_path'
        self.test_size = test_size
        self.random_state = 42
        self.preprocessor = None

    def split(self):
        pass

    def pre_process(self):
        pass

    def save(self):
        pass

    def run(self):
        pass