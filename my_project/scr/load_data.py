import pandas as pd

from abc import ABC, abstractmethod

class Loader(ABC):

    @abstractmethod
    def load_data(self):
        pass

class PandasLoader(Loader):

    def __init__(self, path: str) -> None:
        self.path = path

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

class DataLoader:

    def __init__(self, loader: Loader) -> None:
        self.loader = loader

    def load_data(self) -> pd.DataFrame:
        return self.loader.load_data()