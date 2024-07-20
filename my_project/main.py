from my_project.scr.load_data import DataLoader, PandasLoader
from my_project.scr.metrics import AccuracyMetric
from my_project.scr.models import LightGBMModel
from my_project.scr.model_runner import ModelRunner

def main():
    train_path = '/Users/821069/personal/data_science_project/showcase-data-science-project/data/train.csv'
    test_path = '/Users/821069/personal/data_science_project/showcase-data-science-project/data/test.csv'
    train_loader = DataLoader(loader = PandasLoader(train_path))
    test_loader = DataLoader(loader = PandasLoader(test_path))
    metrics = [
        AccuracyMetric()
    ]
    params = None
    target = ''
    features = [
    ]
    model = LightGBMModel(params = params)

    train_run = ModelRunner(
        train_loader = train_loader,
        test_loader = test_loader,
        model = model,
        target = target,
        features = features,
        metrics = metrics
    )

    fitted_model = train_run.run()


if __name__ == "__main__":
    main()