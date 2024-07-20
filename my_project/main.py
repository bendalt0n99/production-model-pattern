from my_project.scr.load_data import DataLoader, PandasLoader
from my_project.scr.metrics import AccuracyMetric
from my_project.scr.models import LightGBMModel
from my_project.scr.model_runner import ModelRunner

def main():
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    train_loader = DataLoader(loader = PandasLoader(train_path))
    test_loader = DataLoader(loader = PandasLoader(test_path))
    metrics = [
        AccuracyMetric()
    ]
    params = None
    target = 'price_range'
    features = [
        'battery_power',
        'clock_speed',
        'dual_sim',
        'fc',
        'four_g',
        'int_memory',
        'm_dep',
        'mobile_wt',
        'n_cores',
        'pc',
        'px_height',
        'px_width',
        'ram',
        'sc_h',
        'sc_w',
        'talk_time',
        'three_g',
        'touch_screen',
        'wifi'
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