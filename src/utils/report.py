# METRICS AND THE GRAPHS
from typing import Dict
from darts import TimeSeries
from darts.metrics import smape, mae, mse
import matplotlib.pyplot as plt

def evaluate_series(y_true: TimeSeries, y_pred: TimeSeries) -> Dict[str, float]:
    return {
        "smape": float(smape(y_true, y_pred)),
        "mae":   float(mae(y_true, y_pred)),
        "mse":   float(mse(y_true, y_pred)),}

def plot_forecast(train: TimeSeries, test: TimeSeries, forecast: TimeSeries) -> None:
    train.plot(label="TRAIN")
    test.plot(label="TEST")
    forecast.plot(label="PREDICT")
    plt.legend()
    plt.show()
