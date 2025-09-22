# MODEL | ARIMA
from typing import Dict, Any
from darts import TimeSeries
from darts.models import ARIMA
from darts.models import ExponentialSmoothing
from darts.metrics import smape


class TheModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.arima = None

    def arima_model(self) -> ARIMA:
        p, d, q = self.config.get("the_arima_order", [1, 1, 1])
        self.arima = ARIMA(order=(p, d, q))
        return self.arima

    def arima_fit(self, train_y: TimeSeries) -> None:
        if self.arima is None:
            self.arima_model()
        self.arima.fit(train_y)

    def arima_predict(self, horizon: int) -> TimeSeries:
        return self.arima.predict(horizon)

    def arima_score(self, y_true: TimeSeries, y_pred: TimeSeries) -> float:
        return float(smape(y_true, y_pred))
    

class TheSecondModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exponential = None

    def exponential_model(self) -> ExponentialSmoothing:
        self.exponential = ExponentialSmoothing()
        return self.exponential

    def exponential_fit(self, train_y: TimeSeries) -> None:
        if self.exponential is None:
            self.exponential_model()
        self.exponential.fit(train_y)

    def exponential_predict(self, horizon: int) -> TimeSeries:
        return self.exponential.predict(horizon)

    def exponential_score(self, y_true: TimeSeries, y_pred: TimeSeries) -> float:
        return float(smape(y_true, y_pred))