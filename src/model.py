# DARTS ARIMA STARTER (SIMPLE, BEGINNER-FRIENDLY)

from darts.models import ARIMA
from darts import TimeSeries

class TimeSeriesModel:
    # WRAPPER AROUND A SIMPLE ARIMA MODEL

    def __init__(self, series: TimeSeries):
        # SAVE THE INPUT SERIES
        self.series = series
        self.model = None

    def fit_arima(self, order=(2, 1, 1)):
        # FIT ARIMA WITH DEFAULT ORDER
        self.model = ARIMA(order=order)
        self.model.fit(self.series)

    def predict(self, steps: int = 300) -> TimeSeries:
        # PREDICT NEXT STEPS (DEFAULT 300)
        return self.model.predict(steps)
