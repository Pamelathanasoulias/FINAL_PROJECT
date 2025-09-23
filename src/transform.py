import pandas as pd
from pathlib import Path
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller


class WeatherTransform:
    def __init__(self, df, params):
        self.df = df.copy()
        self.params = params
        self.y_series = None
        self.covariates_series = None

    # CLEAN COLUMNS
    def the_columns(self):
        the_time = self.params["the_time"]
        the_target = self.params["the_target"]
        keep = [the_time] + self.params["the_features"]

        self.df[the_time] = pd.to_datetime(self.df[the_time], errors="coerce")
        self.df = self.df[keep].dropna(subset=[the_time]).sort_values(the_time)
        self.df = self.df.drop_duplicates(subset=[the_time])

    # SAVE CLEAN FILE
    def save_clean_file(self):
        clean_path = self.params["data"]["processed_path"]
        Path(clean_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(clean_path, index=False)

    # BUILD SERIES
    def build_series(self):
        the_time = self.params["the_time"]
        the_target = self.params["the_target"]

        # MAKE SERIES FOR TARGET + COVARIATES
        self.y_series = TimeSeries.from_dataframe(
            self.df,
            time_col=the_time,
            value_cols=the_target,
            fill_missing_dates=True,
            freq=None,)

        the_covariates = [c for c in self.params["the_features"] if c != the_target]
        if the_covariates:
            self.covariates_series = TimeSeries.from_dataframe(
                self.df,
                time_col=the_time,
                value_cols=the_covariates,
                fill_missing_dates=True, freq=None,)

        # FILL ANY NANS WITH FORWARD/BACKWARD STRATEGY
        filler = MissingValuesFiller()
        self.y_series = filler.transform(self.y_series)
        if self.covariates_series is not None:
            self.covariates_series = filler.transform(self.covariates_series)

    # TRAIN / TEST SPLIT
    def train_test_split(self):
        test_size = int(self.params["test_size"])
        train = self.y_series[:-test_size]
        test = self.y_series[-test_size:]
        return train, test
