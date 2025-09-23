"""Transform utilities for the weather time-series dataset.

This module prepares the raw dataframe, saves a cleaned CSV to data processed folder, and builds
Darts TimeSeries objects for the target and covariates.
"""

import pandas as pd
from pathlib import Path
from darts import TimeSeries
from darts.dataprocessing.transformers import MissingValuesFiller


class WeatherTransform:
    """
    Parameters
    ----------
    df : pandas.DataFrame
        Raw input dataframe.
    params : dict
        Project parameters : columns, paths, test size.

    Attributes
    ----------
    df : pandas.DataFrame
        Cleaned dataframe.
    params : dict
        Stored project parameters.
    y_series : darts.TimeSeries | None
        Target series : temperature.
    covariates_series : darts.TimeSeries | None
        Covariates series : pressure, humidity.
    """

    def __init__(self, df, params):
        """Init with dataframe and params dictionary."""
        self.df = df.copy()
        self.params = params
        self.y_series = None
        self.covariates_series = None

    def the_columns(self):
        """Kept required columns, clean and sort by time."""
        the_time = self.params["the_time"]
        the_target = self.params["the_target"]
        keep = [the_time] + self.params["the_features"]

        self.df[the_time] = pd.to_datetime(self.df[the_time], errors="coerce")
        self.df = self.df[keep].dropna(subset=[the_time]).sort_values(the_time)
        self.df = self.df.drop_duplicates(subset=[the_time])

    def save_clean_file(self):
        """Save cleaned dataframe to processed path."""
        clean_path = self.params["data"]["processed_path"]
        Path(clean_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(clean_path, index=False)

    def build_series(self):
        """Build Darts TimeSeries for target and covariates."""
        the_time = self.params["the_time"]
        the_target = self.params["the_target"]

        self.y_series = TimeSeries.from_dataframe(
            self.df, time_col=the_time, value_cols=the_target,
            fill_missing_dates=True, freq=None
        )

        covariates = [c for c in self.params["the_features"] if c != the_target]
        if covariates:
            self.covariates_series = TimeSeries.from_dataframe(
                self.df, time_col=the_time, value_cols=covariates,
                fill_missing_dates=True, freq=None
            )

        filler = MissingValuesFiller()
        self.y_series = filler.transform(self.y_series)
        if self.covariates_series is not None:
            self.covariates_series = filler.transform(self.covariates_series)

    def train_test_split(self):
        """Split series into train/test by test_size."""
        test_size = int(self.params["test_size"])
        train = self.y_series[:-test_size]
        test = self.y_series[-test_size:]
        return train, test
