import pandas as pd
from typing import Dict, Any, Tuple
from darts import TimeSeries
from darts.utils.model_selection import train_test_split


# SAVE CONFIG + DATAFRAME
class DataTransformation:
    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]):
        self.config = config
        self.df = df.copy()

        # PLACEHOLDERS
        self.ts_target: TimeSeries | None = None
        self.ts_features: TimeSeries | None = None

    # SELECT ONLY TIME,  TARGET & FEATURES
    def select_columns(self) -> pd.DataFrame:
        needed = [self.config["the_time"], self.config["the_target"]] + self.config["the_features"]
        self.df = self.df[needed].copy()
        return self.df

    # FORMAT TIME COLUMN & SORT 
    def format_time(self) -> pd.DataFrame:
        time_col = self.config["the_time"]
        self.df[time_col] = pd.to_datetime(self.df[time_col])
        self.df = self.df.sort_values(time_col).reset_index(drop=True)
        return self.df

    # CONVERT TO DARTS TIMESERIES
    def build_series(self) -> Tuple[TimeSeries, TimeSeries | None]:
        time_col = self.config["the_time"]
        target_col = self.config["the_target"]
        feature_cols = [c for c in self.config["the_features"] if c != target_col]

        self.ts_target = TimeSeries.from_dataframe(self.df, time_col=time_col, value_cols=target_col)
        self.ts_features = (TimeSeries.from_dataframe(self.df, time_col=time_col, value_cols=feature_cols)
            if len(feature_cols) > 0
            else None)
        
        return self.ts_target, self.ts_features

    # TRAIN/TEST SPLIT
    def divide_series(self, test_size: int | None = None) -> Tuple[TimeSeries, TimeSeries]:
        if self.ts_target is None:
            raise ValueError("CALL build_series() FIRST")

        if test_size is None:
            test_size = int(self.config.get("test_size", 12))

        train, test = train_test_split(self.ts_target, test_size=test_size)
        return train, test
