import pandas as pd
from typing import Dict, Any, Tuple
from darts import TimeSeries
from darts.utils.model_selection import train_test_split


# TRANSFORM
class DataTransformation:
    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]):
        self.config = config
        self.df = df.copy()

        # PLACEHOLDERS FOR DARTS SERIES
        self.series_target: TimeSeries | None = None
        self.series_features: TimeSeries | None = None

    # KEEP  TIME, TARGET & FEATURES | DATETIME + SORT
    def the_columns(self) -> pd.DataFrame:
        t = self.config["the_time"]
        y = self.config["the_target"]
        
        all_features = list(self.config["the_features"])


        the_columns = [t, y] + all_features
       
        self.df = self.df[the_columns].copy()
        self.df[t] = pd.to_datetime(self.df[t])
        self.df = self.df.sort_values(t).reset_index(drop=True)
       
        return self.df


    # BUILD SERIES FOR DARTS
    def build_series(self) -> Tuple[TimeSeries, TimeSeries | None]:
        t = self.config["the_time"]
        y = self.config["the_target"]
        series_features_only = [c for c in self.config["the_features"] if c != y]

        self.series_target = TimeSeries.from_dataframe(self.df, time_col=t, value_cols=y)
        self.series_features = (TimeSeries.from_dataframe(self.df, time_col=t, value_cols=series_features_only)
            if series_features_only else None)
        
        return self.series_target, self.series_features


    # TRAIN / TEST SPLIT
    def divide_series(self, test_size: int | None = None) -> Tuple[TimeSeries, TimeSeries]:
        if self.series_target is None:
            raise ValueError("ERROR: CALL build_series() FIRST")
       
        if test_size is None:
            test_size = int(self.config.get("test_size", 12))
        train, test = train_test_split(self.series_target, test_size=test_size)
        
        return train, test

    # SAVE CLEAN DATAFRAME TO CSV IN data/processed/
    def cleaned_data(self, path: str = "data/processed/CLEAN_WEATHER.csv") -> None:
        self.df.to_csv(path, index=False)
