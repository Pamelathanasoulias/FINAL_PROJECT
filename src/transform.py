import pandas as pd

class DataTransformation:
    # SIMPLE TRANSFORM CLASS FOR THIS PROJECT

    def __init__(self, df: pd.DataFrame):
        # SAVE ORIGINAL DATAFRAME
        self.df = df.copy()

    def to_datetime(self, col: str = "date") -> pd.DataFrame:
        # CONVERT DATE COLUMN TO DATETIME
        self.df[col] = pd.to_datetime(self.df[col])
        return self.df

    def select_features(self, features: list) -> pd.DataFrame:
        # KEEP ONLY NEEDED COLUMNS (DATE, T, P, RH)
        self.df = self.df[features]
        return self.df
