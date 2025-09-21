# TEST MODEL.PY

from src.utils.input import YAMLLoader, CSVLoader
from src.transform import DataTransformation
from src.model import DartsTrainer

if __name__ == "__main__":
    cfg = YAMLLoader().load_file("params.yaml")
    df  = CSVLoader().load_file(cfg["data"]["raw_data"])

    tfm = DataTransformation(df)
    df  = tfm.to_datetime(cfg["time"]["datetime_col"])
    df  = tfm.select_features([cfg["time"]["datetime_col"],
                               cfg["features"]["target"],
                               *cfg["features"]["exogenous"]])
    df  = tfm.fill_na()

    results = DartsTrainer(df, cfg).run_all()
    print("MODEL RESULTS:", results)
