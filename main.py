from src.utils.input import YAMLLoader, CSVLoader
from src.transform import WeatherTransform
from src.model import WeatherModels
from src.utils.report import the_report


def the_main():
    # CONFIG
    params = YAMLLoader().load_file("params.yaml")

    # DATA SOURCE | CLEAN DATASET ONLY (FULL PATH)
    data_path = "/Users/pam/Desktop/MYPROJECTS/FINAL_PROJECT/data/processed/CLEAN_WEATHER.csv"
    print(f"# DATA SOURCE | CLEAN -> {data_path}")

    # LOAD CLEAN DATA
    df = CSVLoader().load_file(data_path)

    # TRANSFORM | ENSURE COLUMNS OK -> SERIES -> SPLIT
    the_transform = WeatherTransform(df, params)
    the_transform.the_columns()          # CLEAN COLUMNS
    the_transform.build_series()
    the_train, the_test = the_transform.train_test_split()

    # MODELS | ARIMA & LINEAR | MLFLOW LOGGING
    the_models = WeatherModels(
        the_train,
        the_test,
        the_transform.covariates_series,
        params
    )
    model_dictionary = the_models.run_the_models()

    # REPORT | PLOTS + METRICS CSV
    df_metrics = the_report(the_train, the_test, model_dictionary, out_dir="docs")

    # SHOW METRICS
    print(df_metrics)


if __name__ == "__main__":
    the_main()
