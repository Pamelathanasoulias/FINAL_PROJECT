import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.utils.input import YAMLLoader, CSVLoader
from src.utils.report import the_report
from src.transform import WeatherTransform
from src.model import WeatherModels

# LOAD PARAMS
params = YAMLLoader().load_file("params.yaml")

# LOAD RAW
df = CSVLoader().load_file(params["data"]["raw_data"])

# TRANSFORM
the_transform = WeatherTransform(df, params)
the_transform.the_columns()
the_transform.save_clean_file()
the_transform.build_series()
the_train, the_test = the_transform.train_test_split()

# MODELS
the_models = WeatherModels(the_train, the_test, the_transform.covariates_series, params)
model_dictionary = the_models.run_the_models()

# REPORT
df_metrics = the_report(the_train, the_test, model_dictionary, out_dir="docs")
print(df_metrics)
