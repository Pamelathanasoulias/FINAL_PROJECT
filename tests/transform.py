import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.input import YAMLLoader, CSVLoader
from src.transform import WeatherTransform

# LOAD PARAMS
params = YAMLLoader().load_file("params.yaml")

# LOAD RAW DATA
df = CSVLoader().load_file(params["data"]["raw_data"])

# RUN TRANSFORM
transform = WeatherTransform(df, params)
transform.the_columns()
transform.save_clean_file()
transform.build_series()
train, test = transform.train_test_split()

print("CLEAN DATA SHAPE:", transform.df.shape)
print("TRAIN LENGTH:", len(train))
print("TEST LENGTH:", len(test))
print("TARGET SERIES:", transform.y_series)
print("COVARIATES SERIES:", transform.covariates_series)

