# TEMPERATURE TIME SERIES ML PROJECT

FINAL SCHOOL PROJECT | TIME SERIES FORECASTING USING PYTHON, DARTS, MLFLOW AND SPHINX.


# DATASET: WEATHER | KAGGLE : LONG TERM TIME SERIES FORECASTING

The WEATHER dataset comes from Kaggle | Alistair King : Long Term Time Series Forecasting.
Collected at the Max Planck Institute weather station and covers the entire year 2020.
Measurements are recorded every 10 minutes over 52,000 rows per feature.

DOWNLOAD WEATHER.CSV FROM KAGGLE AND PLACE IT IN data/raw/ BEFORE RUNNING.


# MAIN FEATURES

We are using temperature T, pressure p, relative humidity rh as the main features for forecasting.


# MAIN COLUMNS

- DATE : timestamp of observation | time index
- T : air temperature in °C | target
- p : atmospheric pressure in millibars mbar
- rh : relative humidity in %

Other columns are available in the raw dataset but not all are used in this project ;
- potential temperature
- dew point
- vapor pressure metrics
- wind speed
- direction rainfall
- solar radiation


# PROJECT FEATURES

- Python 3.11
- Environment : final_env
- Libraries : requirements.txt

- Modular structure : src folder with transform.py and model.py
- Forecasting with Darts | 2 models predicting 300 future time steps
- Plots and metrics saved in report.py

- Experiment tracking with MLflow : parameters and errors
- Documentation with Sphinx
- Version control and collaboration : GitHub


