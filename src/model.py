from typing import Optional, Dict
import mlflow
from darts import TimeSeries
from darts.metrics import mape, mae, mse
from darts.models import ARIMA, LinearRegressionModel



class WeatherModels:
    def __init__(self, the_train: TimeSeries, the_test: TimeSeries, the_covariates: Optional[TimeSeries], params: Dict):

        # SERIES
        self.the_train = the_train
        self.the_test = the_test
        self.the_covariates = the_covariates

        # TEST LENGTH
        self.the_steps = len(the_test)

        # MLFLOW
        self.the_experiment = params.get("the_mlflow", "temperature_forecast")

    # TEMPLATE | TRAIN, PREDICT & LOG
    def model_template(self, the_name: str, the_model):

        mlflow.set_experiment(self.the_experiment)
        with mlflow.start_run(run_name=the_name):

            
            if isinstance(the_model, LinearRegressionModel) and self.the_covariates is not None:
                the_model.fit(self.the_train, past_covariates=self.the_covariates)
                the_pred = the_model.predict(self.the_steps, past_covariates=self.the_covariates)
            else:
                the_model.fit(self.the_train)
                the_pred = the_model.predict(self.the_steps)

            # THE METRICS
            the_mape = float(mape(self.the_test, the_pred))
            the_mae = float(mae(self.the_test, the_pred))
            the_mse = float(mse(self.the_test, the_pred))

            # LOG PARAMS & METRICS
            mlflow.log_param("model", the_name)
            mlflow.log_param("steps", self.the_steps)

            if isinstance(the_model, ARIMA):
                mlflow.log_param("p", getattr(the_model, "p", 2))
                mlflow.log_param("d", getattr(the_model, "d", 0))
                mlflow.log_param("q", getattr(the_model, "q", 2))

            if isinstance(the_model, LinearRegressionModel):
                mlflow.log_param("lags", 6)
                if self.the_covariates is not None:
                    mlflow.log_param("lags_past_covariates", 6)

            mlflow.log_metric("mape", the_mape)
            mlflow.log_metric("mae", the_mae)
            mlflow.log_metric("mse", the_mse)

        return {"pred": the_pred, "metrics": {"mape": the_mape, "mae": the_mae, "mse": the_mse}}


    # RUN THE MODELS
    def run_the_models(self):

        # ARIMA
        arima = ARIMA(p=2, d=0, q=2)
        the_arima = self.model_template("ARIMA", arima)

        # LINEAR
        linear = LinearRegressionModel(
            lags=6,
            lags_past_covariates=6 if self.the_covariates is not None else None)
        the_linear = self.model_template("LINEAR", linear)


        return {"ARIMA": the_arima, "LINEAR": the_linear}
