"""Reporting utilities: save forecast plots and a metrics table to docs/."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def the_report(the_train, the_test, model_dictionary, out_dir="docs"):
    """
    Create plots and a metrics CSV from model outputs.

    Parameters
    ----------
    the_train : darts.TimeSeries
        Training target series.
    the_test : darts.TimeSeries
        Test target series.
    model_dictionary : dict
        Output of WeatherModels.run_the_models().
    out_dir : str
        Base output folder (default: "docs").

    Returns
    -------
    pandas.DataFrame
        Metrics table indexed by MODEL.
    """
    Path(out_dir, "plots").mkdir(parents=True, exist_ok=True)

    rows = []
    for name, info in model_dictionary.items():
        predictions = info["pred"]
        metrics = info["metrics"]

        # FULL PLOT
        plt.figure(figsize=(12, 4))
        the_train.plot(label="TRAIN", lw=1.5, color="blue")
        the_test.plot(label="TEST", lw=2, color="black")
        predictions.plot(label="PRED", lw=2, color="magenta")
        plt.title(f"{name} FORECAST")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(out_dir, "plots", f"{name}.png"), dpi=150)
        plt.close()

        # ZOOM PLOT (TEST vs PRED)
        plt.figure(figsize=(8, 4))
        the_test.plot(label="TEST", lw=2, color="black")
        predictions.plot(label="PRED", lw=2, color="magenta")
        plt.title(f"{name} FORECAST")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(out_dir, "plots", f"{name}_forecast.png"), dpi=150)
        plt.close()

        rows.append({
            "MODEL": name,
            "MAPE": metrics["mape"],
            "MAE": metrics["mae"],
            "MSE": metrics["mse"],})

    df = pd.DataFrame(rows).set_index("MODEL")
    df.to_csv(Path(out_dir, "metrics.csv"), float_format="%.6f")
    return df
