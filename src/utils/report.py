# REPORT.PY | SAVE PLOTS + METRICS
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def the_report(the_train, the_test, model_dictionary, out_dir="docs"):
    Path(out_dir, "plots").mkdir(parents=True, exist_ok=True)

    rows = []
    for name, info in model_dictionary.items():
        predictions = info["pred"]
        metrics = info["metrics"]

        # FULL PLOT | TRAIN, TEST & PRED
        plt.figure(figsize=(12, 4))
        the_train.plot(label="TRAIN", lw=1.5, color="blue")
        the_test.plot(label="TEST", lw=2, color="black")
        predictions.plot(label="PRED", lw=2, color="magenta")
        plt.title(f"{name} FORECAST")
        plt.xlabel("")
        plt.legend()
        plt.tight_layout()

        plot_path = Path(out_dir, "plots", f"{name}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        # ZOOM PLOT | ONLY TEST & PREDICTIONS
        plt.figure(figsize=(8, 4))
        the_test.plot(label="TEST", lw=2, color="black")
        predictions.plot(label="PRED", lw=2, color="magenta")
        plt.title(f"{name} FORECAST")
        plt.xlabel("")
        plt.legend()
        plt.tight_layout()

        zoom_path = Path(out_dir, "plots", f"{name}_forecast.png")
        plt.savefig(zoom_path, dpi=150)
        plt.close()

        # METRICS ROW
        rows.append({"MODEL": name,"MAPE": metrics["mape"],"MAE": metrics["mae"],"MSE": metrics["mse"],})

    # METRICS TABLE
    df = pd.DataFrame(rows).set_index("MODEL")
    metrics_path = Path(out_dir, "metrics.csv")
    df.to_csv(metrics_path, float_format="%.6f")

    return df
