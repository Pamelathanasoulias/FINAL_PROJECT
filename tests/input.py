# TEST INPUT.PY

from src.utils.input import YAMLLoader, CSVLoader

if __name__ == "__main__":
    cfg = YAMLLoader().load_file("params.yaml")
    print("YAML KEYS:", list(cfg.keys()))

    df = CSVLoader().load_file(cfg["data"]["raw_data"])
    print("CSV SHAPE:", df.shape)
    print(df.head(2))
