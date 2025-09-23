# FILE LOADERS YAML & CSV

from pathlib import Path
from typing import Dict, Union
import pandas as pd
from yaml import safe_load
from src.utils.logger import logger


class CSVLoader:

    def load_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load a CSV file and return its contents as a DataFrame.

        Args:
            file_path : Path to the CSV file.

        Returns:
            pd.DataFrame : Loaded CSV content.

        Raises:
            FileNotFoundError : If the file does not exist.
        """
        file_path = Path(file_path)
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError as e:
            logger.error(f"{e}: {file_path}")


class YAMLLoader:

    def load_file(self, file_path: Union[str, Path]) -> Dict:
        """
        Load a YAML file and return its contents as a dictionary.

        Args:
            file_path : Path to the YAML file.

        Returns:
            dict : Loaded YAML content.

        Raises:
            FileNotFoundError : If the file does not exist.
        """
        file_path = Path(file_path)
        try:
            with open(file_path, "r") as yaml_file:
                return safe_load(yaml_file)
        except FileNotFoundError as e:
            logger.error(f"{e}: {file_path}")
