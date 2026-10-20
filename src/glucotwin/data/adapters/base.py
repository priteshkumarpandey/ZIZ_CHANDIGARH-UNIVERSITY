from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseDatasetAdapter(ABC):
    @property
    @abstractmethod
    def dataset_name(self) -> str:
        pass

    @abstractmethod
    def load_patients(self, raw_dir: Path) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_timeseries(self, raw_dir: Path, patient_ids: list[str] | None = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_meals(self, raw_dir: Path, patient_ids: list[str] | None = None) -> pd.DataFrame:
        pass
