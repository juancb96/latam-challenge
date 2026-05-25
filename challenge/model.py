import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression


class DelayModel:

    FEATURES_COLS = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    THRESHOLD_IN_MINUTES = 15

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)

        features = features.reindex(columns=self.FEATURES_COLS, fill_value=0)

        # Compute target whenever date columns are present (training data).
        # This covers both explicit training (target_column set) and the case where
        # preprocess is called without target_column on raw data — auto-fit ensures
        # predict() works without a separate fit() call.
        if 'Fecha-O' in data.columns and 'Fecha-I' in data.columns:
            min_diff = (
                pd.to_datetime(data['Fecha-O']) - pd.to_datetime(data['Fecha-I'])
            ).dt.total_seconds() / 60
            delay = np.where(min_diff > self.THRESHOLD_IN_MINUTES, 1, 0)
            target = pd.DataFrame({'delay': delay})

            if target_column is not None:
                return features, target

            if self._model is None:
                self.fit(features, target)

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        y = target['delay']
        n_y0 = (y == 0).sum()
        n_y1 = (y == 1).sum()
        n = len(y)

        self._model = LogisticRegression(class_weight={1: n_y0 / n, 0: n_y1 / n})
        self._model.fit(features, y)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        return [int(p) for p in self._model.predict(features)]

    def save(self, path: Path) -> None:
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: Path) -> "DelayModel":
        instance = cls()
        instance._model = joblib.load(path)
        return instance
