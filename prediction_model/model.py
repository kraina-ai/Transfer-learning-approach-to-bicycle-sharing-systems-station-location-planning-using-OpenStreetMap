# load data

import os
from abc import ABC, abstractmethod
from random import shuffle
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from db_connector.db_definition import db as sqlite_db
from db_connector.models import Classifier, Vector
from db_connector.models.prediction_result import PredictionResult
from tqdm import tqdm

from .hex_cache import (get_area_embedding_data, get_area_hex_id_data,
                        get_hex_label, load_hexes_to_cache,
                        save_area_embedding_data, save_area_hex_id_data)
from .model_parameters import MODELS_CACHE_DIRECTORY
from .neighbourhood_embedding import generate_neighbourhood_vector
from .normalization import normalize_data


class DefaultModel(ABC):
    show_neighbourhood_generation = False
    save_to_db = False
    save_to_csv = True

    @abstractmethod
    def get_classifier_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def _get_training_data_hex_ids(self) -> List[str]:
        raise NotImplementedError()

    def _get_prediction_data(self, area_id: int) -> np.ndarray:
        hex_ids = get_area_hex_id_data(area_id)
        if hex_ids is None:
            hex_ids = [v.hex_id for v in tqdm(Vector.select().where(Vector.area == area_id).iterator(), desc = f'Loading prediction hexes', disable = False)]
            save_area_hex_id_data(area_id, hex_ids)
        x_data_normalized = get_area_embedding_data(area_id)
        if x_data_normalized is None:
            # print("Prediction data", not self.show_neighbourhood_generation)
            x_data = generate_neighbourhood_vector(hex_ids, "Prediction data", not self.show_neighbourhood_generation)
            x_data_normalized = normalize_data(x_data)
            save_area_embedding_data(area_id, x_data_normalized)
        return hex_ids, x_data_normalized

    def _save_data_to_db(self, area_id: int, prediction_hex_ids: List[str], result: np.ndarray) -> None:
        fields = [PredictionResult.area, PredictionResult.classifier, PredictionResult.hex_id, PredictionResult.value]
        classifier_name, _created = Classifier.get_or_create(name = self.get_classifier_name())
        data = []
        for idx, hex_id in enumerate(prediction_hex_ids):
            data.append((area_id, classifier_name, hex_id, result[idx]))
        with sqlite_db.atomic():
            for idx in tqdm(range(0, len(data), 1000), desc=f'Saving results to DB [{self.get_classifier_name()}] (area_id: {area_id})'):
                PredictionResult.insert_many(data[idx:idx+1000], fields=fields).execute()

    def _save_data_to_csv(self, area_id: int, prediction_hex_ids: List[str], result: np.ndarray) -> None:
        classifier, _created = Classifier.get_or_create(name = self.get_classifier_name())
        dicts = []
        for idx, hex_id in enumerate(prediction_hex_ids):
            dicts.append({
                'hex_id': hex_id,
                'value': round(result[idx], 5)
            })
        if len(dicts) > 0:
            file_name = f'csv_results/{area_id}_{classifier.id}.csv'
            map_df = pd.DataFrame(dicts)
            map_df.to_csv(file_name, index=False)

    def _get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        train_hex_ids = self._get_training_data_hex_ids()
        shuffle(train_hex_ids)
        X_train = generate_neighbourhood_vector(train_hex_ids, 'Training data', True)
        X_train_normalized = normalize_data(X_train)
        Y_train = np.array([int(get_hex_label(hex_id)) for hex_id in train_hex_ids])
        return X_train_normalized, Y_train
    
    def predict_and_save_data(self, area_id: int, iterations: int = 100) -> None:
        load_hexes_to_cache(area_id)
        prediction_hex_ids, predition_data = self._get_prediction_data(area_id)
        result = np.zeros((len(prediction_hex_ids), ));
        for i in tqdm(range(iterations), desc=f'Generating predictions [{self.get_classifier_name()}] (area_id: {area_id})'):
            clf = xgb.XGBRFClassifier(use_label_encoder = False, eval_metric = 'logloss')
            model_filename = f'{MODELS_CACHE_DIRECTORY}/{self.get_classifier_name()}_{i}.json'
            if os.path.isfile(model_filename):
                clf.load_model(model_filename)
            else:
                if not os.path.exists(MODELS_CACHE_DIRECTORY):
                    os.makedirs(MODELS_CACHE_DIRECTORY)
                X_train, Y_train = self._get_train_data()
                clf.fit(X_train, Y_train)
                clf.save_model(model_filename)
            Y_pred = clf.predict_proba(predition_data)
            exists_index = list(clf.classes_).index(1)
            result += Y_pred[:,exists_index]
        result /= iterations
        if self.save_to_db:
            self._save_data_to_db(area_id, prediction_hex_ids, result)
        if self.save_to_csv:
            self._save_data_to_csv(area_id, prediction_hex_ids, result)
        
