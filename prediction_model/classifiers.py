from random import shuffle
from typing import List, Tuple

import numpy as np
from db_connector.models import Area, TrainLabel
from peewee import fn
from tqdm import tqdm

from .hex_cache import get_hex_label, load_all_hexes_to_cache, load_hexes_to_cache
from .model import DefaultModel
from .model_parameters import IMBALANCE_RATIO
from .neighbourhood_embedding import generate_neighbourhood_vector
from .normalization import normalize_data


class OneCityClassifier(DefaultModel):
    def __init__(self, area_id: int) -> None:
        super().__init__()
        assert 1 <= area_id <= 34
        self.area_id = area_id
        self.city = Area[area_id]
        self.station_hexes = []
        load_hexes_to_cache(area_id)
    
    def get_classifier_name(self):
        return self.city.name

    def _load_station_hexes(self) -> List[str]:
        if len(self.station_hexes) == 0:
            self.station_hexes = [label_obj.hex_id for label_obj in tqdm(TrainLabel
                                .select()
                                .where(
                                    TrainLabel.area == self.area_id,
                                    TrainLabel.has_station == True
                                ), desc = f'Loading station hexes', disable = True)
                            ]

    def _get_training_data_hex_ids(self) -> List[str]:
        self._load_station_hexes()
        number_of_stations = len(self.station_hexes)
        non_station_hexes = [label_obj.hex_id for label_obj in tqdm(TrainLabel
                                .select()
                                .where(
                                    TrainLabel.area == self.area_id,
                                    TrainLabel.has_station == False
                                ).order_by(fn.Random())
                                .limit(int(number_of_stations * IMBALANCE_RATIO)), desc = f'Loading non-station hexes', disable = True)
                            ]
        return self.station_hexes + non_station_hexes


class AllCitiesClassifier(DefaultModel):
    def __init__(self) -> None:
        super().__init__()
        self.area_ids = list(range(1, 35))
        self.station_hexes = []
        load_all_hexes_to_cache()
        self.show_neighbourhood_generation = True

    def get_classifier_name(self):
        return 'All cities'

    def _get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x_arrays = []
        y_arrays = []
        for area_id in tqdm(self.area_ids, desc = f'Generating training data', position=0):
            train_hex_ids = self._get_training_data_hex_ids(area_id)
            shuffle(train_hex_ids)
            X_train = generate_neighbourhood_vector(train_hex_ids, 'Training data', True)
            X_train_normalized = normalize_data(X_train)
            x_arrays.append(X_train_normalized)
            Y_train = np.array([int(get_hex_label(hex_id)) for hex_id in train_hex_ids]).reshape((-1, 1))
            y_arrays.append(Y_train)
        x_train_all = np.vstack(x_arrays)
        y_train_all = np.vstack(y_arrays)
        return x_train_all, y_train_all

    def _load_station_hexes(self, area_id: int) -> List[str]:
        self.station_hexes = [label_obj.hex_id for label_obj in TrainLabel
                            .select()
                            .where(
                                TrainLabel.area == area_id,
                                TrainLabel.has_station == True
                            )
                        ]

    def _get_training_data_hex_ids(self, area_id: int) -> List[str]:
        self._load_station_hexes(area_id)
        number_of_stations = len(self.station_hexes)
        non_station_hexes = [label_obj.hex_id for label_obj in TrainLabel
                                .select()
                                .where(
                                    TrainLabel.area == area_id,
                                    TrainLabel.has_station == False
                                ).order_by(fn.Random())
                                .limit(int(number_of_stations * IMBALANCE_RATIO))
                            ]
        return self.station_hexes + non_station_hexes

class AllCitiesNormalizedGloballyClassifier(DefaultModel):
    def __init__(self) -> None:
        super().__init__()
        self.area_ids = list(range(1, 35))
        load_all_hexes_to_cache()
        self.show_neighbourhood_generation = True

    def get_classifier_name(self):
        return 'All cities normalized globally'

    def _load_station_hexes(self, area_id: int) -> List[str]:
        station_hexes = [label_obj.hex_id for label_obj in tqdm(TrainLabel
                            .select()
                            .where(
                                TrainLabel.area == area_id,
                                TrainLabel.has_station == True
                            ), desc = f'Loading station hexes', disable = True)
                        ]
        return station_hexes

    def _get_training_data_hex_ids(self) -> List[str]:
        hexes = []
        for area_id in self.area_ids:
            station_hexes = self._load_station_hexes(area_id)
            number_of_stations = len(station_hexes)
            non_station_hexes = [label_obj.hex_id for label_obj in tqdm(TrainLabel
                                    .select()
                                    .where(
                                        TrainLabel.area == area_id,
                                        TrainLabel.has_station == False
                                    ).order_by(fn.Random())
                                    .limit(int(number_of_stations * IMBALANCE_RATIO)), desc = f'Loading non-station hexes', disable = True)
                                ]
            hexes.extend(station_hexes)
            hexes.extend(non_station_hexes)
        return hexes

