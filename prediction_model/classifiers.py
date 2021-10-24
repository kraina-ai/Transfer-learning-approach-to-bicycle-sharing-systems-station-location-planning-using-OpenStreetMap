from typing import List

import numpy as np
from db_connector.models import Area, TrainLabel
from peewee import fn
from tqdm import tqdm

from .hex_cache import load_hexes_to_cache

from .model import DefaultModel
from .model_parameters import IMBALANCE_RATIO


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
