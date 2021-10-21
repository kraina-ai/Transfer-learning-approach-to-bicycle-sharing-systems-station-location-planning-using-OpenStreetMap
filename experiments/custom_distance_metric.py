from pymongo import ASCENDING, GEOSPHERE, MongoClient
# import pandas as pd
# from alive_progress import alive_bar
# from shapely.geometry import Point, mapping
# from keplergl import KeplerGl
# import shapely
# import json
# from os import listdir
# from os.path import isfile, join
# from tqdm import tqdm
# import geopandas as gpd
from h3 import h3
# import math
# import sklearn
import numpy as np

class DistanceMetric():
    def __init__(self):
        client = MongoClient('mongodb://localhost:27017/')
        db = client.osmDataDB
        self.coll_hexes = db.hexesInCitiesFiltered

    def fit(self, filt):
        filt['has_station'] = True
        self.station_hexes = [r['hex_id'] for r in self.coll_hexes.find(filt)]

    def _find_nearest_distance(self, hex_id):
        found_distance = 1000
        for h in self.station_hexes:
            dist = h3.h3_distance(hex_id, h)
            if dist < found_distance:
                found_distance = dist
            if dist == 1:
                break
        return found_distance

    def calculate(self, y_true, y_pred, hex_list):
        values = []
        for y_t, y_p, h_id in zip(y_true, y_pred, hex_list):
            if y_t == y_p:
                values.append(1.0) # TP and TN
            elif y_t == 1.0:
                values.append(0.0) # FN
            else:
                distance = self._find_nearest_distance(h_id)
                values.append(1 / (distance + 1)) # FP
        return np.average(values)

    def generate_y(self, hex_list):
        pass