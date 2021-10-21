import re
from pymongo import ASCENDING, GEOSPHERE, MongoClient
# import pandas as pd
# from alive_progress import alive_bar
# from shapely.geometry import Point, mapping
# from keplergl import KeplerGl
import shapely
# import json
# from os import listdir
# from os.path import isfile, join
# from tqdm import tqdm
# import geopandas as gpd
from h3 import h3
# import math
# import sklearn
import numpy as np


from abc import ABC, abstractmethod
from embedding_methods import BaseEmbedding

class BaseNeighbourEmbedding(ABC):
    hex_exists_dict = {}
    def __init__(self, embedding_cls: BaseEmbedding):
        self.base_dim = 0
        client = MongoClient('mongodb://localhost:27017/')
        client_2 = MongoClient('mongodb://localhost:27017/')
        db = client.osmDataDB
        db_2 = client_2.osmDataDB
        # self.coll_hexes = db.hexesInCitiesFiltered
        self.coll_hexes = db_2.hexesInCitiesFiltered
        self.coll_neighbour_embedding_cache = db[f'hexNeighbourEmbeddingCache{embedding_cls.__class__.__name__}']
        self.coll_neighbour_embedding_cache_2 = db_2[f'hexNeighbourEmbeddingCache{embedding_cls.__class__.__name__}']
        self.coll_neighbour_embedding_cache.create_index([
            ("hex_id", ASCENDING),
            ("ring_no", ASCENDING)
        ])
        self.coll_neighbour_embedding_cache_2.create_index([
            ("hex_id", ASCENDING),
            ("ring_no", ASCENDING)
        ])
        self.coll_neighbour_embedding_cache_full = db[f'hexNeighbourEmbeddingCache{embedding_cls.__class__.__name__}Full']
        self.coll_neighbour_embedding_cache_full_2 = db_2[f'hexNeighbourEmbeddingCache{embedding_cls.__class__.__name__}Full']
        self.coll_neighbour_embedding_cache_full.create_index([
            ("hex_id", ASCENDING),
            ("neighbours", ASCENDING)
        ])
        self.coll_neighbour_embedding_cache_full_2.create_index([
            ("hex_id", ASCENDING),
            ("neighbours", ASCENDING)
        ])

    def _get_final_vector_from_cache(self, hex, neighbours):
        record = self.coll_neighbour_embedding_cache_full_2.find_one({ "hex_id": hex["hex_id"], "neighbours": neighbours })
        if record is not None:
            return np.array(record['vector'], dtype='float64')

    def _get_vector_from_cache(self, hex, embedding_cls_name, ring_no):
        record = self.coll_neighbour_embedding_cache_2.find_one({ "hex_id": hex["hex_id"], "ring_no": ring_no })
        # if record is None:
        #     record = self.coll_neighbour_embedding_cache.find_one({ "hex_id": hex["hex_id"], "ring_no": ring_no })
        #     if record is not None:
        #         del record['_id']
        #         self.coll_neighbour_embedding_cache_2.insert(record)
        if record is not None:
            return np.array(record['vector'], dtype='float64')

    def _save_final_vector_to_cache(self, hex, neighbours, vector):
        record = {
            "hex_id": hex["hex_id"],
            "neighbours": neighbours,
            "vector": list(vector)
        }
        v = self._get_final_vector_from_cache(hex, neighbours)
        if v is None:
            self.coll_neighbour_embedding_cache_full.insert_one(record)
            self.coll_neighbour_embedding_cache_full_2.insert_one(record)

    def _save_vector_to_cache(self, hex, embedding_cls_name, ring_no, vector):
        record = {
            "hex_id": hex["hex_id"],
            # "embedding_cls": embedding_cls_name,
            "ring_no": ring_no,
            "vector": list(vector)
        }
        v = self._get_vector_from_cache(hex, embedding_cls_name, ring_no)
        if v is None:
            self.coll_neighbour_embedding_cache.insert_one(record)
            self.coll_neighbour_embedding_cache_2.insert_one(record)

    # def _hex_from_cache(self, hex):
    #     if not hex in self.hex_exists_dict:
    #         self.hex_exists_dict[hex] = self.coll_hexes.find_one({ 'hex_id': hex })
    #     return self.hex_exists_dict[hex]
    
    def _average_neighbours_level(self, hex, k, embedding_cls: BaseEmbedding, check_if_exists=True):
        vector = self._get_vector_from_cache(hex, embedding_cls.__class__.__name__, k)
        if vector is None:
            indexes = h3.hex_ring(hex['hex_id'], k)
            if check_if_exists:
                hexes = [h for h in self.coll_hexes.find({ 'hex_id': { '$in': list(indexes) } })]
            else:
                hexes = []
                for h in indexes:
                    polygons = h3.h3_set_to_multi_polygon([h], geo_json=False)
                    outlines = [loop for polygon in polygons for loop in polygon]
                    polyline = [outline + [outline[0]] for outline in outlines][0]
                    polygon = shapely.geometry.Polygon(polyline)
                    hexes.append({'hex_id': h, 'geometry': shapely.geometry.mapping(polygon)})
            if len(hexes) > 0:
                # vectors = embedding_cls.transform_multiple(hexes)
                vectors = [embedding_cls.transform(h) for h in hexes]
                try:
                    vector = np.average(np.array(vectors), axis=0).reshape((-1,))
                except:
                    for v in vectors:
                        print(len(v))
                        print(np.array(v).shape)
                    raise
                # vector = np.average(np.array(vectors), axis=0).reshape((-1,))
            else:
                print('None neighbours!')
                vector = np.zeros(shape=(self.base_dim,), dtype='float64')
            self._save_vector_to_cache(hex, embedding_cls.__class__.__name__, k, vector)
        return vector

    @abstractmethod
    def _calculate_embedding(self, vectors):
        pass

    def get_embedding(self, hex, no_rings, embedding_cls: BaseEmbedding, check_if_exists=True):
        vector = self._get_final_vector_from_cache(hex, no_rings)
        if vector is None:
            base_vector = embedding_cls.transform(hex)
            vectors = [base_vector]
            self.base_dim = base_vector.shape[0]
            for k in range(1, no_rings+1):
                vectors.append(self._average_neighbours_level(hex, k, embedding_cls, check_if_exists))
            try:
                vector = self._calculate_embedding(np.array(vectors))
            except:
                for v in vectors:
                    print(len(v))
                    print(np.array(v).shape)
                raise
            # self._save_final_vector_to_cache(hex, no_rings, vector)
        return vector
        

class ConcatenateNeighbourEmbedding(BaseNeighbourEmbedding):
    def _calculate_embedding(self, vectors):
        vector = np.concatenate(vectors)
        return np.array(vector)

class AverageNeighbourEmbedding(BaseNeighbourEmbedding):
    def _calculate_embedding(self, vectors):
        return np.average(vectors, axis=0)

class AverageDiminishingNeighbourEmbedding(BaseNeighbourEmbedding):
    def _calculate_embedding(self, vectors):
        weights = [1 / (n + 1) for n in range(len(vectors))]
        return np.average(vectors, axis=0, weights=weights)

class AverageDiminishingSquqredNeighbourEmbedding(BaseNeighbourEmbedding):
    def _calculate_embedding(self, vectors):
        weights = [1 / ((n + 1)**2) for n in range(len(vectors))]
        return np.average(vectors, axis=0, weights=weights)