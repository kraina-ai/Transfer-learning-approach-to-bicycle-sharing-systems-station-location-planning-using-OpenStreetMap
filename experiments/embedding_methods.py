from pymongo import ASCENDING, GEOSPHERE, MongoClient
# import pandas as pd
# from alive_progress import alive_bar
from shapely.geometry import shape, Polygon, Point, LineString, MultiPolygon
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
import requests

from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# from autoencoder.autoencoder_runner import get_model

from osm_tags import filtered_osm_tags_dict

CACHE_COLL_NAME = 'hexEmbeddingCache'


class BaseEmbedding(ABC):
    def __init__(self):
        client = MongoClient('mongodb://localhost:27017/')
        client_2 = MongoClient('mongodb://localhost:27017/')
        db = client.osmDataDB
        db_2 = client_2.osmDataDB
        self.coll_cities = db.cities
        self.coll_relations_filtered = db.relationsFiltered
        self.coll_relations = db.relations
        self.coll_embedding_cache = db[f'hexEmbeddingCache{self.__class__.__name__}']
        self.coll_embedding_cache_2 = db_2[f'hexEmbeddingCache{self.__class__.__name__}']
        # self.coll_embedding_cache.create_index([("hex_id", ASCENDING), ("embedding_cls", ASCENDING)])
        self.coll_embedding_cache.create_index([("hex_id", ASCENDING)])
        self.coll_embedding_cache_2.create_index([("hex_id", ASCENDING)])

    def _get_vector_from_cache(self, hex, cls_name):
        vector = self.coll_embedding_cache_2.find_one({"hex_id": hex["hex_id"]})
        # if vector is None:
        #     vector = self.coll_embedding_cache.find_one({ "hex_id": hex["hex_id"] })
        #     if vector is not None:
        #         del vector['_id']
        #         self.coll_embedding_cache_2.insert_one(vector)
        if vector is not None:
            return np.array(vector['vector'], dtype='float64')

    def _get_multiple_vectors_from_cache(self, hexes, cls_name):
        vectors = self.coll_embedding_cache_2.find({
            "hex_id": {
                "$in": [h["hex_id"] for h in hexes]
            },
            # "embedding_cls": cls_name
        })
        return [r for r in vectors]

    def _save_vector_to_cache(self, hex, cls_name, vector):
        record = {
            "hex_id": hex["hex_id"],
            # "embedding_cls": cls_name,
            "vector": [float(v) for v in list(vector)]
        }
        v = self._get_vector_from_cache(hex, cls_name)
        if v is None:
            # print(self.__class__.__name__, f'Saving vector')
            self.coll_embedding_cache.insert_one(record)
            self.coll_embedding_cache_2.insert_one(record)

    @abstractmethod
    def _transform(self, hex):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, relations_filter, filtered=True):
        raise NotImplementedError()

    def transform(self, hex):
        # print(self.__class__.__name__, 'transform')
        vector = self._get_vector_from_cache(hex, self.__class__.__name__)
        if vector is None:
            vector = self._transform(hex)
            self._save_vector_to_cache(hex, self.__class__.__name__, vector)
        return np.array(vector).reshape((-1,))

    def transform_multiple(self, hexes):
        # print(self.__class__.__name__, 'transform_multiple')
        vector_records = self._get_multiple_vectors_from_cache(hexes, self.__class__.__name__)
        vectors = []
        for h in hexes:
            existing_vector = [np.array(r['vector'], dtype='float64') for r in vector_records if r['hex_id'] == h['hex_id']]
            if len(existing_vector) > 0:
                vectors.append(np.array(existing_vector[0]))
            # existing_record = self.coll_embedding_cache.find_one({
            #     "hex_id": h["hex_id"],
            #     # "embedding_cls": self.__class__.__name__
            # })
            # if existing_record is not None:
            #     vectors.append(np.array(existing_record['vector']))
            else:
                vectors.append(self.transform(h))
                # v = np.array(self._transform(h)).reshape((-1,))
                # self._save_vector_to_cache(h, self.__class__.__name__, v)
                # vectors.append(v)
        return vectors

class BaseCountCategoryEmbedding(BaseEmbedding):
    def fit(self, relations_filter, filtered=True):
        # print('BaseCountCategoryEmbedding', filtered)
        self.filtered = filtered
        categories = self.coll_relations.find({}, {'_id':0, 'category': 1}).distinct('category')
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(categories)
       

    def _transform(self, hex):
        if self.filtered:
            # print('BaseCountCategoryEmbedding', 'filtered')
            relations = self.coll_relations_filtered.find({
                "geometry": {
                    "$geoIntersects": {
                        "$geometry": hex['geometry']
                    }
                }
            },
            {
                '_id': 0,
                'category': 1
            })
        else:
            # print('BaseCountCategoryEmbedding', 'nonfiltered')
            relations = self.coll_relations.find({
                "geometry": {
                    "$geoIntersects": {
                        "$geometry": hex['geometry']
                    }
                }
            },
            {
                '_id': 0,
                'category': 1
            })
        categories = [r['category'] for r in relations]
        # print('BaseCountCategoryEmbedding', len(categories))
        if len(categories) > 0:
            functions = ' '.join(categories)
            vector = self.vectorizer.transform([functions]).toarray().reshape((-1,))
        else:
            print(f'No relations for hex {hex["hex_id"]}')
            vector = np.zeros(shape=(len(self.vectorizer.get_feature_names()),))
        return np.array(vector).reshape((-1,))

class BaseShapeAnalyzerEmbedding(BaseEmbedding):
    area_only_categories = ['roads_walk', 'roads_drive', 'roads_bike', 'water']
    def fit(self, relations_filter, filtered=True):
        self.filtered = filtered
        self.categories = self.coll_relations.find({}, {'_id':0, 'category': 1}).distinct('category')
        for cat in self.area_only_categories:
            self.categories.remove(cat)

    def _intersect(self, pol1, pol2):
        try:
            return pol1.intersection(pol2)
        except:
            print('Invalid geometry, using buffer fix')
            return pol1.buffer(0).intersection(pol2.buffer(0))

    def _transform(self, hex):
        if self.filtered:
            relations_cursor = self.coll_relations_filtered.find({
                "geometry": {
                    "$geoIntersects": {
                        "$geometry": hex['geometry']
                    }
                }
            },
            {
                '_id': 0,
                'geometry': 1,
                'category': 1
            })
        else:
            relations_cursor = self.coll_relations.find({
                "geometry": {
                    "$geoIntersects": {
                        "$geometry": hex['geometry']
                    }
                }
            },
            {
                '_id': 0,
                'geometry': 1,
                'category': 1
            })
        relations = []
        for r in relations_cursor:
            r['shape'] = shape(r['geometry'])
            relations.append(r)
        roads_bike = [r for r in relations if r['category'] == 'roads_bike']
        roads_drive =  [r for r in relations if r['category'] == 'roads_drive']
        roads_walk = [r for r in relations if r['category'] == 'roads_walk']
        water = [r for r in relations if r['category'] == 'water']
        vector = []
        # cat_tmp = []
        hex_polygon = shape(hex['geometry'])
        hex_area = hex_polygon.area
        for category in self.categories:
            filtered_relations = [r for r in relations if r['category'] == category]
            # count points
            points = [r for r in filtered_relations if type(r['shape']) == Point]
            vector.append(float(len(points)))
            # cat_tmp.append(f"{category} points")
            # sum area of polygons
            polygons = [r['shape'] for r in filtered_relations if type(r['shape']) == Polygon or type(r['shape']) == MultiPolygon]
            areas = [self._intersect(hex_polygon, polygon).area for polygon in polygons]
            vector.append(float(sum(areas)) / hex_area)
            # cat_tmp.append(f"{category} areas")
        
        vector.append(float(sum([self._intersect(hex_polygon, w['shape']).area for w in water if type(w['shape']) == Polygon or type(w['shape']) == MultiPolygon])) / hex_area)
        # cat_tmp.append(f"water area")
        for roads_coll in [roads_bike, roads_drive, roads_walk]:
            vector.append(float(sum([self._intersect(road['shape'], hex_polygon).length for road in roads_coll if type(road['shape']) == LineString])))
            # cat_tmp.append(f"road length")
        # print(relations)
        # print(vector)
        # print(cat_tmp)
        return np.array(vector).reshape((-1,))

# - Prevoius + tfidf per category
#   - sustenance: amenity tfidf
#   - education: amenity tfidf
#   - transportation: amenity or public_transport
#   - finance: amenity tfidf
#   - healthcare: amenity tfidf
#   - culture_art_entertainment: amenity
#   - other: amenity tfidf (place_of_worship + religion)
#   - historic: historic tfidf
#   - leisure: leisure tfidf
#   - shops: shop tfidf
#   - sport: sport tfidf
#   - tourism: tourism tfidf

class PerCategoryFilteredShapeAnalyzerEmbedding(BaseEmbedding):
    area_only_categories = ['roads_walk', 'roads_drive', 'roads_bike', 'water']
    categories_dict = filtered_osm_tags_dict

    def fit(self, relations_filter, filtered=True):
        self.filtered = filtered

    def _intersect(self, pol1, pol2):
        try:
            return pol1.intersection(pol2)
        except:
            print('Invalid geometry, using buffer fix')
            return pol1.buffer(0).intersection(pol2.buffer(0))

    def _inner_transform(self, hex):
        if self.filtered:
            relations_cursor = self.coll_relations_filtered.find({
                "geometry": {
                    "$geoIntersects": {
                        "$geometry": hex['geometry']
                    }
                }
            },
            {
                '_id': 0,
                'geometry': 1,
                'category': 1,
                'amenity': 1,
                'aerialway': 1,
                'aeroway': 1,
                'public_transport': 1,
                'historic': 1,
                'leisure': 1,
                'shop': 1,
                'sport': 1,
                'tourism': 1
            })
        else:
            relations_cursor = self.coll_relations.find({
                "geometry": {
                    "$geoIntersects": {
                        "$geometry": hex['geometry']
                    }
                }
            },
            {
                '_id': 0,
                'geometry': 1,
                'category': 1,
                'amenity': 1,
                'aerialway': 1,
                'aeroway': 1,
                'public_transport': 1,
                'historic': 1,
                'leisure': 1,
                'shop': 1,
                'sport': 1,
                'tourism': 1
            })
        relations = []
        for r in relations_cursor:
            r['shape'] = shape(r['geometry'])
            relations.append(r)
        roads_bike = [r for r in relations if r['category'] == 'roads_bike']
        roads_drive =  [r for r in relations if r['category'] == 'roads_drive']
        roads_walk = [r for r in relations if r['category'] == 'roads_walk']
        water = [r for r in relations if r['category'] == 'water']
        vector = []
        hex_polygon = shape(hex['geometry'])
        hex_area = hex_polygon.area
        for key_tuple, values in self.categories_dict.items():
            category, key = key_tuple
            for value in values:
                filtered_relations = [r for r in relations if r['category'] == category and key in r and value in r[key]]
                # count points
                points = [r for r in filtered_relations if type(r['shape']) == Point]
                vector.append(float(len(points)))
                # sum area of polygons
                polygons = [r['shape'] for r in filtered_relations if type(r['shape']) == Polygon or type(r['shape']) == MultiPolygon]
                areas = [self._intersect(hex_polygon, polygon).area for polygon in polygons]
                vector.append(float(sum(areas)) / hex_area)
        
        vector.append(float(sum([self._intersect(hex_polygon, w['shape']).area for w in water if type(w['shape']) == Polygon or type(w['shape']) == MultiPolygon])) / hex_area)
        for roads_coll in [roads_bike, roads_drive, roads_walk]:
            vector.append(float(sum([self._intersect(road['shape'], hex_polygon).length for road in roads_coll if type(road['shape']) == LineString])))
        return np.array(vector).reshape((-1,))

    def _transform(self, hex):
        return self._inner_transform(hex)

class PerCategoryShapeAnalyzerEmbedding(BaseEmbedding):
    area_only_categories = ['roads_walk', 'roads_drive', 'roads_bike', 'water']
    per_category_tags = {
        'aerialway': ['aerialway'],
        'airports': ['aeroway'],
        'sustenance': ['amenity'],
        'education': ['amenity'],
        'transportation': ['amenity', 'public_transport'],
        'finance': ['amenity'],
        # 'finances': ['amenity'],
        'healthcare': ['amenity'],
        'culture_art_entertainment': ['amenity'],
        'buildings': [],
        'emergency': [],
        'other': ['amenity'],
        'historic': ['historic'],
        'leisure': ['leisure', 'amenity'],
        'shops': ['shop'],
        'sport': ['sport'],
        'tourism': ['tourism']
    }

    def _split_values(self, values_list):
        values_set = set()
        for value in values_list:
            for pre_word in value.split(';'):
                for word in pre_word.split(','):
                    word = word.strip()
                    if len(word) > 0:
                        if word[0] == '_':
                            word = word[1:]
                        values_set.add(word)
        return list(values_set)

    def fit(self, relations_filter, filtered=True):
        self.filtered = filtered
        self.categories_dict = {}
        for cat, v_list in self.per_category_tags.items():
            if len(v_list) == 0:
                self.categories_dict[(cat, 'category')] = self._split_values(self.coll_relations.find({ 'category': cat }, {'_id':0, 'category': 1}).distinct('category'))
            else:
                for v in v_list:
                    self.categories_dict[(cat, v)] = self._split_values(self.coll_relations.find({ 'category': cat }, {'_id':0, v: 1}).distinct(v))
        # print(self.categories_dict)

    def _intersect(self, pol1, pol2):
        try:
            return pol1.intersection(pol2)
        except:
            print('Invalid geometry, using buffer fix')
            return pol1.buffer(0).intersection(pol2.buffer(0))

    def _inner_transform(self, hex):
        if self.filtered:
            relations_cursor = self.coll_relations_filtered.find({
                "geometry": {
                    "$geoIntersects": {
                        "$geometry": hex['geometry']
                    }
                }
            },
            {
                '_id': 0,
                'geometry': 1,
                'category': 1,
                'amenity': 1,
                'aerialway': 1,
                'aeroway': 1,
                'public_transport': 1,
                'historic': 1,
                'leisure': 1,
                'shop': 1,
                'sport': 1,
                'tourism': 1
            })
        else:
            relations_cursor = self.coll_relations.find({
                "geometry": {
                    "$geoIntersects": {
                        "$geometry": hex['geometry']
                    }
                }
            },
            {
                '_id': 0,
                'geometry': 1,
                'category': 1,
                'amenity': 1,
                'aerialway': 1,
                'aeroway': 1,
                'public_transport': 1,
                'historic': 1,
                'leisure': 1,
                'shop': 1,
                'sport': 1,
                'tourism': 1
            })
        relations = []
        for r in relations_cursor:
            r['shape'] = shape(r['geometry'])
            relations.append(r)
        roads_bike = [r for r in relations if r['category'] == 'roads_bike']
        roads_drive =  [r for r in relations if r['category'] == 'roads_drive']
        roads_walk = [r for r in relations if r['category'] == 'roads_walk']
        water = [r for r in relations if r['category'] == 'water']
        vector = []
        # cat_tmp = []
        hex_polygon = shape(hex['geometry'])
        hex_area = hex_polygon.area
        for key_tuple, values in self.categories_dict.items():
            category, key = key_tuple
            for value in values:
                # print(category, key, value)
                filtered_relations = [r for r in relations if r['category'] == category and key in r and value in r[key]]
                # count points
                points = [r for r in filtered_relations if type(r['shape']) == Point]
                vector.append(float(len(points)))
                # sum area of polygons
                polygons = [r['shape'] for r in filtered_relations if type(r['shape']) == Polygon or type(r['shape']) == MultiPolygon]
                areas = [self._intersect(hex_polygon, polygon).area for polygon in polygons]
                vector.append(float(sum(areas)) / hex_area)
        
        vector.append(float(sum([self._intersect(hex_polygon, w['shape']).area for w in water if type(w['shape']) == Polygon or type(w['shape']) == MultiPolygon])) / hex_area)
        # cat_tmp.append(f"water area")
        for roads_coll in [roads_bike, roads_drive, roads_walk]:
            vector.append(float(sum([self._intersect(road['shape'], hex_polygon).length for road in roads_coll if type(road['shape']) == LineString])))
            # cat_tmp.append(f"road length")
        # print(relations)
        # print(vector)
        # print(cat_tmp)
        # print(len(vector))
        # raise ArithmeticError()
        return np.array(vector).reshape((-1,))

    def _transform(self, hex):
        return self._inner_transform(hex)

class PerCategoryShapeAnalyzerEmbedding300(PerCategoryShapeAnalyzerEmbedding):
    def _transform(self, hex):
        resolution = h3.h3_get_resolution(hex['hex_id'])
        raw_vector = self._inner_transform(hex)
        payload = {
            'filtered': False,
            'resolution': resolution,
            'size': 300,
            'vector': list(raw_vector)
        }
        try:
            r = requests.post('http://127.0.0.1:8000/predict', json=payload)
            embedding_vector = np.array(r.json()).reshape((-1, ))
        except Exception as ex:
            print(ex)
            print(payload)
            raise
        return embedding_vector

class PerCategoryFilteredShapeAnalyzerEmbeddingN(PerCategoryFilteredShapeAnalyzerEmbedding):
    size = 0
    def _transform(self, hex):
        resolution = h3.h3_get_resolution(hex['hex_id'])
        raw_vector = self._inner_transform(hex)
        payload = {
            'filtered': True,
            'resolution': resolution,
            'size': self.size,
            'vector': list(raw_vector)
        }
        try:
            r = requests.post('http://127.0.0.1:8000/predict', json=payload)
            embedding_vector = np.array(r.json()).reshape((-1, ))
        except Exception as ex:
            print(ex)
            print(payload)
            raise
        return embedding_vector

class PerCategoryFilteredShapeAnalyzerEmbedding20(PerCategoryFilteredShapeAnalyzerEmbeddingN):
    size = 20

class PerCategoryFilteredShapeAnalyzerEmbedding32(PerCategoryFilteredShapeAnalyzerEmbeddingN):
    size = 32

class PerCategoryFilteredShapeAnalyzerEmbedding50(PerCategoryFilteredShapeAnalyzerEmbeddingN):
    size = 50

class PerCategoryFilteredShapeAnalyzerEmbedding64(PerCategoryFilteredShapeAnalyzerEmbeddingN):
    size = 64

class PerCategoryFilteredShapeAnalyzerEmbedding100(PerCategoryFilteredShapeAnalyzerEmbeddingN):
    size = 100

class PerCategoryFilteredShapeAnalyzerEmbedding128(PerCategoryFilteredShapeAnalyzerEmbeddingN):
    size = 128

class PerCategoryFilteredShapeAnalyzerEmbedding200(PerCategoryFilteredShapeAnalyzerEmbeddingN):
    size = 200

class PerCategoryFilteredShapeAnalyzerEmbedding256(PerCategoryFilteredShapeAnalyzerEmbeddingN):
    size = 256

class PerCategoryFilteredShapeAnalyzerEmbedding300(PerCategoryFilteredShapeAnalyzerEmbeddingN):
    size = 300

class PerCategoryFilteredShapeAnalyzerEmbedding500(PerCategoryFilteredShapeAnalyzerEmbeddingN):
    size = 500


# class PerCategoryShapeAnalyzerEmbedding300(PerCategoryShapeAnalyzerEmbedding):
#     def _transform(self, hex):
#         resolution = h3.h3_get_resolution(hex['hex_id'])
#         model = get_model(resolution, PerCategoryShapeAnalyzerEmbedding, input_dim=5702, latent_dim=300)
#         raw_vector = self._inner_transform(hex)
#         return model.call(raw_vector.reshape(1,-1)).reshape(-1)