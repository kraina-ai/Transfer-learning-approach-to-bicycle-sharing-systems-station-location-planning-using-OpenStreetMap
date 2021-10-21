from pymongo import ASCENDING, GEOSPHERE, MongoClient
import pandas as pd
from alive_progress import alive_bar
from shapely.geometry import Point, mapping
from keplergl import KeplerGl
import shapely
import json
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import geopandas as gpd
from h3 import h3
import math
import sklearn
import numpy as np
import time

client = MongoClient('mongodb://localhost:27017/')
db = client.osmDataDB
coll_results = db.experimentsResults
coll_results.create_index([
    ("city", ASCENDING),
    ("resolution", ASCENDING),
    ("inbalance_ratio", ASCENDING),
    ("embedding_cls", ASCENDING),
    ("neighbours", ASCENDING),
    ("neighbour_embedding_cls", ASCENDING)
])

# {
#     'city': city['city'],
#     'resolution': resolution,
#     'inbalance_ratio': inbalance_ratio,
#     'embedding_cls': embedding_cls.__name__,
#     'neighbours': neighbours,
#     'neighbour_embedding_cls': neighbour_embedding_cls.__name__,
#     'results': [
#         {
#             'classfier_cls': k,
#             'iteration': iteration + 1,
#             'dataset_type': 'validation',
#             'accuracy': acc,
#             'f1_score': f1,
#         }
#     ]
# }

from os import listdir
from os.path import isfile, join
onlyfiles = [join('results', f) for f in listdir('results') if isfile(join('results', f))]
# print(onlyfiles)

for f in onlyfiles:
    df = pd.read_csv(f)
    # print(df)
    first_row = df.iloc[0]
    results_dict = {
        'city': first_row.city,
        'resolution': int(first_row.resolution),
        'inbalance_ratio': int(first_row.inbalance_ratio),
        'embedding_cls': first_row.embedding_cls,
        'neighbours': int(first_row.neighbours),
        'neighbour_embedding_cls': first_row.neighbour_embedding_cls
    }
    existing_result = coll_results.find_one(results_dict)
    if existing_result is None:
        results = []
        for idx, row in df.iterrows():
            results.append({
                'classfier_cls': row.classfier_cls,
                'iteration': int(row.iteration),
                'dataset_type': row.dataset_type,
                'accuracy': float(row.accuracy),
                'f1_score': float(row.f1_score),
            })
        results_dict['results'] = results
        coll_results.insert_one(results_dict)
