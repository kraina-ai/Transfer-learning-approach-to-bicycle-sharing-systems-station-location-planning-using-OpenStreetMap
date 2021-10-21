import json
import math
import multiprocessing
import time
from functools import partial
from itertools import chain

from pymongo import ASCENDING, GEOSPHERE, MongoClient
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def parse_hex(chunk):
    #define client inside function
    client = client = MongoClient('localhost',27017,maxPoolSize=10000)
    db = client.osmDataDB
    coll_hexes = db.hexesInCities
    coll_relations = db.relations
    coll_hexes_filtered = db.hexesInCitiesFiltered
    coll_relations_filtered = db.relationsFiltered
    for _id in chunk:
        hex = coll_hexes.find_one({ '_id': _id })
        if coll_hexes_filtered.find_one({ 'hex_id': hex['hex_id'] }) is None:
            relations = list(coll_relations.find({
                "geometry": {
                    "$geoIntersects": {
                        "$geometry": hex['geometry']
                    }
                }
            }))
            if len(relations) > 0:
                relations_to_add = []
                for relation in relations:
                    if coll_relations_filtered.find_one({ 'osm_id': relation['osm_id'] }) is None:
                        relation['city_id'] = hex['city_id']
                        del relation['_id']
                        keys_to_delete = []
                        for k, v in relation.items():
                            keys_to_delete.append(k)
                            if '.' in k:
                                k = k.replace('.', '\u002E')
                            relation[k] = v
                        for k in keys_to_delete:
                            del relation[k]
                        # coll_relations_filtered.insert_one(relation, bypass_document_validation=True)
                        relations_to_add.append(relation)
                if len(relations_to_add) > 0:
                    coll_relations_filtered.insert_many(relations_to_add)
                del hex['_id']
                coll_hexes_filtered.insert_one(hex)

client = MongoClient('mongodb://localhost:27017/')
db = client.osmDataDB
coll_hexes = db.hexesInCities
result = coll_hexes.find({"resolution": { "$gt": 9 }},{"_id":1})
hex_ids = [x["_id"] for x in tqdm(result)]
del result
chunks_list = list(chunks(hex_ids,100))
del hex_ids
process_map(parse_hex, chunks_list, max_workers=8)