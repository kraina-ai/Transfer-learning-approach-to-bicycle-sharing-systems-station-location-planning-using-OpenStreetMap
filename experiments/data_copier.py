from pymongo import MongoClient
from tqdm import tqdm

client = MongoClient('mongodb://localhost:27017/')
coll_hex = client.osmDataDB.hexesInCitiesFiltered
coll_hex_cache = client.osmDataDB.hexNeighbourEmbeddingCacheBaseCountCategoryEmbeddingFull

for h in tqdm(coll_hex.find({ 'resolution': 11 }, { "hex_id": 1, "city_id": 1, "has_station": 1 })):
    city_id = h['city_id']
    hex_id = h['hex_id']
    has_station = h['has_station']
    newvalues = { "$set": { "city_id": city_id, "has_station": has_station } }
    coll_hex_cache.update_one({ "hex_id": hex_id }, newvalues)