from pymongo import ASCENDING, GEOSPHERE, MongoClient
import shapely
from shapely.geometry import shape

client = MongoClient('mongodb://localhost:27017/')
db = client.osmDataDB
coll_cities = db.cities
coll_hexes = db.hexesInCitiesFiltered
coll_relations = db.relationsFiltered
# relations_cursor = coll_relations.find({}).limit(100)

# for r in relations_cursor:
#     r['shape'] = shape(r['geometry'])
#     print(r['shape'])

print(coll_relations.find({}).distinct('geometry.type'))