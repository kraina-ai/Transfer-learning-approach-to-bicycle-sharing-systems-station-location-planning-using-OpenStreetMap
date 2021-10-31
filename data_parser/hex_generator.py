from typing import List, Union
from h3 import h3
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely.geometry import mapping

def generate_hexes(shape: Union[Polygon, MultiPolygon]) -> List[str]:
    buffered_polygon = shape.buffer(0.005)
    if type(buffered_polygon) == Polygon:
        indexes = h3.polyfill(mapping(buffered_polygon), 11, geo_json_conformant=True)
    else:
        indexes = []
        for sub_polygon in buffered_polygon.geoms:
            indexes.extend(h3.polyfill(mapping(sub_polygon), 11, geo_json_conformant=True))
    return indexes