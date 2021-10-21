from peewee import AutoField, CharField, DecimalField

from ..db_definition import JSONField
from ._base_model import BaseModel


class Area(BaseModel):
    id = AutoField()
    name = CharField()
    lon = DecimalField()
    lat = DecimalField()
    shape = JSONField()
