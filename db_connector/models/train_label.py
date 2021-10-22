from peewee import BooleanField, CharField, ForeignKeyField

from ._base_model import BaseModel
from .area import Area


class TrainLabel(BaseModel):
    hex_id = CharField(max_length = 15, primary_key = True, index = True)
    area = ForeignKeyField(Area)
    has_station = BooleanField()
