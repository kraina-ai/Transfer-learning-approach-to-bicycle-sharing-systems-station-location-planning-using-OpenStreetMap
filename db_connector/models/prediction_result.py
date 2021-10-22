from peewee import CharField, DecimalField, ForeignKeyField

from ._base_model import BaseModel
from .classifier import Classifier


class PredictionResult(BaseModel):
    hex_id = CharField(max_length = 15, primary_key = True, index = True)
    area = ForeignKeyField(Classifier)
    value = DecimalField()
    