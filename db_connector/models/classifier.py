from peewee import AutoField, CharField
from ._base_model import BaseModel

class Classifier(BaseModel):
    id = AutoField()
    name = CharField()