import numpy as np
from peewee import CharField, ForeignKeyField, IntegerField

from ._base_model import BaseModel
from .area import Area


class Vector(BaseModel):
    hex_id = CharField(max_length = 15, primary_key = True, index = True)
    area = ForeignKeyField(Area)
    v0 = IntegerField()
    v1 = IntegerField()
    v2 = IntegerField()
    v3 = IntegerField()
    v4 = IntegerField()
    v5 = IntegerField()
    v6 = IntegerField()
    v7 = IntegerField()
    v8 = IntegerField()
    v9 = IntegerField()
    v10 = IntegerField()
    v11 = IntegerField()
    v12 = IntegerField()
    v13 = IntegerField()
    v14 = IntegerField()
    v15 = IntegerField()
    v16 = IntegerField()
    v17 = IntegerField()
    v18 = IntegerField()
    v19 = IntegerField()

    def get_vector(self) -> np.ndarray:
        return np.array([
            self.v0,  self.v1,  self.v2,  self.v3,
            self.v4,  self.v5,  self.v6,  self.v7,
            self.v8,  self.v9,  self.v10, self.v11,
            self.v12, self.v13, self.v14, self.v15,
            self.v16, self.v17, self.v18, self.v19,
        ])
