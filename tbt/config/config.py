# config = {
#     "layers":[
#         {"key":"name", "datatype":"string", "info":{}},
#         {"key":"age", "datatype":"int"}
#         ],
#     "content_window":12332
# }

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Literal, Any, List
import datetime
import math
from tbt.translator.translator import Translator

class Layer:
    def __init__(self, key, encode, decode, embedding_dim,datatype,character_set=[], max_len=0, normalizer=1.0):
        self.key = key
        self.encode = encode
        self.decode = decode
        self.embedding_dim = embedding_dim
        self.datatype=datatype
        self.character_set=character_set
        self.max_len=max_len
        self.normalizer=normalizer


class ModelConfig:
    def __init__(self):
        self.layers = {}

    def string(self, key: str, max_len: int, character_set: list, reserved_character="\u0000"):
        t = Translator(datatype="string", info={"max_len": max_len, "character_set": character_set, "reserved_value": reserved_character})
        l = Layer(key, t.encode, t.decode, embedding_dim=max_len, datatype="string", character_set=character_set, max_len=max_len)
        self.layers[key] = l

    def int(self, key: str, normalizer:float=1.0):
        t = Translator(datatype="int")
        l = Layer(key, t.encode, t.decode, embedding_dim=1,datatype="int", normalizer=normalizer)
        self.layers[key] = l

    def float(self, key: str, normalizer:float=1.0):
        t = Translator(datatype="float")
        l = Layer(key, t.encode, t.decode, embedding_dim=1,datatype="float", normalizer=normalizer)
        self.layers[key] = l

    def boolean(self, key: str):
        t = Translator(datatype="boolean")
        l = Layer(key, t.encode, t.decode, embedding_dim=1,datatype="boolean")
        self.layers[key] = l

    def date(self, key: str):
        t = Translator(datatype="date")
        l = Layer(key, t.encode, t.decode, embedding_dim=3,datatype="date")  # 3 for year, month, day
        self.layers[key] = l

    def category(self, key: str, values: list):
        t = Translator(datatype="category", info={"values": values})
        l = Layer(key, t.encode, t.decode, embedding_dim=1, datatype="category")  # Each category is a single value
        self.layers[key] = l