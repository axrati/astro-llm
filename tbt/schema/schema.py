import json
from typing import Type

class Schema:
    def __init__(self):
        self.attributes = {} # {key:type}
    
    def set(self,key:str,val:Type):
        self.attributes[key]=val

    