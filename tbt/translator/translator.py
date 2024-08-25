import torch
from typing import Dict, Union, Literal, Any, List
import datetime

    
class Translator:
    def __init__(self, datatype: Literal["float", "int", "date", "string", "category", "boolean"], info: Any = {}):
        if datatype == "float":
            self.encode = self.encode_float
            self.decode = self.decode_float
        elif datatype == "int":
            self.encode = self.encode_int_as_float
            self.decode = self.decode_int_from_float
        elif datatype == "date":
            self.encode = self.encode_date
            self.decode = self.decode_date
        elif datatype == "string":
            max_len = info.get("max_len", 10)
            self.encode = lambda value: self.encode_string(value, max_len=max_len)
            self.decode = self.decode_string
        elif datatype == "category":
            if "values" in info:
                self.category_map, self.reverse_category_map = self._automap_categories(info["values"])
                self.encode = lambda value: self.encode_category(value, self.category_map)
                self.decode = lambda tensor: self.decode_category(tensor, self.reverse_category_map)
            else:
                raise ValueError("Missing 'categories' in info for category datatype")
        elif datatype == "boolean":
            self.encode = self.encode_boolean_as_float
            self.decode = self.decode_boolean_from_float
    
    def _automap_categories(self, categories: List[str]) -> Union[Dict[str, int], Dict[int, str]]:
        category_map = {category: i for i, category in enumerate(categories)}
        reverse_category_map = {i: category for category, i in category_map.items()}
        return category_map, reverse_category_map

    RESERVED_PATTERN = "\u0000"  # Null character for reserved padding

    def encode_string(self, value: str, max_len: int = 10) -> torch.Tensor:
        truncated_value = value[:max_len]
        padded_value = truncated_value + self.RESERVED_PATTERN * (max_len - len(truncated_value))
        return torch.tensor([ord(c) for c in padded_value], dtype=torch.long)
    
    def encode_int_as_float(self, value: int) -> torch.Tensor:
        return torch.tensor([float(value)], dtype=torch.float)
    
    def encode_float(self, value: float) -> torch.Tensor:
        return torch.tensor([value], dtype=torch.float)
    
    def encode_boolean_as_float(self, value: bool) -> torch.Tensor:
        return torch.tensor([1.0 if value else 0.0], dtype=torch.float)
    
    def encode_date(self, value: Union[str, datetime.date]) -> torch.Tensor:
        if isinstance(value, str):
            value = datetime.datetime.strptime(value, '%Y-%m-%d').date()
        return torch.tensor([value.year, value.month, value.day], dtype=torch.float)
    
    def encode_category(self, value: str, category_map: Dict[str, int]) -> torch.Tensor:
        return torch.tensor([float(category_map[value])], dtype=torch.float)
    
    def decode_string(self, tensor: torch.Tensor) -> str:
        decoded_value = ''.join([chr(int(x)) for x in tensor if x > 0])
        return decoded_value.split(self.RESERVED_PATTERN)[0]
    
    def decode_int_from_float(self, tensor: torch.Tensor, normalization_factor: float = 1.0) -> int:
        return int(torch.mean(tensor).item()) * normalization_factor
    
    def decode_float(self, tensor: torch.Tensor, normalization_factor: float = 1.0) -> float:
        return float(torch.mean(tensor).item()) * normalization_factor
    
    def decode_boolean_from_float(self, tensor: torch.Tensor) -> bool:
        return bool(torch.sigmoid(tensor).item() >= 0.5)
    
    def decode_date(self, tensor: torch.Tensor) -> datetime.date:
        year, month, day = int(tensor[0].item()), int(tensor[1].item()), int(tensor[2].item())
        try:
            return datetime.date(year, month, day)
        except ValueError:
            max_day = (datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)).day
            day = min(day, max_day)
            return datetime.date(year, month, day)
    
    def decode_category(self, tensor: torch.Tensor, reverse_category_map: Dict[int, str]) -> str:
        return reverse_category_map[int(tensor.item())]
    
    
    
    
        
        
        
# # Example Usage for Category
# translator_category = Translator(datatype="category", info={"values": ['cat', 'dog', 'bird']})

# # Encode a category
# encoded_category = translator_category.encode('dog')
# print(f"Encoded category: {encoded_category}")
# # Decode the category
# decoded_category = translator_category.decode(encoded_category)
# print(f"Decoded category: {decoded_category}")



# t = Translator(datatype="int")
# a = t.encode(123523453452345)
# b = t.decode(a)



# t = Translator(datatype="string",info={"max_len":40,"character_set":["a","b","c","d"]})
# a = t.encode("dddcda")
# b = t.decode(a)

# t = Translator(datatype="date")
# a = t.encode("01-02-2025")
# b = t.decode(a)
