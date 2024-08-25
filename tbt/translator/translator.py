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
            self.character_set = info.get("character_set", [])
            self.reserved_value = info.get("reserved_value", "\u0000")
            self.distinct_characters = []
            for c in self.character_set:
                if c not in self.distinct_characters:
                    self.distinct_characters.append(c)
                if self.reserved_value not in self.distinct_characters:
                    self.distinct_characters.append(self.reserved_value)
            self.total_characters = len(self.distinct_characters)
            self.char_to_idx = {char: i for i, char in enumerate(self.distinct_characters)}
            self.idx_to_char = {i: char for char, i in self.char_to_idx.items()}
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
        self.info = info
    
    def _automap_categories(self, categories: List[str]) -> Union[Dict[str, int], Dict[int, str]]:
        category_map = {category: i for i, category in enumerate(categories)}
        reverse_category_map = {i: category for category, i in category_map.items()}
        return category_map, reverse_category_map

    RESERVED_PATTERN = "\u0000"  # Null character for reserved padding

    def encode_string(self, value: str, max_len: int = 10) -> torch.Tensor:
        # Remove the reserved characters from input if any
        clean_value = value.replace(self.reserved_value, "")
        # Truncate the string to max_len if necessary
        truncated_value = clean_value[:max_len]
        # Pad the string with the reserved value if it's shorter than max_len
        padded_value = truncated_value + self.reserved_value * (max_len - len(truncated_value))
        # Convert the string to indices based on the character set
        encoded_indices = [self.char_to_idx[char] for char in padded_value]
        return torch.tensor(encoded_indices, dtype=torch.long)
    
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
        # Convert indices back to characters
        available_indexes = []
        for idx in tensor:
            try:
                recommendation = self.idx_to_char[int(idx)]
                available_indexes.append(recommendation)
            except:
                available_indexes.append(self.char_to_idx(self.reserved_value))
                print("Bad parse on strings...")
                continue

        # Join the characters and strip any trailing reserved values
        output_string = ""
        for avi in available_indexes:
            if avi != self.reserved_value:
                output_string=output_string+avi
        return output_string

    
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
    
    def decode_category(self, value: int, reverse_category_map: Dict[int, str]) -> str:
        return reverse_category_map[int(value)]
    
    
    
    
        
        
        
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
