from collections import Counter
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
            # Info will contain "date_pattern"
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
    
    def encode_date(self, value: Union[str, datetime.date], date_pattern:str) -> torch.Tensor:

        if isinstance(value, str):
            value = datetime.datetime.strptime(value, date_pattern).date()
        year = value.year / 1000.0  # Normalize year to [-1, 1]
        month = (value.month - 1) / 11.0  # Normalize month to [0, 1]
        day = (value.day - 1) / 30.0  # Normalize day to [0, 1]
        return torch.tensor([year, month, day], dtype=torch.float)

    
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
    
    def decode_date(self, tensor: torch.Tensor) -> str:
        decoded_dates = []
        
        for batch in tensor:
            for date_tensor in batch:
                # Extract and denormalize each component
                year = int(date_tensor[0].item() * 1000)  # Assuming normalization to [-1, 1] range
                month = int((date_tensor[1].item() * 11) + 1)  # Map to [1, 12]
                day = int((date_tensor[2].item() * 30) + 1)  # Map to [1, 31]

                # Clip the month and day to ensure they are within valid ranges
                month = max(1, min(12, month))
                day = max(1, min(31, day))

                # Construct the date string
                # decoded_date = f"{year:04d}-{month:02d}-{day:02d}"
                decoded_data = {"year":year,"month":month,"day":day}
                decoded_dates.append(decoded_data)
        result = self.pick_most_frequent_date(decoded_dates)
        return result

    def pick_most_frequent_date(self,dates: List[str]) -> str:
        years = []
        months = []
        days = []
        
        for date in dates:
            years.append(date['year'])
            months.append(date['month'])
            days.append(date['day'])
        
        # Get the mode for year, month, and day
        most_common_year = Counter(years).most_common(1)[0][0]
        most_common_month = Counter(months).most_common(1)[0][0]
        most_common_day = Counter(days).most_common(1)[0][0]
        
        # Combine into the final most frequent date
        most_frequent_date = f"Date: {most_common_month}-{most_common_day} Year: {most_common_year}"
        return {"year":most_common_year, "month":most_common_month,"day":most_common_day}
    

    
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
