from tbt.model.model import DataTransformerModel
from tbt.config.config import ModelConfig

config = ModelConfig()
config.int("age",50)
config.float("grade",2)
config.boolean("valid")
config.category("bucket", values=["a","b","c"])
config.string("name",max_len=20,character_set=['a','b','c','d','e','f'])
config.date("date")

model = DataTransformerModel(
    config=config,
    d_model=8,
    nhead=4,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=256,
    dropout=.1,
    max_len=5000,
    output_scale=1.0
    )

source = [
    {"age":21,"valid": False, "bucket":"a", "grade":1.6, "name":"abd", "date":"01-02-1995", "bucket":"a"},
    {"age":22,"valid": True, "bucket":"b", "grade":1.3, "name":"bda", "date":"01-02-1995", "bucket":"a"},
    {"age":25,"valid": False, "bucket":"c", "grade":1.8, "name":"ddba", "date":"01-02-1995", "bucket":"a"},
]

target = [
    {"age":25,"valid": True, "bucket":"b", "grade":1.8, "name":"dba", "date":"01-02-1995", "bucket":"a"},
    {"age":22,"valid": False, "bucket":"c", "grade":1.2, "name":"dddba", "date":"01-02-1995", "bucket":"a"},
    {"age":21,"valid": True, "bucket":"a", "grade":1.1, "name":"babda", "date":"01-02-1995", "bucket":"a"},
]


output = model(source,target)
predictions = model.decode_output(output)
print(predictions)