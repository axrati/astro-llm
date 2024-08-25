from tbt.model.model import DataTransformerModel
from tbt.config.config import ModelConfig

config = ModelConfig()
config.int("age",50)
config.float("grade",2)
config.boolean("valid")
# config.category("bucket", values=["a","b","c"])
config.string("name",max_len=20,character_set=['a','b','c','d','e','f'])

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
    {"age":21,"valid": False, "bucket":"a", "grade":1.6, "name":"d"},
    {"age":22,"valid": True, "bucket":"b", "grade":1.3, "name":"f"},
    {"age":25,"valid": False, "bucket":"c", "grade":1.8, "name":"b"},
]

target = [
    {"age":25,"valid": True, "bucket":"b", "grade":1.8, "name":"b"},
    {"age":22,"valid": False, "bucket":"c", "grade":1.2, "name":"d"},
    {"age":21,"valid": True, "bucket":"a", "grade":1.1, "name":"b"},
]


output = model(source,target)
predictions = model.decode_output(output)
print(predictions)