from tbt.model.model import DataTransformerModel
from tbt.config.config import ModelConfig

config = ModelConfig()
config.int("age",50)
config.boolean("valid")

model = DataTransformerModel(
    config=config,
    d_model=8,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=256,
    dropout=.1,
    max_len=5000,
    output_scale=1.0
    )

source = [
    {"age":21,"valid": False},
    {"age":22,"valid": True},
    {"age":25,"valid": False},
]

target = [
    {"age":25,"valid": True},
    {"age":22,"valid": False},
    {"age":21,"valid": True},
]


output = model(source,target)
predictions = model.decode_output(output)
print(predictions)