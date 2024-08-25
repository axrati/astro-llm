from tbt.model.model import DataTransformerModel
from tbt.config.config import ModelConfig
from tbt.trainer.trainer import Trainer
from tbt.cli.cli import CLI
import json
import string

numbers = list(string.digits)
uppercase_letters = list(string.ascii_uppercase)
lowercase_letters = list(string.ascii_lowercase)
special_characters = list(string.punctuation)
all_characters = numbers + uppercase_letters + lowercase_letters + special_characters+[" "]

config = ModelConfig()
config.int("age",50)
# config.float("grade",2)
config.boolean("valid")
# config.category("bucket", values=["a","b","c"])
# config.string("name",max_len=30,character_set=['a','b','c','d','e','f','g'])
# config.date("defect_date",'%m/%d/%Y')
# config.string("inspection_method",10,all_characters)

model = DataTransformerModel(
    config=config,
    d_model=4,
    nhead=4,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=64,
    dropout=.1,
    max_len=5000,
    output_scale=1.0
    )




f = open("data/defects_dataset.json","r")
data = json.loads(f.read())
f.close()

source = data['source']
target = data['target']
source = source[:200]
target = target[:200]

source = [
    {"age":21,"valid": False, "bucket":"a", "grade":1.6, "name":"abd", "date":"01-02-1995"},
    {"age":22,"valid": True, "bucket":"b", "grade":1.3, "name":"bda", "date":"01-02-1995"},
    {"age":25,"valid": False, "bucket":"c", "grade":1.8, "name":"ddba", "date":"01-02-1995"},
        {"age":22,"valid": True, "bucket":"b", "grade":1.3, "name":"bda", "date":"01-02-1995"},
    {"age":25,"valid": False, "bucket":"c", "grade":1.8, "name":"ddba", "date":"01-02-1995"},
]

target = [
    {"age":25,"valid": True, "bucket":"b", "grade":1.8, "name":"dba", "date":"01-02-1995"},
    {"age":22,"valid": False, "bucket":"c", "grade":1.2, "name":"dddba", "date":"01-02-1995"},
    {"age":21,"valid": True, "bucket":"a", "grade":1.1, "name":"babda", "date":"01-02-1995"},
        {"age":22,"valid": True, "bucket":"b", "grade":1.3, "name":"bda", "date":"01-02-1995"},
    {"age":25,"valid": False, "bucket":"c", "grade":1.8, "name":"ddba", "date":"01-02-1995"},
]

trainer = Trainer(model, config)
trainer.add_data(source=source, target=target)
trainer.train(epochs=10)

output = model(source,target)
predictions = model.decode_output(output)
print(predictions['original'])


cli = CLI(model,trainer)
cli.start()

# trainer.train(epochs=1000)
# output = model(source,target)
# predictions = model.decode_output(output)
# print(predictions)



# # Test Dataset for all types
# source = [
#     {"age":21,"valid": False, "bucket":"a", "grade":1.6, "name":"abd", "date":"01-02-1995"},
#     {"age":22,"valid": True, "bucket":"b", "grade":1.3, "name":"bda", "date":"01-02-1995"},
#     {"age":25,"valid": False, "bucket":"c", "grade":1.8, "name":"ddba", "date":"01-02-1995"},
#         {"age":22,"valid": True, "bucket":"b", "grade":1.3, "name":"bda", "date":"01-02-1995"},
#     {"age":25,"valid": False, "bucket":"c", "grade":1.8, "name":"ddba", "date":"01-02-1995"},
# ]

# target = [
#     {"age":25,"valid": True, "bucket":"b", "grade":1.8, "name":"dba", "date":"01-02-1995"},
#     {"age":22,"valid": False, "bucket":"c", "grade":1.2, "name":"dddba", "date":"01-02-1995"},
#     {"age":21,"valid": True, "bucket":"a", "grade":1.1, "name":"babda", "date":"01-02-1995"},
#         {"age":22,"valid": True, "bucket":"b", "grade":1.3, "name":"bda", "date":"01-02-1995"},
#     {"age":25,"valid": False, "bucket":"c", "grade":1.8, "name":"ddba", "date":"01-02-1995"},
# ]

# config = ModelConfig()
# config.int("age",50)
# config.float("grade",2)
# config.boolean("valid")
# config.category("bucket", values=["a","b","c"])
# config.string("name",max_len=30,character_set=['a','b','c','d','e','f','g'])
# config.date("date")