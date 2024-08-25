# time-based-transformers

In essence, the model is predicting the next JSON object in a sequence based on both the past sequence (provided by source) and the partially known future sequence (provided by target).

It uses the Transformer architecture to capture dependencies across both sequences, allowing for complex predictions that take into account long-range context and structure.

# Example use

```python
from tbt.model.model import DataTransformer
from tbt.config.config import ModelConfig

config = ModelConfig()
config.int("age",10)
config.boolean("valid")

model = DataTransformerModel(
    config=config,
    d_model=8,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=256,
    dropout=.1,
    max_len=5000
    )

source = [
    {"age":21,"valid": False},
    {"age":21,"valid": True},
    {"age":21,"valid": False},
]

target = [
    {"age":21,"valid": True},
    {"age":21,"valid": False},
    {"age":21,"valid": True},
]

output = model(source,target)
predictions = model.decode_output(output)
print(predictions)
```

# Definitions

### `config` (ModelConfig):

**Description**: This is the configuration object that defines the structure of the JSON objects being processed. It includes information about the data types, embedding dimensions, and specific processing layers for different fields in the JSON objects.

**Role**: It tells the model how to encode, decode, and process each field in the input JSON objects.

_`See example for sample`_

```python
config = ModelConfig()
# Register keys below
config.string()
config.int()
config.boolean()
config.float()
config.categories()
```

### `d_model` (int):

**Description**: The dimensionality of the input and output vectors of the Transformer model. It represents the size of the hidden layers in the Transformer.

**Role**: Determines the width of the Transformer model. A larger d_model allows the model to capture more complex patterns, but it also increases memory usage and computation time.

### `nhead` (int):

Description: The number of attention heads in the multi-head attention mechanism. Each attention head processes the input data differently and independently, allowing the model to focus on different parts of the input sequence.
Role: More attention heads allow the model to learn multiple representations of the data in parallel, improving its ability to capture different types of relationships in the input.

### `num_encoder_layers` (int):

**Description**: The number of layers in the Transformer encoder. The encoder processes the source sequence and encodes it into a hidden representation.

**Role**: More encoder layers allow the model to capture deeper and more complex relationships in the input sequence, but they also increase computation time and memory usage.

### `num_decoder_layers` (int):

**Description**: The number of layers in the Transformer decoder. The decoder processes the target sequence and predicts the next JSON object in the sequence.

**Role**: More decoder layers allow the model to generate more accurate and context-aware predictions, at the cost of higher computation and memory requirements.

### `dim_feedforward` (int):

**Description**: The size of the hidden layer in the feedforward network inside each Transformer encoder/decoder layer. After the attention mechanism, data passes through this feedforward network for further processing.

**Role**: A larger dim_feedforward allows the model to learn more complex transformations, but it increases the model's size and computational load.

### `dropout` (float):

**Description**: The dropout rate applied to the Transformer layers. Dropout is a regularization technique used to prevent overfitting by randomly setting some layer outputs to zero during training.

**Role**: Higher dropout rates increase regularization, helping prevent overfitting but can slow down convergence during training.

### `max_len` (int):

**Description**: The maximum length of the input sequence that the model can handle. This determines the number of positions for which positional encodings are precomputed.

**Role**: Limits the number of JSON objects the model can process in a sequence. Increasing max_len allows the model to handle longer sequences but increases memory usage.

### `output_scale` (float):

**Description**: A scaling factor applied to the numeric outputs of the model, particularly for fields like integers or floats. It ensures that the model's predictions match the expected scale of the data.

**Role**: Adjusts the range of the model's output to align with the original data range. For example, if the model is predicting large numbers, this parameter can be used to scale the output appropriately.
