# Dataflow for ASTRO

## Summary

In general this model today predicts that given a source of A and target of B, it will predict the outcome of A to turn into B.

For example:

```python
A = [
    {"AAA_stock_price":122, "BBB_stock_price":200},
    {"AAA_stock_price":124, "BBB_stock_price":600},
]
B = [
    {"AAA_stock_price":124, "BBB_stock_price":600},
    {"AAA_stock_price":127, "BBB_stock_price":1000},
]
```

In this instance, we are setting up the model to predict the future value illustrated here:

```python
{"AAA_stock_price":122, "BBB_stock_price":200}
--->{"AAA_stock_price":124, "BBB_stock_price":600}
------>{"AAA_stock_price":127, "BBB_stock_price":1000}
---------> ???
```

ASRO operates on the idea that you can align time series data for prediction on next value given N datatypes in an object.

Each data type travels through the same general architecture, with variations in the initial encoding and final decoding. The embeddings allow the model to work with data of different types in a unified format, and the transformer layers capture relationships between data points. The output layers ensure that the model"s predictions align with the expected data type and dimensionality, and the decoding process translates these predictions back into human-readable or usable forms.

### Each datatype has covers these sections:

1. Input Data Encoding: Raw data (e.g., strings, dates, etc.) is encoded into tensor representations.
2. Embedding Layers: These tensors are passed through embedding layers to map the input to a higher-dimensional space (d_model).
3. Positional Encoding: Adds information about the position of data within the sequence, essential for preserving order in sequences.
4. Transformer Network: Data passes through shared transformer layers (with multi-head self-attention, feedforward layers, etc.) to capture relationships between elements.
5. Output Layers: The transformed data is passed through output layers that match the required dimensionality for the respective data type.
6. Decoding: The output is decoded back into a human-readable form.

## String Data Flow

#### Encoding:

Input: A string (e.g., "Hello").

#### Encoding Layer:

Each character in the string is converted to an integer index based on a predefined character set (char_to_idx). This results in a tensor of shape [max_len] (e.g., [10] if max_len=10).

Example: "Hello" → [8, 5, 12, 12, 15, 0, 0, 0, 0, 0] where 0 represents padding.

#### Embedding Layer:

The tensor is passed through an embedding layer, mapping each index to a higher-dimensional space (d_model). The output tensor has a shape of [max_len, d_model].
Transformer Layers:

#### Positional Encoding:

Adds positional information to the embeddings. The tensor retains its shape [max_len, d_model].

#### Transformer Encoder/Decoder:

The tensor is processed by the shared transformer encoder and decoder layers. The output from the decoder has the shape [max_len, d_model].

#### Output Layer:

The output layer reshapes this to [max_len, total_characters] where total_characters is the size of the character set. The logits for each character position are then passed through a softmax to yield probabilities.

#### Decoding:

The tensor is converted back to a string by mapping the highest probability indices back to characters using idx_to_char.

## Date Data Flow

#### Encoding:

Input: A date (e.g., "12/31/2023").

#### Encoding Layer:

The date is split into year, month, and day. These values are normalized and converted to a tensor of shape [3] representing [year, month, day].

#### Embedding Layer:

The tensor [3] is linearly mapped to [d_model].

#### Positional Encoding:

Positional information is added, resulting in a shape [1, d_model].

#### Transformer Encoder/Decoder:

The tensor is processed through the transformer layers. The output from the decoder has a shape [1, d_model].

#### Output Layer:

The output layer maps this to [3] to represent [year, month, day].

#### Decoding:

The tensor is decoded back to a date string format, considering rounding and clipping to valid date ranges.

## Boolean Data Flow

#### Encoding:

Input: A boolean value (True or False).

#### Encoding Layer:

The boolean value is converted to a float tensor of shape [1] with either 1.0 for True or 0.0 for False.

#### Embedding Layer:

The tensor [1] is linearly mapped to [d_model].

#### Positional Encoding:

Positional encoding is applied (though it may not be necessary for a single boolean value). The tensor retains its shape [1, d_model].

#### Transformer Encoder/Decoder:

The tensor is processed by the transformer layers. The output has a shape of [1, d_model].

#### Output Layer:

The output layer maps this to [2] (representing logits for False and True).

#### Decoding:

The output logits are passed through a softmax, and the highest probability determines whether the final output is True or False.

## Category Data Flow

#### Encoding:

Input: A category (e.g., "red", "green", "blue").

#### Encoding Layer:

The category is mapped to an integer index based on a predefined list of categories. This results in a tensor of shape [1].

#### Embedding Layer:

The tensor [1] is linearly mapped to [d_model].

#### Positional Encoding:

Positional encoding is applied, resulting in a shape [1, d_model].

#### Transformer Encoder/Decoder:

The tensor is processed by the transformer layers, resulting in a shape [1, d_model].

#### Output Layer:

The output layer maps this to [num_categories] logits, where num_categories is the number of possible categories.

#### Decoding:

The logits are passed through a softmax, and the highest probability index is mapped back to the corresponding category.

## Int Data Flow

#### Encoding:

Input: An integer value (e.g., 42).

#### Encoding Layer:

The integer is converted to a float tensor of shape [1].

#### Embedding Layer:

The tensor [1] is linearly mapped to [d_model].

#### Positional Encoding:

Positional encoding is applied, resulting in a shape [1, d_model].

#### Transformer Encoder/Decoder:

The tensor is processed by the transformer layers, resulting in a shape [1, d_model].

#### Output Layer:

The output layer maps this to [1], representing the predicted integer value.

#### Decoding:

The output tensor is converted back to an integer.

## Float Data Flow

#### Encoding:

Input: A float value (e.g., 3.14).

#### Encoding Layer:

The float value is directly converted to a tensor of shape [1].

#### Embedding Layer:

The tensor [1] is linearly mapped to [d_model].

#### Positional Encoding:

Positional encoding is applied, resulting in a shape [1, d_model].

#### Transformer Encoder/Decoder:

The tensor is processed by the transformer layers, resulting in a shape [1, d_model].

#### Output Layer:

The output layer maps this to [1], representing the predicted float value.

#### Decoding:

The output tensor is converted back to a float value.
