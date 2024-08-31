# ASTRO Transformers

### Adaptive Structure Transformation and Representation Output (ASTRO)

This neural network model is a specialized data transformer designed to process JSON-like data with multiple data types, such as strings, integers, floats, dates, booleans, and categorical data. The model’s flexibility and modularity make it highly advanced, enabling it to handle diverse data types in a unified architecture while preserving the contextual relationships between data elements.

<br></br>

# Insights from this architecture

The model leverages a single transformer network for all data types, enabling it to learn shared representations and relationships between different types of data, which is crucial for complex, structured data.

### Dynamic Embedding Sizes:

The model uses dynamic embedding sizes tailored to each data type, ensuring that the transformer receives the most relevant information while maintaining a compact and efficient representation.

### Contextual Learning:

The transformer’s self-attention mechanism allows the model to learn contextual dependencies between different fields in the JSON object, making it particularly effective in understanding the relationships between different data elements.

### Robust Output Decoding:

The model employs majority voting and softmax-based decoding for robust output generation, which helps mitigate errors caused by noisy data or outliers.

This neural network model exemplifies an advanced approach to processing structured data by integrating specialized encoding methods for various data types with a powerful transformer architecture. Its ability to handle diverse data types within a unified framework while maintaining the contextual relationships between them makes it highly effective for tasks involving complex JSON-like data.

<br></br>

# Overview of the Model Architecture

The core architecture of the model is based on a transformer encoder-decoder framework, which is generally known for its effectiveness in handling sequence data. However, this model extends the transformer architecture to process structured data types typically found in JSON objects. The model consists of:

## Embedding Layers:

Each data type has its own specialized encoding mechanism that translates the raw data into a vector representation (embedding) that the transformer can process. These embeddings are then passed through a shared transformer network.

## Positional Encoding:

Positional encoding is added to the embeddings to retain the order of elements within the data, which is crucial for maintaining the structure and meaning of the JSON data.

## Transformer Encoder and Decoder:

The shared transformer model processes the embeddings through multiple layers of self-attention and feedforward networks to learn contextual relationships.

## Output Layers:

Separate output layers are used for each data type, ensuring that the final predictions are tailored to the specific nature of each data type.

Data Type Processing Pipeline
Each data type in the JSON object is handled uniquely, leveraging different encoding strategies before being passed into the shared transformer network. The methodologies for each data type are outlined below:

### 1. `Strings`

#### Encoding:

Strings are first tokenized into characters, which are then mapped to indices based on a predefined character set. These indices are converted into a fixed-length tensor, padded with a reserved character if necessary.

#### Transformation:

The string tensor is passed through a linear layer that maps it into a higher-dimensional embedding suitable for the transformer.

#### Output:

The transformer decodes the string by predicting the most likely character sequence. Softmax is applied to the output logits to determine the probability distribution over the possible characters, and the final string is reconstructed.

### 2. `Integers`

#### Encoding:

Integers are encoded as single-element tensors, typically as floating-point numbers to ensure compatibility with the transformer’s operations.

#### Transformation:

The integer tensor is passed through a linear layer, producing a fixed-size embedding.

#### Output:

After passing through the transformer, the output is rescaled to the original integer range. The model takes the mean of the outputs and converts them back to integers.

### 3. `Floats`

#### Encoding:

Floats are directly converted into single-element tensors.

#### Transformation:

Similar to integers, the float tensor is mapped into a higher-dimensional embedding through a linear layer.

#### Output:

The final output is rescaled to its original float value range using a normalization factor.

### 4. `Booleans`

#### Encoding:

Booleans are encoded as single-element tensors with values of 0.0 (False) or 1.0 (True).

#### Transformation:

These tensors are passed through a linear layer to produce embeddings.

#### Output:

The transformer output is passed through a softmax function to determine the probability of True vs. False. Then confidence based decisioning happens. If entropy is below a certain threshold (indicating high confidence), we perform a majority vote across the batch using the predicted classes. If entropy is high, we average the probabilities across the batch and select the final output based on these averaged probabilities.

### 5. `Dates`

#### Encoding:

Dates are encoded into a 3-element tensor representing the year, month, and day, each normalized to a specific range.

#### Transformation:

The date tensor is passed through a linear layer that converts it into a higher-dimensional embedding.

#### Output:

The model decodes the date by predicting the most likely year, month, and day. The final output is selected based on the most common predictions across the batch, ensuring robustness against noise.

### 6. `Categorical Data`

#### Encoding:

Categories are mapped to integer indices based on a predefined mapping. These indices are then converted into single-element tensors.

#### Transformation:

The categorical tensor is passed through a linear layer to produce an embedding.

#### Output:

The transformer output is passed through a softmax layer to predict the most likely category. The model uses a majority vote across the batch to determine the final category output.
