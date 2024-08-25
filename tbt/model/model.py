# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import math
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, List
from tbt.config.config import ModelConfig
from tbt.utils.utils import stringdate

class PositionalEncoding(nn.Module):
    """
    Max len here is the context window size
    """
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Data transformer    
class DataTransformerModel(nn.Module):
    def __init__(self, config:ModelConfig, d_model=64, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=256, dropout=0.1, max_len=5000, output_scale=1.0):
        super(DataTransformerModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.output_scale = output_scale  # Scaling factor for numeric outputs

        # Embedding layers dictionary
        self.embeddings = nn.ModuleDict()

        # Shared Transformer model for all layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len)

        # Output layers for each key
        self.output_layers = nn.ModuleDict({
            key: self._get_output_layer(layer, d_model) for key, layer in config.layers.items()
        })

        for key, layer in config.layers.items():
            # Create embedding linear layer
            self.embeddings[key] = nn.Linear(layer.embedding_dim, d_model)
        # Move whole model to appropriate device
        self.to(self.device)

    def _get_output_layer(self, layer, d_model):
        """ Return the appropriate output layer based on the data type. """
        if layer.datatype == "boolean":
            return nn.Linear(d_model, 2)  # Two outputs: one for False, one for True
        elif layer.datatype == "int":
            return nn.Linear(d_model, 1)  # Output a single value
        elif layer.datatype == "float":
            return nn.Linear(d_model, 1)  # Output a single float value
        elif layer.datatype == "string":
            return nn.Linear(d_model, layer.total_characters * layer.max_len)  # Output logits for each character in the sequence
        elif layer.datatype == "date":
            return nn.Linear(d_model, 3)  # 3 outputs for year, month, day
        elif layer.datatype == "category":
            return nn.Linear(d_model, len(layer.values))  # Output logits for each category
        else:
            return nn.Linear(d_model, layer.embedding_dim)  # Default for other types



    def forward(self, src: List[Dict[str, Any]], tgt: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """ Forward pass through the model, combined embeddings for all layers. """
        
        # Initialize combined embeddings for source and target
        combined_src_embedded = None
        combined_tgt_embedded = None

        for key, layer in self.config.layers.items():
            try:
                # Check if obj[key] is giving the right result
                if layer.datatype=="date":
                    encoded_src = torch.stack([layer.encode(obj[key], layer.date_pattern).float().to(self.device) for obj in src])  # Shape: (batch_size, seq_len, embedding_dim)
                    encoded_tgt = torch.stack([layer.encode(obj[key], layer.date_pattern).float().to(self.device) for obj in tgt])
                else:
                    encoded_src = torch.stack([layer.encode(obj[key]).float().to(self.device) for obj in src])  # Shape: (batch_size, seq_len, embedding_dim)
                    encoded_tgt = torch.stack([layer.encode(obj[key]).float().to(self.device) for obj in tgt])
            except Exception as e:
                print(f"Error processing key: {key} with error: {str(e)}")
                print(f"Source data for key: {src}")
                print(f"Target data for key: {tgt}")
                raise e

            # Pass through embedding layers
            src_embedded = self.embeddings[key](encoded_src)  # Shape: (batch_size, seq_len, d_model)
            tgt_embedded = self.embeddings[key](encoded_tgt)

            # Combine the embeddings across all layers
            if combined_src_embedded is None:
                combined_src_embedded = src_embedded
                combined_tgt_embedded = tgt_embedded
            else:
                combined_src_embedded += src_embedded
                combined_tgt_embedded += tgt_embedded

        # Apply positional encoding
        combined_src_embedded = self.positional_encoding(combined_src_embedded)
        combined_tgt_embedded = self.positional_encoding(combined_tgt_embedded)

        # Shared Transformer forward pass
        memory = self.encoder(combined_src_embedded)
        output = self.decoder(combined_tgt_embedded, memory)

        # Split the output for each layer
        results = {}
        for key in self.config.layers.keys():
            layer_output = self.output_layers[key](output)

            # Scale the output for numeric types
            if self.config.layers[key].datatype in ['int', 'float']:
                layer_output = layer_output * self.config.layers[key].normalizer

            results[key] = layer_output

        return results

    def entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        """ Calculate the entropy of the probability distribution. """
        return -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=-1)


    def decode_output(self, output: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """ 
        Decode the model's output back to human-readable form. 
        Outputs 2 different versions
        
        "original": Represents the original data input
        "model"   : Represents an easier to parse version based on datatype (mostly date)
        """
        decoded_output = {}
        original_output = {}
        for key, tensor in output.items():
            layer = self.config.layers[key]
            if layer.datatype == "string":
                # print("string")
                # Reshape for view
                reshaped_tensor = tensor.view(tensor.size(0), layer.max_len, -1, layer.total_characters)
                probabilities = F.softmax(reshaped_tensor, dim=-1)
                averaged_probabilities = probabilities.mean(dim=2)
                predicted_indices = torch.argmax(averaged_probabilities, dim=-1) # will be of shape [batch, max_len],
                # Mode of results across the batch
                mode_result, _ = torch.mode(predicted_indices, dim=0)
                decoded = layer.decode(mode_result.tolist())
                decoded_output[key] = decoded
                original_output[key] = decoded

            elif layer.datatype == "boolean":
                probabilities = F.softmax(tensor, dim=-1)
                # Calculate entropy
                entropy = self.entropy(probabilities)
                # Calculate the average entropy across the batch
                average_entropy = torch.mean(entropy).item()
                # Define a threshold for entropy. Lower threshold means more confident predictions.
                entropy_threshold = 0.5  # This can be tuned
                
                if average_entropy < entropy_threshold:
                    # If entropy is low, rely on the highest probability
                    predicted_classes = torch.argmax(probabilities, dim=-1)
                    final_result = predicted_classes.bool().mean().item() >= 0.5.item()
                else:
                    # If entropy is high, fallback strategy (e.g., averaging the probabilities)
                    averaged_probabilities = probabilities.mean(dim=0)
                    final_result = averaged_probabilities[0][1] > averaged_probabilities[0][0].item()
                decoded_output[key] = final_result.item()
                original_output[key] = final_result.item()
            elif layer.datatype == "int":
                # print("int")
                decoded_output[key] = layer.decode(tensor)
                original_output[key] = layer.decode(tensor)
            elif layer.datatype == "float":
                # print("float")
                decoded_output[key] = layer.decode(tensor)
                original_output[key] = layer.decode(tensor)
            elif layer.datatype == "category":
                # print("category")
                # Apply softmax to convert logits to probabilities
                probabilities = F.softmax(tensor, dim=-1)
                # Step 2: Use argmax to select the index with the highest probability
                selected_indices = torch.argmax(probabilities, dim=-1)
                # Calculate the mode (most frequent value) along the column (axis=0)
                right_enum, _ = torch.mode(selected_indices, dim=0)
                right_enum_list = right_enum.tolist()
                counts = {}
                for index in right_enum_list:
                    if index in counts:
                        counts[index] += 1
                    else:
                        counts[index] = 1
                max_count = 0
                right_enum_index = right_enum_list[0]  
                for index, count in counts.items():
                    if count > max_count:
                        max_count = count
                        right_enum_index = index
                # print(f"Correct enum is {right_enum_index}")
                decoded_output[key] = layer.decode(right_enum_index)  # Likelihood for each category
                original_output[key] = layer.decode(right_enum_index)  # Likelihood for each category
            elif layer.datatype == "date":
                decoded_output[key] = layer.decode(tensor)
                original_output[key] = stringdate(layer.decode(tensor), layer.date_pattern)
            else:
                print("UNCAUGHT DATATYPE")
                decoded_output[key] = layer.decode(tensor.squeeze(0))
                original_output[key] = layer.decode(tensor.squeeze(0))

        return {"original":original_output, "model":decoded_output}
    
    
