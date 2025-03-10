import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any




class Trainer:
    def __init__(self, model, config):
        self.model = model.to(self.get_device())
        self.config = config
        self.source = []
        self.target = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add_data(self, source, target):
        self.source = source
        self.target = target

    def train(self, epochs=10):
        if len(self.source) != len(self.target):
            raise Exception(f"Source to Target mappings must be equal in length. Source is {len(self.source)} and Taget is {len(self.target)}")

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler() # Use Automatic Mixed Precision for efficiency
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            output = self.model(self.source, self.target)
            loss = 0

            for key in output:
                datatype = self.config.layers[key].datatype
                target_tensor = self.get_target_tensor(self.target, key, datatype, self.config.layers[key]).to(self.device)
                loss += self.compute_loss(output[key], target_tensor, datatype, self.config.layers[key])

            # Backward pass using the scaler
            scaler.scale(loss).backward()
            # Optimizer step using the scaler
            scaler.step(optimizer)
            # Update the scaler for the next iteration
            scaler.update()

            total_loss += loss.item()
            print("                                          ", end="\r")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(self.source)}", end="\r")
        print("Training complete.                                                      ")
    def get_target_tensor(self, target, key, datatype, layer):
        if datatype == 'category':
            # Convert string category to index
            # print('category')
            target_tensor = torch.stack([self.config.layers[key].encode(target[i][key]).to(self.device) for i in range(len(target))], dim=0)
            # print(target_tensor.size())
            target_tensor = target_tensor.view(-1)
            target_tensor = target_tensor.repeat(len(target)).view(-1)
            # target_tensor = torch.tensor([self.config.layers[key].encode(target[i][key]).item() for i in range(len(target))], dtype=torch.long)
        elif datatype == 'string':
            # Convert each string in the batch to a tensor of indices
            # print('string')
            target_tensor = torch.stack([self.config.layers[key].encode(target[i][key]).to(self.device) for i in range(len(target))], dim=0)
            target_tensor = target_tensor.view(-1)
        elif datatype == 'date':
            # print('date')
            # Convert each date string in the batch to a normalized tensor of [year, month, day]
            target_tensor = torch.stack([self.config.layers[key].encode(target[i][key], layer.date_pattern).to(self.device) for i in range(len(target))], dim=0)
        elif datatype in ['int', 'float', 'boolean']:
            # Handle numeric and boolean types
            target_tensor = torch.tensor([target[i][key] for i in range(len(target))], dtype=torch.float).to(self.device)
        else:
            raise ValueError(f"Unsupported datatype: {datatype}")
        return target_tensor

    def compute_loss(self, output, target, datatype, layer):
        if datatype == "boolean":
            output = output.view(-1, 2)
            target = target.repeat_interleave(len(target)).long().to(self.device)  # Adjust target size to match output
            loss = F.cross_entropy(output, target)
        elif datatype == "int" or datatype == "float":
            target = target.unsqueeze(1).expand(-1, target.size(0)).to(self.device)
            loss = F.mse_loss(output.squeeze(-1), target)
            # loss = F.mse_loss(output.view_as(target), target)
        elif datatype == "category":
            output = output.view(-1, output.size(-1)).to(self.device)
            target = target.view(-1).long().to(self.device)
            loss = F.cross_entropy(output, target)
        elif datatype == "string":
            # Adjust target tensor to match output shape
            batch_size = output.size(0)
            sequence_length = output.size(0) # batch_size
            output = output.view(batch_size * sequence_length * layer.max_len, layer.total_characters).to(self.device)
            # Reshape target to match the number of required predictions
            target = target.repeat_interleave(batch_size).to(self.device)  # Repeating N times to match the expected N*batch elements
            if output.size(0) != target.size(0):
                raise ValueError(f"Output and target batch sizes do not match: {output.size(0)} vs {target.size(0)}")
            loss = F.cross_entropy(output, target)
        elif datatype == "date":
            target = target.unsqueeze(1).expand(target.size(0), target.size(0), target.size(1)).to(self.device)
            loss = F.mse_loss(output, target.float())
        else:
            raise ValueError(f"Unsupported datatype: {datatype}")
        return loss
    