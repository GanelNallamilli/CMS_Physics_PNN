# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:38:55 2023

@author: drpla
"""
# =============================================================================
# import os
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# 
# =============================================================================
# =============================================================================
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")
# 
# 
# =============================================================================


#class NeuralNetwork(nn.Module)


import torch
import torch.nn as nn
import torch.optim as optim
import json
import pandas as pd

df = pd.read_parquet(r'C:\Users\drpla\Desktop\ICL-PHYSICS-YEAR-4\Masters Project\Data\New folder\merged_nominal.parquet')
#%%
with open(r'C:\Users\drpla\Desktop\ICL-PHYSICS-YEAR-4\Masters Project\Data\New folder\summary.json', "r") as f:
  proc_dict = json.load(f)["sample_id_map"]
  
  
sig = df[df.process_id == proc_dict["GluGluToRadionToHHTo2G2Tau_M-300"]] # just one signal process, mass of X is 1000 GeV
#%%  
#bkg = df[df.process_id == proc_dict["DiPhoton"]] # just one of the background processes

# Define a simple neural network class
class SimpleNN(nn.Module):
    def __init__(self, input_size1, hidden_size1, output_size1,input_size2, hidden_size2, output_size2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size1, hidden_size1)  # Input layer to hidden layer
        self.relu1 = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size1, output_size1)  # Hidden layer to output layer
        self.fc3 = nn.Linear(input_size2, hidden_size2)  # Input layer to hidden layer
        self.relu2 = nn.ReLU()  # Activation function
        self.fc4 = nn.Linear(hidden_size2, output_size2)  # Hidden layer to output layer



    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Create an instance of the neural network
input_size1 = 66645  # Change this according to your input data size
hidden_size1 = 100  # Change this according to the number of neurons in the hidden layer
output_size1 = 10  # Change this according to the number of output classes or regression output size
input_size2 = output_size1
hidden_size2 = 20
output_size2 = 5
model = SimpleNN(input_size1, hidden_size1, output_size1, input_size2, hidden_size2, output_size2)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use appropriate loss function based on the task
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training loop (example)
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    inputs = torch.randn(32, input_size1)  # Replace with your input data
    outputs = model(inputs)

    # Example target labels for demonstration
    targets = torch.randint(0, output_size2, (32,))  # Replace with your target labels

    # Calculate loss
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    
#%%

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.relu1 = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden layer to output layer
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    

    def weightedBCELoss(self, input, target, weight):
      x, y, w = input, target, weight
      log = lambda x: torch.log(x*(1-1e-8) + 1e-8)
      #return torch.mean(-w * (y*log(x) + (1-y)*log(1-x)))
      return -w * (y*log(x) + (1-y)*log(1-x))


"""
Writing practicing
"""
# =============================================================================
#     def batch_weightedBCELoss(self, input, target, weight, batch_size):
# #batch_weightedBCELoss(self, train, train_labels_tensor, train_weights_tensor, batch_size)
#         self.batch_size=batch_size
# 
#        # train=model.forward(input)
# 
#         total_batch_err=torch.empty(0,1)
#         output_length=input.shape[0]
#         batch_remainder=output_length%batch_size
# 
#         for i in range(0, output_length//batch_size):
#             weights = weight[i*(batch_size):(i+1)*(batch_size), :]
#             labels = target[i*(batch_size):(i+1)*(batch_size), :]
#             inputs = input[i*(batch_size):(i+1)*(batch_size), :]
# 
#             loss=self.weightedBCELoss(inputs, labels, weights)
# 
#             total_batch_err=torch.cat((total_batch_err,loss)) 
#             print(total_batch_err.shape[0])
# 
#         if batch_remainder > 0:
#             weights = weight[(output_length//batch_size)*batch_size:, :]
#             labels = target[(output_length//batch_size)*batch_size:, :]
#             inputs = input[(output_length//batch_size)*batch_size:, :]
# 
#             loss=self.weightedBCELoss(inputs, labels, weights)
# 
#             #weights = train_weights_tensor[(train_weights_tensor.shape[0]//batch_size)*batch_size:, :]
#             total_batch_err=torch.cat((total_batch_err,loss))
#             print(total_batch_err.shape[0])
#             
#         return torch.mean(total_batch_err)
# =============================================================================
