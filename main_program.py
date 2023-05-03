"""
Main experiment file
"""

import os
import torch
import torch.utils.data

import torchvision.datasets as dset

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from ultralytics import YOLO

from torchvision.datasets import ImageFolder
import torchvision.transforms as T


class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(32 * 32 * 3, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 10)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


if __name__ == '__main__':
  
  # Model
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  
  # Set fixed random number seed
  torch.manual_seed(42)
  
  # Prepare COCO dataset
  path2data="./datasets/COCO17/images/train2017"
  path2json="./datasets/COCO17/annotations/instances_train2017.json"
  coco_train = dset.CocoDetection(root = path2data,
                                  annFile = path2json)
  print('Number of samples: ', len(coco_train))
  #dataset = pathlib.Path('./datasets/COCO14/', transform=transforms.ToTensor())
  transform = T.ToTensor()
  dataset = ImageFolder(root=path2data, transform=transform)
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
  
  # Initialize the MLP
  mlp = MLP()
  
  # Define the loss function and optimizer
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
  
  # Run the training loop
  for epoch in range(0, 2): # 5 epochs at maximum
    
    # Print epoch
    print(f'Starting epoch {epoch+1}')
    
    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader, 0):
      
      # Get inputs
      inputs, targets = data
      
      # Zero the gradients
      optimizer.zero_grad()
      
      # Perform forward pass
      outputs = mlp(inputs)
      
      # Compute loss
      loss = loss_function(outputs, targets)
      
      # Perform backward pass
      loss.backward()
      
      # Perform optimization
      optimizer.step()
      
      # Print statistics
      current_loss += loss.item()
      if i % 500 == 499:
          print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 500))
          current_loss = 0.0

  # Process is complete.
  print('Training process has finished.')


