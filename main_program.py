"""
Main experiment file. Code adapted from TokenCut: https://github.com/YangtaoWANG95/TokenCut
"""

import os
import argparse
import random
import pickle

import torch
import datetime
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from PIL import Image
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

from ultralytics import YOLO
import matplotlib.pyplot as plt
import time

experiment = Experiment(
  api_key = "sOsGqgERgDT04PBLtSGQgU91t",
  project_name = "p6",
  workspace="j2kjonas"
)

# Report multiple hyperparameters using a dictionary:
hyper_params = {
   "learning_rate": 0.5,
   "steps": 100000,
   "batch_size": 50,
}
experiment.log_parameters(hyper_params)

# Initialize and train your model
# model = TheModelClass()
# train(model)

# Seamlessly log your Pytorch model
log_model(experiment, model_name="TheModel")


if __name__ == "__main__":

    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device('cuda')
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO("yolov8n.pt")    # load a pretrained model (recommended for training)
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="coco.yaml", batch=1, epochs=1)  # Train the model


