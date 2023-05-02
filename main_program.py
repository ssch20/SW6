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

from ultralytics import YOLO
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":

    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device('cuda')
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO("yolov8n.pt")    # load a pretrained model (recommended for training)
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="coco.yaml", batch=1, epochs=1)  # Train the model


