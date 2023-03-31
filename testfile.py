import torch
import textwrap
import fiftyone as fo
import fiftyone.zoo as foz




downloadedDataset = foz.load_zoo_dataset("coco-2014", max_samples = 50)
session = fo.launch_app(downloadedDataset)
