import torch
import textwrap
import fiftyone as fo
import fiftyone.zoo as foz

from Images.DatasetAPI import DatasetApi



if __name__ == "__main__":
    dataSet = DatasetApi("coco-2014")
    dataSet.listdatasets()
    dataSet.get_dataset_info("coco-2014")





    
    
