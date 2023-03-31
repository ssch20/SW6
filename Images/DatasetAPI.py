import torch
import textwrap
import fiftyone as fo
import fiftyone.zoo as foz

class DatasetApi():

    def __init__(self, nameOfDataset) -> None:
        self.nameOfDataset = nameOfDataset
    
    def listdatasets(self):
        available_datasets = foz.list_zoo_datasets(tags = ["image", "detection" ])
        print(available_datasets)
    
    def viewDataset(self, dataset):
        dataset = 
        
    
    def downloadDataset(self, nameOfDataset: str):
        downloadedDataset = foz.load_zoo_dataset(nameOfDataset, max_samples = 50)
        return downloadedDataset

        
    

    def get_dataset_info(self, nameOfDataset):
        print(foz.ZooDatasetInfo(nameOfDataset))

    



""" 
    Coco_14_dataset = foz.get_zoo_dataset("coco-2014")
    Coco_17_dataset = foz.get_zoo_dataset("coco-2017")

    print("***** Dataset description *****")
    print(textwrap.dedent("    " + Coco_14_dataset.__doc__))

    print("***** Tags *****")
    print("%s\n" % ", ".join(Coco_14_dataset.tags))

    print("***** Supported splits *****")
    print("%s\n" % ", ".join(Coco_14_dataset.supported_splits))

    print(Coco_14_dataset)
    print(Coco_17_dataset)

    dataset = foz.download_zoo_dataset("coco-2014", split="train")
    dataset = foz.download_zoo_dataset("coco-2017", split="train")

    # Visualize the in the App
    session = fo.launch_app(dataset) """