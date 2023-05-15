import torch
from ultralytics import YOLO

if __name__ == "__main__":

    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = YOLO('yolov8n.yaml')  # build a new model from YAML
    model = YOLO("yolov8n.pt")    # load a pretrained model (recommended for training)
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights