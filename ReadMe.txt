CS-23-SW-6-22 - How to setup
Specifications with these commands tested:
Processor: I9900k
Graphics card: NVIDIA RTX 3070 TI
Ram: 32 GB 
Disk space needed for both datasets: 51.1 GB
------------------------------------------------------
Step 1. Download repository locally and COCO 2014 and COCO 2017 datasets, Anaconda prompt (needed for commands below).

Step 2. create a folder and rename it "datasets" in the root of the repository if it is not already created. Place datasets in folder, and name them COCO14 and COCO17, in them is the following folders are needed:
annotations: "instances_train<Year>.json", and "instances_val<Year>.json" needs to be included, replace year with 2014 and 2017.
images: train<Year>, val<Year>, and test<Year> folders.
labels: train<Year> and val<Year> folders.

Step 3. Check if train<Year> and val<Year> .txt files in the root of the both datasets are available, if so then proceed to next step. 

Step 4. Make sure that the following files are available in the root of the datasets folder:
COCO14
COCO17
coco14.yaml
coco17.yaml
yolov8n.pt

Step 5. In the Anaconda prompt, navigate to the root of the repository and execude the following command:
pip install -r requirements.txt

Step 6. In the Anaconda prompt, navigate to the datasets folder and if the specifications of your computer is similar in terms of ram, the following command can be executed in the Anaconda prompt: 
yolo mode=train model=yolov8n.pt data=coco14.yaml imgsz=640 epochs=1
or
yolo mode=train model=yolov8n.pt data=coco17.yaml imgsz=640 epochs=1 
if complications happen because of ram (out of memory) then the following command can be added: yolo mode=train model=yolov8n.pt data=coco17.yaml imgsz=640 epochs=1 batch=16

Step 7. After completion of previous step and training the model with 1 epoch, the results of it will appear in the repo/datasets/runs.

Done.