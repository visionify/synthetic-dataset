# Synthetic Dataset Generate

This repo generates the sythetic dataset from the sessions image folder created by download_sessions.py from the oosdetection-edge repo.

## Steps to create synthetic dataset

* Clone the repository recursively:

`git clone https://github.com/visionify/synthetic-dataset.git`

`cd synthetic-dataset/`

* Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt], including albumentation. 
To install, run:

`python3 -m pip install -r requirements.txt`

* To generate the synthtic dataset use:

`python3 synthetic_dataset_create.py --dataset-folder synthetic-dataset --session-images <path/to/your/oos-test-images-folder>`

* To split the dataset use:

`python split_dataset.py --dataset-folder synthetic-dataset --splitted-dataset splitted-synthetic-dataset`

Your sythetic dataset for resnet-18 classification training is saved at:

`synthetic-dataset/splitted-sythetic-dataset/`