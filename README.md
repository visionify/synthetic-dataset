# Synthetic Dataset Generate

This repo generates the sythetic dataset from the sessions image folder created by download_sessions.py from the oosdetection-edge repo.

## Before you run the tracker

1. Clone the repository recursively:

`git clone https://github.com/visionify/synthetic-dataset.git`

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt], including albementation. To install, run:

`pip install -r requirements.txt`

3. To generate the synthtic dataset use:

`python sythetic_dataset_create.py --dataset-folder store-dataset --session-images oos-test-images`

4. To split the dataset use:

`python split_dataset.py --dataset-folder store-dataset --splitted-dataset new-splitted-dataset`
