# The implementation of SHGNN for submission 1106 in KDD 2023.

## Requirements

* python 3.x
* torch == 1.7.1 
* dgl == 0.6.1


## Data
We provide the processed public dataset in the crime prediction (CP) task in our experiments. The raw data can be downloaded from the [NYC open data website](opendata.cityofnewyork.us).


The dataset consists of 4 files:

* urban_graph.dgl - the urban graph with nodes and edges.
* label_array.npy - the ground truth crime count data of each node.
* features.npy - the node features constructed based on the POI data.
* mask.json - a python dict that records the node id in train / val / test set. 


## Model Training
Set hyper-parameters in the file 'script.sh', and then train SHGNN by:
```
sh script.sh
```
The training log and the trained model will be saved in log/ and save_model/ respectively.
