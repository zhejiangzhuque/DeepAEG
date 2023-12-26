# DeepAEG
DeepAEG: A model for predicting cancer drug response based on data enhancement and edge-collaborative update strategies
![image](https://github.com/zhejiangzhuque/DeepEAG/blob/main/model.jpg)
# Requirements
[environment prepare](http://www.cnblogs.com/sxdcgaq8080/p/7894828.html)
# Installation
DeepAEG can be downloaded by  

```git clone git@github.com:zhejiangzhuque/DeepAEG.git```  

Installation has been tested in a Linux/MacOS platform.
# Environment
```conda env create -f DeepAEG.yml```

```conda activate DeepAEG```
# Model implementation
## Step 1: gene data Preparing

Four types of raw data are required to generate genomic mutation matrix(the order is: copy number,  Gene expression, Gene methlation, Gene mutation).

Data of the same class were saved as after normalization（[data/CCLE/Result_StandardScaler.csv](https://github.com/zhejiangzhuque/DeepAEG/blob/main/data/CCLE/Result_StandardScaler.csv)）, Four items of data were combined into a four-element group.

The Four types of raw data files can be downloaded from CCLE database.

## Step 2: drug data Preparing and Feature extraction

Each drug in our study will be represented as a graph with nodes and edges, and we collected a total of 221 drugs. Here, we use the [deepchem](https://github.com/deepchem/deepchem) library to extract node features and graphs of drugs.

drugfeature=[node_features adj_np, adj_np_01,smiles_feature]

node_features：features of all atoms within a drug with size 50

adj_np：adjacent list of all atoms within a drug， The elements therein are replaced by the eigenvector of that edge with size 11.

adj_np_01: adjacent list of all atoms within a drug. It denotes the all the neighboring atoms indexs

smiles_feature: The resulting features are extracted directly from SMILES by Transformer

### Selection of feature extraction for data enhancement or not

```python extract_drug_features_auc.py -use_aug True -aug_num 2```

```python extract_drug_features_auc.py -use_aug False```

[-use_aug] Whether to conduct data augmentation (default: Fasle)

[-aug_num] The number of each SMILES recombined into new virtual SMILES (default: 2)

## Step 3: Conduct training
```python run_DeepAEG_newaug2.py```





