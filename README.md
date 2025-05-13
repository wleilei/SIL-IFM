# Subgraph Invariant Learning Towards Large-Scale Graph Node Classification [(AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/35412)

# Getting Started 
```
conda env create -f environment.yml 
```

# Datsets
All the datasets are downloaded using torch_geometric.datasets into /datasets and the processing code is in data.py.

# Model Training
Codes of our model are writen in ifm.py, with synthetic.py for the the synthetic datasets and real-world.py for the real-world datasets. And the hyper-parameters are set in config.py.
