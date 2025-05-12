import torch
from torch_geometric.data  import Data
from torch_geometric.datasets import Reddit
from torch_geometric.loader import ClusterData, ClusterLoader
from torch.utils.data import Dataset
from collections import defaultdict

class SubGraphData(Dataset):
    def __init__(self,dataset=None,num_parts=1500,indices=None,train=False):
        super(SubGraphData,self).__init__()
        data = dataset[0]
        cluster_data = ClusterData(data, num_parts=num_parts, recursive=False, save_dir=dataset.processed_dir)
        self.data_list = [cluster_data[i] for i in indices]
        if train:
            self.nodes_by_label = self.get_labeled_nodes(self.data_list)
    
    def get_labeled_nodes(self,data_list):
        nodes_by_label = defaultdict(list)  
        for data in data_list:       
            for i in range(len(data.y)):
                label = int(data.y[i].item())
                nodes_by_label[label].append(data.x[i])
        return nodes_by_label 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,index):
        return self.data_list[index]

import pickle
from tqdm import tqdm

class PreSubGraphData(Dataset):
    def __init__(self,split,mean,std):
        self.mean, self.std = mean, std
        # if split == "train":
        #     self.get_mean_std(split)
        # self.data_list = self.get_data(split)
        self.data_list = torch.load(f"/data/IFM/datasets/coco/{split}.pt")
    
    def label_remap_coco(self):
        # Util function for name 'COCO-SP'
        # to remap the labels as the original label idxs are not contiguous
        original_label_idx = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78,
            79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]

        label_map = {}
        for i, key in enumerate(original_label_idx):
            label_map[key] = i

        return label_map
    def get_data(self,split):
        graph_list = []
        file_path = f"/data/IFM/datasets/coco/{split}.pickle"
        label_map = self.label_remap_coco()
        with open(file_path, "rb") as f:
            graphs = pickle.load(f)
            for graph in tqdm(graphs, desc=f'Processing {split} dataset'):
                x = graph[0].to(torch.float)
                x = (x - self.mean) / self.std
                y = torch.LongTensor(graph[3])
                for i, label in enumerate(y):
                    y[i] = label_map[label.item()]
                edge_index = torch.tensor(graph[2], dtype=torch.long)
                # y = torch.tensor(y, dtype=torch.long)
                data = Data(x=x, edge_index=edge_index, y=y)
                graph_list.append(data)
        torch.save(graph_list, f"/data/IFM/datasets/coco/{split}.pt")
        return graph_list    
    def get_mean_std(self,split):
        node_list = []
        file_path = f"/data/IFM/datasets/coco/{split}.pickle"
        with open(file_path, "rb") as f:
            graphs = pickle.load(f)
            for graph in tqdm(graphs, desc=f'Processing {split} dataset'):
                node_list.append(torch.tensor(graph[0], dtype=torch.float32))
        node = torch.cat(node_list,dim=0)
        mean_vals = node.mean(dim=0)
        std_vals = node.std(dim=0)
        self.mean = mean_vals
        self.std = std_vals        
        return 0
    def get_labeled_nodes(self):
        data_list = self.data_list
        nodes_by_label = defaultdict(list)  
        for data in data_list:       
            for i in range(len(data.y)):
                label = int(data.y[i].item())
                nodes_by_label[label].append(data.x[i])
        return nodes_by_label 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

class SubClassData(Dataset):
    def __init__(self,nodes_by_label,min_length,label = 0):
        super(SubClassData,self).__init__()
        node = nodes_by_label[label]  
        self.features = self.expand_tensor(node,min_length)

    def expand_tensor(self,node,min_length):
        n = len(node)
        if n >= min_length:
            return node
        else:
            extra_samples_count = min_length - n  
            indices = torch.randint(0, n, (extra_samples_count,))
            extra_samples = [node[i] for i in indices]
            expanded_node = node + extra_samples
            return expanded_node

    def __len__(self):
        return len(self.features)

    def __getitem__(self,index):

        return self.features[index].squeeze(0)

def get_indices(num_parts,train_ratio,val_ratio):
    train_size = int(train_ratio * num_parts)
    val_size = int(val_ratio * num_parts)
    test_size = num_parts - train_size - val_size
    all_indices = torch.randperm(num_parts)
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    return train_indices,val_indices,test_indices

from torch_geometric.data import InMemoryDataset
class MyDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None):
        super(MyDataset, self).__init__('.', transform)
        self.data_list = data_list
        self.data, self.slices = self.collate(data_list)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        return data


def get_dataset(data_dir):
    if data_dir == "/data/IFM/datasets/Reddit":
        from torch_geometric.datasets import Reddit
        dataset = Reddit(data_dir)
        num_classes = dataset.num_classes
        return dataset, num_classes
    elif data_dir == "/data/IFM/datasets/AmazonProducts":
        from torch_geometric.datasets import AmazonProducts
        dataset = AmazonProducts(data_dir)
        nodes_by_label = defaultdict(list) 
        data = dataset[0]
        num_classes = dataset.num_classes
        if data.y.dim() != 1:
            data.y = torch.argmax(data.y,dim=1)
            for i in range(len(data.y)):
                label = int(data.y[i].item())
                nodes_by_label[label].append(data.x[i])
            index = [] 
            for i in range(num_classes):
                if len(nodes_by_label[i]) != 0:
                    index.append(i)
            data.y = torch.index_select(data.y, 1, torch.tensor(index))
            data.y = torch.argmax(data.y,dim=1)
            num_classes = len(index)
            dataset.save([data],dataset.processed_paths[0])
        return dataset, num_classes

    else:
        return 0
