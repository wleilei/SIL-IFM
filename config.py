class Config:
    def __init__(self):
        # self.device = 0
        self.data_config_amaz = {
            "data_dir":"/data/IFM/datasets/AmazonProducts",
            "num_parts":2000,
            "train_ratio":0.2, "test_ratio":0.4,
            "min_length":1024, 
            "class_batch_size":512, "num_workers": 1
            }
        self.model_config_amaz = {
            "convs":"transformer",
            "in_feats":200,
            "h_feats":128,
            "num_heads":8,
            "num_layers_cls":12,
            "num_layers_node":12,
            "num_cls":75,
            "invar":True
        }
        self.data_config_voc = {
            "min_length":1024*8, 
            "class_batch_size":1024, "num_workers": 1
            }
        self.model_config_voc = {
            "convs":"transformer",
            "in_feats":14,
            "h_feats":32,
            "num_heads":8,
            "num_layers_cls":12,
            "num_layers_node":12,
            "num_cls":21,
            "invar":True
        }
        self.device = 6
        self.data_config_coco = {
            "min_length":1024*8, 
            "class_batch_size":128, "num_workers": 1
            }
        self.model_config_coco = {
            "convs":"gatv2",
            "in_feats":14,
            "h_feats":64,
            "num_heads":8,
            "num_layers_cls":12,
            "num_layers_node":12,
            "num_cls":81,
            "invar":True
        }
        # self.device = 1
        self.entro = False
        self.data_config_reddit = {
            "data_dir":"/data/IFM/datasets/Reddit",
            "num_parts":400,
            "train_ratio":0.2, "test_ratio":0.4,
            "min_length":1024, 
            "class_batch_size":256, "num_workers": 1
        }
        self.model_config_reddit = {
            "convs":"gcn",
            "in_feats":602,
            "h_feats":256,
            "num_heads":4,
            "num_layers_cls":12,
            "num_layers_node":12,
            "num_cls":41,
            "invar":False
        }
