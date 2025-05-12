import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.basicConfig(filename='/data/IFM/metrics.log', level=logging.INFO)

############## Config Loading ##############
from config import Config
arg = Config()

############## Data Loading ##############
from data import PreSubGraphData,SubClassData, get_indices, get_dataset
from torch_geometric.loader import DataLoader

def load_data(num_classes, min_length, class_batch_size, num_workers):    
    # train_dataloader
    train_data = PreSubGraphData(split="train",mean=None,std=None)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=num_workers, pin_memory=True)
    # val_dataloader
    val_data = PreSubGraphData(split="val",mean=None,std=None)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    # test_dataloader
    test_data = PreSubGraphData(split="test",mean=None,std=None)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    # class_dataloader
    nodes_by_label = train_data.get_labeled_nodes()
    in_feats = train_data[0].x.shape[1]
    H = torch.cat([torch.mean(torch.cat(nodes_by_label[i]).reshape(-1,in_feats),dim=0) for i in range(num_classes)]).reshape(-1,in_feats)
    class_data = [SubClassData(nodes_by_label, min_length=min_length, label=i) for i in range(num_classes)]
    class_dataloader = [DataLoader(data, batch_size=class_batch_size, shuffle=True, pin_memory=True) for data in class_data]
    return train_dataloader, val_dataloader, test_dataloader, class_dataloader,H

############## Model Loading ##############
import torch.nn as nn

def load_model(convs,in_feats,h_feats,num_heads,num_layers_cls,num_layers_node,num_cls,invar,H):    
    from ifm import IFM,get_convs
    node_reps = get_convs(convs, h_feats, num_heads, num_layers_node)
    model = IFM(in_feats=in_feats,h_feats=h_feats,num_cls=num_cls,
                num_heads=num_heads,num_layers_cls=num_layers_cls,num_layers_node=num_layers_node,
               node_reps=node_reps,invar=invar,H=H)
     
    return model

############## Train and Test Setting ##############
from utils import Entropy
import torch.nn.functional as F
import torch.nn as nn
import wandb
import numpy as np
from tqdm import tqdm
from utils import evalution,weighted_cross_entropy
from torchmetrics import AUROC
from torchmetrics.classification import F1Score

criterion = nn.CrossEntropyLoss(reduction='none')

def train(wb,epochs,device,train_dataloader,val_dataloader,class_dataloader,model,optimizer,scheduler,num_classes,invar,model_dir):
    best_f1_mean = 0
    patience = 10
    patience_counter = 0
    epoch = 0
    f1_weighted = F1Score(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    for epoch in range(1,epochs+1):
        model.train()
        train_f1=[]
        # cls_samples = [next(iter(dataloader)).to(device) for dataloader in class_dataloader]
        if invar:
            cls_samples = [next(iter(dataloader)).unsqueeze(0) for dataloader in class_dataloader]
            cls_samples = torch.cat(cls_samples,dim=0).to(device)
        for step,graph in enumerate(tqdm(train_dataloader,desc=f"Epoch:{epoch}/{epochs}",unit='batch')):
            if invar:
                x_n,y_n = graph.x,graph.y
                x_n,y_n = x_n.to(device),y_n.to(device)
                edge_index = graph.edge_index.to(device)
                x_h, H = model(node=x_n,edge_index=edge_index, cls_samples=cls_samples)
                x_h, H_n = F.normalize(x_h, p=2, dim=1), F.normalize(H, p=2, dim=1)
                logits_n = torch.einsum("bcd,cd->bc",x_h.unsqueeze(1),H_n) / 0.07
                # loss_n = criterion(logits_n,y_n)
                loss_n = weighted_cross_entropy(logits_n,y_n)
                entropy = Entropy(x_h)
                loss =  loss_n - entropy
                loss = loss.mean()
            else:
                x_n,y_n = graph.x,graph.y
                x_n,y_n = x_n.to(device),y_n.to(device)
                edge_index = graph.edge_index.to(device)
                x_h = model(node=x_n,edge_index=edge_index) 
                loss = weighted_cross_entropy(x_h,y_n)
                loss = loss.mean()
                logits_n = x_h
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            p = torch.argmax(logits_n, dim=1)
            macro_f1_score = f1_weighted(p, y_n)
            train_f1.append(macro_f1_score.cpu().detach().numpy())
            # torch.cuda.empty_cache()
        train_f1_mean, train_f1_std = np.mean(train_f1), np.std(train_f1)
        print(f"train_macro_f1_mean:{train_f1_mean},train_macro_f1_std:{train_f1_std}")
        torch.cuda.empty_cache()
        model.eval()
        val_f1=[]
        for step,graph in enumerate(tqdm(val_dataloader,desc=f"Epoch:{epoch}/{epochs}",unit='batch')):
            x_n,y_n = graph.x,graph.y
            x_n,y_n = x_n.to(device),y_n.to(device)
            edge_index = graph.edge_index
            edge_index = edge_index.to(device)
            if invar:
                x_h, H = model(node=x_n,edge_index=edge_index)
                x_n, H_n = F.normalize(x_h, p=2, dim=1), F.normalize(H, p=2, dim=1)
                logits = torch.einsum("bcd,cd->bc",x_n.unsqueeze(1),H_n) / 0.07
            else:
                x_h = model(node=x_n,edge_index=edge_index)
                logits = x_h
            p = torch.argmax(logits, dim=1)
            macro_f1_score = f1_weighted(p, y_n)
            val_f1.append(macro_f1_score.cpu().detach().numpy()) 
            torch.cuda.empty_cache()            
        val_f1_mean, val_f1_std = np.mean(val_f1), np.std(val_f1)
        print(f"val_macro_f1_mean:{val_f1_mean},val_macro_f1_std:{val_f1_std}") 
        if best_f1_mean <= val_f1_mean:
            state = {'model': model.state_dict(),'optimizer': optimizer.state_dict()}
            torch.save(state, model_dir)
            best_f1_mean, best_f1_std = val_f1_mean, val_f1_std
            patience_counter = 0 
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    logging.info(f'name: coco-{arg.model_config_coco["convs"]}-400-{invar}, '
                 f'best_f1_mean: {best_f1_mean}, best_f1_std: {best_f1_std}')
    return 0

def test(device,test_dataloader,model,num_classes,invar):
    model.eval()
    f1_weighted = F1Score(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    auroc_weighted = AUROC(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    y_true, y_pred = [],[]
    test_f1,test_auc=[],[]
    for step,graph in enumerate(tqdm(test_dataloader,desc=f"testing",unit='batch')):
        x_n,y_n = graph.x,graph.y
        x_n,y_n = x_n.to(device),y_n.to(device)
        edge_index = graph.edge_index
        edge_index = edge_index.to(device)
        if invar:
            x_h, H = model(node=x_n,edge_index=edge_index)
            x_n, H_n = F.normalize(x_h, p=2, dim=1), F.normalize(H, p=2, dim=1)
            logits = torch.einsum("bcd,cd->bc",x_n.unsqueeze(1),H_n) / 0.07
        else:
            x_h = model(node=x_n,edge_index=edge_index)
            logits = x_h
        p = torch.argmax(logits, dim=1)
        macro_f1_score = f1_weighted(p, y_n)
        auc_ovr_macro = auroc_weighted(logits,y_n)
        test_f1.append(macro_f1_score.cpu().detach().numpy())    
        test_auc.append(auc_ovr_macro.cpu().detach().numpy())   
        torch.cuda.empty_cache()     
    test_f1_mean, test_f1_std = np.mean(test_f1), np.std(test_f1)
    print(f"test_macro_f1_mean:{test_f1_mean},test_macro_f1_std:{test_f1_std}") 
    test_auc_mean, test_auc_std = np.mean(test_auc), np.std(test_auc)
    print(f"test_auc_mean:{test_auc_mean},test_acc_std:{test_auc_std}") 
    logging.info(f'name: {arg.model_config_coco["convs"]}-400-{invar}, '
                 f'test_f1_mean: {test_f1_mean}, test_f1_std: {test_f1_std},'
                 f'test_auc_mean: {test_auc_mean}, test_auc_std: {test_auc_std},')
    return 0



############## Main ##############
import torch
from transformers import get_cosine_schedule_with_warmup
from utils import seed_everything
import pdb

def main():
    wb=False
    seed_everything(42)
    torch.cuda.set_device(arg.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = arg.model_config_coco["num_cls"]
    invar = arg.model_config_coco["invar"]
    train_dataloader, val_dataloader, test_dataloader, class_dataloader, H = load_data(num_classes,**arg.data_config_coco)
    model = load_model(**arg.model_config_coco,H=H)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=5e-4)
    epochs = 100
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=len(train_dataloader)*0.1*epochs,
                                                num_training_steps=len(train_dataloader)*epochs)
    model_dir = f"models/{ arg.model_config_coco['convs']}-{invar}.pth"
    train(wb,epochs,device,train_dataloader,val_dataloader,class_dataloader,model,optimizer,scheduler,num_classes,invar,model_dir)
    model_state = torch.load(model_dir,map_location=device)
    model.load_state_dict(model_state["model"])
    test(device,test_dataloader,model,num_classes,invar)


if __name__ == '__main__':
    main()




















