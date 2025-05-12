import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def get_convs(convs, h_feats, num_heads, num_layers_node):
    if convs == "gat":
        from torch_geometric.nn.conv import GATConv
        node_reps = nn.ModuleList([GATConv(h_feats,h_feats//num_heads,num_heads,add_self_loops=False) for _ in range(num_layers_node)])
        return node_reps
    elif convs == "gcn":
        from torch_geometric.nn.conv import GCNConv
        node_reps = nn.ModuleList([GCNConv(h_feats,h_feats,normalize=False,add_self_loops=False) for _ in range(num_layers_node)])
        return node_reps
    elif convs == "gatv2":
        from torch_geometric.nn.conv import GATv2Conv
        node_reps = nn.ModuleList([GATv2Conv(h_feats,h_feats//num_heads,num_heads,add_self_loops=False) for _ in range(num_layers_node)])
        return node_reps
    elif convs == "transformer":
        from torch_geometric.nn.conv import TransformerConv
        node_reps = nn.ModuleList([TransformerConv(h_feats,h_feats//num_heads,num_heads, beta=True) for _ in range(num_layers_node)])
        return node_reps
    elif convs == "sage":
        from torch_geometric.nn.conv import SAGEConv
        node_reps = nn.ModuleList([SAGEConv(h_feats,h_feats,normalize=False) for _ in range(num_layers_node)])
        return node_reps
    elif convs == "sgc":
        from torch_geometric.nn.conv import SGConv
        node_reps = nn.ModuleList([SGConv(h_feats,h_feats,add_self_loops=False) for _ in range(num_layers_node)])
        return node_reps
    else:
        return 0

class FFN(nn.Module):
    def __init__(self, h_feats):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(h_feats, h_feats*4)
        self.fc2 = nn.Linear(h_feats*4, h_feats)
        self.act = nn.Sequential(nn.SiLU())
        self.ln = nn.LayerNorm(h_feats)
    
    def forward(self,x):
        h = self.act(self.fc1(x))
        x = self.ln(x+self.fc2(h))
        return x

class Compressor(nn.Module):
    def __init__(self,h_feats,num_heads):
        super(Compressor,self).__init__()
        self.num_heads = num_heads
        self.depth = h_feats // num_heads
        self.w_q = nn.Linear(h_feats, h_feats, bias=False)
        self.w_k = nn.Linear(h_feats, h_feats, bias=False)
        self.w_v = nn.Linear(h_feats, h_feats, bias=False)
        self.ln_sq = nn.LayerNorm(h_feats)

    def forward(self,h,cls_samples):
        batch_size = h.shape[0]
        cls_samples = torch.cat([h.unsqueeze(1),cls_samples],dim=1)
        q = self.w_q(h).view(batch_size, -1, self.num_heads, self.depth) 
        k = self.w_k(cls_samples).view(batch_size, -1, self.num_heads, self.depth)
        v = self.w_v(cls_samples).view(batch_size, -1, self.num_heads, self.depth)
        scores = torch.einsum('abcd,abcd->abc',q,k)/(self.depth**0.5)
        attention_weights = F.softmax(scores, dim=1)
        h_sq = torch.einsum('abc,abcd->acd',attention_weights,v).reshape(batch_size,-1)
        return self.ln_sq(h_sq)

class Centralizer(nn.Module):
    def __init__(self,h_feats,num_heads):
        super(Centralizer,self).__init__()
        self.num_heads = num_heads
        self.depth = h_feats // num_heads
        self.w_q = nn.Linear(h_feats, h_feats, bias=False)
        self.w_k = nn.Linear(h_feats, h_feats, bias=False)
        self.w_v = nn.Linear(h_feats, h_feats, bias=False)
        self.ln_fuse = nn.LayerNorm(h_feats)
    
    def forward(self,node,H):
        batch_size = node.shape[0]
        H = H.unsqueeze(0).repeat(batch_size,1,1)
        q = self.w_q(node).view(batch_size, -1, self.num_heads, self.depth) 
        k = self.w_k(H).view(batch_size, -1, self.num_heads, self.depth)
        v = self.w_v(H).view(batch_size, -1, self.num_heads, self.depth)
        scores = torch.einsum('abcd,abcd->abc',q,k)/(self.depth**0.5)
        attention_weights = F.softmax(scores, dim=1)
        node_h = torch.einsum('abc,abcd->acd',attention_weights,v)
        node_h = node_h.reshape(batch_size,-1)
        return self.ln_fuse(node+node_h)

class IFM(nn.Module):
    def __init__(self,in_feats,h_feats,num_heads,num_layers_cls,num_layers_node,num_cls,node_reps,invar,H):
        super(IFM,self).__init__()
        self.invar = invar
        self.in_map = nn.Sequential(
                nn.LayerNorm(in_feats),
                nn.Linear(in_feats,h_feats),
                nn.SiLU(),
                )
        
        self.num_cls = num_cls
        self.num_layers_cls = num_layers_cls
        self.num_layers_node = num_layers_node
        self.invar = invar

        self.cls_reps = nn.ModuleList([Compressor(h_feats,num_heads) for _ in range(num_layers_cls)])
        self.cls_ffns = nn.ModuleList([FFN(h_feats) for _ in range(num_layers_cls)])

        self.node_reps = node_reps
        self.node_lns = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(num_layers_node)])
        self.node_ffns = nn.ModuleList([FFN(h_feats) for _ in range(num_layers_node)])
        
        if H == None:
            self.cls_init = torch.randn(num_cls,h_feats*4)
        else:
            self.cls_init = H
        self.cls_map = nn.Sequential(nn.LayerNorm(in_feats),nn.Linear(in_feats,h_feats),nn.SiLU())

        self.cls_final = nn.Parameter(torch.randn(num_cls,h_feats),requires_grad=False)
        self.centralizer = nn.ModuleList([Centralizer(h_feats,num_heads) for _ in range(num_layers_node)])

        self.final_map = nn.Sequential(nn.Linear(h_feats,h_feats),nn.SiLU(),nn.LayerNorm(h_feats))
        self.classifier = nn.Sequential(nn.Linear(h_feats,num_cls))

    def forward(self,node=None,edge_index=None,cls_samples=None):
        # node:b*d; cls_samples:C*B*d
        # pdb.set_trace()
        node = self.in_map(node)
        if self.training:              
            if self.invar:    
                cls_samples= self.in_map(cls_samples) 
                H = self.cls_init.to(node.device)
                H = self.cls_map(H)
                for i in range(self.num_layers_cls):
                    H = H+self.cls_reps[i](H,cls_samples)
                    H = self.cls_ffns[i](H)
                # H = H + self.gru(H,self.cls_final.data)
                H_d = H.detach()
                for i in range(self.num_layers_node):
                    node = self.centralizer[i](node,H_d)
                    node = node+self.node_lns[i](self.node_reps[i](node,edge_index))
                    node = self.node_ffns[i](node)    
                self.cls_final.data = H.data
                node = self.final_map(node)
                H = self.final_map(H)
                return node,H
            else:
                for i in range(self.num_layers_node):
                    node = node+self.node_lns[i](self.node_reps[i](node,edge_index))
                    node =  self.node_ffns[i](node)
                return self.classifier(node)
           
        else:
            if self.invar:
                H = self.cls_final.data
                for i in range(self.num_layers_node):
                    node = self.centralizer[i](node,H)
                    node = node+self.node_lns[i](self.node_reps[i](node,edge_index))
                    node =  self.node_ffns[i](node)
                node = self.final_map(node)
                H = self.final_map(H)
                return node, H
            else:
                for i in range(self.num_layers_node):
                    node = node+self.node_lns[i](self.node_reps[i](node,edge_index))
                    node =  self.node_ffns[i](node)
                return self.classifier(node)
