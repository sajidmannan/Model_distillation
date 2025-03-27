import torch 
import torch.nn as nn
from torch.autograd import grad 
import torch_geometric 
from torch_geometric.nn.pool import global_add_pool
import sys 
sys.path.append("egnn/")
from models.egnn_clean import egnn_clean as eg


class EGNN_Ener(nn.Module):
    def __init__(
        self, 
        node_input_dim=3, 
        node_output_dim=16, 
        edge_input_dim=3,
        hidden_dim=32, 
        num_layers=2
        ):
        super().__init__()
        self.proj_layer = nn.Linear(3, node_output_dim, bias=False)
        self.egnn = eg.EGNN(
                            in_node_nf=node_input_dim, 
                            hidden_nf=hidden_dim, 
                            out_node_nf=node_output_dim, 
                            in_edge_nf=edge_input_dim, 
                            n_layers=num_layers
                        )
        self.linear = nn.Linear(node_output_dim, 3, bias=False)
        # self.linear = nn.Linear(3, 3, bias=False)


    def forward(self, batch, device):
        h = batch['node_attrs']
        x_orig = batch['positions'].to(device) 
        x_proj = self.proj_layer(x_orig)    # project coordinates into embeddings 
        edges = [row for row in batch['edge_index']]
        edge_attr = x_orig[edges[0]]-x_orig[edges[1]]
        h_new, x_new = self.egnn(h, x_proj, edges, edge_attr)
        pred_forces = self.linear(x_new)
        return pred_forces