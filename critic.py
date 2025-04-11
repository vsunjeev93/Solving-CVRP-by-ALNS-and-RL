import torch
from torch_geometric.data import Data
from GNN import GIN
class critic(torch.nn.Module):
    def __init__(self,num_in_features: int,num_embedding:int,num_GNNs:int,num_out_layers:int):
        super().__init__()
        self.embed=torch.nn.Linear(num_in_features,num_embedding)
        self.GNN_layers=torch.nn.ModuleList()
        for i in range(num_GNNs):
            if i==0:
                self.GNN_layers.append(GIN(num_embedding,num_embedding))
            else:
                self.GNN_layers.append(GIN(num_embedding*2,num_embedding)) #skip connections
        
        out_layers=[]
        for i in range(num_out_layers):
            if i==num_out_layers-1:
                out_layers.append(torch.nn.Linear(num_embedding,1))
            else:
                out_layers.append(torch.nn.Linear(num_embedding,num_embedding))
        self.out_layers=torch.nn.Sequential(*out_layers)
    def forward(self,graph:Data,inf:float=10**6)-> torch.Tensor:
        x=graph.x
        x_in=self.embed(x)
        for i,GNN_layer in enumerate(self.GNN_layers):
            if i==0:
                x=GNN_layer(x_in,graph)
            else:
                x=torch.cat([x_in,x],dim=1)
                x=GNN_layer(x,graph)
        x=x[graph.center_node_index,:]
        x=self.out_layers(x)
        # print(x.size())
        return x
    
if __name__=='__main__':
    from graph_data import data_generator
    batch=data_generator(10,4, instances=2, batch_size=1)
    for graph in batch:
        mask=torch.tensor([True]*5+[False]*4)
        model=critic(6,64,2,3)
        print(model(graph,mask))
            

