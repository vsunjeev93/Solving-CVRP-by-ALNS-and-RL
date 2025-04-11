import torch
from torch_geometric.data import Data
from GNN import GIN
from torch.distributions import Categorical
from torch_geometric.nn import norm
class actor(torch.nn.Module):
    def __init__(self,num_in_features: int,num_embedding:int,num_GNNs:int,num_out_layers:int,num_actions:int):
        super().__init__()
        self.embed=torch.nn.Linear(num_in_features,num_embedding)
        self.GNN_layers=torch.nn.ModuleList()  # Use ModuleList instead of list
        for i in range(num_GNNs):
            if i==0:
                self.GNN_layers.append(GIN(num_embedding,num_embedding))
            else:
                self.GNN_layers.append(GIN(num_embedding*2,num_embedding)) #skip connections
        
        out_layers=[]
        self.out_layers=torch.nn.ModuleList()
        for i in range(num_out_layers):
            if i==num_out_layers-1:
                out_layers.append(torch.nn.Linear(num_embedding*2,num_actions))
            elif i==0:
                out_layers.append(torch.nn.Linear(num_embedding,num_embedding))
            else:
                out_layers.append(torch.nn.Linear(num_embedding*2,num_embedding))
        self.out_layers.extend(out_layers)
        self.out_layers=torch.nn.Sequential(*out_layers)
        self.bn1=norm.BatchNorm(num_embedding,track_running_stats=False)
    def forward(self,graph:Data,mask:torch.Tensor,inf:float=10**6,greedy=False)-> torch.Tensor:
        x=graph.x
        x_in=self.embed(x)
        x_in=self.bn1(x_in)
        for i,GNN_layer in enumerate(self.GNN_layers):
            if i==0:
                x=GNN_layer(x_in,graph)
            else:
                x=torch.cat([x_in,x],dim=1)
                x=GNN_layer(x,graph)

        x_o=x[graph.center_node_index,:]
        for i, out_layer in enumerate(self.out_layers):
            if i==0:
                x=out_layer(x_o)
            else:
                x=torch.cat([x,x_o],dim=1)
                x=out_layer(x)
        # PyTorch will automatically broadcast mask to match x's shape
        x = x.masked_fill(~mask, -inf)
        # print(x,'logits',x.size())
        x=torch.nn.functional.softmax(x,dim=-1)
        # print(x,'probs')
        m=Categorical(x)
        if not greedy:
            sample = m.sample()
        else:
            sample = torch.argmax(x, dim=1)
        # log_prob=m.log_prob(sample)
        # entropy=torch.bmm(torch.nn.functional.log_softmax(x).unsqueeze(-1).permute(0,2,1),x.unsqueeze(-1))
        # print(m.log_prob(sample),'log_acton',x)
        # assert 1==2
        return sample,m.log_prob(sample)
    
if __name__=='__main__':
    from graph_data import data_generator
    batch=data_generator(10,4, instances=2, batch_size=1)
    for graph in batch:
        mask=torch.tensor([True]*5+[False]*4)
        model=actor(6,64,2,3,9)
        print(model(graph,mask))
            

