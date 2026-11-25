"""Original early tanh MLP/TinyCNN and Yang BatchNorm CNN, fixed CPU training."""
import numpy as np

def network(kind,p,features):
 import torch
 from torch import nn
 if kind=='YangCNN':
  layers=[];prev=1;old=64;stride_available=True
  for width in [64,64,128,128,256]:
   stride=2 if stride_available and width>old else 1
   if stride==2:stride_available=False
   layers.extend([nn.Conv1d(prev,width,3,stride=stride,padding=1,bias=False),nn.BatchNorm1d(width),nn.ReLU()]);prev=width;old=width
  return nn.Sequential(nn.Unflatten(1,(1,features)),*layers,nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(256,1))
 widths=p['architecture']
 class Early(nn.Module):
  def __init__(self):
   super().__init__();self.cnn=kind=='TinyCNN';self.skip=nn.Linear(features,1) if kind=='EarlyResidualMLP' else None;self.layers=nn.ModuleList();self.drop=p.get('dropout',0.)
   if self.cnn:
    a,b,h=widths;self.conv=nn.Sequential(nn.Conv1d(1,a,3,padding=1),nn.Tanh(),nn.Conv1d(a,b,3,padding=1),nn.Tanh());dims=[8*b+features-8,h,1]
   else:dims=[features]+widths+[1]
   for a,b in zip(dims,dims[1:]):self.layers.append(nn.Linear(a,b))
  def forward(self,x):
   if self.cnn:
    z=self.conv(x[:,:8,None].transpose(1,2)).flatten(1)
    if x.shape[1]>8:z=torch.cat([z,x[:,8:]],1)
   else:z=x
   for i,layer in enumerate(self.layers):
    z=layer(z)
    if i<len(self.layers)-1:
     z=torch.tanh(z)
     if self.drop:z=nn.functional.dropout(z,self.drop,self.training)
   return z+(self.skip(x) if self.skip is not None else 0)
 return Early()

class TorchRegressor:
 def __init__(self,kind,params):self.kind=kind;self.params=params
 def fit(self,x,y):
  import torch
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
  self.dim=x.shape[1];self.ym=float(np.mean(y));self.ys=max(float(np.std(y)),1e-8);self.states=[];self.losses=[]
  xx=torch.tensor(x,dtype=torch.float32);yy=torch.tensor((y-self.ym)/self.ys,dtype=torch.float32)[:,None]
  for seed in self.params.get('seeds',list(range(42,47))):
   torch.manual_seed(seed);net=network(self.kind,self.params,self.dim);net.train();opt=torch.optim.Adam(net.parameters(),lr=self.params['lr'],weight_decay=self.params['weight_decay'])
   for _ in range(self.params['epochs']):
    opt.zero_grad();loss=((net(xx)-yy)**2).mean();loss.backward();opt.step()
   if not torch.isfinite(loss):raise ValueError('NONFINITE_NEURAL_LOSS')
   self.losses.append(float(loss.detach()));self.states.append({k:v.detach().numpy().copy() for k,v in net.state_dict().items()})
  return self
 def predict(self,x):
  import torch
  torch.set_num_threads(1);net=network(self.kind,self.params,self.dim);xx=torch.tensor(x,dtype=torch.float32);pp=[]
  with torch.no_grad():
   for state in self.states:
    net.load_state_dict({k:torch.from_numpy(v) for k,v in state.items()});net.eval();pp.append(net(xx).numpy()[:,0])
  return np.mean(pp,axis=0)*self.ys+self.ym
