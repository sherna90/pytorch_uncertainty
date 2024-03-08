from dataset import *
from model_gaussian import *
from torch.distributions.normal import Normal

def loss(y,y_hat):
    m = Normal(y_hat[0],y_hat[1])
    nll=-1.0*m.log_prob(y).sum()
    return nll
    

data_loader=PennFudanDataLoader(True,32)
model=GaussNet()
num_epochs=100


