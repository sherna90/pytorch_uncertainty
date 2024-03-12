from dataset import *
from model_gaussian import *
from torch.distributions.normal import Normal
from torch import optim
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.ops import box_convert,remove_small_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
import numpy as np
import pickle 
from utils import *

if torch.backends.cuda.is_built():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_built():
    device = torch.device('mps')
else:
    device =  torch.device('cpu') 

data_loader=TACODataLoader(True,16)
model=GaussNet(backbone='resnet34')
model.to(device)
num_epochs=10
optimizer = optim.SGD(model.parameters(), lr=1e-5,momentum=0.9)
history=list()
for epoch in range(num_epochs):
    train_loss=0.0
    for iter,(input,target) in enumerate(data_loader):
        input=torch.concatenate(input).to(device)
        target=torch.concatenate(target).to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = gaussian_nll(target,output)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    train_loss/=iter
    history.append(train_loss)
    if epoch % (num_epochs//10)==0:
        print("epoch: %d, train loss: %.6f" %(epoch, train_loss))


model.save(model.state_dict(),'gaussian_resnet34.pth')

test_data_loader=UAVVasteDataLoader(False,32)
test_loss=0.0
for iter,(input,target) in enumerate(data_loader):
    input=torch.concatenate(input).to(device)
    target=torch.concatenate(target).to(device)
    with torch.no_grad():
        output = model(input)
    loss = gaussian_nll(target,output)
    test_loss+=loss.item()
test_loss/=iter

data = {
    'train_loss': history,
    'test_loss': test_loss,
    'epochs': num_epochs
}
with open('gaussian_resnet34.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f)

#results,boxes,scores=predict("PennFudanPed/PNGImages/FudanPed00001.png",model)
#show(results)
#plt.show()
