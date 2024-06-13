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

def train_model(backbone):
    if torch.backends.cuda.is_built():
        torch.cuda.empty_cache()
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device =  torch.device('cpu') 

    if backbone=='resnet34':
        model=GaussNet(backbone='resnet34')
    elif backbone=='vgg16':
        model=GaussNet(backbone='vgg16')
    elif backbone=='mobilenet_v3':
        model=GaussNet(backbone='mobilenet_v3')
    
    data_loader=TACODataLoader(True,32)
    
    model.to(device)
    num_epochs=100
    optimizer = optim.SGD(model.parameters(), lr=1e-5,momentum=0.9)
    history=list()
    train_entropy=list()
    print('-------------------------------------------')
    for epoch in range(num_epochs):
        train_loss=0.0
        for iter,(input,target) in enumerate(data_loader):
            input=torch.concatenate(input).to(device)
            target=torch.concatenate(target).to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = gaussian_nll(target,output)
            entropy = gaussian_entropy(output)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            if epoch==num_epochs-1:
                train_entropy.append(entropy.item())
        train_loss/=iter
        history.append(train_loss)
        if epoch % (num_epochs//10)==0:
            print("epoch: %d, train loss: %.6f" %(epoch, train_loss))

    print("-----------------------------------------------------------------")
    print("backbone: {0}, train entropy (mean): {1}, train entropy (std): {2}".format(backbone, np.mean(train_entropy),np.std(train_entropy)))
    torch.save(model.state_dict(),''.join(['gaussian_',backbone,'.pth']))

    test_data_loader=UAVVasteDataLoader(False,32)
    test_loss=0.0
    test_entropy=list()
    for iter,(input,target) in enumerate(test_data_loader):
        input=torch.concatenate(input).to(device)
        target=torch.concatenate(target).to(device)
        with torch.no_grad():
            output = model(input)
        loss = gaussian_nll(target,output)
        entropy = gaussian_entropy(output)
        test_entropy.append(entropy.item())
        test_loss+=loss.item()
    test_loss/=iter
    print("backbone: {0}, test entropy (mean): {1}, test entropy (std): {2}".format(backbone, np.mean(test_entropy),np.std(test_entropy)))
    data = {
        'train_loss': history,
        'train_entropy':train_entropy,
        'test_entropy':test_entropy,
        'test_loss': test_loss,
        'epochs': num_epochs
    }
    with open(''.join(['gaussian_',backbone,'.pickle']), 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f)

if __name__ == "__main__":
    models=['resnet34','mobilenet_v3','vgg16']
    for m in models:
        train_model(m)
