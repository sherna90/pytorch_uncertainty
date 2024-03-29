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
import sys 

def run_test(backbone,num_samples):
    if torch.backends.cuda.is_built():
        torch.cuda.empty_cache()
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device =  torch.device('cpu') 
    order=2

    if backbone=='resnet34':
        model=GaussNet(backbone=backbone)
        model.to(device)
        model.load_state_dict(torch.load('results/gaussian_resnet34.pth'))
    elif backbone=='mobilenet_v3':
        model=GaussNet(backbone=backbone)
        model.to(device)
        model.load_state_dict(torch.load('results/gaussian_mobilenet.pth'))
    elif backbone=='vgg16':
        model=GaussNet(backbone=backbone)
        model.to(device)
        model.load_state_dict(torch.load('results/gaussian_vgg16.pth'))


    data_loader=TACODataLoader(True,32)


    train_data_loader=UAVVasteDataLoader(False,32)
    train_loss=0.0
    train_ospa_error=list()
    for iter,(input,target) in enumerate(train_data_loader):
        for X,y in zip(input,target):
            img=X[0].unsqueeze(0).to(device)
            labels=y.to(device)
            with torch.no_grad():
                theta = model(img)
            m=Normal(theta[0],theta[1])
            boxes=m.sample_n(num_samples).squeeze(1)
            metric=evaluate_ospa(boxes,labels,order)
            train_ospa_error.append(metric)
    print("-----------------------------------------------------------------")
    print("backbone: {0}, num_samples: {1}, train OSPA (mean): {2}, train OSPA (std): {3}".format(backbone,num_samples, np.mean(train_ospa_error),np.std(train_ospa_error)))
    

    test_data_loader=UAVVasteDataLoader(False,32)
    test_loss=0.0
    ospa_error=list()
    for iter,(input,target) in enumerate(test_data_loader):
        for X,y in zip(input,target):
            img=X[0].unsqueeze(0).to(device)
            labels=y.to(device)
            with torch.no_grad():
                theta = model(img)
            m=Normal(theta[0],theta[1])
            boxes=m.sample_n(num_samples).squeeze(1)
            metric=evaluate_ospa(boxes,labels,order)
            ospa_error.append(metric)
    print("-----------------------------------------------------------------")
    print("backbone: {0}, num_samples: {1}, test OSPA (mean): {2}, test OSPA (std): {3}".format(backbone,num_samples, np.mean(ospa_error),np.std(ospa_error)))
        
    data = {
        'test_ospa': ospa_error,
        'num_samples': num_samples,
        'mean_ospa': np.mean(ospa_error),
        'std_ospa': np.std(ospa_error),
        'train_ospa': train_ospa_error,
        'mean_ospa': np.mean(train_ospa_error),
        'std_ospa': np.std(train_ospa_error)
    }

    with open(''.join(['eval_',backbone,'_',str(num_samples),'.pickle']), 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f)       
    
if __name__ == "__main__":
    models=['resnet34','mobilenet_v3','vgg16']
    samples=[100,10_000,100_000]
    for m in models:
        for s in samples:
            run_test(m,s)