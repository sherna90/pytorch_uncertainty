from dataset import *
from model_gaussian import *
from torch.distributions.normal import Normal
from torch import optim
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
import numpy as np

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def nll(y,y_hat):
    m = Normal(y_hat[0],y_hat[1])
    nll=-1.0*m.log_prob(y).mean()
    return nll

if torch.backends.cuda.is_built():
    device = torch.device('cuda:0')
else:
    device =  torch.device('cpu') 

data_loader=PennFudanDataLoader(True,32)
model=GaussNet()
model.to(device)
num_epochs=100
optimizer = optim.SGD(model.parameters(), lr=1e-5,momentum=0.9)
history=list()
for epoch in range(num_epochs):
    train_loss=0.0
    for iter,(input,target) in enumerate(data_loader):
        input=torch.concatenate(input).to(device)
        target=torch.concatenate(target).to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = nll(target,output)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    train_loss/=iter
    history.append(train_loss)
    if epoch % (num_epochs//10)==0:
        print("epoch: %d, train loss: %.6f" %(epoch, train_loss))



image = read_image("PennFudanPed/PNGImages/FudanPed00046.png")
w,h=image.shape[1:]
with torch.no_grad():
    theta=model(image.unsqueeze(0).to(device))
m=Normal(theta[0],theta[1])
boxes=m.sample_n(10).squeeze(1)
boxes = torch.stack([boxes[:,0]*w,boxes[:,1]*h,boxes[:,2]*w,boxes[:,3]*h],axis=0)
boxes=torch.transpose(boxes,0,1).to(int)
results = draw_bounding_boxes(image, boxes, width=5)
