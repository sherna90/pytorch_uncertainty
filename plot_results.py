import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns 
import pickle 

params = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [12, 10],
    # Set line widths
    'axes.linewidth' : 0.5,
    'grid.linewidth' : 0.5,
    'lines.linewidth' : 1.,
    # Remove legend frame
    'legend.frameon' : False,
    # Always save as 'tight'
    'savefig.bbox' : 'tight',
    'savefig.pad_inches' : 0.05
   }
mpl.rcParams.update(params)



def plot_entropy(myax, inclass, outclass, label, bins = np.logspace(-8, 0.5, num=30), show_legend = False, show_xlabel=False, show_ylabel=False):
    myax.set_title(str(label), fontsize=12)
    myax.yaxis.grid(True)
    myax.boxplot([inclass, outclass],labels=['in-distribution','out-of-distribution'],vert=True)
    #if show_xlabel:
    #    myax.set_xlabel('Entropy')
    if show_ylabel:
        myax.set_ylabel('Entropy')
    if show_legend:
        myax.legend()

def plot_loss(myax,epochs,loss, label, show_xlabel=False, show_ylabel=False):
    myax.set_title(str(label), fontsize=12)
    myax.yaxis.grid(True)
    myax.plot(epochs,loss)
    if show_xlabel:
        myax.set_xlabel('Epochs')
    if show_ylabel:
        myax.set_ylabel('Negative log-likelihood')



file=open('results/gaussian_resnet34.pickle','rb')
resnet_data=pickle.load(file)
file.close()

file=open('results/gaussian_mobilenet_v3.pickle','rb')
mobilenet_data=pickle.load(file)
file.close()

file=open('results/gaussian_vgg16.pickle','rb')
vgg_data=pickle.load(file)
file.close()


fig, axes = plt.subplots(nrows=1, ncols=3)
plot_entropy(axes[0], resnet_data['train_entropy'][-25:], resnet_data['test_entropy'], 'Resnet34', show_ylabel=True)
plot_entropy(axes[1], mobilenet_data['train_entropy'][-25:], mobilenet_data['train_entropy'], 'MobileNet')
plot_entropy(axes[2], vgg_data['train_entropy'][-25:], vgg_data['train_entropy'], 'VGG16')
fig.subplots_adjust(left=0.2)
plt.savefig('entropy_plots.pdf')
plt.close()

fig, axes = plt.subplots(nrows=1, ncols=3,sharey=True,)
plot_loss(axes[0], range(1,101),resnet_data['train_loss'], 'Resnet34', show_ylabel=True)
plot_loss(axes[1], range(1,101),mobilenet_data['train_loss'], 'MobileNet')
plot_loss(axes[2], range(1,101),vgg_data['train_loss'], 'VGG16')
fig.subplots_adjust(left=0.2)
plt.savefig('nll_plots.pdf')
plt.close()
