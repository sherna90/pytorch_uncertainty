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



def plot_entropy(myax, inclass, outclass, label, bins = np.logspace(-8, 0.5, num=30), show_legend = False, show_xlabel=False, show_ylabel=False,txt='Entropy'):
    myax.set_title(str(label), fontsize=12)
    myax.yaxis.grid(True)
    myax.boxplot([inclass, outclass],labels=['in-distribution','out-of-distribution'],vert=True)
    #if show_xlabel:
    #    myax.set_xlabel('Entropy')
    if show_ylabel:
        myax.set_ylabel(txt)
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

if __name__ == "__main__":
    file=open('results/gaussian_resnet34.pickle','rb')
    resnet_data=pickle.load(file)
    file.close()

    file=open('results/gaussian_mobilenet_v3.pickle','rb')
    mobilenet_data=pickle.load(file)
    file.close()

    file=open('results/gaussian_vgg16.pickle','rb')
    vgg_data=pickle.load(file)
    file.close()


    file=open('results/eval_resnet34_100.pickle','rb')
    resnet_100_data=pickle.load(file)
    file.close()

    file=open('results/eval_resnet34_10000.pickle','rb')
    resnet_10000_data=pickle.load(file)
    file.close()

    file=open('results/eval_resnet34_100000.pickle','rb')
    resnet_100000_data=pickle.load(file)
    file.close()

    file=open('results/eval_vgg16_100.pickle','rb')
    vgg16_100_data=pickle.load(file)
    file.close()

    file=open('results/eval_vgg16_10000.pickle','rb')
    vgg16_10000_data=pickle.load(file)
    file.close()

    file=open('results/eval_vgg16_100000.pickle','rb')
    vgg16_100000_data=pickle.load(file)
    file.close()


    file=open('results/eval_mobilenet_v3_100.pickle','rb')
    mobilenet_100_data=pickle.load(file)
    file.close()

    file=open('results/eval_mobilenet_v3_10000.pickle','rb')
    mobilenet_10000_data=pickle.load(file)
    file.close()

    file=open('results/eval_mobilenet_v3_100000.pickle','rb')
    mobilenet_100000_data=pickle.load(file)
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

    fig, axes = plt.subplots(nrows=1, ncols=3,sharey=True,label='OSPA')
    plot_entropy(axes[0], resnet_100_data['train_ospa'], resnet_100_data['test_ospa'], 'Resnet34 (100)', show_ylabel=True,txt='OSPA')
    plot_entropy(axes[1], resnet_10000_data['train_ospa'], resnet_10000_data['test_ospa'], 'Resnet34 (10_000)', show_ylabel=True,txt='OSPA')
    plot_entropy(axes[2], resnet_100000_data['train_ospa'], resnet_100000_data['test_ospa'], 'Resnet34 (100_000)', show_ylabel=True,txt='OSPA')
    fig.subplots_adjust(left=0.2)
    plt.savefig('resnet_ospa.pdf')
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=3,sharey=True,label='OSPA')
    plot_entropy(axes[0], vgg16_100_data['train_ospa'], vgg16_100_data['test_ospa'], 'VGG16 (100)', show_ylabel=True,txt='OSPA')
    plot_entropy(axes[1], vgg16_10000_data['train_ospa'], vgg16_10000_data['test_ospa'], 'VGG16 (10_000)', show_ylabel=True,txt='OSPA')
    plot_entropy(axes[2], vgg16_100000_data['train_ospa'], vgg16_100000_data['test_ospa'], 'VGG16 (100_000)', show_ylabel=True,txt='OSPA')
    fig.subplots_adjust(left=0.2)
    plt.savefig('vgg16_ospa.pdf')
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=3,sharey=True,label='OSPA')
    plot_entropy(axes[0], mobilenet_100_data['train_ospa'], mobilenet_100_data['test_ospa'], 'MobileNet (100)', show_ylabel=True,txt='OSPA')
    plot_entropy(axes[1], mobilenet_10000_data['train_ospa'], mobilenet_10000_data['test_ospa'], 'MobileNet (10_000)', show_ylabel=True,txt='OSPA')
    plot_entropy(axes[2], mobilenet_100000_data['train_ospa'], mobilenet_100000_data['test_ospa'], 'MobileNet (100_000)', show_ylabel=True,txt='OSPA')
    fig.subplots_adjust(left=0.2)
    plt.savefig('mobilenet_ospa.pdf')
    plt.close()