import torch
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.ops import box_convert,remove_small_boxes,box_iou
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1
        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()
        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))
        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]
    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].int()
    return keep

def predict(image_path,model):
    image = read_image(image_path)
    w,h=image.shape[1:]
    with torch.no_grad():
        theta=model(image.unsqueeze(0).to(device))
    m=Normal(theta[0],theta[1])
    boxes=m.sample_n(100).squeeze(1)
    boxes = torch.stack([boxes[:,0]*w,boxes[:,1]*h,boxes[:,2]*w,boxes[:,3]*h],axis=0)
    boxes=torch.transpose(boxes,0,1).to(int)
    boxes=torch.abs(box_convert(boxes, 'xyxy', 'xywh'))
    boxes=box_convert(boxes, 'xywh', 'xyxy')
    scores=-1.0*m.log_prob(boxes).mean(axis=1)
    scores=(scores-torch.min(scores))/(torch.max(scores)-torch.min(scores))
    idx=soft_nms_pytorch(boxes,scores,thresh=0.2,cuda=1)
    boxes=boxes[idx]
    idx=remove_small_boxes(boxes,min_size=300)
    results = draw_bounding_boxes(image, boxes[idx], width=2)
    return results,boxes[idx],scores[idx]


def evaluate_ospa(boxes,labels,order=2):
    boxes=torch.abs(box_convert(boxes, 'xyxy', 'xywh'))
    boxes=box_convert(boxes, 'xywh', 'xyxy')
    cost_matrix=box_iou(boxes,labels).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix,maximize=True)
    # Length of longest set of states
    n = labels.shape[0]
    # Calculate metric
    distance = ((1/n) * np.sum(cost_matrix[row_ind, col_ind]**order))**(1/order)
    return distance

    
def gaussian_nll(y,y_hat):
    m = Normal(y_hat[0],y_hat[1])
    nll=-1.0*m.log_prob(y).mean()
    return nll

def gaussian_entropy(y_hat):
    m = Normal(y_hat[0],y_hat[1])
    entropy=m.entropy().sum()
    return entropy

def regular_theta(theta, mode='180', start=-np.pi/2):
    assert mode in ['360', '180']
    cycle = 2 * np.pi if mode == '360' else np.pi
    theta = theta - start
    theta = theta % cycle
    return theta + start