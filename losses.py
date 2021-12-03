import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import monai
import kornia
from monai.networks.utils import one_hot
import skimage
import skimage.morphology
import pylab as plt
import utils

def torch_dilation(tensor,theta0=1):
    '''
    binary 
    '''
    result = F.max_pool2d(tensor, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    result[result!=0]=1
    return result

def torch_erosion(tensor,theta0=1):
    '''
    binary 
    '''
    result = -F.max_pool2d(- tensor, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
#     result[result!=0]=1
#     print(torch.unique(result))
    return result

def torch_closed(tensor,theta0=1):
    '''
    binary 
    '''
    result = torch_dilate(tensor,theta0)
    result = torch_erosion(result,theta0)
    return result

def torch_skeleton(x, thresh_width=10):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x

# label_dilate = torch_dilate(y,2)
# label_erosion = torch_erosion(y,2)
# label_closed = torch_closed(y,2)

# label_skel = torch_dilate(soft_skeletonize(y),2)

# class twLoss(nn.Module):
#     def __init__(self):
#         super(twLoss, self).__init__()
#         self.kernel = torch.ones(1, 1)
#         self.erosion = kornia.morphology.erosion
        
#     def forward(self,yhat,y):

#         for idx in range():
            
#         x = -max_pool (-x)
#                 # boundary map
#                 gt_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
#         loss = self.ce(yhat,y)
#         return loss 

def skel_iweight(tensor):
    '''
    input shape : B X C X H X W
    '''
    numpy = tensor.cpu().detach().numpy()
    result = np.zeros_like(numpy)
    for idx_b in range(tensor.shape[0]):
        for idx_c in range(tensor.shape[1]):
            skel,dst = skimage.morphology.medial_axis(numpy[idx_b,idx_c], return_distance=True)
            temp = skel*dst
            temp[temp==0] = 255
            result[idx_b,idx_c] = 1/temp
    return torch.tensor(result).cuda()


# class skel_FocalLoss(nn.Module):
#     def __init__(self):
#         super(skel_FocalLoss, self).__init__()
#         self.ce = monai.losses.FocalLoss(to_onehot_y = True, gamma = 2.0)
        
#     def forward(self,yhat,y):
#         loss = self.ce(yhat,y)
        
#         kernel = torch.ones(7,7).cuda()
#         yhat_ = kornia.morphology.dilation(torch_skeleton(yhat),kernel,)
#         y_ = kornia.morphology.dilation(torch_skeleton(y),kernel)
#         loss_iw = self.ce(yhat_,y_)
#         return loss + loss_iw

class skel_FocalLoss(nn.Module):
    def __init__(self, activation_T=1):
        super(skel_FocalLoss, self).__init__()
        self.ce = monai.losses.FocalLoss(to_onehot_y = True, gamma = 2.0)
        self.dice = monai.losses.GeneralizedDiceLoss(to_onehot_y=True, softmax=False)
        self.activation_T = activation_T
#         self.dice = monai.losses.TverskyLoss(to_onehot_y=True, alpha= 0.8, softmax=False)
        
    def forward(self,yhat,y):
        yhat = utils.Activation(yhat, self.activation_T)
        loss_ce = self.ce(yhat,y)
        loss_dice = self.dice(yhat,y)
        
        kernel = torch.ones(3,3).cuda()
        yhat_skel = kornia.morphology.dilation(torch_skeleton(yhat),kernel)
        y_skel = kornia.morphology.dilation(torch_skeleton(y),kernel)
        
        loss_ce_skel = self.ce(yhat_skel,y_skel)
        loss_dice_skel = self.dice(yhat_skel,y_skel)
        return loss_ce + loss_ce_skel + loss_dice + loss_dice_skel

    
class FocalLoss(nn.Module):
    def __init__(self, activation_T=1):
        super(FocalLoss, self).__init__()
        self.activation_T = activation_T
        self.ce = monai.losses.FocalLoss(to_onehot_y = True, gamma = 4.0)
        
    def forward(self,yhat,y):
        yhat = utils.Activation(yhat,self.activation_T)
        loss = self.ce(yhat,y)
        return loss 
    
class BCELoss(nn.Module):
    def __init__(self, activation_T=1):
        super(BCELoss, self).__init__()
        self.activation_T = activation_T
        self.ce = nn.BCELoss()
        
    def forward(self,yhat,y):
        yhat = utils.Activation(yhat,self.activation_T)
        loss = self.ce(yhat,y)
        return loss 

class CELoss(nn.Module):
    def __init__(self, activation_T=1):
        super(CELoss, self).__init__()
        self.activation_T = activation_T
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self,yhat,y):
        yhat = utils.Activation(yhat,self.activation_T)
        y = y[:,0].long()
        loss = self.ce(yhat,y)
        return loss 

class DiceCELoss(nn.Module):
    def __init__(self, activation_T=1):
        super(DiceCELoss, self).__init__()
        self.activation_T = activation_T
        self.dice = monai.losses.GeneralizedDiceLoss(to_onehot_y=True, softmax=False)
        self.ce = CELoss()        
        
    def forward(self,yhat,y):
        yhat = utils.Activation(yhat,self.activation_T)
        dice = self.dice(yhat,y)
        ce = self.ce(yhat,y)
        return dice+ce
    
# Reconstruction loss
class MSELoss(nn.Module):
    def __init__(self):        
        super(MSELoss, self).__init__()
        self.MSE = nn.MSELoss()
        
    def forward(self,yhat,y):
        MSE = self.MSE(yhat,y)
        return MSE
    
class L1Loss(nn.Module):
    def __init__(self):        
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()
        
    def forward(self,yhat,y):
        loss = self.loss(yhat,y)
        return loss

class SSIMLoss(nn.Module):
    def __init__(self):        
        super(SSIMLoss, self).__init__()
        self.loss = kornia.losses.SSIMLoss(9)
        
    def forward(self,yhat,y):
        loss = self.loss(yhat,y)
        return loss

class ReconLoss(nn.Module):
    def __init__(self):        
        super(ReconLoss, self).__init__()
        self.SSIMLoss = kornia.losses.SSIMLoss(11)
#         self.PSNRLoss = kornia.losses.SSIMLoss(11)
        self.L1Loss = nn.L1Loss()
        self.MSELoss = nn.MSELoss()
        
    def forward(self,yhat,y):
#         SSIMLoss = self.SSIMLoss(yhat,y)
        L1Loss = self.L1Loss(yhat,y)
        MSELoss = self.MSELoss(yhat,y)
        return MSELoss+L1Loss

class BoundaryCELoss(nn.Module):
    def __init__(self,activation_T=1):        
        super(BoundaryCELoss, self).__init__()
        self.activation_T = activation_T
        self.ce = CELoss()
        self.boundary = BoundaryLoss()

    def forward(self,yhat,y):
        yhat = utils.Activation(yhat,self.activation_T)
        ce = self.ce(yhat,y) 
        boundary = self.boundary(yhat,y)        
        return ce+boundary

class BoundaryFocalLoss(nn.Module):
    def __init__(self,activation_T=1):        
        super(BoundaryFocalLoss, self).__init__()
        self.activation_T = activation_T
        self.ce = monai.losses.FocalLoss(to_onehot_y = True, gamma = 2.0)
        self.boundary = BoundaryLoss()
        self.dice = monai.losses.GeneralizedDiceLoss(to_onehot_y=True, softmax=False)
#         self.dice = monai.losses.TverskyLoss(to_onehot_y=True, alpha= 0.8, softmax=False)

    def forward(self,yhat,y):
        yhat = utils.Activation(yhat,self.activation_T)
        ce = self.ce(yhat,y) 
        boundary = self.boundary(yhat,y)
        dice =  self.dice(yhat,y)
        
        kernel = torch.ones(3,3).cuda()
        yhat_skel = kornia.morphology.dilation(torch_skeleton(yhat),kernel)
        y_skel = kornia.morphology.dilation(torch_skeleton(y),kernel)
        
        dice_skel = self.dice(yhat_skel,y_skel)
        return ce+boundary#+dice_skel+dice
    
class BoundaryFocalSkelDiceLoss(nn.Module):
    def __init__(self,activation_T=1):        
        super(BoundaryFocalSkelDiceLoss, self).__init__()
        self.activation_T = activation_T
        self.ce = monai.losses.FocalLoss(to_onehot_y = True, gamma = 2.0)
        self.boundary = BoundaryLoss()
        self.dice = monai.losses.GeneralizedDiceLoss(to_onehot_y=True, softmax=False)
#         self.dice = monai.losses.TverskyLoss(to_onehot_y=True, alpha= 0.8, softmax=False)

    def forward(self,yhat,y):
        yhat = utils.Activation(yhat,self.activation_T)
        ce = self.ce(yhat,y) 
        boundary = self.boundary(yhat,y)
#         dice =  self.dice(yhat,y)
        
        kernel = torch.ones(3,3).cuda()
        yhat_skel = kornia.morphology.dilation(torch_skeleton(yhat),kernel)
        y_skel = kornia.morphology.dilation(torch_skeleton(y),kernel)
        
        dice_skel = self.dice(yhat_skel,y_skel)
        return ce+boundary+dice_skel#+dice

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """
#     def __init__(self, theta0=3, theta=3, alpha = 0.7, gamma = 0.75): #DRIVE
#     def __init__(self, theta0=3, theta=15, alpha = 0.7, gamma = 0.75):  #AMC 이거로도 잘되었음
    def __init__(self, theta0=3, theta=15, alpha = 0.7, gamma = 0.75):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, 1, H, w)
        Return:
            - boundary loss, averaged over mini-batch
        """
        
        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = one_hot(gt, c)

        # boundary map
        gt_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        
#         # to check hyper-parameter
#         idx= 0
#         print('boundary_loss')
#         print(torch.unique(gt_b),torch.unique(gt_b_ext))
#         plt.figure(figsize=(24,8))
#         plt.subplot(231);plt.title('gt');plt.imshow(gt[idx,0].cpu().detach().numpy())
#         plt.subplot(232);plt.title('gt_boundary');plt.imshow(gt_b[idx,0].cpu().detach().numpy())
#         plt.subplot(233);plt.title('gt_boundary_ext');plt.imshow(gt_b_ext[0,idx].cpu().detach().numpy())
#         plt.subplot(234);plt.title('pred');plt.imshow(pred[idx,1].cpu().detach().numpy())
#         plt.subplot(235);plt.title('pred_boundary');plt.imshow(pred_b[idx,0].cpu().detach().numpy())
#         plt.subplot(236);plt.title('pred_boundary_ext');plt.imshow(pred_b_ext[idx,0].cpu().detach().numpy())
#         plt.show()
        
        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        smooth = 1e-7
#         original impliment
        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + smooth)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + smooth)
        
        # Boundary F1 Score
        smooth = 1e-7
        BF1 = (2 * P * R) / (P + R + smooth)
#         BF1 = (2 * self.alpha * (1-self.alpha) * P * R + smooth) / (self.alpha*P + (1-self.alpha)*R + smooth)
        # summing BF1 Score for each class and average over mini-batch
#         loss = torch.mean(1 - BF1)
        loss = torch.mean(torch.pow(1 - BF1, self.gamma))
        
        return loss


from sklearn.metrics import *
import numpy as np
import cv2
import torch

# def soft_skeletonize(x, thresh_width=20):
#     '''
#     Differenciable aproximation of morphological skelitonization operaton
#     thresh_width - maximal expected width of vessel
#     '''

#     for i in range(thresh_width):
#         min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
#         contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
#         x = torch.nn.functional.relu(x - contour)
#     return x

# def norm_intersection(center_line, vessel):
#     '''
#     inputs shape  (batch, channel, height, width)
#     intersection formalized by first ares
#     x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
#     '''
#     smooth = 1.
# #     print(center_line.shape,vessel.shape)
# #     clf = center_line.view(*center_line.shape[:2], -1)
# #     vf = vessel.view(*vessel.shape[:2], -1)
# #     print(clf.shape,vf.shape)
#     clf = center_line.flatten()
#     vf = vessel.flatten()
#     intersection = (clf * vf).sum(-1)
#     print(intersection.shape)
#     return (intersection + smooth) / (clf.sum(-1) + smooth)

# def soft_cldice_loss(pred, target, target_skeleton=None):
#     '''
#     inputs shape  (batch, channel, height, width).
#     calculate clDice loss
#     Because pred and target at moment of loss calculation will be a torch tensors
#     it is preferable to calculate target_skeleton on the step of batch forming,
#     when it will be in numpy array format by means of opencv
#     '''
#     if len(pred.shape)==4 and pred.shape[1]>1:
#         pred = torch.argmax(pred,1).unsqueeze(1).float()
#     elif len(pred.shape)==4 and pred.shape[1]==1:
#         pred = pred.float()
            
#     cl_pred = soft_skeletonize(pred)
#     if target_skeleton is None:
#         target_skeleton = soft_skeletonize(target)
        
# #     print('soft_cldice_loss')
# #     plt.figure(figsize=(24,24))
# #     plt.subplot(141)
# #     plt.imshow(target[0,0].cpu().detach())
# #     plt.subplot(142)
# #     plt.imshow(target_skeleton[0,0].cpu().detach())
# #     plt.subplot(143)
# #     plt.imshow(pred[0,0].cpu().detach())
# #     plt.subplot(144)
# #     plt.imshow(cl_pred[0,0].cpu().detach())
# #     plt.show()    
    
#     iflat = norm_intersection(cl_pred, target)
#     tflat = norm_intersection(target_skeleton, pred)
#     intersection = iflat * tflat
#     loss = -((2. * intersection) / (iflat + tflat))
    
#     return loss

class clDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        return soft_cldice_loss(pred,gt)#.squeeze()

import numpy as np
import cv2
import torch

def opencv_skelitonize(img):
    skel = np.zeros(img.shape, np.uint8)
    img = img.astype(np.uint8)
    size = np.size(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel

def dice_loss(pred, target):
    '''
    inputs shape  (batch, channel, height, width).
    calculate dice loss per batch and channel of sample.
    E.g. if batch shape is [64, 1, 128, 128] -> [64, 1]
    '''
    smooth = 1.
    iflat = pred.view(*pred.shape[:2], -1) #batch, channel, -1
    tflat = target.view(*target.shape[:2], -1)
    intersection = (iflat * tflat).sum(-1)
    return -((2. * intersection + smooth) /
              (iflat.sum(-1) + tflat.sum(-1) + smooth))

def soft_skeletonize(x, thresh_width=10):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x*-1, (3, 3), 1, 1)*-1
        contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x

def norm_intersection(center_line, vessel):
    '''
    inputs shape  (batch, channel, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    '''
    smooth = 1.
    # original impliment
#     clf = center_line.view(*center_line.shape[:2], -1)
#     vf = vessel.view(*vessel.shape[:2], -1)
    clf = center_line.view(center_line.shape[0],center_line.shape[1], -1)
    vf = vessel.reshape(vessel.shape[0],vessel.shape[1], -1)
    intersection = (clf * vf).sum(-1)
    return (intersection + smooth) / (clf.sum(-1) + smooth)

def soft_cldice_loss(pred, target, target_skeleton=None):
    '''
    inputs shape  (batch, channel, height, width).
    calculate clDice loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv
    '''
    cl_pred = soft_skeletonize(pred)
    if target_skeleton is None:
        target_skeleton = soft_skeletonize(target)
        
#     plt.figure(figsize=(10,10))
#     plt.subplot(121)
#     plt.title('pred')
#     plt.imshow(pred[0,0])
#     plt.subplot(122)
#     plt.title('cl_pred')
#     plt.imshow(cl_pred[0,0])
#     plt.show()
    
#     plt.figure(figsize=(10,10))
#     plt.subplot(121)
#     plt.title('target')
#     plt.imshow(target[0,0])
#     plt.subplot(122)
#     plt.title('target_skeleton')
#     plt.imshow(target_skeleton[0,0])
#     plt.show()
#     plt.imshow(torch_closed(target_skeleton,2)[0,0])
#     plt.show()
#     print(pred.shape,cl_pred.shape,target.shape,target_skeleton.shape)
    iflat = norm_intersection(cl_pred, target)
    tflat = norm_intersection(target_skeleton, pred)
    intersection = iflat * tflat
    return torch.mean(-((2. * intersection) / (iflat + tflat)))

import cv2 as cv
import numpy as np

import torch
from torch import nn

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
copy pasted from - all credit goes to original authors:
https://github.com/SilmarilBearer/HausdorffLoss
"""

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().detach().numpy())).float().cuda()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().detach().numpy())).float().cuda()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.cpu().numpy(), target.cpu().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss