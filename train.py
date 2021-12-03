import warnings
warnings.filterwarnings(action='ignore')

import argparse
from argparse import ArgumentParser, Namespace

import os, random, glob
from natsort import natsorted
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import kornia 

import pytorch_lightning as pl

import nets
import losses
import datasets
import utils
import wandb
import torchmetrics

import multiprocessing
# print(multiprocessing.cpu_count())

import monai
from monai.inferers import sliding_window_inference

class SegModel(pl.LightningModule):
    def __init__(self,  project: str,
                        data_dir: str,
                        activation_T: int = 1,
                        batch_size: int = 1,
                        data_module = 'dataset',
                        data_padsize= None,
                        data_cropsize= None,
                        data_resize= None,
                        data_patchsize= None,
                        experiment_name = None,
                        gpus = -1,
                        lossfn = 'CELoss',
                        lr = 1e-3,
                        net_name = 'unet',
                        net_norm = 'batch',
                        net_encoder_name = 'resnet50',
                        net_inputch = 1,
                        net_outputch = 2,
                        net_bayesian=None,
                        net_ckpt = None,
                        net_nnblock=False,
                        net_rcnn=False,
                        net_skipatt=False,
                        net_supervision=False,
                        net_wavelet=False,
                        precision = 32,
                        **kwargs):
                
        super().__init__(**kwargs)
        self.activation_T = activation_T 
        self.data_module = data_module
        self.data_padsize = data_padsize
        self.data_cropsize = data_cropsize
        self.data_resize = data_resize
        self.data_patchsize = data_patchsize
        self.net_name = net_name
        self.net_inputch = net_inputch
        self.net_outputch = net_outputch
        self.net_encoder_name = net_encoder_name
        self.net_bayesian = net_bayesian
        self.net_norm = net_norm
        self.net_nnblock = net_nnblock
        self.net_rcnn = net_rcnn
        self.net_skipatt = net_skipatt
        self.net_supervision = net_supervision
        self.net_wavelet = net_wavelet
        self.precision = precision
        self.project = project
        self.lr = lr
        
        # loss       
        fn_call = getattr(losses, lossfn)
        self.lossfn = fn_call(activation_T = self.activation_T)

        # net
        fn_call = getattr(nets, net_name)
        if 'smp' in net_name:
            self.net = fn_call(net_inputch=self.net_inputch, net_outputch=self.net_outputch, net_encoder_name = self.net_encoder_name, net_bayesian = self.net_bayesian)
        elif 'R2AttU' in net_name:
            self.net = fn_call(net_inputch=self.net_inputch, net_outputch=self.net_outputch, net_bayesian = self.net_bayesian)
        elif 'monai' in net_name:
            self.net = fn_call(net_inputch=self.net_inputch, net_outputch=self.net_outputch, net_bayesian = self.net_bayesian) 
        else:
            self.net = fn_call(net_inputch=self.net_inputch, 
                               net_outputch=self.net_outputch, 
                               net_skipatt = self.net_skipatt,
                               net_nnblock=self.net_nnblock, 
                               net_rcnn = self.net_rcnn,
                               net_supervision = self.net_supervision,
                               net_wavelet = self.net_wavelet)

        if self.net_norm == 'instance':
            self.net = nets.bn2instance(self.net)
        elif self.net_norm == 'group':
            self.net = nets.bn2group(self.net)
      
#         if self.net_activation == 'leakyrelu':
#             self.net = nets.relu2lrelu(self.net)
#         elif self.net_activation == 'gelu':
#             self.net = nets.relu2gelu(self.net)
                    
        # metric
        self.metric = metric = monai.metrics.DiceMetric(include_background=False, reduction='mean_channel', get_not_nans=False)
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x,y,mask = batch['x'], batch['y'], batch['mask']

        yhat = self(x)
        if isinstance(yhat,tuple):
            yhat, xhat = yhat
            xhat = F.sigmoid(xhat)
#             xhat = 1-xhat
    
#             loss_reconstruction = F.binary_cross_entropy_with_logits(xhat, x) 
            loss_reconstruction = F.binary_cross_entropy(xhat, x) 
            self.logger.experiment.log({'images_recon_train' : [wb_image(x, 'x'), wb_image(xhat, 'xhat')]})

        def torch_delete(tensor, indices):
            mask = torch.ones(len(tensor), dtype=torch.bool)
            for idx in range(len(indices)):
                if indices[idx] == 0:
                    mask[idx] = False
            return tensor[mask]
        
#         print('before',y.shape,yhat.shape)
        y = torch_delete(y,mask)
        yhat = torch_delete(yhat,mask)
#         print('after',y.shape,yhat.shape)
        
        if len(y)>=1:
#             yhat = utils.Activation(yhat)
            self.logger.experiment.log({'segmentation_train' : wb_mask(x, yhat, y),})
            try:
                loss = self.lossfn(yhat, y) + loss_reconstruction
            except:
                loss = self.lossfn(yhat, y)                
        else:
            loss = loss_reconstruction
    
        self.log('loss', loss)
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        x,y,mask = batch['x'], batch['y'], batch['mask']
        
#         yhat = self(x) # changed to sliding window method
        def predictor(x, return_idx=0): # in case of prediction is type of list
            result = self.net(x)
            if isinstance(result, list) or isinstance(result, tuple):
                return result[return_idx]
            else:
                return result

        roi_size = int(self.data_patchsize) if len(self.data_patchsize.split('_'))==1 else (int(self.data_patchsize.split('_')[0]),int(self.data_patchsize.split('_')[1]))
        yhat = sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=4, predictor=predictor, overlap=0.75, mode='constant')
#         yhat = utils.Activation(yhat)
        loss = self.lossfn(yhat, y)     
        
        if 'Rec' in self.net_name:
            def predictor(x, return_idx=1): # in case of prediction is type of list
                result = self.net(x)
                if isinstance(result, list) or isinstance(result, tuple):
                    return result[return_idx]
                else:
                    return result
                
            xhat = sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=4, predictor=predictor, overlap=0.75, mode='constant')
            xhat = F.sigmoid(xhat)
#             xhat = 1-xhat
            
#             loss_reconstruction = F.mse_loss(xhat, x)
            loss_reconstruction = F.binary_cross_entropy(xhat, x) 
            self.logger.experiment.log({'images_recon_val' : [wb_image(x, 'x'), wb_image(xhat, 'xhat')]})
            loss = loss + loss_reconstruction

        if yhat.shape[1] > 1:
            # multi-class
            yhat_temp = monai.networks.utils.one_hot(torch.argmax(yhat,1).unsqueeze(1), num_classes=self.net_outputch)
            y_temp = monai.networks.utils.one_hot(y, num_classes=self.net_outputch)

            metric = self.metric(yhat_temp,y_temp)[0][-1]
        else:
            # single-class
            metric = self.metric(yhat.round(),y.round())[0][-1]
            
        self.log('loss_val', loss, prog_bar=True)
        self.log('metric_val', metric)
        self.logger.experiment.log({'segmentation_val' : wb_mask(x, yhat, y),})
#                   'images_val' : [wb_image(x, 'x'), wb_image(y, 'y'), wb_image(yhat, 'yhat')]})
    
        return {'loss_val': loss}    

    def test_step(self, batch, batch_idx):
        x,y,mask = batch['x'], batch['y'], batch['mask']
        
#         yhat = self(x) # changed to sliding window method
        def predictor(x, return_idx=0): # in case of prediction is type of list
            result = self.net(x)
            if isinstance(result, list) or isinstance(result, tuple):
                return result[return_idx]
            else:
                return result

        roi_size = int(self.data_patchsize) if len(self.data_patchsize.split('_'))==1 else (int(self.data_patchsize.split('_')[0]),int(self.data_patchsize.split('_')[1]))
        yhat = sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=4, predictor=predictor, overlap=0.75, mode='constant')
#         yhat = utils.Activation(yhat)
        loss = self.lossfn(yhat, y)     
        
        if 'Rec' in self.net_name:
            def predictor(x, return_idx=1): # in case of prediction is type of list
                result = self.net(x)
                if isinstance(result, list) or isinstance(result, tuple):
                    return result[return_idx]
                else:
                    return result
                
            xhat = sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=4, predictor=predictor, overlap=0.75, mode='constant')
            xhat = F.sigmoid(xhat)
#             xhat = F.sigmoid(1-xhat)
                
#             loss_reconstruction = F.mse_loss(xhat, x)
            loss_reconstruction = F.binary_cross_entropy(xhat, x) 
            loss = loss + loss_reconstruction

        if yhat.shape[1] > 1:
            # multi-class
            yhat_temp = monai.networks.utils.one_hot(torch.argmax(yhat,1).unsqueeze(1), num_classes=self.net_outputch)
            y_temp = monai.networks.utils.one_hot(y, num_classes=self.net_outputch)

            metric = self.metric(yhat_temp,y_temp)[0][-1]
        else:
            # single-class
            metric = self.metric(yhat.round(),y)[0][-1]
        return {'loss_test': loss, 'metric_test':metric, 'yhat':yhat}    
    
    def test_epoch_end(self, outputs):
        metrics = list()
        yhats = list()
        
        for output in outputs:
            yhat = output['yhat']
            metric = output['metric_test']
            metrics.append(metric)
        
        yhats = torch.tensor(yhats)
        metrics = torch.tensor(metrics)
        print('metric_mean:', torch.mean(metrics))
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        
        x,y = batch['x'], batch['y']
        
#         yhat = self(x) # changed to sliding window method
        def predictor(x, return_idx=0): # in case of prediction is type of list
            result = self.net(x)
            if isinstance(result, list) or isinstance(result, tuple):
                return result[return_idx]
            else:
                return result

        roi_size = int(self.data_patchsize) if len(self.data_patchsize.split('_'))==1 else (int(self.data_patchsize.split('_')[0]),int(self.data_patchsize.split('_')[1]))
        yhat = sliding_window_inference(inputs=x, roi_size=roi_size, sw_batch_size=4, predictor=predictor, overlap=0.5, mode='constant')
#         yhat = utils.Activation(yhat)
        
        return {'x':x,'y':y,'yhat':yhat}
        
    
    def configure_optimizers(self):
        """
        mode : lr is given --> Adam with lr with given lr
        mode : lr is not given --> CosineAnnealingWarmup (default), SGD with varying lr
        """
        if self.lr > 1e-4:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=20, min_lr=6e-6)

            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': scheduler,'monitor': 'loss_val'}}
        else:
            self.lr = 1e-5
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

            return {'optimizer': optimizer}
    
    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        def str2bool(v):
            if v == 'None':
                return None
            elif v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                return v
            
        parser.add_argument("--project", type=str, help="wandb project name, this will set your wandb project")
        parser.add_argument("--data_dir", type=str, help="path where dataset is stored, subfolders name should be x_train, y_train")
        parser.add_argument("--data_module", type=str,default='dataset', help="Data Module, see datasets.py")
        parser.add_argument("--data_padsize", type=str2bool, default=None, help="input like this 'height_width' : pad - crop - resize - patch")
        parser.add_argument("--data_cropsize", type=str2bool, default=None, help="input like this 'height_width' : pad - crop - resize - patch")
        parser.add_argument("--data_resize", type=str2bool, default=None, help="input like this 'height_width' : pad - crop - resize - patch")
        parser.add_argument("--data_patchsize", type=str2bool, default=None, help="input like this 'height_width' : pad - crop - resize - patch: recommand 'A * 2^n'")
        parser.add_argument("--batch_size", type=int, default=None, help="batch_size, if None, searching will be done")
        parser.add_argument("--lossfn", type=str2bool, default='CE', help="class of the loss function[CELoss, DiceCELoss, MSE, ...], see losses.py")
        parser.add_argument("--net_name", type=str2bool, default='smp_unet', help="Class of the Networks, see nets.py")
        parser.add_argument("--net_inputch", type=int, default=1, help='dimensions of network input channel, see nets.py')
        parser.add_argument("--net_outputch", type=int, default=2, help='dimensions of network output channel, see nets.py')
        parser.add_argument("--net_bayesian", type=float, default=0.2, help='Dropout in the Bottleneck, see nets.py')
        parser.add_argument("--net_norm", type=str2bool, default='batch', help='net normalization, [batch,instance,group], see nets.py')          
        parser.add_argument("--net_ckpt", type=str2bool, default=None, help='path to checkpoint, ex) logs/[PROJECT]/[ID]')         
        parser.add_argument("--activation_T", type=int, default=1, help='Temperature of activation')        
        parser.add_argument("--net_nnblock", type=str2bool, default=False, help='nnblock')              
        parser.add_argument("--net_rcnn", type=str2bool, default=False, help='net_rcnn')      
        parser.add_argument("--net_skipatt", type=str2bool, default=False, help='net_skipatt')      
        parser.add_argument("--net_supervision", type=str2bool, default=False, help='supervision')      
        parser.add_argument("--net_wavelet", type=str2bool, default=False, help='net_wavelet')        
        parser.add_argument("--net_encoder_name", type=str, default='resnet50', help='encoder_name of segmentation_model_pytorch')
        parser.add_argument("--precision", type=int, default=32, help='amp will be set when 16 is given')
        parser.add_argument("--lr", type=float, default=1e-3, help="Set learning rate of Adam optimzer.")        
        parser.add_argument("--experiment_name", type=str, default=None, help='Postfix name of experiment')         
        return parser

class MyDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "path/to/dir", 
                       data_module ='dataset', 
                       batch_size: int = 1, 
                       data_padsize = None, 
                       data_cropsize= None, 
                       data_resize= None, 
                       data_patchsize = None, 
                       num_workers: int = int(multiprocessing.cpu_count()/8)):

        super().__init__()
        self.data_dir = data_dir
        self.data_module = data_module
        self.batch_size = batch_size 
        self.data_padsize = data_padsize
        self.data_cropsize = data_cropsize
        self.data_resize = data_resize
        self.data_patchsize = data_patchsize
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        
        fn_call = getattr(datasets, self.data_module)
        self.trainset = fn_call(self.data_dir, 
                                'train',
                                transform_spatial = datasets.augmentation_imagesize(data_padsize = self.data_padsize,
                                                                                 data_cropsize = self.data_cropsize,
                                                                                 data_resize = self.data_resize,
                                                                                 data_patchsize = self.data_patchsize,), 
                                transform=datasets.augmentation_train(),
#                                 transform=datasets.augmentation_valid(),
                                adaptive_hist_range= False)
        
        self.validset = fn_call(self.data_dir, 
                                'valid',
                                transform_spatial = datasets.augmentation_imagesize(data_padsize = self.data_padsize,
                                                                                 data_cropsize = self.data_cropsize,
                                                                                 data_resize = self.data_resize), 
                                transform=datasets.augmentation_valid(),
                                adaptive_hist_range= False)
        
        self.testset = fn_call(self.data_dir, 
                               'test',
                                transform_spatial = datasets.augmentation_imagesize(data_padsize = self.data_padsize,
                                                                                 data_cropsize = self.data_cropsize,
                                                                                 data_resize = self.data_resize), 
                                transform=datasets.augmentation_valid(),
                                adaptive_hist_range= False)
        
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size)


# wandb image visualization
segmentation_classes = ['black', 'class1', 'class2', 'class3']

def labels():
    l = {}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    return l

def wb_mask(x, yhat, y, samples=4):
    
    x = torchvision.utils.make_grid(x[:samples].cpu().detach(),normalize=True).permute(1,2,0)
    y = torchvision.utils.make_grid(y[:samples].cpu().detach()).permute(1,2,0).round()
    yhat = torchvision.utils.make_grid(torch.argmax(yhat[:samples],1).unsqueeze(1).cpu()).permute(1,2,0) if yhat.shape[1]>1 else \
           torchvision.utils.make_grid(yhat[:samples].round().cpu()).permute(1,2,0)

    x = (x*255).numpy().astype(np.uint8)        # 0 ~ 255
    yhat = yhat[...,0].numpy().astype(np.uint8) # 0 ~ n_class 
    y = y[...,0].numpy().astype(np.uint8)       # 0 ~ n_class
    
    return wandb.Image(x, masks={
    "prediction" : {"mask_data" : yhat, "class_labels" : labels()},
    "ground truth" : {"mask_data" : y, "class_labels" : labels()}})

def wb_image(x, caption, samples=4):
    if x.shape[1] != 1 and x.shape[1] !=3:
        x = torch.argmax(x,1).unsqueeze(1)
    if x.max() <= 1:
        x = (x*255).long()
    else:
        if len(torch.unique(x))<5 and x.max()<10:
            scale = 255//x.max()
            x = (x*scale).long()
    try:
        x = torchvision.utils.make_grid(x[:samples].cpu().detach(),normalize=True).permute(1,2,0)
    except:
        x = torchvision.utils.make_grid(x[:samples].cpu().detach()).permute(1,2,0)

    x = x.numpy()
    x = x.astype(np.uint16)
        
    return wandb.Image(x, caption = caption)

def main(args: Namespace):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SegModel(**vars(args))
    if args.net_ckpt is not None:
        ckpt = natsorted(glob.glob(args.net_ckpt+'/**/*.ckpt'))
        model = SegModel.load_from_checkpoint(checkpoint_path = ckpt[-1], strict=False, **vars(args))
        print(ckpt[-1],'is loaded')
    assert args.project != None, "You should set wandb-logger project name by option --project [PROJECT_NAME]"
    print('project', args.project)
    
    # ------------------------
    # 2 SET LOGGER
    # ------------------------    
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor, StochasticWeightAveraging, LambdaCallback, EarlyStopping
    
    args.experiment_name = "Dataset{}_Net{}_NetEncoder{}_Loss{}_Precision{}_Patchsize{}_Prefix{}_"\
    .format(args.data_dir.split('/')[-1], args.net_name, args.net_encoder_name, args.lossfn, args.precision, args.data_patchsize, args.experiment_name)
    print('Current Experiment:',args.experiment_name,'\n','*'*100)
    
    os.makedirs('logs',mode=0o777, exist_ok=True)
    wb_logger = pl_loggers.WandbLogger(save_dir='logs/', name=args.experiment_name, project=args.project) #, log_model = "all")
    wb_logger.log_hyperparams(args)
    # wb_logger.watch(model,log="all", log_freq=100)
    wb_logger.watch(model, log_graph=False)
    Checkpoint_callback = ModelCheckpoint(verbose=True, 
                                          monitor='loss_val',
                                          mode='min',
                                          filename='{epoch:04d}-{loss_val:.4f}-{metric_val:.4f}',
                                          save_top_k=1,)

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer.from_argparse_args(args,
                                            amp_backend='native',
                                            auto_scale_batch_size='power',
                                            callbacks=[Checkpoint_callback,
                                                       LearningRateMonitor(),
                                                       StochasticWeightAveraging(),
                                                       EarlyStopping(monitor='loss_val', patience=200),
                                                      ],
                                            # deterministic=True,
                                            deterministic=False,
                                            gpus = -1,
                                            logger = wb_logger,
                                            log_every_n_steps=1,
                                            max_epochs = 2000,
                                            num_processes = 0,
                                            # stochastic_weight_avg = True,
                                            sync_batchnorm = True,
                                            weights_summary = 'top', 
                                           )
    
    myData = MyDataModule.from_argparse_args(args)
    if args.batch_size == None:
        trainer.tune(model,datamodule=myData)
    trainer.fit(model,datamodule=myData)
    
    # ------------------------
    # 5 START TRAINING
    # ------------------------
if __name__ == '__main__':
    
    parser = ArgumentParser(add_help=False)
    parser = SegModel.add_model_specific_args(parser)
    
    args = parser.parse_args()
    print('args:',args,'\n')
    
    main(args) 