import warnings
warnings.filterwarnings(action='ignore')

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

import pytorch_lightning as pl

import monai
import nets
import losses
import datasets
import utils
import wandb
import torchmetrics

from monai.inferers import sliding_window_inference

class SegModel(pl.LightningModule):
    def __init__(self,  data_dir: str,
                        project: str,
                        batch_size: int = 1,
                        data_module = 'dataset',
                        data_padsize= None,
                        data_cropsize= None,
                        data_resize= None,
                        data_patchsize= None,
                        experiment_name = None,
                        gpus = -1,
                        lossfn = 'CELoss',
                        lr = None,
                        net = 'unet_eb5_batch',
                        net_inputch = 1,
                        net_outputch = 1,
                        net_norm = 'batch',
                        net_ckpt = None,
                        precision = 32,
                        **kwargs):
                
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.data_module = data_module
        self.data_padsize = data_padsize
        self.data_cropsize = data_cropsize
        self.data_resize = data_resize
        self.data_patchsize = data_patchsize
        self.net_inputch = net_inputch
        self.net_outputch = net_outputch
        self.net_norm = net_norm
        self.net_ckpt = net_ckpt
        self.precision = precision
        self.project = project
        self.lr =lr
        
        # loss       
        fn_call = getattr(losses, lossfn)
        self.lossfn = fn_call()

        # net
        fn_call = getattr(nets, net)
        self.net = fn_call(net_inputch=self.net_inputch, net_outputch=self.net_outputch)
        
        if self.net_norm == 'instance':
            self.net = nets.bn2instance(self.net)
            print('net_norms were replaced to instance normalizations')
        elif self.net_norm == 'group':
            self.net = nets.bn2group(self.net)
            print('net_norms were replaced to group normalizations')           
        
        # metric
        self.metric = torchmetrics.F1(self.net_outputch)
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x,y  = batch['x'], batch['y']

        yhat = self(x)
        yhat = utils.Activation(yhat)
        loss = self.lossfn(yhat, y)
        metric = self.metric(torch.argmax(yhat,1).cpu().int().flatten(),y.cpu().int().flatten())

        self.log('loss', loss, prog_bar=True)
        self.log('metric', metric, prog_bar=True)
        self.logger.experiment.log({'image_train' : wb_mask(x, yhat, y)}) # wandb.log({'train' : wb_mask(x, yhat, y)})
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        x,y  = batch['x'], batch['y']
        
#         yhat = self(x) # changed to sliding window method
        roi_size = int(self.data_patchsize) if len(self.data_patchsize.split('_'))==1 else (int(self.data_patchsize.split('_')[0]),int(self.data_patchsize.split('_')[1]))
        yhat = sliding_window_inference(inputs=x,roi_size=roi_size,sw_batch_size=4,predictor=self.net,overlap=0.5,mode='constant')
        yhat = utils.Activation(yhat)
        loss = self.lossfn(yhat, y)
        metric = self.metric(torch.argmax(yhat,1).cpu().int().flatten(),y.cpu().int().flatten())

        self.log('loss_val', loss, prog_bar=True)
        self.log('metric_val', metric, prog_bar=True)
        self.logger.experiment.log({'image_val' : wb_mask(x, yhat, y)})
        return {'loss_val': loss}    

    def configure_optimizers(self):
        """
        mode : lr is given --> Adam with lr with given lr
        mode : lr is not given --> CosineAnnealingWarmup (default), SGD with varying lr
        """
        if self.lr != 0:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=100, min_lr=1e-7)    
        else:
            optimizer = torch.optim.SGD(self.net.parameters(), lr=1e-7, weight_decay = 0.0005, momentum=0.9)
            scheduler = utils.CosineAnnealingWarmUpRestarts(optimizer, T_0=100, T_mult=1, eta_max=0.01, T_up=10, gamma=0.5)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'loss_val'}
                }
    
    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = ArgumentParser(parents=[parent_parser])
        def none_or_str(value):
            if value == 'None':
                return None
            return value
        parser.add_argument("--project", type=str, help="wandb project name, this will set your wandb project")
        parser.add_argument("--data_dir", type=str, help="path where dataset is stored, subfolders name should be x_train, y_train")
        parser.add_argument("--data_module", type=str,default='dataset', help="Data Module, see datasets.py")
        parser.add_argument("--data_padsize", type=none_or_str, default=None, help="input like this (height_width) : pad - crop - resize - patch")
        parser.add_argument("--data_cropsize", type=none_or_str, default=None, help="input like this (height_width) : pad - crop - resize - patch")
        parser.add_argument("--data_resize", type=none_or_str, default=None, help="input like this (height_width) : pad - crop - resize - patch")
        parser.add_argument("--data_patchsize", type=none_or_str, default=None, help="input like this (height_width) : pad - crop - resize - patch: recommand (A * 2^n)")
        parser.add_argument("--batch_size", type=int, default=None, help="batch_size, if None, searching will be done")
        parser.add_argument("--lossfn", type=str, default='CE', help="class of the loss function[CELoss, DiceCELoss, MSE, ...], see losses.py")
        parser.add_argument("--net", type=str, default='unet_eb5_batch', help="Class of the Networks, see nets.py")
        parser.add_argument("--net_inputch", type=int, default=1, help='dimensions of network input channel')
        parser.add_argument("--net_outputch", type=int, default=2, help='dimensions of network output channel')          
        parser.add_argument("--net_norm", type=str, default='batch', help='net normalization, [batch,instance,group]')          
        parser.add_argument("--net_ckpt", type=str, default=None, help='path to checkpoint, ex) logs/[PROJECT]/[ID]')          
        parser.add_argument("--precision", type=int, default=32, help='amp will be set when 16 is given')
        parser.add_argument("--lr", type=float, default=0, help="Set learning rate of Adam optimzer.")        
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
                       num_workers: int = 4):

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
                                transform=datasets.augmentation_train())
        
        self.validset = fn_call(self.data_dir, 
                                'valid',
                                transform_spatial = datasets.augmentation_imagesize(data_padsize = self.data_padsize,
                                                                                 data_cropsize = self.data_cropsize,
                                                                                 data_resize = self.data_resize), 
                                transform=datasets.augmentation_valid())
        
        self.testset = fn_call(self.data_dir, 
                               'test',
                                transform_spatial = datasets.augmentation_imagesize(data_padsize = self.data_padsize,
                                                                                 data_cropsize = self.data_cropsize,
                                                                                 data_resize = self.data_resize), 
                                transform=datasets.augmentation_valid())
        
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


# wandb image visualization
segmentation_classes = ['black', 'class1', 'class2', 'class3', 'class4', 'class5']

def labels():
    l = {}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    return l

def wb_mask(x, yhat, y, samples=2):
    
    x = torchvision.utils.make_grid(x[:samples].cpu().detach(),normalize=True).permute(1,2,0)
    y = torchvision.utils.make_grid(y[:samples].cpu().detach()).permute(1,2,0)
    yhat = torchvision.utils.make_grid(torch.argmax(yhat[:samples],1).unsqueeze(1).cpu()).permute(1,2,0) if yhat.shape[1]>1 else \
           torchvision.utils.make_grid(yhat[:samples].round().cpu()).permute(1,2,0)

    x = (x*255).numpy().astype(np.uint8)        # 0 ~ 255
    yhat = yhat[...,0].numpy().astype(np.uint8) # 0 ~ n_class 
    y = y[...,0].numpy().astype(np.uint8)       # 0 ~ n_class
    
    return wandb.Image(x, masks={
    "prediction" : {"mask_data" : yhat, "class_labels" : labels()},
    "ground truth" : {"mask_data" : y, "class_labels" : labels()}})


def main(args: Namespace):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SegModel(**vars(args))
    if args.net_ckpt is not None:
        ckpt = natsorted(glob.glob(args.net_ckpt+'**/*.ckpt'))
        model = SegModel.load_from_checkpoint(checkpoint_path = ckpt[-1],**vars(args))
        print(ckpt[-1],'is loaded')
    assert args.project != None, "You should set wandb-logger project name by option --project [PROJECT_NAME]"
    print('project', args.project)
    
    # ------------------------
    # 2 SET LOGGER
    # ------------------------    
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor, StochasticWeightAveraging, LambdaCallback, EarlyStopping
    
    args.experiment_name = "Dataset{}_Net{}_Netinputch{}_Netoutputch{}_Loss{}_Lr{}_Precision{}_Patchsize{}_Prefix{}_"\
    .format(args.data_dir.split('/')[-1], args.net, args.net_inputch, args.net_outputch, args.lossfn, args.lr, args.precision,args.data_patchsize,args.experiment_name)
    print('Current Experiment:',args.experiment_name,'\n','*'*100)
    
    os.makedirs('logs',mode=0o777, exist_ok=True)
    wb_logger = pl_loggers.WandbLogger(save_dir='logs/', name=args.experiment_name, project=args.project, log_model = "all")
    wb_logger.log_hyperparams(args)
    wb_logger.watch(model,log="all", log_freq=1)
        
    Checkpoint_callback = ModelCheckpoint(verbose=True, 
                                          monitor='loss_val',
                                          mode='min',
#                                           monitor='metric_val',
#                                           mode='max',
                                          filename='{epoch:04d}-{loss_val:.4f}-{metric_val:.4f}',
                                          save_top_k=3,)
    
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer.from_argparse_args(args,
                                            amp_backend='native',
                                            auto_scale_batch_size='power',
                                            callbacks=[Checkpoint_callback,
                                                       LearningRateMonitor(),
                                                       StochasticWeightAveraging(),
                                                       EarlyStopping(monitor='loss_val',patience=200),
                                                      ],
                                            deterministic=True,
                                            gpus = -1,
                                            gradient_clip_val=0.5,
                                            log_gpu_memory='min_max',
                                            logger=wb_logger,
                                            max_epochs=2000,
                                            num_processes=4,
                                            stochastic_weight_avg=True,
                                            sync_batchnorm =True,
                                            weights_summary='top', 
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
    