# Mytorch for segmentation
## Key features
- pytorch-lighting based segmentation model with tricks
- easy to train segmentation
- support sweeps using wandb

# INSTALL
```
# git clone
git clone https://github.com/kevinkwshin/Mytorch/
%cd Mytorch
# install packages
# !apt updates
!pip install -r requirements.txt --user --upgrade #--quiet -U
```

# HOW TO USE 
## Training
```python 
usage: train.py [-h] [--project PROJECT] [--data_dir DATA_DIR]
                [--data_module DATA_MODULE] [--data_padsize DATA_PADSIZE]
                [--data_cropsize DATA_CROPSIZE] [--data_resize DATA_RESIZE]
                [--data_patchsize DATA_PATCHSIZE] [--batch_size BATCH_SIZE]
                [--lossfn LOSSFN] [--net_name NET_NAME]
                [--net_inputch NET_INPUTCH] [--net_outputch NET_OUTPUTCH]
                [--net_baysian NET_BAYSIAN] [--net_norm NET_NORM]
                [--net_ckpt NET_CKPT] [--net_encoder_name NET_ENCODER_NAME]
                [--precision PRECISION] [--lr LR]
                [--experiment_name EXPERIMENT_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --project PROJECT     wandb project name, this will set your wandb project
  --data_dir DATA_DIR   path where dataset is stored, subfolders name should
                        be x_train, y_train
  --data_module DATA_MODULE
                        Data Module, see datasets.py
  --data_padsize DATA_PADSIZE
                        input like this (height_width) : pad - crop - resize -
                        patch
  --data_cropsize DATA_CROPSIZE
                        input like this (height_width) : pad - crop - resize -
                        patch
  --data_resize DATA_RESIZE
                        input like this (height_width) : pad - crop - resize -
                        patch
  --data_patchsize DATA_PATCHSIZE
                        input like this (height_width) : pad - crop - resize -
                        patch: recommand (A * 2^n)
  --batch_size BATCH_SIZE
                        batch_size, if None, searching will be done
  --lossfn LOSSFN       class of the loss function[CELoss, DiceCELoss, MSE,
                        ...], see losses.py
  --net_name NET_NAME   Class of the Networks, see nets.py
  --net_inputch NET_INPUTCH
                        dimensions of network input channel
  --net_outputch NET_OUTPUTCH
                        dimensions of network output channel
  --net_baysian NET_BAYSIAN
                        Dropout in the Bottleneck
  --net_norm NET_NORM   net normalization, [batch,instance,group]
  --net_ckpt NET_CKPT   path to checkpoint, ex) logs/[PROJECT]/[ID]
  --net_encoder_name NET_ENCODER_NAME
                        encoder__name
  --precision PRECISION
                        amp will be set when 16 is given
  --lr LR               Set learning rate of Adam optimzer.
  --experiment_name EXPERIMENT_NAME
                        Postfix name of experiment```

## Testing
