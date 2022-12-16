from model.hooknet_se_resnext101_32x4d  import hooknet_se_resnext101_32x4d
from model.hooknet                      import hooknet
from functional                         import train_epoch, test_epoch
from datagen                            import dataloader_setting
from datetime                           import date  
from config                             import *
import torch.optim                      as optim
import segmentation_models_pytorch      as smp
import torch
import os


def model_setting():
    ### Model Setting 
    if ENCODER_NAME == 'se_resnext101_32x4d':
        model = hooknet_se_resnext101_32x4d(class_num=CLASSES, hook_indice=INDICE)
        model = torch.nn.DataParallel(model, device_ids=[0,1])
        print('hooknet_se_resnext101_32x4d')
    else:
        model = hooknet(in_channels=3, class_num=CLASSES, hook_indice=INDICE)
    model = model.to(DEVICE)
    return model


def name_weight():
    mode = 'binary_class' if CLASSES==1 else 'multi_class'
    name = f'{mode}_{MODEL}_{ENCODER_NAME}_{LOSS}_{DESCRIPTION}_{date.today()}'
    weight_path = '/workspace/log/weight/pytorch'
    os.makedirs(weight_path, exist_ok = True)
    return f'{weight_path}/{name}.pth'


def train_dataset():
    # model setting
    model = model_setting()
    
    # dataloader setting
    train_loader, valid_loader, test_loader = dataloader_setting()

    # weight and log setting
    weight = name_weight()

    # loss, metrics, optimizer setting
    loss = getattr(smp.losses, LOSS)(mode=smp.losses.MULTILABEL_MODE, alpha=0.5, beta=0.5)
    metrics = [smp.utils.metrics.IoU(), smp.utils.metrics.Fscore(),]
    optimizer = getattr(optim, OPTIMIZER)(params=model.parameters(), lr=LR)

    max_score = 0
    for i in range(0, EPOCH):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch(DEVICE, train_loader, model, loss, metrics, optimizer)
        valid_logs = test_epoch(DEVICE, valid_loader, model, loss, metrics, 'valid')

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, weight)
            print('Model saved!')

    best_model = torch.load(weight)
    test_logs = test_epoch(DEVICE, test_loader, best_model, loss, metrics, 'test')


if __name__ == '__main__':
    train_dataset()