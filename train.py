from datagen                            import PathSplit_Hooknet, PathSplit_QuadScaleHooknet, PathToDataset, TensorData_scale1, TensorData_scale2
from functional                         import name_weight
from torchsampler                       import ImbalancedDatasetSampler
from torch.utils.tensorboard            import SummaryWriter
from model.hooknet                      import hooknet, quad_scale_hooknet
from model.hooknet_se_resnext101_32x4d  import hooknet_se_resnext101_32x4d  
import segmentation_models_pytorch      as smp
import matplotlib.pyplot                as plt
import pandas                           as pd
import numpy                            as np
import argparse
import torch
import tqdm
import glob
import os
import cv2

def train(BASE_PATH, BATCH_SIZE, INPUT_SHAPE, CLASSES, INDICE, EPOCHS, MODEL, ENCODER, LOSS_FUNCTION, DESCRIPTION):
    ### Using GPU Device
    GPU = True
    device = "cuda" if GPU and torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')

    ### Model Setting
    if MODEL == 'hooknet':
        if ENCODER == 'normal':
            model = hooknet(in_channels=3, class_num=CLASSES, hook_indice=INDICE)
            print('hooknet')
        elif ENCODER == 'se_resnext101_32x4d':
            model = hooknet_se_resnext101_32x4d(class_num=CLASSES, hook_indice=INDICE)
            print('hooknet_se_resnext101_32x4d')
        else:
            raise NameError('Choose between normal and se_resnext101_32x4d.')
        ### Path Setting
        mode = 1
        Path = PathSplit_Hooknet(BASE_PATH, INDICE)
        TRAIN_ZIP, VALID_ZIP, TEST_ZIP = Path.Split()
        ### Dataset, DataLoader Customizing
        train_dataset = PathToDataset(TRAIN_ZIP, INPUT_SHAPE, CLASSES, mode)
        valid_dataset = PathToDataset(VALID_ZIP, INPUT_SHAPE, CLASSES, mode)
        test_dataset = PathToDataset(TEST_ZIP, INPUT_SHAPE, CLASSES, mode)
        ### Dataset
        train_x, train_y, train_ph = train_dataset.NumpyDataset()
        train_data = TensorData_scale1(train_x, train_y, train_ph, INPUT_SHAPE, augmentation=True)
        valid_x, valid_y, valid_ph = valid_dataset.NumpyDataset()
        valid_data = TensorData_scale1(valid_x, valid_y, valid_ph, INPUT_SHAPE)                  
        test_x, test_y, test_ph = test_dataset.NumpyDataset()
        test_data = TensorData_scale1(test_x, test_y, test_ph, INPUT_SHAPE)
    elif MODEL == 'quad_scale_hooknet':
        model = quad_scale_hooknet(in_channels=3, class_num=CLASSES)
        print('quad_scale_hooknet')
        ### Path Setting
        mode = 2
        Path = PathSplit_QuadScaleHooknet(BASE_PATH, INDICE)
        TRAIN_ZIP, VALID_ZIP, TEST_ZIP = Path.Split()
        ### Dataset, DataLoader Customizing
        train_dataset = PathToDataset(TRAIN_ZIP, INPUT_SHAPE, CLASSES, mode)
        valid_dataset = PathToDataset(VALID_ZIP, INPUT_SHAPE, CLASSES, mode)
        test_dataset = PathToDataset(TEST_ZIP, INPUT_SHAPE, CLASSES, mode)
        ### Dataset
        train_x, train_y, train_ph = train_dataset.NumpyDataset()
        train_data = TensorData_scale2(train_x, train_y, train_ph, INPUT_SHAPE, augmentation=True)
        valid_x, valid_y, valid_ph = valid_dataset.NumpyDataset()
        valid_data = TensorData_scale2(valid_x, valid_y, valid_ph, INPUT_SHAPE)                  
        test_x, test_y, test_ph = test_dataset.NumpyDataset()
        test_data = TensorData_scale2(test_x, test_y, test_ph, INPUT_SHAPE)
    else: 
        raise NameError('Choose between hooknet and quad_scale_hooknet.')


    model.cuda()

    ### DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        sampler=ImbalancedDatasetSampler(train_data),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Loss, Metrics, Optimizer Setting
    loss = smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(), smp.utils.metrics.Fscore(), smp.utils.metrics.Accuracy(), smp.utils.metrics.Recall(), smp.utils.metrics.Precision(),
    ]
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )

    weight = name_weight(frame='pytorch', model=MODEL, classes=CLASSES, description=DESCRIPTION)
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    ### Using Tensorboard
    os.makedirs('../log/tensorboard', exist_ok = True)
    log_dir = f'../log/tensorboard/{MODEL}_{ENCODER}_{CLASSES}_{LOSS_FUNCTION}_{DESCRIPTION}'
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=DESCRIPTION)
    writer.add_graph(model, images.cuda())

    max_score = 0
    for i in range(0, EPOCHS):    
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        writer.add_scalars('Loss', {'train_loss':train_logs['dice_loss'],
                            'valid_loss':valid_logs['dice_loss']}, i)
        writer.add_scalars('IoU', {'train_loss':train_logs['iou_score'],
                                    'valid_loss':valid_logs['iou_score']}, i)
        writer.add_scalars('Fscore', {'train_loss':train_logs['fscore'],
                                    'valid_loss':valid_logs['fscore']}, i)
        
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, weight)
            print('Model saved!')        
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    ### Summary writer closing
    # writer.close()

    ### Test best saved model
    best_model = torch.load(weight)
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=device,
    )
    logs = test_epoch.run(test_loader)


if __name__ == '__main__':
    ### argparse setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--BASE_PATH', required=True, help='Input the patch path. It should be like ../slideset/patientnum/mask/patches.png.')
    parser.add_argument('--BATCH_SIZE', default=8, type=int, help='Input the batch size.')
    parser.add_argument('--INPUT_SHAPE', required=True, choices=[284, 512], type=int, help='Input the image shape. It should be like 512.')
    parser.add_argument('--CLASSES', default=4, required=True, choices=[3, 4],type=int, help='Input the class of the patches. It should be 1 or >2.')
    parser.add_argument('--TARGET_INDICE', default=4, choices=[1, 2, 3, 4], type=int, help='Input the target indice. It should be 5 > target > context > 1.')
    parser.add_argument('--CONTEXT_INDICE', default=2, choices=[1, 2, 3], type=int, help='Input the context indice. It should be 5 > target > context >1.')
    parser.add_argument('--EPOCHS', default=50, type=int, help='Input the epoches.')
    parser.add_argument('--MODEL', required=True, choices=['hooknet', 'quad_scale_hooknet'], type=str, help='Input the model(hooknet or quad_scale_hooknet).')
    parser.add_argument('--ENCODER', required=True, choices=['normal', 'se_resnext101_32x4d'], type=str, help='Input the encoder model(backbone model). It should be normal or seresnext101_32x4d')
    parser.add_argument('--LOSS_FUNCTION', default='celoss', type=str, help='Input the loss function for model. It should be celoss or diceloss.')
    parser.add_argument('--DESCRIPTION', required=True, help='Input the description of the training. It will tagged on model weight and tensorboard log')
    args = parser.parse_args()

    ### argparse 
    BASE_PATH = args.BASE_PATH
    BATCH_SIZE = args.BATCH_SIZE
    INPUT_SHAPE = (args.INPUT_SHAPE, args.INPUT_SHAPE)
    CLASSES = args.CLASSES
    INDICE = (args.TARGET_INDICE, args.CONTEXT_INDICE)
    EPOCHS = args.EPOCHS
    MODEL = args.MODEL
    ENCODER = args.ENCODER
    LOSS_FUNCTION = args.LOSS_FUNCTION
    DESCRIPTION = args.DESCRIPTION

    train(BASE_PATH, BATCH_SIZE, INPUT_SHAPE, CLASSES, INDICE, EPOCHS, MODEL, ENCODER, LOSS_FUNCTION, DESCRIPTION)