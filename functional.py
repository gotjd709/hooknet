from datetime                     import date
import matplotlib.pyplot          as plt
import numpy                      as np
import torch
import cv2
import os

def name_weight(frame=None, model=None, classes=None, description=None):
    mode = 'binary_class' if classes==1 else 'multi_class'
    if frame == 'tensorflow' and model != None:
        try:
            os.makedirs('../log/tensorflow', exist_ok = True)
            return '../log/tensorflow/' + f'{mode}_{model}_{description}_{date.today()}.h5'
        except:
            return '../log/tensorflow/' + f'{mode}_{model}_{description}_{date.today()}.h5'
    elif frame == 'pytorch' and model != None:
        try:
            os.makedirs('../log/pytorch', exist_ok = True)
            return '../log/pytorch/' + f'{mode}_{model}_{description}_{date.today()}.pth'
        except:
            return '../log/pytorch/' + f'{mode}_{model}_{description}_{date.today()}.pth'
    else:
        raise NameError('Check frame or model. frame should be tensorflow or pytorch. model should be model`s name.')

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, vmin=0, vmax=3, cmap='Oranges')
    plt.show()

def predict(TEST_ZIP, MODEL, device, best_model, INPUT_SHAPE):
    for i in range(len(TEST_ZIP)):
        
        image_vis = cv2.imread(TEST_ZIP[i][0]).astype('uint8')
        
        if MODEL == 'hooknet':
            image_batch = np.zeros((2,) + INPUT_SHAPE + (3,), dtype='float32')
            image_batch[0,...] = cv2.imread(TEST_ZIP[i][0]).astype(np.float32) 
            image_batch[1,...] = cv2.imread(TEST_ZIP[i][1]).astype(np.float32)
            gt_mask = cv2.imread(TEST_ZIP[i][2], 0)
            image_batch = torch.from_numpy(image_batch).float().to(device).unsqueeze(0).permute(0,1,4,2,3)
        elif MODEL == 'quad_scale_hooknet':
            image_batch = np.zeros((4,) + INPUT_SHAPE + (3,), dtype='float32')
            image_batch[0,...] = cv2.imread(TEST_ZIP[i][0]).astype(np.float32) 
            image_batch[1,...] = cv2.imread(TEST_ZIP[i][1]).astype(np.float32) 
            image_batch[2,...] = cv2.imread(TEST_ZIP[i][2]).astype(np.float32)
            image_batch[3,...] = cv2.imread(TEST_ZIP[i][3]).astype(np.float32)
            gt_mask = cv2.imread(TEST_ZIP[i][4], 0)           
            image_batch = torch.from_numpy(image_batch).float().to(device).unsqueeze(0).permute(0,1,4,2,3)
        pr_mask = best_model(image_batch)
        pr_mask = pr_mask.squeeze().permute(1,2,0).cpu().detach().numpy().round()
        pr_mask = np.argmax(pr_mask, axis=2)

        visualize(
            image=image_vis, 
            ground_truth_mask=gt_mask, 
            predicted_mask=pr_mask
        )