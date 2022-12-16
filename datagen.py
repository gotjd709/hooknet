from albumentations                           import Compose, OneOf, HorizontalFlip, VerticalFlip
from torchsampler                             import ImbalancedDatasetSampler
from torch.utils.data                         import Dataset
from sklearn.utils                            import shuffle
from config                                   import *
import numpy                                  as np
import torch
import glob
import cv2


class PathSplit(object):
    def __init__(self, base_path, hook_indice, magnification):
        self.base_path = base_path
        self.train_target_path = shuffle(self.base_path, random_state=321)[:int(0.60*len(self.base_path))]
        self.valid_target_path = shuffle(self.base_path, random_state=321)[int(0.60*len(self.base_path)):int(0.80*len(self.base_path))]
        self.test_target_path = shuffle(self.base_path, random_state=321)[int(0.80*len(self.base_path)):]
        self.hook_indice   = hook_indice
        self.magnification = magnification

    def Split(self):
        # split path
        target = 100 if self.magnification == 100 else 50
        context = target//2 if (self.hook_indice[1] - self.hook_indice[0]) == 0 else target//4
        
        train_zip = shuffle(
        [('/'.join(x.split('/')[:-2])+f'/input_x{str(target)}/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+f'/input_x{str(context)}/'+x.split('/')[-1], x, '/'.join(x.split('/')[:-2])+f'/input_y{str(context)}/'+x.split('/')[-1]) for x in self.train_target_path],
            random_state=333
        )
        valid_zip = shuffle(
        [('/'.join(x.split('/')[:-2])+f'/input_x{str(target)}/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+f'/input_x{str(context)}/'+x.split('/')[-1], x, '/'.join(x.split('/')[:-2])+f'/input_y{str(context)}/'+x.split('/')[-1]) for x in self.valid_target_path],
            random_state=333
        )
        test_zip = shuffle(
        [('/'.join(x.split('/')[:-2])+f'/input_x{str(target)}/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+f'/input_x{str(context)}/'+x.split('/')[-1], x, '/'.join(x.split('/')[:-2])+f'/input_y{str(context)}/'+x.split('/')[-1]) for x in self.test_target_path],
            random_state=333
        )

        return train_zip, valid_zip, test_zip

class TensorData(Dataset):    
    def __init__(self, path_list, image_size, classes, augmentation=None):
        self.path_list = path_list
        self.image_size = image_size
        self.mask_size = (70,70) if image_size == (284,284) else (512,512)
        self.augmentation2 = train_aug() if augmentation else test_aug()
        self.classes = classes

    def get_labels(self):
        label_list = []
        for path in self.path_list:
            label_list.append(path[1].split('/')[-3])
        return label_list
    
    def __len__(self):
        return len(self.path_list)
        
    def __getitem__(self, index):
        batch_x = np.zeros((2,) + self.image_size + (3,), dtype='float32') 
        x_data = batch_x.copy()
        batch_y = np.zeros((2,) + self.mask_size + (int(self.classes),), dtype='uint8')
        y_data = batch_y.copy()
        img_path1, img_path2, mask_path1, mask_path2 = self.path_list[index]
        batch_x[0] = cv2.imread(img_path1)/255
        batch_x[1] = cv2.imread(img_path2)/255
        mask1 = cv2.imread(mask_path1, 0)
        mask2 = cv2.imread(mask_path2, 0)
        for i in range(int(self.classes)):
            batch_y[0,...,i] = np.where(mask1==i, 1, 0)
            batch_y[1,...,i] = np.where(mask2==i, 1, 0)
        sample = self.augmentation2(image=batch_x[0], image1=batch_x[1], mask=batch_y[0], mask1=batch_y[1]) 
        x_data[0], x_data[1], y_data[0], y_data[1] = sample['image'], sample['image1'], sample['mask'], sample['mask1']
        x_data = torch.FloatTensor(x_data)
        x_data = x_data.permute(0,3,1,2)
        y_data = torch.FloatTensor(y_data)
        y_data = y_data.permute(0,3,1,2)
        return x_data, y_data



def train_aug():
    ret = Compose(
        [
            OneOf([
                HorizontalFlip(p=1),
                VerticalFlip(p=1),
            ],p=0.667)
        ],
        additional_targets={'image1':'image', 'mask1':'mask'}
    )
    return ret

def test_aug():
    ret = Compose(
        [],
        additional_targets={'image1':'image', 'mask1':'mask'}
    )
    return ret


def dataloader_setting():
    # Path Setting
    path = PathSplit(glob.glob(BASE_PATH), INDICE, MAGNIFICATION)
    train_zip, valid_zip, test_zip = path.Split()
    train_data = TensorData(train_zip, INPUT_SHAPE, CLASSES, augmentation=True)
    valid_data = TensorData(valid_zip, INPUT_SHAPE, CLASSES)                  
    test_data = TensorData(test_zip, INPUT_SHAPE, CLASSES)
    ### DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        sampler = ImbalancedDatasetSampler(train_data) if SAMPLER else None,
        batch_size=BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKER,
        pin_memory = True,
        prefetch_factor = 2*NUM_WORKER
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKER,
        pin_memory = True,
        prefetch_factor = 2*NUM_WORKER
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle = False,
        num_workers = NUM_WORKER,
        pin_memory = True,
        prefetch_factor = 2*NUM_WORKER
    )
    return train_loader, valid_loader, test_loader