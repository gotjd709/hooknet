from fileinput import hook_compressed
from torch.utils.data                         import Dataset
from sklearn.utils                            import shuffle
from albumentations                           import *
import numpy                                  as np
import torch
import glob
import cv2


class PathSplit:
    def __init__(self, base_path, hook_indice):
        self.base_path = glob.glob(base_path)
        self.hook_indice = hook_indice
    def Path(self):
        self.TRAIN_TARGET_PATH = shuffle(self.base_path, random_state=321)[:int(0.60*len(self.base_path))]
        self.VALID_TARGET_PATH = shuffle(self.base_path, random_state=321)[int(0.60*len(self.base_path)):int(0.80*len(self.base_path))]
        self.TEST_TARGET_PATH = shuffle(self.base_path, random_state=321)[int(0.80*len(self.base_path)):]
        return self.TRAIN_TARGET_PATH, self.VALID_TARGET_PATH, self.TEST_TARGET_PATH

    def Split(self):
        raise NotImplementedError


class PathSplit_Hooknet(PathSplit):
    def __init__(self, base_path, hook_indice):
        super(PathSplit_Hooknet, self).__init__(
            base_path = base_path,
            hook_indice = hook_indice
        )

    def Split(self):
        # split path
        target_indice = 2 ** (4-int(self.hook_indice[0])) 
        context_indice = 2 ** (4-int(self.hook_indice[1])) 
        TRAIN_ZIP = shuffle(
        [('/'.join(x.split('/')[:-2])+f'/input_x{target_indice}/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+f'/input_x{context_indice}/'+x.split('/')[-1], x) for x in self.Path()[0]],
            random_state=333
        )
        VALID_ZIP = shuffle(
        [('/'.join(x.split('/')[:-2])+f'/input_x{target_indice}/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+f'/input_x{context_indice}/'+x.split('/')[-1], x) for x in self.Path()[1]],
            random_state=333
        )
        TEST_ZIP = shuffle(
        [('/'.join(x.split('/')[:-2])+f'/input_x{target_indice}/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+f'/input_x{context_indice}/'+x.split('/')[-1], x) for x in self.Path()[2]],
            random_state=333
        )
        return TRAIN_ZIP, VALID_ZIP, TEST_ZIP


class PathSplit_QuadScaleHooknet(PathSplit): 
    def __init__(self, base_path, hook_indice):
        super(PathSplit_QuadScaleHooknet, self).__init__(
            base_path = base_path,
            hook_indice = hook_indice
        )

    def Split(self):
        TRAIN_ZIP = shuffle(
        [('/'.join(x.split('/')[:-2])+'/input_x1/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x2/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x4/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x8/'+x.split('/')[-1], x) for x in self.Path()[0]],
            random_state=333
        )
        VALID_ZIP = shuffle(
        [('/'.join(x.split('/')[:-2])+'/input_x1/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x2/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x4/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x8/'+x.split('/')[-1], x) for x in self.Path()[1]],
            random_state=333
        )
        TEST_ZIP = shuffle(
        [('/'.join(x.split('/')[:-2])+'/input_x1/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x2/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x4/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x8/'+x.split('/')[-1], x) for x in self.Path()[2]],
            random_state=333
        )
        return TRAIN_ZIP, VALID_ZIP, TEST_ZIP


class TensorData_scale1(Dataset):    
    def __init__(self, path_list, image_size, classes, augmentation=None):
        self.path_list = path_list
        self.image_size = image_size
        self.mask_size = (70,70) if image_size == (284,284) else (512,512)
        self.augmentation1 = multi_train_aug() if augmentation else multi_test_aug()
        self.classes = classes

    def get_labels(self):
        label_list = []
        for path in self.path_list:
            label_list.append(path[1][-6:-4])
        return label_list
    
    def __len__(self):
        return len(self.path_list)
        
    def __getitem__(self, index):
        batch_x = np.zeros((2,) + self.image_size + (3,), dtype='float32') 
        x_data = batch_x.copy()
        batch_y = np.zeros(self.mask_size + (int(self.classes),), dtype='float32')
        img_path1, img_path2, mask_path = self.path_list[index]
        batch_x[0] = cv2.imread(img_path1)
        batch_x[1] = cv2.imread(img_path2)
        mask = cv2.imread(mask_path, 0) 
        for i in range(int(self.classes)):
            batch_y[...,i] = np.where(mask==i, 1, 0)
        sample = self.augmentation1(image=batch_x[0], image1=batch_x[1], mask=batch_y) 
        x_data[0], x_data[1], y_data = sample['image'], sample['image1'], sample['mask']
        x_data = torch.FloatTensor(x_data)
        x_data = x_data.permute(0,3,1,2)
        y_data = torch.FloatTensor(y_data)
        y_data = y_data.permute(2,0,1)
        return x_data, y_data


class TensorData_scale2(Dataset):    
    def __init__(self, path_list, image_size, classes, augmentation=None):
        self.path_list = path_list
        self.image_size = image_size
        self.mask_size = (70,70) 
        self.augmentation2 = quad_train_aug() if augmentation else quad_test_aug()
        self.classes = classes

    def get_labels(self):
        label_list = []
        for path in self.path_list:
            label_list.append(path[1][-6:-4])
        return label_list
    
    def __len__(self):
        return len(self.path_list)
        
    def __getitem__(self, index):
        batch_x = np.zeros((4,) + self.image_size + (3,), dtype='float32')
        x_data = batch_x.copy()
        batch_y = np.zeros(self.mask_size + (int(self.classes),), dtype='float32')
        img_path1, img_path2, img_path3, img_path4, mask_path = self.path_list[index]
        batch_x[0] = cv2.imread(img_path1)
        batch_x[1] = cv2.imread(img_path2)
        batch_x[2] = cv2.imread(img_path3)
        batch_x[3] = cv2.imread(img_path4)
        mask = cv2.imread(mask_path, 0) 
        for i in range(int(self.classes)):
            batch_y[...,i] = np.where(mask==i, 1, 0)
        sample = self.augmentation2(image=batch_x[0], image1=batch_x[1], image2=batch_x[2], image3=batch_x[3], mask=batch_y)
        x_data[0], x_data[1], x_data[2], x_data[3], y_data = sample['image'], sample['image1'], sample['image2'], sample['image3'], sample['mask']
        x_data = torch.FloatTensor(x_data)
        x_data = x_data.permute(0,3,1,2)
        y_data = torch.FloatTensor(y_data)
        y_data = y_data.permute(2,0,1)
        return x_data, y_data


def multi_train_aug():
    ret = Compose(
        [
            OneOf([
                HorizontalFlip(p=1),
                VerticalFlip(p=1),
                ShiftScaleRotate(p=1)
            ],p=0.75)
        ],
        additional_targets={'image1':'image'}
    )
    return ret

def multi_test_aug():
    ret = Compose(
        [],
        additional_targets={'image1':'image'}
    )
    return ret

def quad_train_aug():
    ret = Compose(
        [
            OneOf([
                HorizontalFlip(p=1),
                VerticalFlip(p=1),
                ShiftScaleRotate(p=1)
            ],p=0.75)
        ],
        additional_targets={'image1':'image', 'image2':'image', 'image3':'image'}
    )
    return ret

def quad_test_aug():
    ret = Compose(
        [],
        additional_targets={'image1':'image', 'image2':'image', 'image3':'image'}
    )
    return ret