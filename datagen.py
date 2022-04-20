from torch.utils.data                         import Dataset
from sklearn.utils                            import shuffle
from albumentations                           import *
import numpy                                  as np
import torch
import glob
import cv2


class PathSplit:
    def __init__(self, base_path):
        self.base_path = glob.glob(base_path)

    def Path(self):
        self.TRAIN_TARGET_PATH = shuffle(self.base_path, random_state=321)[:int(0.60*len(self.base_path))]
        self.VALID_TARGET_PATH = shuffle(self.base_path, random_state=321)[int(0.60*len(self.base_path)):int(0.80*len(self.base_path))]
        self.TEST_TARGET_PATH = shuffle(self.base_path, random_state=321)[int(0.80*len(self.base_path)):]
        return self.TRAIN_TARGET_PATH, self.VALID_TARGET_PATH, self.TEST_TARGET_PATH

    def Split(self):
        raise NotImplementedError


class PathSplit_Hooknet(PathSplit):
    def __init__(self, base_path):
        super(PathSplit_Hooknet, self).__init__(
            base_path = base_path
        )

    def Split(self):
        # split path
        TRAIN_ZIP = shuffle(
        [('/'.join(x.split('/')[:-2])+'/input_x1/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x4/'+x.split('/')[-1], x) for x in self.Path()[0]],
            random_state=333
        )
        VALID_ZIP = shuffle(
        [('/'.join(x.split('/')[:-2])+'/input_x1/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x4/'+x.split('/')[-1], x) for x in self.Path()[1]],
            random_state=333
        )
        TEST_ZIP = shuffle(
        [('/'.join(x.split('/')[:-2])+'/input_x1/'+x.split('/')[-1], '/'.join(x.split('/')[:-2])+'/input_x4/'+x.split('/')[-1], x) for x in self.Path()[2]],
            random_state=333
        )
        return TRAIN_ZIP, VALID_ZIP, TEST_ZIP


class PathSplit_QuadScaleHooknet(PathSplit): 
    def __init__(self, base_path):
        super(PathSplit_QuadScaleHooknet, self).__init__(
            base_path = base_path
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




class PathToDataset:
    def __init__(self, path_list, image_size, mode):
        self.path_list = path_list
        self.image_size = image_size
        self.mask_size = (70,70) if image_size == 284 else (512,512)
        self.mode = mode

    def NumpyDataset(self):
        batch_x = np.zeros((len(self.path_list),) + (2,) + self.image_size + (3,), dtype='float32') if self.mode == 1 else np.zeros((len(self.path_list),) + (4,) + self.image_size + (3,), dtype='float32')
        batch_y = np.zeros((len(self.path_list),) + self.mask_size + (4,), dtype='float32')
        for j, path in enumerate(self.path_list):
            img_path1 = path[0]
            img_path2 = path[1]
            mask_path = path[2] 
            if self.mode == 2:
                img_path3 = path[2]
                img_path4 = path[3]
                mask_path = path[4]
                batch_x[j][2] = cv2.imread(img_path3).astype(np.float32)
                batch_x[j][3] = cv2.imread(img_path4).astype(np.float32)
            mask = cv2.imread(mask_path, 0)[107:177,107:177] if self.mask_size == (70,70) else cv2.imread(mask_path, 0)
            batch_x[j][0] = cv2.imread(img_path1).astype(np.float32)
            batch_x[j][1] = cv2.imread(img_path2).astype(np.float32)
            batch_y[j][:,:,0] = np.where(mask==0, 1, 0)
            batch_y[j][:,:,1] = np.where(mask==1, 1, 0)
            batch_y[j][:,:,2] = np.where(mask==2, 1, 0)
            batch_y[j][:,:,3] = np.where(mask==3, 1, 0)
        return batch_x, batch_y, self.path_list



class TensorData_scale1(Dataset):    
    def __init__(self, x_data, y_data, path_list, image_size, augmentation=None):
        self.x_data = x_data
        self.y_data = y_data
        self.path_list = path_list
        self.image_size = image_size
        self.len = self.y_data.shape[0]
        self.augmentation1 = multi_train_aug() if augmentation else multi_test_aug()
        
    def get_labels(self):
        label_list = []
        for path in self.path_list:
            label_list.append(path[1][-6:-4])
        return label_list
    
    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        sample = self.augmentation1(image=self.x_data[index][0], image1=self.x_data[index][1], mask=self.y_data[index])
        x_data_s = np.zeros((2,) + self.image_size + (3,), dtype='float32')
        x_data_s[0], x_data_s[1], y_data_s = sample['image'], sample['image1'], sample['mask']
        x_data_s = torch.FloatTensor(x_data_s)
        x_data_s = x_data_s.permute(0,3,1,2)
        y_data_s = torch.FloatTensor(y_data_s)
        y_data_s = y_data_s.permute(2,0,1)
        return x_data_s, y_data_s

class TensorData_scale2(Dataset):    
    def __init__(self, x_data, y_data, path_list, image_size, augmentation=None):
        self.x_data = x_data
        self.y_data = y_data
        self.path_list = path_list
        self.image_size = image_size
        self.len = self.y_data.shape[0]
        self.augmentation2 = quad_train_aug() if augmentation else quad_test_aug()
        
    def get_labels(self):
        label_list = []
        for path in self.path_list:
            label_list.append(path[1][-6:-4])
        return label_list
    
    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        sample = self.augmentation2(image=self.x_data[index][0], image1=self.x_data[index][1], image2=self.x_data[index][2], image3=self.x_data[index][3], mask=self.y_data[index])
        x_data_s = np.zeros((4,) + self.image_size + (3,), dtype='float32')
        x_data_s[0], x_data_s[1], x_data_s[2], x_data_s[3], y_data_s = sample['image'], sample['image1'], sample['image2'], sample['image3'], sample['mask']
        x_data_s = torch.FloatTensor(x_data_s)
        x_data_s = x_data_s.permute(0,3,1,2)
        y_data_s = torch.FloatTensor(y_data_s)
        y_data_s = y_data_s.permute(2,0,1)
        return x_data_s, y_data_s


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