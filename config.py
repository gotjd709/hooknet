import torch

MODEL = 'HookNet'
#ENCODER_NAME = 'se_resnext101_32x4d'
ENCODER_NAME = 'vgg16'
CLASSES = 6
BASE_PATH = '/data/AGGC_2022/Subset3_Train_9_Zeiss/input_y100/*.png'
#BASE_PATH = '/data/AGGC_2022/multi_tissue30_tumor05/Subset3_Train_9_Zeiss/input_y100/*.png'
INPUT_SHAPE = (284,284)
#INPUT_SHAPE = (512,512)
SAMPLER = None
BATCH_SIZE = 4
NUM_WORKER = 4
LOSS = 'TverskyLoss'
DESCRIPTION = 'test'
LR = 1e-4
OPTIMIZER = 'Adam'
EPOCH = 100
INDICE = (1, 2)
MAGNIFICATION = 100

GPU = True
DEVICE = "cuda" if GPU and torch.cuda.is_available() else "cpu"