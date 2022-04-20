# HookNet & Quad-scale HookNet

This is **my personal implementation(with Torch) of HookNet**. I also summarize [HookNet paper](https://arxiv.org/pdf/2006.12230.pdf) in [my blog](https://biology-statistics-programming.tistory.com/154).

### HookNet Structure
> We propose HookNet, a semantic segmentation model for histopathology whole-slide images, which combines
context and details via multiple branches of encoder-decoder convolutional neural networks. Concentric patches at
multiple resolutions with different fields of view are used to feed different branches of HookNet, and intermediate
representations are combined via a *hooking* mechanism. - [Mart van Rijthoven. (2020)](https://arxiv.org/pdf/2006.12230.pdf)

<p align="center"><img src="https://user-images.githubusercontent.com/70703320/148633495-02263c48-67f8-4d2f-b7da-e18f83dad2cf.png" width="60%" height=30%"></p>

Also, I tried to make Quad-scale HookNet, which is a new model that transformed HookNet.

### Quad-scale HookNet Structure
This model uses the same *hooking* mechanism as hooknet, but performs concat from 3 context structures with different mpps to 1 target structure.
<p align="center"><img src="https://user-images.githubusercontent.com/70703320/148635257-fbecc29d-d641-4ff3-af9e-d584604a1736.png" width="60%" height=30%"></p>



### Environment
```
pip install -r requirements.txt
```
Above, I install python 3.6 with CUDA 11.4

# Description

### Repository Structure
- `model/hooknet.py`: main HookNet and Quad-scale HookNet model script
- `model/hooknet_se_resnext101_32x4d.py`: main HookNet_SE-ResNeXt101_32x4d model script
- `datagen.py`: the data dataloader and augmentation script
- `functional.py`: naming a weight of model and converting outputs to images script 
- `test_view.py`: visualizing outputs script
- `train.py`: main training script

### Training

##### Data Preparation
```
HookNet_Data
    ├ slide_num_1
    |       ├ input_x1
    |       ├ input_x2
    |       ├ input_x4
    |       ├ input_x8
    |       └ input_y1
    .
    .
    .
    └ slide_num_n
            ├ input_x1
            ├ input_x2
            ├ input_x4
            ├ input_x8
            └ input_y1    
```
- input_x1: mpp=1 image patches(284x284x3) directory
- input_x2: mpp=2 image patches(284x284x3) directory
- input_x4: mpp=4 image patches(284x284x3) directory
- input_x8: mpp=8 image patches(284x284x3) directory
- input_y1: mpp=1 mask patches(284x284) directory (The shape of the mask patches will be 70x70 by cropping.)

</br>


##### Train Example
```
python train.py --BASE_PATH './HookNet_Data/*/input_y1/*.png' --INPUT_SHAPE 284 --CLASSES 3  --TARGET_INDICE 4 --CONTEXT_INDICE 2 --MODEL 'hooknet'--ENCODER 'normal' --LOSS_FUNCTION 'dice_loss' --DESCRIPTION 'HookNet_test'
```

##### Train Option
- `--BASE_PATH`: The path of input_y1 mask patches 
- `--BATCH_SIZE`: The batch size of training model.
- `--INPUT_SHAPE`: The input shape of the patch.
- `--CLASSES`: The number of output classes.
- `--TARGET_INDICE`: The number of target indice of Hooknet. It should be 5 > target > context > 1.
- `--CONTEXT_INDICE`: The number of context indice of HookNet. It should be 5 > target > context > 1.
- `--EPOCHS`: The epochs batch size of training model.
- `--MODEL`: Choose the model either hooknet or quad_scale_hooknet
- `--ENCODER`: Choose the encoder model either normal or se_resnext101_32x4d
- `--LOSS_FUNCTION`: Choose the loss function either celoss or diceloss
- `--DESCRIPTION`: Add the name of a training model weight.

# Reference

### paper
- [Model Structure](https://arxiv.org/pdf/2006.12230.pdf)
- [Encoder model structure](https://arxiv.org/pdf/1709.01507v4.pdf)

### code
- [Public implementation(with Tensorflow)](https://github.com/DIAGNijmegen/pathology-hooknet)
- [SE-ResNext101_32x4d](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py)
- [Training](https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb)
