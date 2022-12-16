# HookNet

This is **my personal implementation(with Torch) of HookNet**. I also summarize [HookNet paper](https://arxiv.org/pdf/2006.12230.pdf) in [my blog](https://biology-statistics-programming.tistory.com/154).

### HookNet Structure
> We propose HookNet, a semantic segmentation model for histopathology whole-slide images, which combines
context and details via multiple branches of encoder-decoder convolutional neural networks. Concentric patches at
multiple resolutions with different fields of view are used to feed different branches of HookNet, and intermediate
representations are combined via a *hooking* mechanism. - [Mart van Rijthoven. (2020)](https://arxiv.org/pdf/2006.12230.pdf)

<p align="center"><img src="https://user-images.githubusercontent.com/70703320/148633495-02263c48-67f8-4d2f-b7da-e18f83dad2cf.png" width="60%" height=30%"></p>

### Environment
```
pip install -r requirements.txt
```
Above, I install python 3.6 with CUDA 11.4

# Description

### Repository Structure
- `model/hooknet.py`: main HookNet [input_image_shape :: (284x284x3) x 2, input_mask_shape :: (70x70) x 1]
- `model/hooknet_se_resnext101_32x4d.py`: main HookNet_SE-ResNeXt101_32x4d model script [input_image_shape :: (512x512x3) x 2, input_mask_shape :: (512x512) x 2]
- `datagen.py`: the data dataloader and augmentation script
- `functional.py`: naming a weight of model and converting outputs to images script 
- `train.py`: main training script

### Training

##### Data Preparation
```
Hooknet
    ├ slide_num_1
    |       ├ input_x100
    |       ├ input_x50
    |       ├ input_x25
    |       ├ input_x12
    |       └ input_y1
    .
    .
    .
    └ slide_num_n
            ├ input_x100
            ├ input_x50
            ├ input_x25
            ├ input_x12
            └ input_y1    
```
if you want to train with hooknet(normal) or quad_scale_hooknet... (This model trains with valid padding.)
- input_x1: mpp=1 image patches(284x284x3) directory
- input_x2: mpp=2 image patches(284x284x3) directory
- input_x4: mpp=4 image patches(284x284x3) directory
- input_x8: mpp=8 image patches(284x284x3) directory
- input_y1: mpp=1 mask patches(70x70) directory 

if you want to train with hooknet(se_resnext101_32x4d)... (This model trains with same padding.)
- input_x1: mpp=1 image patches(512x512x3) directory
- input_x2: mpp=2 image patches(512x512x3) directory
- input_x4: mpp=4 image patches(512x512x3) directory
- input_x8: mpp=8 image patches(512x512x3) directory
- input_y1: mpp=1 mask patches(512x512) directory 

</br>

You can get this data structure by using [util_multi.py](https://github.com/CODiPAI-CMC/wsi_processing)

##### Train Example
```
python train.py 
```
You can adjust hyper parameters in config.py

# Reference

### paper
- [Model Structure](https://arxiv.org/pdf/2006.12230.pdf)
- [Encoder model structure](https://arxiv.org/pdf/1709.01507v4.pdf)

### code
- [Public implementation(with Tensorflow)](https://github.com/DIAGNijmegen/pathology-hooknet)
- [SE-ResNext101_32x4d](https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py)
- [Training](https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb)
