from xml.etree.ElementInclude import default_loader
from datagen                         import PathSplit_Hooknet, PathSplit_QuadScaleHooknet, PathToDataset, TensorData_scale1, TensorData_scale2
from model.hooknet                   import hooknet, quad_scale_hooknet
import segmentation_models_pytorch   as smp
import argparse
import torch

def test(BASE_PATH, CLASSES, INDICE, MODEL, INPUT_SHAPE, BATCH_SIZE, WEIGHT_PATH):
    ### Using GPU Device
    GPU = True
    device = "cuda" if GPU and torch.cuda.is_available() else "cpu"
    ### Model Setting && Path Setting
    if MODEL == 'hooknet':
        print('hooknet')
        Path = PathSplit_Hooknet(BASE_PATH, INDICE)
        _, _, TEST_ZIP = Path.Split()
        mode=1
        test_dataset = PathToDataset(TEST_ZIP, INPUT_SHAPE, CLASSES, mode)
        test_x, test_y, test_ph = test_dataset.NumpyDataset()
        test_data = TensorData_scale1(test_x, test_y, test_ph, INPUT_SHAPE)
    elif MODEL == 'quad_scale_hooknet':
        print('quad_scale_hooknet')
        Path = PathSplit_QuadScaleHooknet(BASE_PATH)
        _, _, TEST_ZIP = Path.Split()
        mode=2
        test_dataset = PathToDataset(TEST_ZIP, INPUT_SHAPE, CLASSES, mode)
        test_x, test_y, test_ph = test_dataset.NumpyDataset()
        test_data = TensorData_scale2(test_x, test_y, test_ph, INPUT_SHAPE) 

    ### Dataset, DataLoader Customizing               
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    ### Loss, Metrics, Optimizer Setting
    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(), smp.utils.metrics.Fscore(), smp.utils.metrics.Accuracy(), smp.utils.metrics.Recall(), smp.utils.metrics.Precision(),
    ]

    ### Test best saved model
    best_model = torch.load(WEIGHT_PATH)
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=device,
    )
    logs = test_epoch.run(test_loader)
    print(logs)

    return TEST_ZIP, MODEL, device, best_model, INPUT_SHAPE

if __name__ == '__main__':
    ### argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--BASE_PATH', required=True, help='Input the patch path. It should be like ../slideset/patientnum/mask/patches.png.')
    parser.add_argument('--CLASSES', default=4, required=True, choices=[3, 4],type=int, help='Input the class of the patches. It should be 1 or >2.')
    parser.add_argument('--TARGET_INDICE', default=4, choices=[1, 2, 3, 4], type=int, help='Input the target indice. It should be 5 > target > context > 1.')
    parser.add_argument('--CONTEXT_INDICE', default=2, choices=[1, 2, 3], type=int, help='Input the context indice. It should be 5 > target > context >1.')
    parser.add_argument('--MODEL', required=True, help='Input the model within hooknet or quad_scale_hooknet.')
    parser.add_argument('--INPUT_SHAPE', default=284, help='Input the input shape.')
    parser.add_argument('--BATCH_SIZE', default=8, help='Input the batch size.')
    parser.add_argument('--WEIGHT_PATH', required=True, help='Input the weight path.')
    args = parser.parse_args()

    ### argparse 
    BASE_PATH = args.BASE_PATH
    CLASSES = args.CLASSES
    INDICE = (int(args.TARGET_INDICE), int(args.CONTEXT_INDICE))
    MODEL = args.MODEL
    INPUT_SHAPE = (args.INPUT_SHAPE, args.INPUT_SHAPE)
    BATCH_SIZE = args.BATCH_SIZE
    WEIGHT_PATH = args.WEIGHT_PATH

    ### Let's test
    test(BASE_PATH, CLASSES, INDICE, MODEL, INPUT_SHAPE, BATCH_SIZE, WEIGHT_PATH)