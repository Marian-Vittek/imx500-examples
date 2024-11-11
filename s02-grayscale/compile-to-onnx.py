import torch
from torch.utils.data import DataLoader, Dataset
from model_compression_toolkit.core.pytorch.utils import get_working_device
import model_compression_toolkit as mct

# import my "neural network" class
import theNeuralNetwork

# instantiate it into model
model = theNeuralNetwork.NeuralNetwork() 
#model.eval() # not sure what this was supposed to do

# Move to device
device = get_working_device()
model.to(device)


# mct seems to require a 'dataset', so create a dummy random dataset of 1 element
dataset = [torch.randn((3,224,224))]
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
def representative_dataset_gen2():
    ds_iter = iter(dataloader)
    for _ in range(1):
        yield [next(ds_iter)]
        


# Set IMX500 TPC
tpc = mct.get_target_platform_capabilities(fw_name="pytorch",
                                           target_platform_name='imx500',
                                           target_platform_version='v3')

# Perform post training quantization
quant_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=model,
                                                            representative_data_gen=representative_dataset_gen2,
                                                            target_platform_capabilities=tpc)


#export model to .onnx file
mct.exporter.pytorch_export_model(model=quant_model,
                                  save_model_path='./model.onnx',
                                  repr_dataset=representative_dataset_gen2)

