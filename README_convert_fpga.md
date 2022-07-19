# How to convert trained model to FPGA?

## Install requirements
Setup environment (e.g. using conda) with python 3.6 tensorflow==1.10.0 and keras 2.1.6

## Train the model
E.g.:
    
    python train_cnn.py -j configs/train_cnn_config_csv.json --model_type novel

Make sure that method `model_save_json_h5` is used.
You should get three files as the result, e.g.: models/novel_20220719_174137.h5, models/novel_20220719_174137.json, which are needed for the next steps, and the last not used in the further process: models/novel_20220719_174137_default.h5

## Convert keras model to tensorflow 
Use `keras_2_tf.py` from: https://github.com/695kede/xilinx-edge-ai/tree/master/docs/Keras-freeze to convert the keras model to pure tf. For example:
    
    python keras_2_tf.py --keras_json=models/novel_20220719_165733.json --keras_hdf5=models/novel_20220719_165733.h5 --tfckpt=models/novel_20220719_165733.ckpt --tf_graph=models/novel_20220719_165733.pb

You need to provide: 
* *.json file, 
* .h5 file,

and as the results you will get:
* \*.ckpt\* files (do not worry about weird strings after ckpt),
* *.pb file. 


## Freezing the graph

Quantization using other environment where the DNNDK is installed:

 freeze_graph --input_graph=models/novel_20220717_113357.pb \
             --input_checkpoint=models/novel_20220717_113357.ckpt \
             --input_binary=true \
             --output_graph=./models/novel_20220717_113357_frozen_graph.pb \
             --output_node_names=dense_1/Softmax

Here you provide: 
* \*.ckpt\* files,
* *.pb file, 

and you get:
* *\_frozen_graph.pb file (frozen\_graph in the filename is just a name convention). 

Frozen graph should be evaluated, then quantized and finally converted to the DPU.