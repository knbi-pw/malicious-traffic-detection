# malicious-traffic-detection

This is a project of members of KNBI of Warsaw University of Technology.
The project's aim is to train CNN/Deep Learning network models with web traffic and predict whether this traffic is malicious or not.

# Modules' description and example usage

Most of the parameters used to the execution of the scripts are stored in a .json file. 
An example .json file is in _/config directory.

Example CNN usage:

_preprocess_pcaps_cnn.py_: `python3 preprocess_pcaps_cnn.py -j config/config.json`

_train_cnn.py_: `python3 train_cnn.py -j  config/config.json`

_predict.py_: `python3 predict.py -j config/config.json`

_json_to_csv.py_: `python3 json_to_csv.py -j  config/config.json`
