# Sign Detection

To train the model, use the output from the dataset generator under `datasets/bfmc-generated`. Train a model with main.py, and convert to `.onnx` using `convert.py`. Once this file exists, HAILO's docker container from the H8 HAT must be used to compile the model to `.hef`. The compilation is done with `compile.py`, which must be placed in the container, along with `config.json` in this repository.

8.4.21
