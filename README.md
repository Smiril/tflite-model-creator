## Intro

This project provides scripts inspired by [coral.ai](https://coral.ai) to setup and run on MacOS.

## Setup

**Clone and setup**

1. Clone this repository (`git clone https://github.com/Smiril/tflite-model-creator.git`)
2. Run script `./setup.sh` to create virtual env and install necessary Python dependencies. This may take several minutes to run.

Now you can put your `data_train.jpeg/jpg/png/bmp` and `data_validate.jpeg/jpg/png/bmp` files into the `image/` dir and start running the script.
For 100 species make in `image/train` and `image/validate` 100 sub dirs for each species once.


## Run the Code
```
python creator.py image --num_classes 100 --image_size 255 --batch_size 32 --epochs 1000 --output_model_path output_model.tflite
```
## Links

```
https://github.com/Smiril/coral-ai-edge-tpu-video-watcher.git
``
