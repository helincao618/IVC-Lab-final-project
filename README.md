# IVC-Lab-final-project
Image Quality Improvement Based on GAN
## Install
The latest codes are tested on Ubuntu 16.04, CUDA10.0, PyTorch 1.4 and Python 3.8:
```shell
conda install pytorch==1.4.0 cudatoolkit=10.0 -c pytorch
```
## Data and Model
Download data in [BaiduDisk](https://pan.baidu.com/s/1zFpuZ7sMO6jEN2jXcmkhLg) with extract code 9p8j.

Download model in [BaiduDisk](https://pan.baidu.com/s/1QspiFHhonanihUe0Ja-66w ) with extract code t3mk
## Train and Test
### Train
```shell
python train.py
```
After trained in all epoch, the model will be saved in the same path.
### Test
```shell
python test.py
```
