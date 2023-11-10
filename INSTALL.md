# Installation

We provide installation instructions for all classification experiments here.

## Dependency Setup
Create an new conda virtual environment
```
conda create -n alpha-SVRG python=3.8 -y
conda activate alpha-SVRG
```

Install [Pytorch](https://pytorch.org/)>=1.13.1, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.14.1 following official instructions. For example:
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Clone this repo and install required packages:
```
git clone https://github.com/abcdef-12345-afk/alpha-SVRG
pip install timm==0.6.12 tensorboardX six
```

The results in the paper are produced with `torch==1.13.1+cu117 torchvision==0.14.1+cu117 timm==0.6.12`.

## Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```
All other small dataest will be downloaded automatically when running the code.