# Unsupervised Domain Adaptation for Semantic Segmentation with Latent Search

## Setup

git clone https://github.com/prinshul/UDA_Latent_Search_Segmentation.git

cd UDA_Latent_Search_Segmentation

conda create -y -n ltp python=3.10

conda activate ltp

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

pip install numpy==1.26.4 opencv-python-headless==4.10.0.84

## Data setup

cd synthia_cityscapes_deeplabv2_resnet101

Download SYNTHIA-RAND-CITYSCAPES from http://synthia-dataset.cvc.uab.cat/SYNTHIA_RAND_CITYSCAPES.rar

unrar x SYNTHIA_RAND_CITYSCAPES.rar 'RAND_CITYSCAPES/RGB/*' 'RAND_CITYSCAPES/GT/LABELS/*' synthia/

Download leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip from https://www.cityscapes-dataset.com/downloads/

unzip leftImg8bit_trainvaltest.zip -d cityscapes

unzip gtFine_trainvaltest.zip -d cityscapes

Download DeepLab_resnet_pretrained_init-f81d91e8.pth from https://drive.google.com/file/d/0BxhUwxvLPO7TVFJQU1dwbXhHdEk/view?usp=sharing&resourcekey=0-7UxnHrm5eDCyvz2G35aKgA

bash run.sh
