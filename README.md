# mmcv
mmcv


train
single gpu
    python tools/train.py configs/yolact/yolact_r101_1x8_food.py 

multi gpu
    ./tools/dist_train.sh configs/yolact/yolact_r101_1x8_food.py 1

use custom_data

1. change config file
2. change 'env lib site package mmdet dataset class' 
