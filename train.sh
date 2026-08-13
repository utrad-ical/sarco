#!/bin/bash

mkdir -p ./temp_l1
mkdir -p ./temp_l3
mkdir -p ./train_temp

python ./bin/tools/listbuilder.py ./train_data/list.txt ./train_temp
cp ./train_temp/list_l1.txt ./temp_l1/list.txt
cp ./train_temp/list_l3.txt ./temp_l3/list.txt

# 1st
# python ./bin/lumbar.py --mode=train --volume_list=./temp_l1/list.txt --labelused=1 --model_file_name=model/l1_lumbar.ckpt --batch_size=40

# python ./bin/lumbar.py --mode=train --volume_list=./temp_l3/list.txt --labelused=1 --model_file_name=model/l3_lumbar.ckpt --batch_size=40

# 2nd
python ./bin/lumbarmodifier.py --mode=train --volume_list=./temp_l1/list.txt --labelused=1 --model_file_name=model/l1_modifier.ckpt --batch_size=60 --n_epoch 25

python ./bin/lumbarmodifier.py --mode=train --volume_list=./temp_l3/list.txt --labelused=1 --model_file_name=model/l3_modifier.ckpt --batch_size=60 --n_epoch 25

# 3rd
python ./bin/sarcosegmentation.py --mode=train --volume_list=./temp_l1/list.txt --labelused=1 --model_file_name=./model/l1_segmentation.ckpt --batch_size=40

python ./bin/sarcosegmentation.py --mode=train --volume_list=./temp_l3/list.txt --labelused=1 --model_file_name=./model/l3_segmentation.ckpt --batch_size=40

